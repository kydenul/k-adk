package postgres

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/bytedance/sonic"
	discardlog "github.com/kydenul/k-adk/internal/discard_log"
	"github.com/kydenul/log"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/genai"

	_ "github.com/lib/pq" // PostgreSQL driver
)

// EmbeddingModel is an interface for generating embeddings from text.
type EmbeddingModel interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	Dimension() int
}

var _ memory.Service = (*PostgresMemoryService)(nil)

// PostgresMemoryService implements memory.Service using PostgresSQL with pgvector
type PostgresMemoryService struct {
	db             *sql.DB
	logger         log.Logger
	embeddingModel EmbeddingModel
	embeddingDim   int
}

// PgMemSvrConfig holds configuration for PostgresMemoryService.
type PgMemSvrConfig struct {
	// ConnString is the PostgreSQL connection string
	// e.g., "postgres://user:pass@localhost:5432/dbname?sslmode=disable"
	ConnStr string

	// EmbeddingModel is used to generate embeddings for semantic search (optional)
	EmbeddingModel EmbeddingModel

	// Optional. Falls back to DiscardLog if nil.
	Logger log.Logger
}

// NewPostgresMemoryService creates a new PostgreSQL-backed memory service.
func NewPostgresMemoryService(
	ctx context.Context,
	cfg PgMemSvrConfig,
) (*PostgresMemoryService, error) {
	// NOTE: Set Logger
	if cfg.Logger == nil {
		cfg.Logger = &discardlog.DiscardLog{}
	}

	// NOTE: Open and connect to PostgresSQL
	db, err := sql.Open("postgres", cfg.ConnStr)
	if err != nil {
		cfg.Logger.Errorf("failed to open database: %v", err)
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	if err := db.PingContext(ctx); err != nil {
		cfg.Logger.Errorf("failed to connect to database: %v", err)
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// NOTE: Embedding Dimension
	embeddingDim := 0
	if cfg.EmbeddingModel != nil {
		embeddingDim = cfg.EmbeddingModel.Dimension()

		if embeddingDim == 0 {
			embedding, err := cfg.EmbeddingModel.Embed(ctx, "dimension probe")
			if err != nil {
				cfg.Logger.Errorf("failed to probe embedding dimension: %v", err)
				return nil, fmt.Errorf("failed to probe embedding dimension: %w", err)
			}

			embeddingDim = len(embedding)
		}
	}

	svc := &PostgresMemoryService{
		db:             db,
		embeddingModel: cfg.EmbeddingModel,
		embeddingDim:   embeddingDim,
		logger:         cfg.Logger,
	}

	if err := svc.initSchema(ctx); err != nil {
		svc.logger.Errorf("failed to initialize schema: %v", err)
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	svc.logger.Info("PostgresMemoryService initialized")

	return svc, nil
}

// initSchema creates the necessary tables and extensions.
func (s *PostgresMemoryService) initSchema(ctx context.Context) error {
	// Base schema without vector column
	baseSchema := `
		-- Memory entries table
		CREATE TABLE IF NOT EXISTS memory_entries (
			id SERIAL PRIMARY KEY,
			app_name VARCHAR(255) NOT NULL,
			user_id VARCHAR(255) NOT NULL,
			session_id VARCHAR(255) NOT NULL,
			event_id VARCHAR(255) NOT NULL,
			author VARCHAR(255),
			content JSONB NOT NULL,
			content_text TEXT NOT NULL,
			timestamp TIMESTAMPTZ NOT NULL,
			created_at TIMESTAMPTZ DEFAULT NOW(),
			UNIQUE(app_name, user_id, session_id, event_id)
		);

		-- Indexes for efficient querying
		CREATE INDEX IF NOT EXISTS idx_memory_app_user ON memory_entries(app_name, user_id);
		CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_entries(session_id);
		CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory_entries(timestamp);
		CREATE INDEX IF NOT EXISTS idx_memory_content_text ON memory_entries USING gin(to_tsvector('english', content_text));
	`

	if _, err := s.db.ExecContext(ctx, baseSchema); err != nil {
		s.logger.Errorf("failed to create base schema: %v", err)
		return fmt.Errorf("failed to create base schema: %w", err)
	}

	// Add vector column if embedding model is configured
	if s.embeddingDim > 0 {
		vectorSchema := fmt.Sprintf(`
			-- Enable pgvector extension
			CREATE EXTENSION IF NOT EXISTS vector;

			-- Add embedding column if not exists
			DO $$
			BEGIN
				IF NOT EXISTS (
					SELECT 1 FROM information_schema.columns
					WHERE table_name = 'memory_entries' AND column_name = 'embedding'
				) THEN
					ALTER TABLE memory_entries ADD COLUMN embedding vector(%d);
				END IF;
			END $$;

			-- Vector similarity index (IVFFlat for approximate nearest neighbor)
			DO $$
			BEGIN
				IF NOT EXISTS (
					SELECT 1 FROM pg_indexes WHERE indexname = 'idx_memory_embedding'
				) THEN
					CREATE INDEX idx_memory_embedding ON memory_entries
					USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
				END IF;
			END $$;
		`, s.embeddingDim)

		if _, err := s.db.ExecContext(ctx, vectorSchema); err != nil {
			s.logger.Errorf("failed to create vector schema: %v", err)
			return fmt.Errorf("failed to create vector schema: %w", err)
		}
	}

	return nil
}

// AddSession extracts memory entries from a session and stores them.
func (s *PostgresMemoryService) AddSession(ctx context.Context, sess session.Session) error {
	events := sess.Events()
	if events == nil || events.Len() == 0 {
		s.logger.Warn("no events found in session")
		return nil
	}

	s.logger.Debugf("adding session to memory: app=%s, user=%s, session=%s, events=%d",
		sess.AppName(), sess.UserID(), sess.ID(), events.Len())

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		s.logger.Errorf("failed to begin transaction: %v", err)
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	// Prepare statement base on whether we have embeddings
	var stmt *sql.Stmt
	if s.embeddingModel != nil {
		stmt, err = tx.PrepareContext(ctx, `
			INSERT INTO memory_entries
			(app_name, user_id, session_id, event_id, author, content, content_text, embedding, timestamp)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
			ON CONFLICT (app_name, user_id, session_id, event_id) DO UPDATE
			SET content = EXCLUDED.content, content_text = EXCLUDED.content_text, embedding = EXCLUDED.embedding
		`)
	} else {
		stmt, err = tx.PrepareContext(ctx, `
			INSERT INTO memory_entries (app_name, user_id, session_id, event_id, author, content, content_text, timestamp)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
			ON CONFLICT (app_name, user_id, session_id, event_id) DO UPDATE
			SET content = EXCLUDED.content, content_text = EXCLUDED.content_text
		`)
	}

	if err != nil {
		s.logger.Errorf("failed to prepare statement: %v", err)
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer func() { _ = stmt.Close() }()

	insertedCount := 0
	skippedCount := 0
	errorCount := 0

	for event := range events.All() {
		if event.Content == nil || len(event.Content.Parts) == 0 {
			skippedCount++
			continue
		}

		// Extract text content
		text := extractTextFromContent(event.Content)
		if text == "" {
			skippedCount++
			continue
		}

		// Serialize content to JSON
		contentJSON, err := sonic.Marshal(event.Content)
		if err != nil {
			errorCount++
			continue
		}

		timestamp := event.Timestamp
		if timestamp.IsZero() {
			timestamp = time.Now()
		}

		eventID := event.ID
		if eventID == "" {
			eventID = fmt.Sprintf("%s-%d", event.InvocationID, timestamp.UnixNano())
		}

		if s.embeddingModel != nil {
			// Generate embedding
			var embeddingStr *string
			embedding, embErr := s.embeddingModel.Embed(ctx, text)
			if embErr == nil && len(embedding) > 0 {
				embStr := vectorToString(embedding)
				embeddingStr = &embStr
			} else if embErr != nil {
				s.logger.Debugf("failed to generate embedding for event %s: %v", eventID, embErr)
			}

			_, err = stmt.ExecContext(ctx,
				sess.AppName(),
				sess.UserID(),
				sess.ID(),
				eventID,
				event.Author,
				contentJSON,
				text,
				embeddingStr,
				timestamp,
			)
		} else {
			_, err = stmt.ExecContext(ctx,
				sess.AppName(),
				sess.UserID(),
				sess.ID(),
				eventID,
				event.Author,
				contentJSON,
				text,
				timestamp,
			)
		}
		if err != nil {
			// Log but continue with other events
			s.logger.Errorf("failed to insert memory entry: %v", err)
			errorCount++
			continue
		}
		insertedCount++
	}

	if err := tx.Commit(); err != nil {
		s.logger.Errorf("failed to commit transaction: %v", err)
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	s.logger.Infof("session added to memory: session=%s, inserted=%d, skipped=%d, errors=%d",
		sess.ID(), insertedCount, skippedCount, errorCount)

	return nil
}

// Close closes the database connection.
func (s *PostgresMemoryService) Close() error { return s.db.Close() }

// DB returns the underlying database connection for testing purposes.
func (s *PostgresMemoryService) DB() *sql.DB { return s.db }

// Search finds relevant memory entries for a query.
func (s *PostgresMemoryService) Search(
	ctx context.Context,
	req *memory.SearchRequest,
) (*memory.SearchResponse, error) {
	s.logger.Debugf("searching memories: app=%s, user=%s, query=%q",
		req.AppName, req.UserID, req.Query)

	var (
		memories   []memory.Entry
		err        error
		searchType string
	)

	// NOTE: If we have an embedding model and a query, try vector search first
	if s.embeddingModel != nil && req.Query != "" {
		embedding, embErr := s.embeddingModel.Embed(ctx, req.Query)
		if embErr == nil && len(embedding) > 0 {
			memories, err = s.searchByVector(ctx, req, embedding)
			if err != nil {
				s.logger.Errorf("failed to search by vector: %v", err)
				return nil, err
			}
			searchType = "vector"
		}
	}

	// NOTE: Fallback to text search if no results or no embedding model
	if len(memories) == 0 && req.Query != "" {
		memories, err = s.searchByText(ctx, req)
		if err != nil {
			s.logger.Errorf("failed to search by text: %v", err)
			return nil, err
		}
		searchType = "text"
	}

	// NOTE: If still no results and query is empty, return recent entries
	if len(memories) == 0 {
		memories, err = s.searchRecent(ctx, req)
		if err != nil {
			s.logger.Errorf("failed to search recent: %v", err)
			return nil, err
		}
		searchType = "recent"
	}

	s.logger.Debugf("search completed: type=%s, results=%d", searchType, len(memories))

	return &memory.SearchResponse{Memories: memories}, nil
}

// searchByVector performs semantic similarity search.
func (s *PostgresMemoryService) searchByVector(
	ctx context.Context,
	req *memory.SearchRequest,
	embedding []float32,
) ([]memory.Entry, error) {
	s.logger.Debugf("searching by vector: app=%s, user=%s, embedding_dim=%d",
		req.AppName, req.UserID, len(embedding))

	query := `
		SELECT content, author, timestamp
		FROM memory_entries
		WHERE app_name = $1 AND user_id = $2 AND embedding IS NOT NULL
		ORDER BY embedding <=> $3
		LIMIT 10
	`

	embeddingStr := vectorToString(embedding)
	rows, err := s.db.QueryContext(ctx, query, req.AppName, req.UserID, embeddingStr)
	if err != nil {
		s.logger.Errorf("failed to search by vector: %v", err)
		return nil, fmt.Errorf("failed to search by vector: %w", err)
	}
	defer func() { _ = rows.Close() }()

	return s.scanMemories(rows)
}

// searchByText performs full-text search using PostgreSQL's tsvector.
func (s *PostgresMemoryService) searchByText(
	ctx context.Context,
	req *memory.SearchRequest,
) ([]memory.Entry, error) {
	s.logger.Debugf("searching by text: app=%s, user=%s, query=%q",
		req.AppName, req.UserID, req.Query)

	query := `
		SELECT content, author, timestamp
		FROM memory_entries
		WHERE app_name = $1 AND user_id = $2
		AND to_tsvector('english', content_text) @@ plainto_tsquery('english', $3)
		ORDER BY ts_rank(to_tsvector('english', content_text), plainto_tsquery('english', $3)) DESC,
		         timestamp DESC
		LIMIT 10
		`
	rows, err := s.db.QueryContext(ctx, query, req.AppName, req.UserID, req.Query)
	if err != nil {
		s.logger.Errorf("failed to search by text: %v", err)
		return nil, fmt.Errorf("failed to search by text: %w", err)
	}
	defer func() { _ = rows.Close() }()

	return s.scanMemories(rows)
}

// searchRecent returns the most recent memory entries.
func (s *PostgresMemoryService) searchRecent(
	ctx context.Context,
	req *memory.SearchRequest,
) ([]memory.Entry, error) {
	s.logger.Debugf("searching recent entries: app=%s, user=%s", req.AppName, req.UserID)

	query := `
		SELECT content, author, timestamp
		FROM memory_entries
		WHERE app_name = $1 AND user_id = $2
		ORDER BY timestamp DESC
		LIMIT 10
	`

	rows, err := s.db.QueryContext(ctx, query, req.AppName, req.UserID)
	if err != nil {
		s.logger.Errorf("failed to search recent: %v", err)
		return nil, fmt.Errorf("failed to search recent: %w", err)
	}
	defer func() { _ = rows.Close() }()

	return s.scanMemories(rows)
}

// scanMemories converts database rows to memory entries.
func (*PostgresMemoryService) scanMemories(rows *sql.Rows) ([]memory.Entry, error) {
	var memories []memory.Entry

	for rows.Next() {
		var contentJSON []byte
		var author sql.NullString
		var timestamp time.Time

		if err := rows.Scan(&contentJSON, &author, &timestamp); err != nil {
			continue
		}

		var content genai.Content
		if err := sonic.Unmarshal(contentJSON, &content); err != nil {
			continue
		}

		entry := memory.Entry{
			Content:   &content,
			Timestamp: timestamp,
		}
		if author.Valid {
			entry.Author = author.String
		}

		memories = append(memories, entry)
	}

	return memories, rows.Err()
}
