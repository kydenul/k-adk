package postgres

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sync"
	"time"

	"github.com/bytedance/sonic"
	discardlog "github.com/kydenul/k-adk/internal/discard_log"
	ksess "github.com/kydenul/k-adk/session"
	"github.com/kydenul/log"
	"google.golang.org/adk/session"
)

var _ ksess.Persister = (*SessionPersister)(nil)

// Default configuration values.
const (
	DefaultAsyncBufferSize = 1000
	DefaultAsyncOpTimeout  = 30 * time.Second
)

// SessionPersister implements Persister for PostgreSQL session persistence.
// It is designed to work alongside RedisSessionService as a long-term storage backend.
type SessionPersister struct {
	client    *Client
	logger    log.Logger
	asyncChan chan asyncOp // Channel for async operations
	wg        sync.WaitGroup
	closed    bool
	mu        sync.Mutex
}

type asyncOp struct {
	opType    string // "session", "event", "delete"
	sess      session.Session
	evt       *session.Event
	appName   string
	userID    string
	sessionID string
}

// PersisterOption configures the SessionPersister.
type PersisterOption func(*SessionPersister)

// WithAsyncBufferSize sets the buffer size for async operations.
// Default is 1000. Set to 0 to disable async mode (all operations are synchronous).
func WithAsyncBufferSize(size int) PersisterOption {
	return func(p *SessionPersister) {
		if size > 0 {
			p.asyncChan = make(chan asyncOp, size)
		} else {
			p.asyncChan = nil
		}
	}
}

// NewSessionPersister creates a new PostgreSQL session persister.
func NewSessionPersister(
	ctx context.Context,
	client *Client,
	opts ...PersisterOption,
) (*SessionPersister, error) {
	if client == nil {
		return nil, errors.New("postgres client cannot be nil")
	}

	logger := client.Logger()
	if logger == nil {
		logger = discardlog.NewDiscardLog()
	}

	p := &SessionPersister{
		client:    client,
		logger:    logger,
		asyncChan: make(chan asyncOp, DefaultAsyncBufferSize),
	}

	// Apply options
	for _, opt := range opts {
		opt(p)
	}

	// Initialize database schema
	if err := p.initSchema(ctx); err != nil {
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	// Start async worker if async mode is enabled
	if p.asyncChan != nil {
		p.wg.Add(1)
		go p.asyncWorker() //nolint:contextcheck // async worker manages its own context
	}

	logger.Info("PostgreSQL session persister initialized")

	return p, nil
}

// initSchema creates the necessary tables and indexes.
func (p *SessionPersister) initSchema(ctx context.Context) error {
	// Create sessions table
	sessionsSchema := `
		CREATE TABLE IF NOT EXISTS sessions (
			id VARCHAR(255) NOT NULL,
			app_name VARCHAR(255) NOT NULL,
			user_id VARCHAR(255) NOT NULL,
			state JSONB NOT NULL DEFAULT '{}',
			last_update_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			PRIMARY KEY (app_name, user_id, id)
		);

		CREATE INDEX IF NOT EXISTS idx_sessions_app_user ON sessions(app_name, user_id);
		CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
		CREATE INDEX IF NOT EXISTS idx_sessions_last_update ON sessions(last_update_time);
	`

	if _, err := p.client.DB().ExecContext(ctx, sessionsSchema); err != nil {
		p.logger.Errorf("failed to create sessions table: %v", err)
		return fmt.Errorf("failed to create sessions table: %w", err)
	}

	// Create sharded events tables
	for i := range p.client.ShardCount() {
		eventsSchema := fmt.Sprintf(`
			CREATE TABLE IF NOT EXISTS session_events_%d (
				id VARCHAR(255) NOT NULL,
				app_name VARCHAR(255) NOT NULL,
				user_id VARCHAR(255) NOT NULL,
				session_id VARCHAR(255) NOT NULL,
				event_order INT NOT NULL,
				content JSONB NOT NULL,
				author VARCHAR(255),
				timestamp TIMESTAMPTZ NOT NULL,
				created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
				PRIMARY KEY (app_name, user_id, session_id, event_order)
			);

			CREATE INDEX IF NOT EXISTS idx_events_%d_session
				ON session_events_%d(app_name, user_id, session_id);
			CREATE INDEX IF NOT EXISTS idx_events_%d_timestamp
				ON session_events_%d(timestamp);
		`, i, i, i, i, i)

		if _, err := p.client.DB().ExecContext(ctx, eventsSchema); err != nil {
			p.logger.Errorf("failed to create events shard table %d: %v", i, err)
			return fmt.Errorf("failed to create events shard table %d: %w", i, err)
		}
	}

	p.logger.Infof("schema initialized with %d event shards", p.client.ShardCount())

	return nil
}

// asyncWorker processes async operations from the channel.
func (p *SessionPersister) asyncWorker() {
	defer p.wg.Done()

	for op := range p.asyncChan {
		p.processAsyncOp(op)
	}
}

// processAsyncOp processes a single async operation with proper context management.
func (p *SessionPersister) processAsyncOp(op asyncOp) {
	ctx, cancel := context.WithTimeout(context.Background(), DefaultAsyncOpTimeout)
	defer cancel()

	var err error
	switch op.opType {
	case "session":
		err = p.persistSessionSync(ctx, op.sess)
	case "event":
		err = p.persistEventSync(ctx, op.sess, op.evt)
	case "delete":
		err = p.deleteSessionSync(ctx, op.appName, op.userID, op.sessionID)
	}

	if err != nil {
		p.logger.Errorf("async %s operation failed: %v", op.opType, err)
	}
}

// PersistSession saves or updates a session in PostgreSQL.
// If async mode is enabled, the operation is queued and returns immediately.
func (p *SessionPersister) PersistSession(ctx context.Context, sess session.Session) error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return errors.New("persister is closed")
	}
	p.mu.Unlock()

	if p.asyncChan != nil {
		select {
		case p.asyncChan <- asyncOp{opType: "session", sess: sess}:
			return nil
		default:
			p.logger.Warn("async channel full, falling back to sync persist")
		}
	}
	return p.persistSessionSync(ctx, sess)
}

func (p *SessionPersister) persistSessionSync(ctx context.Context, sess session.Session) error {
	// Collect state
	var stateJSON []byte
	if state := sess.State(); state != nil {
		stateMap := maps.Collect(state.All())
		var err error
		stateJSON, err = sonic.Marshal(stateMap)
		if err != nil {
			stateJSON = []byte("{}")
		}
	} else {
		stateJSON = []byte("{}")
	}

	query := `
		INSERT INTO sessions (id, app_name, user_id, state, last_update_time, created_at)
		VALUES ($1, $2, $3, $4, $5, NOW())
		ON CONFLICT (app_name, user_id, id) DO UPDATE
		SET state = EXCLUDED.state, last_update_time = EXCLUDED.last_update_time
	`

	_, err := p.client.DB().ExecContext(ctx, query,
		sess.ID(), sess.AppName(), sess.UserID(), stateJSON, sess.LastUpdateTime())
	if err != nil {
		p.logger.Errorf("failed to persist session %s: %v", sess.ID(), err)
		return fmt.Errorf("failed to persist session: %w", err)
	}

	p.logger.Debugf("session persisted: %s", sess.ID())
	return nil
}

// PersistEvent saves a single event to PostgreSQL (real-time sync).
// If async mode is enabled, the operation is queued and returns immediately.
func (p *SessionPersister) PersistEvent(
	ctx context.Context,
	sess session.Session,
	evt *session.Event,
) error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return errors.New("persister is closed")
	}
	p.mu.Unlock()

	if p.asyncChan != nil {
		select {
		case p.asyncChan <- asyncOp{opType: "event", sess: sess, evt: evt}:
			return nil
		default:
			p.logger.Warn("async channel full, falling back to sync persist")
		}
	}
	return p.persistEventSync(ctx, sess, evt)
}

func (p *SessionPersister) persistEventSync(
	ctx context.Context,
	sess session.Session,
	evt *session.Event,
) error {
	// Serialize event
	evtData, err := sonic.Marshal(evt)
	if err != nil {
		p.logger.Errorf("failed to marshal event %s: %v", evt.ID, err)
		return fmt.Errorf("failed to marshal event: %w", err)
	}

	tableName := p.client.GetEventsTableName(sess.UserID())

	// Use transaction to ensure atomicity when getting next order and inserting
	tx, err := p.client.DB().BeginTx(ctx, nil)
	if err != nil {
		p.logger.Errorf("failed to begin transaction: %v", err)
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	// Lock the session row to serialize event inserts for this session
	lockQuery := `SELECT id FROM sessions WHERE app_name = $1 AND user_id = $2 AND id = $3 FOR UPDATE`
	var lockedID string
	_ = tx.QueryRowContext(ctx, lockQuery, sess.AppName(), sess.UserID(), sess.ID()).Scan(&lockedID)
	// Ignore error - session may not exist yet, but we still need the order

	// Get next event order
	//nolint:gosec // table name is generated internally
	orderQuery := `SELECT COALESCE(MAX(event_order), -1) + 1 FROM ` + tableName +
		` WHERE app_name = $1 AND user_id = $2 AND session_id = $3`

	var nextOrder int
	err = tx.QueryRowContext(ctx, orderQuery,
		sess.AppName(), sess.UserID(), sess.ID()).Scan(&nextOrder)
	if err != nil {
		p.logger.Errorf("failed to get next event order: %v", err)
		return fmt.Errorf("failed to get next event order: %w", err)
	}

	// Insert event
	//nolint:gosec // table name is generated internally
	insertQuery := `INSERT INTO ` + tableName +
		` (id, app_name, user_id, session_id, event_order, content, author, timestamp, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())`

	_, err = tx.ExecContext(ctx, insertQuery,
		evt.ID, sess.AppName(), sess.UserID(), sess.ID(),
		nextOrder, evtData, evt.Author, evt.Timestamp)
	if err != nil {
		p.logger.Errorf("failed to insert event %s: %v", evt.ID, err)
		return fmt.Errorf("failed to insert event: %w", err)
	}

	// Also update session's last_update_time
	updateQuery := `UPDATE sessions SET last_update_time = $1 WHERE app_name = $2 AND user_id = $3 AND id = $4`
	if _, err = tx.ExecContext(ctx, updateQuery,
		evt.Timestamp, sess.AppName(), sess.UserID(), sess.ID()); err != nil {
		p.logger.Warnf("failed to update session last_update_time: %v", err)
		// Don't fail the whole operation for this
	}

	if err := tx.Commit(); err != nil {
		p.logger.Errorf("failed to commit transaction: %v", err)
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	p.logger.Debugf("event persisted: session=%s, event=%s, shard=%s",
		sess.ID(), evt.ID, tableName)
	return nil
}

// DeleteSession removes a session and all its events from PostgreSQL.
// If async mode is enabled, the operation is queued and returns immediately.
func (p *SessionPersister) DeleteSession(
	ctx context.Context,
	appName, userID, sessionID string,
) error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return errors.New("persister is closed")
	}
	p.mu.Unlock()

	if p.asyncChan != nil {
		select {
		case p.asyncChan <- asyncOp{
			opType:    "delete",
			appName:   appName,
			userID:    userID,
			sessionID: sessionID,
		}:
			return nil
		default:
			p.logger.Warn("async channel full, falling back to sync delete")
		}
	}
	return p.deleteSessionSync(ctx, appName, userID, sessionID)
}

func (p *SessionPersister) deleteSessionSync(
	ctx context.Context,
	appName, userID, sessionID string,
) error {
	tx, err := p.client.DB().BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	// Delete events from sharded table
	tableName := p.client.GetEventsTableName(userID)
	//nolint:gosec // table name is generated internally
	eventsQuery := `DELETE FROM ` + tableName + ` WHERE app_name = $1 AND user_id = $2 AND session_id = $3`
	_, err = tx.ExecContext(ctx, eventsQuery, appName, userID, sessionID)
	if err != nil {
		return fmt.Errorf("failed to delete events: %w", err)
	}

	// Delete session
	sessionQuery := `DELETE FROM sessions WHERE app_name = $1 AND user_id = $2 AND id = $3`
	_, err = tx.ExecContext(ctx, sessionQuery, appName, userID, sessionID)
	if err != nil {
		return fmt.Errorf("failed to delete session: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	p.logger.Debugf("session deleted from postgres: %s", sessionID)
	return nil
}

// Close closes the persister and releases resources.
// It waits for all pending async operations to complete before returning.
func (p *SessionPersister) Close() error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil
	}
	p.closed = true
	p.mu.Unlock()

	if p.asyncChan != nil {
		close(p.asyncChan)
		p.wg.Wait() // Wait for all async operations to complete
	}

	p.logger.Info("PostgreSQL session persister closed")
	return nil
}

// Client returns the underlying PostgreSQL client.
func (p *SessionPersister) Client() *Client { return p.client }
