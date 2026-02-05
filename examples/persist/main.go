// Package main demonstrates a multi-agent system with Redis + PostgreSQL
// hybrid session persistence using Google ADK.
//
// This example provides two modes:
//   - demo: Runs a persistence demonstration that validates the hybrid storage functionality
//   - serve: Starts the full web server with the weather_time_agent
//
// Usage:
//
//	go run main.go demo   # Run persistence demo
//	go run main.go serve  # Start web server (default)
package main

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/bytedance/sonic"
	"github.com/kydenul/log"
	"github.com/spf13/viper"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
	"google.golang.org/genai"

	pg "github.com/kydenul/k-adk/session/postgres"
	rsess "github.com/kydenul/k-adk/session/redis"
)

// TTL defines the session expiration time in Redis.
const TTL = 10 * time.Minute

// Demo constants
const (
	demoAppName = "persist_demo"
	demoUserID  = "demo_user_001"
)

// logger is the global logger instance used throughout the application.
var logger log.Logger

// init initializes the global logger by loading configuration from config.yaml.
// Panics if the configuration file cannot be loaded.
func init() {
	opt, err := log.LoadFromFile(filepath.Join(".", "config.yaml"))
	if err != nil {
		panic(fmt.Sprintf("Failed to load log config from file: %v", err))
	}
	logger = log.NewLog(opt)

	// Viper Config
	viper.AddConfigPath(".")
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	if err := viper.ReadInConfig(); err != nil {
		log.Fatalf("Failed to read config file: %v", err)
	}
}

// redisConfig loads Redis configuration from config.yaml using viper.
// Fatally exits if configuration loading fails.
func redisConfig() *rsess.RedisConfig {
	redisConfig := &rsess.RedisConfig{}
	if err := viper.UnmarshalKey("RedisProd", redisConfig); err != nil {
		log.Fatalf("Failed to unmarshal Redis config: %v", err)
	}

	logger.Info(redisConfig)

	return redisConfig
}

// postgresConfig loads PostgreSQL configuration from config.yaml using viper.
// Fatally exits if configuration loading fails.
func postgresConfig() *pg.Config {
	cfg := pg.DefaultConfig()
	if err := viper.UnmarshalKey("Postgres", cfg); err != nil {
		log.Fatalf("Failed to unmarshal Postgres config: %v", err)
	}

	cfg.Logger = logger
	logger.Infof("Postgres config: %s", cfg)

	return cfg
}

func main() {
	mode := "serve"
	if len(os.Args) > 1 {
		mode = os.Args[1]
	}

	switch mode {
	case "demo":
		runDemo()

	case "serve":
		runServer()

	default:
		fmt.Printf("Unknown mode: %s\n", mode)
		fmt.Println("Usage: go run main.go [demo|serve]")
		fmt.Println("  demo  - Run persistence demonstration")
		fmt.Println("  serve - Start web server (default)")
		os.Exit(1)
	}
}

// runDemo demonstrates the Redis + PostgreSQL hybrid persistence functionality.
// It performs a series of tests to validate that sessions and events are correctly
// persisted to PostgreSQL and can survive Redis cache expiration.
func runDemo() {
	ctx := context.Background()

	logger.Info("=== Starting Persistence Demo ===")
	logger.Info("This demo validates the Redis + PostgreSQL hybrid session persistence")

	// Initialize clients
	rdb, err := rsess.NewRedisClient(redisConfig())
	if err != nil {
		log.Fatalf("Failed to create redis client: %v", err)
	}
	defer func() {
		if err := rdb.Close(); err != nil {
			logger.Errorf("Failed to close redis client: %v", err)
		}
	}()

	pgClient, err := pg.NewClient(ctx, postgresConfig())
	if err != nil {
		log.Fatalf("Failed to create postgres client: %v", err)
	}
	defer func() {
		if err := pgClient.Close(); err != nil {
			logger.Errorf("Failed to close postgres client: %v", err)
		}
	}()

	pgPersister, err := pg.NewSessionPersister(ctx, pgClient)
	if err != nil {
		log.Fatalf("Failed to create postgres persister: %v", err)
	}
	defer func() {
		if err := pgPersister.Close(); err != nil {
			logger.Errorf("Failed to close postgres persister: %v", err)
		}
	}()

	sessService, err := rsess.NewRedisSessionService(rdb, TTL, logger,
		rsess.WithPersister(pgPersister))
	if err != nil {
		log.Fatalf("Failed to create session service: %v", err)
	}

	// Run demo scenarios
	printDivider("Demo 1: Create Session and Events")
	sessionID := runCreateSessionDemo(ctx, sessService)

	printDivider("Demo 2: Verify Data in Redis")
	runVerifyRedisDemo(ctx, sessService, sessionID)

	printDivider("Demo 3: Verify Data in PostgreSQL")
	runVerifyPostgresDemo(ctx, pgClient, sessionID)

	printDivider("Demo 4: Simulate Redis Expiration")
	runSimulateExpirationDemo(ctx, rdb, sessService, pgClient, sessionID)

	printDivider("Demo 5: Multiple Sessions Test")
	runMultipleSessionsDemo(ctx, sessService, pgClient)

	printDivider("Demo 6: Cleanup")
	runCleanupDemo(ctx, sessService, pgClient, sessionID)

	logger.Info("=== Persistence Demo Completed Successfully ===")
}

// runCreateSessionDemo creates a session and adds multiple events.
func runCreateSessionDemo(ctx context.Context, sessService *rsess.RedisSessionService) string {
	logger.Info("Creating new session...")

	// Create session with initial state
	createResp, err := sessService.Create(ctx, &session.CreateRequest{
		AppName: demoAppName,
		UserID:  demoUserID,
		State: map[string]any{
			"demo_key":   "demo_value",
			"created_at": time.Now().Format(time.RFC3339),
			"counter":    0,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}

	sess := createResp.Session
	sessionID := sess.ID()
	logger.Infof("✓ Session created: %s", sessionID)

	// Add multiple events to simulate a conversation
	events := []struct {
		author  string
		content string
	}{
		{"user", "Hello, what's the weather like today?"},
		{"agent", "I can help you check the weather. Which city are you interested in?"},
		{"user", "I'm in Beijing."},
		{"agent", "The weather in Beijing today is sunny with a high of 25°C."},
		{"user", "Thanks! What about tomorrow?"},
		{"agent", "Tomorrow in Beijing will be partly cloudy with a high of 23°C."},
	}

	for i, e := range events {
		evt := &session.Event{
			Author: e.author,
		}
		evt.Content = &genai.Content{
			Parts: []*genai.Part{
				{Text: e.content},
			},
		}

		if err := sessService.AppendEvent(ctx, sess, evt); err != nil {
			log.Fatalf("Failed to append event %d: %v", i, err)
		}
		logger.Infof("✓ Event %d appended: [%s] %s", i+1, e.author, truncateString(e.content, 50))

		// Small delay to ensure async persistence has time to process
		time.Sleep(100 * time.Millisecond)
	}

	logger.Infof("✓ Total %d events added to session", len(events))

	return sessionID
}

// runVerifyRedisDemo verifies the session data exists in Redis.
func runVerifyRedisDemo(
	ctx context.Context,
	sessService *rsess.RedisSessionService,
	sessionID string,
) {
	logger.Info("Verifying session in Redis...")

	getResp, err := sessService.Get(ctx, &session.GetRequest{
		AppName:   demoAppName,
		UserID:    demoUserID,
		SessionID: sessionID,
	})
	if err != nil {
		log.Fatalf("Failed to get session from Redis: %v", err)
	}

	sess := getResp.Session
	events := sess.Events()
	eventCount := 0
	for range events.All() {
		eventCount++
	}

	logger.Infof("✓ Session found in Redis: %s", sess.ID())
	logger.Infof("✓ Events count: %d", eventCount)

	// Print state
	state := sess.State()
	if state != nil {
		logger.Info("✓ Session state:")
		for k, v := range state.All() {
			logger.Infof("    %s: %v", k, v)
		}
	}
}

// runVerifyPostgresDemo directly queries PostgreSQL to verify persisted data.
func runVerifyPostgresDemo(ctx context.Context, pgClient *pg.Client, sessionID string) {
	logger.Info("Verifying session in PostgreSQL...")

	// Query session
	var id, appName, userID string
	var stateJSON []byte
	var lastUpdateTime time.Time

	err := pgClient.DB().QueryRowContext(ctx,
		`SELECT id, app_name, user_id, state, last_update_time
		 FROM sessions
		 WHERE app_name = $1 AND user_id = $2 AND id = $3`,
		demoAppName, demoUserID, sessionID,
	).Scan(&id, &appName, &userID, &stateJSON, &lastUpdateTime)
	if err != nil {
		log.Fatalf("Failed to query session from PostgreSQL: %v", err)
	}

	logger.Infof("✓ Session found in PostgreSQL:")
	logger.Infof("    ID: %s", id)
	logger.Infof("    AppName: %s", appName)
	logger.Infof("    UserID: %s", userID)
	logger.Infof("    LastUpdate: %s", lastUpdateTime.Format(time.RFC3339))

	// Parse and print state
	var state map[string]any
	if err := sonic.Unmarshal(stateJSON, &state); err == nil {
		logger.Info("    State:")
		for k, v := range state {
			logger.Infof("        %s: %v", k, v)
		}
	}

	// Query events
	tableName := pgClient.GetEventsTableName(demoUserID)
	rows, err := pgClient.DB().QueryContext(ctx,
		fmt.Sprintf(`SELECT id, author, content, timestamp
		 FROM %s
		 WHERE app_name = $1 AND user_id = $2 AND session_id = $3
		 ORDER BY event_order`, tableName),
		demoAppName, demoUserID, sessionID,
	)
	if err != nil {
		log.Fatalf("Failed to query events from PostgreSQL: %v", err)
	}
	defer func() { _ = rows.Close() }()

	logger.Infof("✓ Events in PostgreSQL (table: %s):", tableName)
	eventCount := 0
	for rows.Next() {
		var evtID, author string
		var contentJSON []byte
		var timestamp time.Time
		if err := rows.Scan(&evtID, &author, &contentJSON, &timestamp); err != nil {
			logger.Warnf("Failed to scan event row: %v", err)
			continue
		}
		eventCount++

		// Extract text from content
		var evt session.Event
		unmarshalErr := sonic.Unmarshal(contentJSON, &evt)
		if unmarshalErr == nil && evt.Content != nil && len(evt.Content.Parts) > 0 {
			text := evt.Content.Parts[0].Text
			logger.Infof(
				"    [%d] %s: %s: %s",
				eventCount, timestamp.Format("15:04:05"), author, truncateString(text, 40),
			)
		}
	}

	if err := rows.Err(); err != nil {
		log.Fatalf("Error iterating events: %v", err)
	}

	logger.Infof("✓ Total %d events persisted to PostgreSQL", eventCount)
}

// runSimulateExpirationDemo simulates Redis cache expiration and recovery from PostgreSQL.
func runSimulateExpirationDemo(
	ctx context.Context,
	rdb *rsess.RedisClient,
	sessService *rsess.RedisSessionService,
	pgClient *pg.Client,
	sessionID string,
) {
	logger.Info("Simulating Redis cache expiration...")

	// Delete session from Redis (simulate TTL expiration)
	sessionKey := fmt.Sprintf("session:%s:%s:%s", demoAppName, demoUserID, sessionID)
	eventsKey := fmt.Sprintf("events:%s:%s:%s", demoAppName, demoUserID, sessionID)

	if err := rdb.Del(ctx, sessionKey, eventsKey).Err(); err != nil {
		log.Fatalf("Failed to delete keys from Redis: %v", err)
	}
	logger.Info("✓ Session deleted from Redis (simulating TTL expiration)")

	// Verify session is gone from Redis
	_, err := sessService.Get(ctx, &session.GetRequest{
		AppName:   demoAppName,
		UserID:    demoUserID,
		SessionID: sessionID,
	})
	if err != nil {
		logger.Infof("✓ Confirmed: Session not found in Redis (expected): %v", err)
	} else {
		logger.Warn("⚠ Unexpected: Session still exists in Redis")
	}

	// Verify data still exists in PostgreSQL
	logger.Info("Verifying data persists in PostgreSQL after Redis expiration...")

	var count int
	err = pgClient.DB().QueryRowContext(ctx,
		`SELECT COUNT(*) FROM sessions WHERE app_name = $1 AND user_id = $2 AND id = $3`,
		demoAppName, demoUserID, sessionID,
	).Scan(&count)
	if err != nil || count == 0 {
		log.Fatalf("Session not found in PostgreSQL: %v", err)
	}
	logger.Infof("✓ Session still exists in PostgreSQL (count: %d)", count)

	// Count events in PostgreSQL
	tableName := pgClient.GetEventsTableName(demoUserID)
	query := fmt.Sprintf(
		`SELECT COUNT(*) FROM %s WHERE app_name = $1 AND user_id = $2 AND session_id = $3`,
		tableName,
	)
	err = pgClient.DB().QueryRowContext(ctx, query, demoAppName, demoUserID, sessionID).Scan(&count)
	if err != nil {
		log.Fatalf("Failed to count events in PostgreSQL: %v", err)
	}
	logger.Infof("✓ Events still exist in PostgreSQL (count: %d)", count)

	logger.Info("✓ Simulation complete: Data survives Redis expiration via PostgreSQL persistence")
}

// runMultipleSessionsDemo tests creating multiple sessions for the same user.
func runMultipleSessionsDemo(
	ctx context.Context,
	sessService *rsess.RedisSessionService,
	pgClient *pg.Client,
) {
	logger.Info("Testing multiple sessions for the same user...")

	sessionIDs := make([]string, 3)

	// Create multiple sessions
	for i := range 3 {
		createResp, err := sessService.Create(ctx, &session.CreateRequest{
			AppName: demoAppName,
			UserID:  demoUserID,
			State: map[string]any{
				"session_number": i + 1,
			},
		})
		if err != nil {
			log.Fatalf("Failed to create session %d: %v", i+1, err)
		}
		sessionIDs[i] = createResp.Session.ID()
		logger.Infof("✓ Created session %d: %s", i+1, sessionIDs[i])

		// Add an event to each session
		evt := &session.Event{
			Author: "user",
		}
		evt.Content = &genai.Content{
			Parts: []*genai.Part{
				{Text: fmt.Sprintf("Message from session %d", i+1)},
			},
		}
		if err := sessService.AppendEvent(ctx, createResp.Session, evt); err != nil {
			log.Fatalf("Failed to append event to session %d: %v", i+1, err)
		}
	}

	// Wait for async persistence
	time.Sleep(500 * time.Millisecond)

	// Verify all sessions in PostgreSQL
	var count int
	err := pgClient.DB().QueryRowContext(ctx,
		`SELECT COUNT(DISTINCT id) FROM sessions WHERE app_name = $1 AND user_id = $2`,
		demoAppName, demoUserID,
	).Scan(&count)
	if err != nil {
		log.Fatalf("Failed to count sessions: %v", err)
	}
	logger.Infof("✓ Total sessions in PostgreSQL for user %s: %d", demoUserID, count)

	// List sessions via service
	listResp, err := sessService.List(ctx, &session.ListRequest{
		AppName: demoAppName,
		UserID:  demoUserID,
	})
	if err != nil {
		log.Fatalf("Failed to list sessions: %v", err)
	}
	logger.Infof("✓ Sessions returned by service: %d", len(listResp.Sessions))

	// Cleanup: delete the test sessions
	for _, sid := range sessionIDs {
		if err := sessService.Delete(ctx, &session.DeleteRequest{
			AppName:   demoAppName,
			UserID:    demoUserID,
			SessionID: sid,
		}); err != nil {
			logger.Warnf("Failed to delete session %s: %v", sid, err)
		}
	}

	// Wait for async deletion
	time.Sleep(500 * time.Millisecond)

	logger.Info("✓ Multiple sessions test completed")
}

// runCleanupDemo cleans up demo data from both Redis and PostgreSQL.
func runCleanupDemo(
	ctx context.Context,
	sessService *rsess.RedisSessionService,
	pgClient *pg.Client,
	sessionID string,
) {
	logger.Info("Cleaning up demo data...")

	// Try to delete via service (handles Redis + PostgreSQL)
	err := sessService.Delete(ctx, &session.DeleteRequest{
		AppName:   demoAppName,
		UserID:    demoUserID,
		SessionID: sessionID,
	})
	if err != nil {
		logger.Warnf("Service delete returned error (expected if Redis key doesn't exist): %v", err)
	}

	// Wait for async operations
	time.Sleep(500 * time.Millisecond)

	// Direct cleanup from PostgreSQL for any remaining data
	tableName := pgClient.GetEventsTableName(demoUserID)

	result, err := pgClient.DB().ExecContext(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE app_name = $1 AND user_id = $2`, tableName),
		demoAppName, demoUserID,
	)
	if err != nil {
		logger.Warnf("Failed to delete events from PostgreSQL: %v", err)
	} else {
		rowsAffected, _ := result.RowsAffected()
		logger.Infof("✓ Deleted %d events from PostgreSQL", rowsAffected)
	}

	result, err = pgClient.DB().ExecContext(ctx,
		`DELETE FROM sessions WHERE app_name = $1 AND user_id = $2`,
		demoAppName, demoUserID,
	)
	if err != nil {
		logger.Warnf("Failed to delete sessions from PostgreSQL: %v", err)
	} else {
		rowsAffected, _ := result.RowsAffected()
		logger.Infof("✓ Deleted %d sessions from PostgreSQL", rowsAffected)
	}

	// Verify cleanup
	var count int
	err = pgClient.DB().QueryRowContext(ctx,
		`SELECT COUNT(*) FROM sessions WHERE app_name = $1 AND user_id = $2`,
		demoAppName, demoUserID,
	).Scan(&count)
	if err != nil && err != sql.ErrNoRows {
		logger.Warnf("Failed to verify cleanup: %v", err)
	} else {
		logger.Infof("✓ Remaining sessions in PostgreSQL: %d", count)
	}

	logger.Info("✓ Cleanup completed")
}

// runServer starts the full web server with the weather_time_agent.
func runServer() {
	ctx := context.Background()
	apiKey := os.Getenv("GOOGLE_API_KEY")

	// Initialize the Gemini model for LLM operations
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{APIKey: apiKey})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Set up Redis client for primary session storage
	rdb, err := rsess.NewRedisClient(redisConfig())
	if err != nil {
		log.Fatalf("Failed to create redis client: %v", err)
	}

	// Set up PostgreSQL client for persistent session storage
	pgClient, err := pg.NewClient(ctx, postgresConfig())
	if err != nil {
		log.Fatalf("Failed to create postgres client: %v", err)
	}
	defer func() {
		if err := pgClient.Close(); err != nil {
			logger.Errorf("Failed to close postgres client: %v", err)
		}
	}()

	// Create PostgreSQL session persister
	pgPersister, err := pg.NewSessionPersister(ctx, pgClient)
	if err != nil {
		log.Fatalf("Failed to create postgres persister: %v", err)
	}
	defer func() {
		if err := pgPersister.Close(); err != nil {
			logger.Errorf("Failed to close postgres persister: %v", err)
		}
	}()

	// Create Redis-backed session service with PostgreSQL persistence
	// Sessions are stored in Redis for fast access and automatically synced to PostgreSQL
	sessService, err := rsess.NewRedisSessionService(
		rdb,
		TTL,
		logger,
		rsess.WithPersister(pgPersister),
	)
	if err != nil {
		log.Fatalf("Failed to create session service: %v", err)
	}

	// Create the root agent with weather/time capabilities and Google Search tool
	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "weather_time_agent",
		Model:       model,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "I can answer your questions about the time and weather in a city.",
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Create a multi-loader to manage agents
	agentLoader, err := agent.NewMultiLoader(rootAgent)
	if err != nil {
		log.Fatalf("Failed to create multi loader: %v", err)
	}

	// Use in-memory artifact storage for generated content
	artifactService := artifact.InMemoryService()

	// Configure the launcher with all services
	config := &launcher.Config{
		SessionService:  sessService,
		ArtifactService: artifactService,
		AgentLoader:     agentLoader,
	}

	logger.Info("Starting application with Redis + PostgreSQL hybrid session persistence")

	// Execute the full launcher (includes web server)
	l := full.NewLauncher()
	if err := l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

// Helper functions

func printDivider(title string) {
	divider := strings.Repeat("=", 60)
	logger.Info("")
	logger.Info(divider)
	logger.Info(title)
	logger.Info(divider)
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
