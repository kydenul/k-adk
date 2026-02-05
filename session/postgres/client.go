package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"hash/fnv"
	"time"

	discardlog "github.com/kydenul/k-adk/internal/discard_log"
	"github.com/kydenul/log"

	_ "github.com/lib/pq" // PostgreSQL driver
)

const (
	DefaultConnMaxIdleTime = 10 * time.Minute
	DefaultConnMaxLifetime = 30 * time.Minute
	DefaultMaxOpenConns    = 25
	DefaultMaxIdleConns    = 10
	DefaultPingRetries     = 3
	DefaultPingTimeout     = 3 * time.Second
	DefaultShardCount      = 8
)

// Config holds PostgreSQL connection configuration for session persistence.
type Config struct {
	// ConnStr is the PostgreSQL connection string
	// e.g., "postgres://user:pass@localhost:5432/dbname?sslmode=disable"
	ConnStr string `mapstructure:"conn_str"`

	// Connection pool settings
	MaxOpenConns    int           `mapstructure:"max_open_conns"`
	MaxIdleConns    int           `mapstructure:"max_idle_conns"`
	ConnMaxIdleTime time.Duration `mapstructure:"conn_max_idle_time"`
	ConnMaxLifetime time.Duration `mapstructure:"conn_max_lifetime"`

	// PingRetries is the number of ping retries during connection validation.
	PingRetries int `mapstructure:"ping_retries"`
	// PingTimeout is the timeout for each ping attempt.
	PingTimeout time.Duration `mapstructure:"ping_timeout"`

	// ShardCount is the number of table shards for events.
	// Events are distributed across shards based on user_id hash.
	// Must be a power of 2 (e.g., 4, 8, 16). Default: 8
	ShardCount int `mapstructure:"shard_count"`

	// Logger is an optional custom logger. If nil, DiscardLog will be used.
	Logger log.Logger `mapstructure:"-"`
}

func (c *Config) String() string {
	maskedConnStr := "[REDACTED]"
	if c.ConnStr == "" {
		maskedConnStr = "(empty)"
	}

	return fmt.Sprintf(
		"PostgresConfig ==> ConnStr: %s, MaxOpenConns: %d, MaxIdleConns: %d, "+
			"ConnMaxIdleTime: %s, ConnMaxLifetime: %s, PingRetries: %d, "+
			"PingTimeout: %s, ShardCount: %d",
		maskedConnStr,
		c.MaxOpenConns,
		c.MaxIdleConns,
		c.ConnMaxIdleTime,
		c.ConnMaxLifetime,
		c.PingRetries,
		c.PingTimeout,
		c.ShardCount,
	)
}

// DefaultConfig returns a Config with default values.
func DefaultConfig() *Config {
	return &Config{
		ConnStr:         "",
		MaxOpenConns:    DefaultMaxOpenConns,
		MaxIdleConns:    DefaultMaxIdleConns,
		ConnMaxIdleTime: DefaultConnMaxIdleTime,
		ConnMaxLifetime: DefaultConnMaxLifetime,
		PingRetries:     DefaultPingRetries,
		PingTimeout:     DefaultPingTimeout,
		ShardCount:      DefaultShardCount,
		Logger:          discardlog.NewDiscardLog(),
	}
}

// Client wraps a PostgreSQL database connection for session persistence.
type Client struct {
	db         *sql.DB
	logger     log.Logger
	shardCount int
}

// NewClient creates a new PostgreSQL client with the given configuration.
// The caller is responsible for closing the client when done.
func NewClient(ctx context.Context, cfg *Config) (*Client, error) {
	if cfg == nil {
		return nil, errors.New("postgres config cannot be nil")
	}

	if cfg.ConnStr == "" {
		return nil, errors.New("postgres connection string cannot be empty")
	}

	// Apply defaults for zero values
	pingRetries := cfg.PingRetries
	if pingRetries <= 0 {
		pingRetries = DefaultPingRetries
	}

	pingTimeout := cfg.PingTimeout
	if pingTimeout <= 0 {
		pingTimeout = DefaultPingTimeout
	}

	shardCount := cfg.ShardCount
	if shardCount <= 0 {
		shardCount = DefaultShardCount
	}
	// Ensure shard count is power of 2
	if !isPowerOfTwo(shardCount) {
		shardCount = nextPowerOfTwo(shardCount)
	}

	// Use DiscardLog if no custom logger is provided
	logger := cfg.Logger
	if logger == nil {
		logger = discardlog.NewDiscardLog()
	}

	// Open database connection
	db, err := sql.Open("postgres", cfg.ConnStr)
	if err != nil {
		logger.Errorf("failed to open postgres connection: %v", err)
		return nil, fmt.Errorf("failed to open postgres connection: %w", err)
	}

	// Configure connection pool
	if cfg.MaxOpenConns > 0 {
		db.SetMaxOpenConns(cfg.MaxOpenConns)
	}
	if cfg.MaxIdleConns > 0 {
		db.SetMaxIdleConns(cfg.MaxIdleConns)
	}
	if cfg.ConnMaxIdleTime > 0 {
		db.SetConnMaxIdleTime(cfg.ConnMaxIdleTime)
	}
	if cfg.ConnMaxLifetime > 0 {
		db.SetConnMaxLifetime(cfg.ConnMaxLifetime)
	}

	// Validate connection with retries
	var pingErr error
	for i := range pingRetries {
		pingCtx, cancel := context.WithTimeout(ctx, pingTimeout)
		pingErr = db.PingContext(pingCtx)
		cancel()

		if pingErr == nil {
			break
		}

		logger.Errorf("postgres ping failed (attempt %d/%d): %v",
			i+1, pingRetries, pingErr)

		if i < pingRetries-1 {
			// Exponential backoff
			time.Sleep(time.Duration(i+1) * time.Second)
		}
	}

	if pingErr != nil {
		if closeErr := db.Close(); closeErr != nil {
			logger.Errorf("failed to close postgres connection: %v", closeErr)
		}
		return nil, fmt.Errorf("postgres ping failed after %d retries: %w",
			pingRetries, pingErr)
	}

	logger.Info("postgres client initialized successfully")

	return &Client{
		db:         db,
		logger:     logger,
		shardCount: shardCount,
	}, nil
}

// DB returns the underlying database connection.
func (c *Client) DB() *sql.DB { return c.db }

// Logger returns the logger instance.
func (c *Client) Logger() log.Logger { return c.logger }

// ShardCount returns the number of event table shards.
func (c *Client) ShardCount() int { return c.shardCount }

// Close closes the database connection.
func (c *Client) Close() error {
	if c.db == nil {
		return nil
	}
	return c.db.Close()
}

// GetShardIndex calculates the shard index for a given user ID.
// Uses FNV-1a hash for consistent distribution.
func (c *Client) GetShardIndex(userID string) int {
	h := fnv.New32a()
	_, _ = h.Write([]byte(userID))
	return int(h.Sum32()) & (c.shardCount - 1) // Bitwise AND for power-of-2 modulo
}

// GetEventsTableName returns the sharded events table name for a user.
func (c *Client) GetEventsTableName(userID string) string {
	shardIdx := c.GetShardIndex(userID)
	return fmt.Sprintf("session_events_%d", shardIdx)
}

// isPowerOfTwo checks if n is a power of 2.
func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

// nextPowerOfTwo returns the next power of 2 >= n.
func nextPowerOfTwo(n int) int {
	if n <= 1 {
		return 1
	}
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	return n + 1
}
