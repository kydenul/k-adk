package redis

import (
	"context"
	"errors"
	"fmt"
	"time"

	discardlog "github.com/kydenul/k-adk/internal/discard_log"
	"github.com/kydenul/log"
	"github.com/redis/go-redis/v9"
	"github.com/spf13/cast"
	"github.com/spf13/viper"
)

const (
	DefaultRedisHost           = "127.0.0.1"
	DefaultRedisPort           = 6379
	DefaultRedisPassword       = ""
	DefaultPoolSize            = 100
	DefaultMaxIdleConns        = 30
	DefaultMinIdleConns        = 15
	DefaultConnMaxIdleTime     = 10 * time.Minute
	DefaultConnMaxLifetime     = 30 * time.Minute
	DefaultPingRetries         = 3
	DefaultPingTimeout         = 3 * time.Second
	DefaultPoolMonitorInterval = 1 * time.Minute
)

// RedisConfig holds Redis connection configuration.
type RedisConfig struct {
	Host     string `mapstructure:"host"`
	Port     uint16 `mapstructure:"port"`
	Password string `mapstructure:"password"`

	PoolSize        int           `mapstructure:"pool_size"`
	MaxIdleConns    int           `mapstructure:"max_idle_conns"`
	MinIdleConns    int           `mapstructure:"min_idle_conns"`
	ConnMaxIdleTime time.Duration `mapstructure:"conn_max_idle_time"`
	ConnMaxLifetime time.Duration `mapstructure:"conn_max_lifetime"`

	// PingRetries is the number of ping retries during connection validation.
	PingRetries int `mapstructure:"ping_retries"`
	// PingTimeout is the timeout for each ping attempt.
	PingTimeout time.Duration `mapstructure:"ping_timeout"`

	// EnablePoolMonitor enables pool statistics monitoring.
	// When enabled, pool stats will be logged at PoolMonitorInterval.
	EnablePoolMonitor bool `mapstructure:"enable_pool_monitor"`
	// PoolMonitorInterval is the interval for logging pool statistics.
	// Only used when EnablePoolMonitor is true.
	PoolMonitorInterval time.Duration `mapstructure:"pool_monitor_interval"`

	// Logger is an optional custom logger. If nil, DiscardLog will be used.
	Logger log.Logger `mapstructure:"-"`
}

func (c *RedisConfig) String() string {
	maskedPassword := "[REDACTED]"
	if c.Password == "" {
		maskedPassword = "(empty)"
	}

	return fmt.Sprintf(
		"RedisConfig ==> Host: %s, Port: %d, Password: %s, PoolSize: %d, "+
			"MaxIdleConns: %d, MinIdleConns: %d, ConnMaxIdleTime: %s, ConnMaxLifetime: %s, "+
			"PingRetries: %d, PingTimeout: %s, EnablePoolMonitor: %v, PoolMonitorInterval: %s",
		c.Host,
		c.Port,
		maskedPassword,
		c.PoolSize,
		c.MaxIdleConns,
		c.MinIdleConns,
		c.ConnMaxIdleTime,
		c.ConnMaxLifetime,
		c.PingRetries,
		c.PingTimeout,
		c.EnablePoolMonitor,
		c.PoolMonitorInterval,
	)
}

// DefaultRedisConfig returns a RedisConfig with default values.
func DefaultRedisConfig() *RedisConfig {
	return &RedisConfig{
		Host:                DefaultRedisHost,
		Port:                DefaultRedisPort,
		Password:            DefaultRedisPassword,
		PoolSize:            DefaultPoolSize,
		MaxIdleConns:        DefaultMaxIdleConns,
		MinIdleConns:        DefaultMinIdleConns,
		ConnMaxIdleTime:     DefaultConnMaxIdleTime,
		ConnMaxLifetime:     DefaultConnMaxLifetime,
		PingRetries:         DefaultPingRetries,
		PingTimeout:         DefaultPingTimeout,
		EnablePoolMonitor:   false,
		PoolMonitorInterval: DefaultPoolMonitorInterval,
		Logger:              nil,
	}
}

// RedisClient wraps a Redis client with optional pool monitoring.
type RedisClient struct {
	redis.UniversalClient
	cancelMonitor context.CancelFunc
	logger        log.Logger
}

// NewRedisClient creates a new Redis client with the given configuration.
// The caller is responsible for closing the client when done.
//
// Example usage:
//
//	cfg := redis.DefaultRedisConfig()
//	cfg.Host = "localhost"
//	cfg.Port = 6379
//	cfg.EnablePoolMonitor = true
//
//	client, err := redis.NewRedisClient(cfg)
//	if err != nil {
//	    return err
//	}
//	defer client.Close()
//
//	sessionService, err := redis.NewRedisSessionService(client, 24*time.Hour)
func NewRedisClient(cfg *RedisConfig) (*RedisClient, error) {
	if cfg == nil {
		return nil, errors.New("redis config cannot be nil")
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

	poolMonitorInterval := cfg.PoolMonitorInterval
	if poolMonitorInterval <= 0 {
		poolMonitorInterval = DefaultPoolMonitorInterval
	}

	// Use DiscardLog if no custom logger is provided
	logger := cfg.Logger
	if logger == nil {
		logger = discardlog.NewDiscardLog()
	}

	rdb := redis.NewUniversalClient(&redis.UniversalOptions{
		Addrs:           []string{cfg.Host + ":" + cast.ToString(cfg.Port)},
		Password:        cfg.Password,
		PoolSize:        cfg.PoolSize,
		MaxIdleConns:    cfg.MaxIdleConns,
		MinIdleConns:    cfg.MinIdleConns,
		ConnMaxIdleTime: cfg.ConnMaxIdleTime,
		ConnMaxLifetime: cfg.ConnMaxLifetime,
	})

	// NOTE: Validate connection with retries
	var pingErr error
	for i := range pingRetries {
		ctx, cancel := context.WithTimeout(context.Background(), pingTimeout)
		pingErr = rdb.Ping(ctx).Err()
		cancel()

		if pingErr == nil {
			break
		}

		logger.Errorf("redis ping failed (attempt %d/%d): %s",
			i+1, pingRetries, pingErr.Error())

		if i < pingRetries-1 {
			// Exponential backoff
			time.Sleep(time.Duration(i+1) * time.Second)
		}
	}

	if pingErr != nil {
		if closeErr := rdb.Close(); closeErr != nil {
			logger.Errorf("failed to close redis client: %s", closeErr.Error())
		}
		return nil, fmt.Errorf("redis ping failed after %d retries: %w",
			pingRetries, pingErr)
	}

	logger.Info("redis client initialized successfully")

	client := &RedisClient{
		UniversalClient: rdb,
		logger:          logger,
	}

	// NOTE: Start pool monitor if enabled
	if cfg.EnablePoolMonitor {
		ctx, cancel := context.WithCancel(context.Background())
		client.cancelMonitor = cancel

		go client.runPoolMonitor(ctx, poolMonitorInterval)
	}

	return client, nil
}

// Client returns the underlying Redis client.
// The returned client shares the same connection pool and should not be closed separately.
func (c *RedisClient) Client() redis.UniversalClient { return c.UniversalClient }

// Close closes the Redis client and stops the pool monitor if running.
func (c *RedisClient) Close() error {
	if c.cancelMonitor != nil {
		c.cancelMonitor()
		c.cancelMonitor = nil
	}

	if c.UniversalClient == nil {
		return nil
	}

	return c.UniversalClient.Close()
}

// runPoolMonitor logs pool statistics at the specified interval.
func (c *RedisClient) runPoolMonitor(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			c.logger.Info("redis pool monitor stopped")
			return

		case <-ticker.C:
			stats := c.PoolStats()
			c.logger.Infof("Redis Pool: Hits=%d Misses=%d Timeouts=%d "+
				"TotalConns=%d IdleConns=%d StaleConns=%d",
				stats.Hits, stats.Misses, stats.Timeouts,
				stats.TotalConns, stats.IdleConns, stats.StaleConns)
		}
	}
}

// "RedisProd"

// LoadRedisConfigFromFile loads redis config from file.
//
//   - configFile: The path to the configuration file.
//   - key: The key in the configuration file where the Redis configuration is located.
//     Recommonded: `Test` / `Pre-Release` / `Production`
func LoadRedisConfigFromFile(configFile, key string) *RedisConfig {
	v := viper.New()
	v.SetConfigFile(configFile)

	if err := v.ReadInConfig(); err != nil {
		log.Fatalf("Failed to read config file: %v", err)
	}

	redisConfig := &RedisConfig{}
	if err := v.UnmarshalKey(key, redisConfig); err != nil {
		log.Fatalf("Failed to unmarshal Redis config: %v", err)
	}

	log.Info(redisConfig)

	return redisConfig
}
