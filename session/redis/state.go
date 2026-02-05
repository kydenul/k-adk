package redis

import (
	"context"
	"fmt"
	"iter"
	"sync"
	"time"

	"github.com/bytedance/sonic"
	discardlog "github.com/kydenul/k-adk/internal/discard_log"
	"github.com/kydenul/log"
	"github.com/redis/go-redis/v9"
	"google.golang.org/adk/session"
)

var _ session.State = (*redisState)(nil)

// updateStateScript is a Lua script that atomically updates the session state.
// It performs a read-modify-write operation atomically to prevent race conditions.
//
// KEYS[1]: session key
// ARGV[1]: new state JSON
// ARGV[2]: TTL in seconds
// ARGV[3]: last_update_time (RFC3339 formatted string from Go)
//
// Returns: "OK" on success, error message on failure
//
// Note: We pass the timestamp from Go (ARGV[3]) instead of using Lua's os.date()
// to ensure consistent time format parsing between Go and Redis.
var updateStateScript = redis.NewScript(`
local data = redis.call('GET', KEYS[1])
if not data then
    return {err = "session not found"}
end

local session = cjson.decode(data)
local newState = cjson.decode(ARGV[1])

session.state = newState
session.last_update_time = ARGV[3]

local updated = cjson.encode(session)
local ttl = tonumber(ARGV[2])

if ttl > 0 then
    redis.call('SET', KEYS[1], updated, 'EX', ttl)
else
    redis.call('SET', KEYS[1], updated)
end

return "OK"
`)

// redisState implements the session.State interface with Redis persistence.
// It uses sync.Map for thread-safe concurrent access.
type redisState struct {
	data   sync.Map
	client redis.UniversalClient
	key    string
	ttl    time.Duration
	logger log.Logger
}

func newRedisState(
	initial map[string]any,
	rdb redis.UniversalClient,
	key string,
	ttl time.Duration,
	logger log.Logger,
) *redisState {
	if logger == nil {
		logger = &discardlog.DiscardLog{}
	}

	s := &redisState{
		client: rdb,
		key:    key,
		ttl:    ttl,
		logger: logger,
	}

	// Copy initial data to sync.Map
	for k, v := range initial {
		s.data.Store(k, v)
	}

	return s
}

func (s *redisState) Get(key string) (any, error) {
	if val, ok := s.data.Load(key); ok {
		return val, nil
	}

	return nil, session.ErrStateKeyNotExist
}

func (s *redisState) Set(key string, value any) error {
	s.data.Store(key, value)

	// Persist to Redis atomically using Lua script.
	if err := s.persistAtomic(context.Background()); err != nil {
		s.logger.Warnf("failed to persist state for key %s: %v", s.key, err)
		return err
	}

	return nil
}

// persistAtomic uses a Lua script to atomically update the session state,
// preventing race conditions in concurrent scenarios.
func (s *redisState) persistAtomic(ctx context.Context) error {
	if s.client == nil {
		return nil
	}

	stateMap := s.toMap()
	stateJSON, err := sonic.Marshal(stateMap)
	if err != nil {
		return fmt.Errorf("failed to marshal state: %w", err)
	}

	// Pass RFC3339 formatted timestamp to ensure consistent parsing with Go's time.Time
	timestamp := time.Now().Format(time.RFC3339)

	result, err := updateStateScript.Run(ctx, s.client, []string{s.key},
		string(stateJSON), int64(s.ttl.Seconds()), timestamp).Result()
	if err != nil {
		if err == redis.Nil {
			// Session does not exist in Redis yet, this is acceptable for new sessions
			return nil
		}
		return fmt.Errorf("failed to persist state atomically: %w", err)
	}

	if result != "OK" {
		return fmt.Errorf("unexpected result from state update script: %v", result)
	}

	return nil
}

// toMap converts sync.Map to a regular map for serialization.
func (s *redisState) toMap() map[string]any {
	result := make(map[string]any)
	s.data.Range(func(key, value any) bool {
		if k, ok := key.(string); ok {
			result[k] = value
		}
		return true
	})
	return result
}

func (s *redisState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		s.data.Range(func(key, value any) bool {
			if k, ok := key.(string); ok {
				return yield(k, value)
			}
			return true
		})
	}
}
