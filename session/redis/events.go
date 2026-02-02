package session

import (
	"context"
	"iter"
	"sync"

	"github.com/bytedance/sonic"
	discardlog "github.com/kydenul/k-adk/internal/discard_log"
	"github.com/kydenul/log"
	"github.com/redis/go-redis/v9"
	"google.golang.org/adk/session"
)

var _ session.Events = (*redisEvents)(nil)

// redisEvents implements session.Events with live Redis reads.
// It is thread-safe and uses sync.RWMutex to protect the cached events.
type redisEvents struct {
	client redis.UniversalClient
	key    string
	logger log.Logger

	// mu protects cached for concurrent access.
	mu sync.RWMutex
	// cached events loaded from Redis or provided at creation time.
	cached []*session.Event
}

func newRedisEvents(
	events []*session.Event,
	rdb redis.UniversalClient,
	key string,
	logger log.Logger,
) *redisEvents {
	if events == nil {
		events = make([]*session.Event, 0)
	}

	if logger == nil {
		logger = &discardlog.DiscardLog{}
	}

	return &redisEvents{
		client: rdb,
		key:    key,
		logger: logger,
		cached: events,
	}
}

// refreshCache reloads events from Redis and updates the cache.
// The caller must hold the write lock (mu.Lock()).
func (e *redisEvents) refreshCacheLocked(ctx context.Context) {
	if e.client == nil || e.key == "" {
		return
	}

	eventData, err := e.client.LRange(ctx, e.key, 0, -1).Result()
	if err != nil {
		e.logger.Warnf("failed to load events from redis key %s: %v", e.key, err)
		return
	}

	events := make([]*session.Event, 0, len(eventData))
	for i, ed := range eventData {
		var evt session.Event
		if err := sonic.Unmarshal([]byte(ed), &evt); err != nil {
			e.logger.Warnf("failed to unmarshal event at index %d from key %s: %v", i, e.key, err)
			continue
		}

		events = append(events, &evt)
	}

	e.cached = events
}

func (e *redisEvents) All() iter.Seq[*session.Event] {
	// Interface doesn't allow context parameter, use background context
	e.mu.Lock()
	e.refreshCacheLocked(context.Background())
	// Take a snapshot of cached events while holding the lock
	snapshot := make([]*session.Event, len(e.cached))
	copy(snapshot, e.cached)
	e.mu.Unlock()

	return func(yield func(*session.Event) bool) {
		for _, evt := range snapshot {
			if !yield(evt) {
				return
			}
		}
	}
}

// Len returns the number of cached events.
// Note: Call All() first to ensure the cache is up-to-date.
func (e *redisEvents) Len() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.cached)
}

// At returns the event at the given index from the cache.
// Note: Call All() first to ensure the cache is up-to-date.
func (e *redisEvents) At(idx int) *session.Event {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if idx < 0 || idx >= len(e.cached) {
		return nil
	}

	return e.cached[idx]
}
