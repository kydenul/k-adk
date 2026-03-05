package redis

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"maps"
	"time"

	"github.com/bytedance/sonic"
	discardlog "github.com/kydenul/k-adk/internal/discard_log"
	ksess "github.com/kydenul/k-adk/session"
	"github.com/kydenul/log"
	"github.com/redis/go-redis/v9"
	"github.com/spf13/cast"
	"google.golang.org/adk/session"
)

var _ session.Service = (*RedisSessionService)(nil)

const (
	// defaultSessionTTL is the default session expiration time.
	// Recommended: 7 days to keep sessions accessible for a reasonable window.
	defaultSessionTTL = 7 * 24 * time.Hour
	// sessionIDByteLength defines the length of the session ID in bytes.
	sessionIDByteLength = 16
)

var (
	ErrSessionNotFound = errors.New("session not found")
	ErrNilSession      = errors.New("session cannot be nil")
	ErrNilRedisClient  = errors.New("redis client cannot be nil")
)

// RedisSessionService implements session.Service with Redis as the backend.
type RedisSessionService struct {
	rdb redis.UniversalClient

	// Optional.
	logger log.Logger
	// Optional. PostgreSQL persister for long-term storage
	persister ksess.Persister
	// ttl is the session expiration time (default: 7 days).
	ttl time.Duration
}

// ServiceOption configures the RedisSessionService.
type ServiceOption func(*RedisSessionService)

// WithPersister sets the optional persister for long-term session storage.
// When set, all session operations will be automatically synced to the persister.
func WithPersister(p ksess.Persister) ServiceOption {
	return func(s *RedisSessionService) { s.persister = p }
}

// WithLogger sets the optional logger for the RedisSessionService.
func WithLogger(logger log.Logger) ServiceOption {
	return func(s *RedisSessionService) { s.logger = logger }
}

// WithTTL sets the session expiration time.
// If ttl is <= 0, the default session TTL (7 days) will be used instead.
func WithTTL(ttl time.Duration) ServiceOption {
	return func(s *RedisSessionService) { s.ttl = ttl }
}

// NewRedisSessionService creates a new RedisSessionService.
// If ttl is <= 0, DefaultSessionTTL (7 days) will be used.
// If logger is nil, a no-op logger will be used internally.
// Returns an error if rdb is nil.
func NewRedisSessionService(
	rdb redis.UniversalClient,
	opts ...ServiceOption,
) (*RedisSessionService, error) {
	if rdb == nil {
		return nil, ErrNilRedisClient
	}

	svc := &RedisSessionService{rdb: rdb}

	// Apply options
	for _, opt := range opts {
		opt(svc)
	}

	// Check
	if svc.ttl <= 0 {
		svc.ttl = defaultSessionTTL
	}
	if svc.logger == nil {
		svc.logger = &discardlog.DiscardLog{}
	}

	if svc.persister != nil {
		svc.logger.Info("PostgreSQL persister enabled for long-term session storage")
	}

	return svc, nil
}

func buildSessionKey(appName, userID, sessionID string) string {
	return fmt.Sprintf("session:%s:%s:%s", appName, userID, sessionID)
}

func buildSessionIndexKey(appName, userID string) string {
	return fmt.Sprintf("session:%s:%s", appName, userID)
}

func buildEventsKey(appName, userID, sessionID string) string {
	return fmt.Sprintf("events:%s:%s:%s", appName, userID, sessionID)
}

// generateSessionID generates a unique session ID using crypto/rand.
func generateSessionID() string {
	b := make([]byte, sessionIDByteLength)

	if _, err := rand.Read(b); err != nil {
		// Fallback to timestamp if crypto/rand fails
		return cast.ToString(time.Now().UnixNano())
	}

	return hex.EncodeToString(b)
}

// Create creates a new session.
func (s *RedisSessionService) Create(
	ctx context.Context,
	req *session.CreateRequest,
) (*session.CreateResponse, error) {
	// NOTE: build redis session
	sessionID := req.SessionID
	if sessionID == "" {
		sessionID = generateSessionID()
	}

	s.logger.Debugf("creating session: app=%s, user=%s, session=%s",
		req.AppName, req.UserID, sessionID)

	key := buildSessionKey(req.AppName, req.UserID, sessionID)
	evKey := buildEventsKey(req.AppName, req.UserID, sessionID)

	sess := &redisSession{
		id:             sessionID,
		appName:        req.AppName,
		userID:         req.UserID,
		state:          newRedisState(req.State, s.rdb, key, s.ttl, s.logger),
		events:         newRedisEvents(nil, s.rdb, evKey, s.logger),
		lastUpdateTime: time.Now(),
	}

	// NOTE: Marshal and Set session to redis
	data, err := sonic.Marshal(sess.toStorable())
	if err != nil {
		s.logger.Errorf("failed to marshal session %s: %v", sessionID, err)
		return nil, fmt.Errorf("failed to marshal session: %w", err)
	}

	if err := s.rdb.Set(ctx, key, data, s.ttl).Err(); err != nil {
		s.logger.Errorf("failed to set session %s in redis: %v", sessionID, err)
		return nil, fmt.Errorf("failed to set session: %w", err)
	}

	s.logger.Infof("session stored in redis success: key=%s, ttl=%s, data=%s", key, s.ttl, data)

	// NOTE: Add to session index
	indexKey := buildSessionIndexKey(req.AppName, req.UserID)
	if err := s.rdb.SAdd(ctx, indexKey, sessionID).Err(); err != nil {
		s.logger.Errorf("failed to add session %s to index: %v", sessionID, err)
		return nil, fmt.Errorf("failed to add session to index: %w", err)
	}

	if err := s.rdb.Expire(ctx, indexKey, s.ttl).Err(); err != nil {
		s.logger.Warnf("failed to set expire for index key %s: %v", indexKey, err)
	}

	s.logger.Infof("session added to index success: key=%s, session=%s", indexKey, sessionID)

	// NOTE: Persist to PostgreSQL if persister is configured
	if s.persister != nil {
		if err := s.persister.PersistSession(ctx, sess); err != nil {
			s.logger.Warnf("failed to persist session %s to postgres: %v", sessionID, err)
			// Don't fail the request, Redis is the primary storage
		}

		s.logger.Infof("session persisted to postgres success")
	}

	return &session.CreateResponse{Session: sess}, nil
}

// Get retrieves a session by ID from Redis only.
//
// Design note: Get does NOT fall back to PostgreSQL when the session is missing from Redis.
// Redis is the sole read source; the PostgreSQL persister (if configured) is write-only and
// serves as a durable archive for auditing, analytics, or feeding the memory service.
// Once a session's Redis TTL expires, it becomes inaccessible through this service.
//
// Recommended: set the Redis TTL to at least 7 days (e.g., ksess.WithTTL(7 * 24 * time.Hour))
// to keep sessions available for a reasonable window. Sessions older than the TTL will no longer
// be retrievable, though their data remains in PostgreSQL if a persister was configured.
func (s *RedisSessionService) Get(
	ctx context.Context,
	req *session.GetRequest,
) (*session.GetResponse, error) {
	s.logger.Debugf("getting session: app=%s, user=%s, session=%s",
		req.AppName, req.UserID, req.SessionID)

	// NOTE: Get session from redis
	key := buildSessionKey(req.AppName, req.UserID, req.SessionID)

	data, err := s.rdb.Get(ctx, key).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			s.logger.Errorf("session not found: %s", req.SessionID)
			return nil, fmt.Errorf("%w: %s", ErrSessionNotFound, req.SessionID)
		}

		s.logger.Errorf("failed to get session %s: %v", req.SessionID, err)
		return nil, fmt.Errorf("failed to get session: %w", err)
	}

	s.logger.Debug("Loaded session from redis success")

	// NOTE:
	var storable storableSession
	if err := sonic.Unmarshal(data, &storable); err != nil {
		s.logger.Errorf("failed to unmarshal session %s: %v", req.SessionID, err)
		return nil, fmt.Errorf("failed to unmarshal session: %w", err)
	}

	// NOTE: Load events
	evKey := buildEventsKey(req.AppName, req.UserID, req.SessionID)
	eventData, err := s.rdb.LRange(ctx, evKey, 0, -1).Result()
	if err != nil && !errors.Is(err, redis.Nil) {
		s.logger.Errorf("failed to get events for session %s: %v", req.SessionID, err)
		return nil, fmt.Errorf("failed to get events: %w", err)
	}

	var events []*session.Event
	var unmarshalErrors []error
	for i, ed := range eventData {
		var evt session.Event
		if err := sonic.Unmarshal([]byte(ed), &evt); err != nil {
			unmarshalErrors = append(unmarshalErrors, fmt.Errorf("event at index %d: %w", i, err))
			continue
		}
		events = append(events, &evt)
	}

	// Log warning if some events failed to unmarshal
	if len(unmarshalErrors) > 0 {
		s.logger.Warnf("failed to unmarshal %d events for session %s: %v",
			len(unmarshalErrors), req.SessionID, errors.Join(unmarshalErrors...))
	}

	// Apply filters
	if req.NumRecentEvents > 0 && len(events) > req.NumRecentEvents {
		events = events[len(events)-req.NumRecentEvents:]
	}
	if !req.After.IsZero() {
		var filtered []*session.Event
		for _, evt := range events {
			if !evt.Timestamp.Before(req.After) {
				filtered = append(filtered, evt)
			}
		}
		events = filtered
	}

	// NOTE: build session
	sess := &redisSession{
		id:             storable.ID,
		appName:        storable.AppName,
		userID:         storable.UserID,
		state:          newRedisState(storable.State, s.rdb, key, s.ttl, s.logger),
		events:         newRedisEvents(events, s.rdb, evKey, s.logger),
		lastUpdateTime: storable.LastUpdateTime,
	}

	s.logger.Infof("session retrieved: session=%s, events=%d", req.SessionID, len(events))

	return &session.GetResponse{Session: sess}, nil
}

// List returns all sessions for a user using pipeline for batch fetching.
func (s *RedisSessionService) List(
	ctx context.Context,
	req *session.ListRequest,
) (*session.ListResponse, error) {
	s.logger.Debugf("listing sessions: app=%s, user=%s", req.AppName, req.UserID)

	// NOTE: List sessions
	indexKey := buildSessionIndexKey(req.AppName, req.UserID)

	sessionIDs, err := s.rdb.SMembers(ctx, indexKey).Result()
	if err != nil {
		s.logger.Errorf("failed to list sessions for user %s: %v", req.UserID, err)
		return nil, fmt.Errorf("failed to list sessions: %w", err)
	}

	if len(sessionIDs) == 0 {
		s.logger.Debugf("no sessions found for user %s", req.UserID)
		return &session.ListResponse{Sessions: nil}, nil
	}

	// NOTE: Use pipeline to batch fetch all session data
	pipe := s.rdb.Pipeline()
	sessionCmds := make(map[string]*redis.StringCmd, len(sessionIDs))
	for _, sessionID := range sessionIDs {
		key := buildSessionKey(req.AppName, req.UserID, sessionID)
		sessionCmds[sessionID] = pipe.Get(ctx, key)
	}

	if _, err := pipe.Exec(ctx); err != nil && !errors.Is(err, redis.Nil) {
		s.logger.Errorf("failed to batch get sessions: %v", err)
		return nil, fmt.Errorf("failed to batch get sessions: %w", err)
	}

	// NOTE: Parse results and collect stale session IDs for cleanup
	sessions := make([]session.Session, 0, len(sessionIDs))
	var staleIDs []string
	for _, sessionID := range sessionIDs {
		cmd := sessionCmds[sessionID]
		data, err := cmd.Bytes()
		if err != nil {
			if errors.Is(err, redis.Nil) {
				s.logger.Warnf("session %s not found in redis, marking for cleanup", sessionID)
				staleIDs = append(staleIDs, sessionID)
			} else {
				s.logger.Warnf("failed to get session %s: %v", sessionID, err)
			}
			continue
		}

		var storable storableSession
		if err := sonic.Unmarshal(data, &storable); err != nil {
			s.logger.Warnf("failed to unmarshal session %s: %v", sessionID, err)
			continue
		}

		key := buildSessionKey(req.AppName, req.UserID, sessionID)
		evKey := buildEventsKey(req.AppName, req.UserID, sessionID)

		sess := &redisSession{
			id:             storable.ID,
			appName:        storable.AppName,
			userID:         storable.UserID,
			state:          newRedisState(storable.State, s.rdb, key, s.ttl, s.logger),
			events:         newRedisEvents(nil, s.rdb, evKey, s.logger),
			lastUpdateTime: storable.LastUpdateTime,
		}
		sessions = append(sessions, sess)
	}

	s.logger.Infof("listed %d sessions for user %s", len(sessions), req.UserID)

	// NOTE: Clean up stale session IDs from the index set asynchronously.
	// Uses a Lua script to atomically check whether each session key still exists
	// before removing it from the index, preventing a race where a concurrent Create()
	// re-creates a session between our pipeline GET and the cleanup SRem.
	if len(staleIDs) > 0 {
		staleIDsCopy := staleIDs
		appName := req.AppName
		userID := req.UserID
		go func() {
			cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
			defer cancel()
			s.cleanStaleSessionIDs(cleanupCtx, appName, userID, staleIDsCopy)
		}()
	}

	return &session.ListResponse{Sessions: sessions}, nil
}

// Delete removes a session.
func (s *RedisSessionService) Delete(ctx context.Context, req *session.DeleteRequest) error {
	s.logger.Debugf("deleting session: app=%s, user=%s, session=%s",
		req.AppName, req.UserID, req.SessionID)

	// NOTE: Delete session
	key := buildSessionKey(req.AppName, req.UserID, req.SessionID)
	evKey := buildEventsKey(req.AppName, req.UserID, req.SessionID)
	indexKey := buildSessionIndexKey(req.AppName, req.UserID)

	pipe := s.rdb.Pipeline()
	pipe.Del(ctx, key)
	pipe.Del(ctx, evKey)
	pipe.SRem(ctx, indexKey, req.SessionID)

	if _, err := pipe.Exec(ctx); err != nil {
		s.logger.Errorf("failed to delete session %s: %v", req.SessionID, err)
		return fmt.Errorf("failed to delete session: %w", err)
	}

	// NOTE: Delete from PostgreSQL if persister is configured
	if s.persister != nil {
		err := s.persister.DeleteSession(ctx, req.AppName, req.UserID, req.SessionID)
		if err != nil {
			s.logger.Warnf("failed to delete session %s from postgres: %v", req.SessionID, err)
			// Don't fail the request, Redis deletion succeeded
		}

		s.logger.Info("session deleted from postgres success")
	}

	s.logger.Infof("session deleted: app=%s, user=%s, session=%s",
		req.AppName, req.UserID, req.SessionID)

	return nil
}

// AppendEvent appends an event to a session.
func (s *RedisSessionService) AppendEvent(
	ctx context.Context,
	sess session.Session,
	evt *session.Event,
) error {
	if sess == nil {
		return ErrNilSession
	}

	evt.Timestamp = time.Now()
	if evt.ID == "" {
		evt.ID = generateSessionID()
	}

	s.logger.Debugf("appending event to session %s: event_id=%s, author=%s",
		sess.ID(), evt.ID, evt.Author)

	data, err := sonic.Marshal(evt)
	if err != nil {
		s.logger.Errorf("failed to marshal event %s: %v", evt.ID, err)
		return fmt.Errorf("failed to marshal event: %w", err)
	}

	evKey := buildEventsKey(sess.AppName(), sess.UserID(), sess.ID())
	if err := s.rdb.RPush(ctx, evKey, data).Err(); err != nil {
		s.logger.Errorf("failed to append event %s to session %s: %v", evt.ID, sess.ID(), err)
		return fmt.Errorf("failed to append event: %w", err)
	}

	s.logger.Infof("event stored in redis: key=%s, event_id=%s", evKey, evt.ID)

	if err := s.rdb.Expire(ctx, evKey, s.ttl).Err(); err != nil {
		s.logger.Warnf("failed to set expire for events key %s: %v", evKey, err)
	}

	// NOTE: Update session's last update time and persist current state
	key := buildSessionKey(sess.AppName(), sess.UserID(), sess.ID())
	sessData, err := s.rdb.Get(ctx, key).Bytes()
	if err != nil {
		s.logger.Errorf("failed to get session %s for update: %v", sess.ID(), err)
		return fmt.Errorf("failed to get session for update: %w", err)
	}

	var storable storableSession
	if err := sonic.Unmarshal(sessData, &storable); err != nil {
		s.logger.Errorf("failed to unmarshal session %s: %v", sess.ID(), err)
		return fmt.Errorf("failed to unmarshal session: %w", err)
	}

	// Sync state from session to storable
	state := sess.State()
	if state != nil {
		storable.State = maps.Collect(state.All())
	}

	storable.LastUpdateTime = time.Now()
	updatedData, err := sonic.Marshal(storable)
	if err != nil {
		s.logger.Errorf("failed to marshal updated session %s: %v", sess.ID(), err)
		return fmt.Errorf("failed to marshal updated session: %w", err)
	}

	if err := s.rdb.Set(ctx, key, updatedData, s.ttl).Err(); err != nil {
		s.logger.Errorf("failed to update session %s: %v", sess.ID(), err)
		return fmt.Errorf("failed to update session: %w", err)
	}

	s.logger.Debugf("session updated in redis: key=%s", key)

	// NOTE: Refresh index key TTL to keep it aligned with active sessions
	indexKey := buildSessionIndexKey(sess.AppName(), sess.UserID())
	if err := s.rdb.Expire(ctx, indexKey, s.ttl).Err(); err != nil {
		s.logger.Warnf("failed to refresh expire for index key %s: %v", indexKey, err)
	}

	// NOTE: Real-time sync to PostgreSQL if persister is configured
	if s.persister != nil {
		if err := s.persister.PersistEvent(ctx, sess, evt); err != nil {
			s.logger.Warnf("failed to persist event %s to postgres: %v", evt.ID, err)
			// Don't fail the request, Redis is the primary storage
		}

		s.logger.Info("event persisted to postgres success")
	}

	s.logger.Infof("event appended: session=%s, event=%s", sess.ID(), evt.ID)

	return nil
}

// cleanStaleSessionIDs atomically removes session IDs from the index set only if
// their corresponding session keys no longer exist in Redis. This prevents a race
// condition where a concurrent Create() re-creates a session between the pipeline
// GET (returning redis.Nil) and the cleanup removal.
var cleanStaleScript = redis.NewScript(`
local indexKey = KEYS[1]
local prefix = ARGV[1]
local removed = 0
for i = 2, #ARGV do
    local sessionKey = prefix .. ARGV[i]
    if redis.call('EXISTS', sessionKey) == 0 then
        redis.call('SREM', indexKey, ARGV[i])
        removed = removed + 1
    end
end
return removed
`)

func (s *RedisSessionService) cleanStaleSessionIDs(
	ctx context.Context,
	appName, userID string,
	staleIDs []string,
) {
	indexKey := buildSessionIndexKey(appName, userID)
	// Session keys are "session:{appName}:{userID}:{sessionID}", so the prefix
	// up to (and including) the last colon lets the Lua script reconstruct each key.
	keyPrefix := fmt.Sprintf("session:%s:%s:", appName, userID)

	args := make([]any, 0, len(staleIDs)+1)
	args = append(args, keyPrefix)
	for _, id := range staleIDs {
		args = append(args, id)
	}

	result, err := cleanStaleScript.Run(ctx, s.rdb, []string{indexKey}, args...).Int()
	if err != nil {
		s.logger.Warnf("failed to clean up stale session IDs from index: %v", err)
		return
	}

	if result > 0 {
		s.logger.Infof("cleaned up %d stale session IDs from index", result)
	}
}
