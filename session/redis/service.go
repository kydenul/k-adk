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
	"github.com/kydenul/log"
	"github.com/redis/go-redis/v9"
	"github.com/spf13/cast"
	"google.golang.org/adk/session"
)

var _ session.Service = (*RedisSessionService)(nil)

// DefaultSessionTTL is the default session expiration time.
const (
	DefaultSessionTTL   = 24 * time.Hour
	sessionIDByteLength = 16
)

var (
	ErrSessionNotFound = errors.New("session not found")
	ErrNilSession      = errors.New("session cannot be nil")
	ErrNilRedisClient  = errors.New("redis client cannot be nil")
)

// RedisSessionService implements session.Service with Redis as the backend.
type RedisSessionService struct {
	rdb    redis.UniversalClient
	logger log.Logger

	// ttl is the session expiration time (default: 24 hours).
	ttl time.Duration
}

// NewRedisSessionService creates a new RedisSessionService.
// If ttl is <= 0, DefaultSessionTTL (24 hours) will be used.
// If logger is nil, a no-op logger will be used internally.
// Returns an error if rdb is nil.
func NewRedisSessionService(
	rdb redis.UniversalClient,
	ttl time.Duration,
	logger log.Logger,
) (*RedisSessionService, error) {
	if rdb == nil {
		return nil, ErrNilRedisClient
	}

	if ttl <= 0 {
		ttl = DefaultSessionTTL
	}

	if logger == nil {
		logger = &discardlog.DiscardLog{}
	}

	return &RedisSessionService{
		rdb:    rdb,
		ttl:    ttl,
		logger: logger,
	}, nil
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

	data, err := sonic.Marshal(sess.toStorable())
	if err != nil {
		s.logger.Errorf("failed to marshal session %s: %v", sessionID, err)
		return nil, fmt.Errorf("failed to marshal session: %w", err)
	}

	if err := s.rdb.Set(ctx, key, data, s.ttl).Err(); err != nil {
		s.logger.Errorf("failed to set session %s in redis: %v", sessionID, err)
		return nil, fmt.Errorf("failed to set session: %w", err)
	}

	// Add to session index
	indexKey := buildSessionIndexKey(req.AppName, req.UserID)
	if err := s.rdb.SAdd(ctx, indexKey, sessionID).Err(); err != nil {
		s.logger.Errorf("failed to add session %s to index: %v", sessionID, err)
		return nil, fmt.Errorf("failed to add session to index: %w", err)
	}

	if err := s.rdb.Expire(ctx, indexKey, s.ttl).Err(); err != nil {
		s.logger.Warnf("failed to set expire for index key %s: %v", indexKey, err)
	}

	s.logger.Infof("session created: app=%s, user=%s, session=%s, ttl=%s",
		req.AppName, req.UserID, sessionID, s.ttl)

	return &session.CreateResponse{Session: sess}, nil
}

// Get retrieves a session by ID.
func (s *RedisSessionService) Get(
	ctx context.Context,
	req *session.GetRequest,
) (*session.GetResponse, error) {
	s.logger.Debugf("getting session: app=%s, user=%s, session=%s",
		req.AppName, req.UserID, req.SessionID)

	key := buildSessionKey(req.AppName, req.UserID, req.SessionID)

	data, err := s.rdb.Get(ctx, key).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			s.logger.Debugf("session not found: %s", req.SessionID)
			return nil, fmt.Errorf("%w: %s", ErrSessionNotFound, req.SessionID)
		}
		s.logger.Errorf("failed to get session %s: %v", req.SessionID, err)
		return nil, fmt.Errorf("failed to get session: %w", err)
	}

	var storable storableSession
	if err := sonic.Unmarshal(data, &storable); err != nil {
		s.logger.Errorf("failed to unmarshal session %s: %v", req.SessionID, err)
		return nil, fmt.Errorf("failed to unmarshal session: %w", err)
	}

	// Load events
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
			unmarshalErrors = append(unmarshalErrors,
				fmt.Errorf("event at index %d: %w", i, err))
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

	sess := &redisSession{
		id:             storable.ID,
		appName:        storable.AppName,
		userID:         storable.UserID,
		state:          newRedisState(storable.State, s.rdb, key, s.ttl, s.logger),
		events:         newRedisEvents(events, s.rdb, evKey, s.logger),
		lastUpdateTime: storable.LastUpdateTime,
	}

	s.logger.Debugf("session retrieved: session=%s, events=%d", req.SessionID, len(events))

	return &session.GetResponse{Session: sess}, nil
}

// List returns all sessions for a user using pipeline for batch fetching.
func (s *RedisSessionService) List(
	ctx context.Context,
	req *session.ListRequest,
) (*session.ListResponse, error) {
	s.logger.Debugf("listing sessions: app=%s, user=%s", req.AppName, req.UserID)

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

	// NOTE: Parse results
	sessions := make([]session.Session, 0, len(sessionIDs))
	for _, sessionID := range sessionIDs {
		cmd := sessionCmds[sessionID]
		data, err := cmd.Bytes()
		if err != nil {
			if errors.Is(err, redis.Nil) {
				s.logger.Warnf("session %s not found in redis, skipping", sessionID)
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

	s.logger.Debugf("listed %d sessions for user %s", len(sessions), req.UserID)

	return &session.ListResponse{Sessions: sessions}, nil
}

// Delete removes a session.
func (s *RedisSessionService) Delete(ctx context.Context, req *session.DeleteRequest) error {
	s.logger.Debugf("deleting session: app=%s, user=%s, session=%s",
		req.AppName, req.UserID, req.SessionID)

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

	if err := s.rdb.Expire(ctx, evKey, s.ttl).Err(); err != nil {
		s.logger.Warnf("failed to set expire for events key %s: %v", evKey, err)
	}

	// Update session's last update time and persist current state
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

	s.logger.Debugf("event appended: session=%s, event=%s", sess.ID(), evt.ID)

	return nil
}
