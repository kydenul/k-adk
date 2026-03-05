package redis

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/bytedance/sonic"
	"github.com/redis/go-redis/v9"
	"google.golang.org/adk/session"
)

// getTestRedisAddr returns the Redis address for testing.
// Falls back to localhost:6379 if TEST_REDIS_ADDR is not set.
func getTestRedisAddr() string {
	if addr := os.Getenv("TEST_REDIS_ADDR"); addr != "" {
		return addr
	}
	return "localhost:6379"
}

// setupTestRedis creates a Redis client and RedisSessionService for testing.
// It skips the test if Redis is not available.
func setupTestRedis(t *testing.T, opts ...ServiceOption) (*RedisSessionService, redis.UniversalClient) {
	t.Helper()

	rdb := redis.NewUniversalClient(&redis.UniversalOptions{
		Addrs: []string{getTestRedisAddr()},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := rdb.Ping(ctx).Err(); err != nil {
		t.Skipf("Redis not available at %s, skipping test: %v", getTestRedisAddr(), err)
	}

	svc, err := NewRedisSessionService(rdb, opts...)
	if err != nil {
		rdb.Close()
		t.Fatalf("failed to create RedisSessionService: %v", err)
	}

	t.Cleanup(func() {
		rdb.Close()
	})

	return svc, rdb
}

// cleanupTestKeys removes all keys matching test prefixes.
func cleanupTestKeys(t *testing.T, rdb redis.UniversalClient, patterns ...string) {
	t.Helper()
	ctx := context.Background()
	for _, pattern := range patterns {
		keys, _ := rdb.Keys(ctx, pattern).Result()
		if len(keys) > 0 {
			rdb.Del(ctx, keys...)
		}
	}
}

// --- NewRedisSessionService ---

func TestNewRedisSessionService(t *testing.T) {
	t.Run("nil client returns error", func(t *testing.T) {
		_, err := NewRedisSessionService(nil)
		if err == nil {
			t.Fatal("expected error for nil client")
		}
		if err != ErrNilRedisClient {
			t.Errorf("expected ErrNilRedisClient, got: %v", err)
		}
	})

	t.Run("default TTL is 7 days", func(t *testing.T) {
		svc, _ := setupTestRedis(t)
		if svc.ttl != 7*24*time.Hour {
			t.Errorf("expected default TTL 7 days, got: %v", svc.ttl)
		}
	})

	t.Run("WithTTL option", func(t *testing.T) {
		svc, _ := setupTestRedis(t, WithTTL(1*time.Hour))
		if svc.ttl != 1*time.Hour {
			t.Errorf("expected TTL 1 hour, got: %v", svc.ttl)
		}
	})

	t.Run("WithTTL zero falls back to default", func(t *testing.T) {
		svc, _ := setupTestRedis(t, WithTTL(0))
		if svc.ttl != defaultSessionTTL {
			t.Errorf("expected default TTL, got: %v", svc.ttl)
		}
	})

	t.Run("WithTTL negative falls back to default", func(t *testing.T) {
		svc, _ := setupTestRedis(t, WithTTL(-1*time.Second))
		if svc.ttl != defaultSessionTTL {
			t.Errorf("expected default TTL, got: %v", svc.ttl)
		}
	})

	t.Run("WithLogger option", func(t *testing.T) {
		svc, _ := setupTestRedis(t)
		// Logger should be set (discard log by default, not nil)
		if svc.logger == nil {
			t.Error("expected non-nil logger")
		}
	})

	t.Run("WithPersister option", func(t *testing.T) {
		svc, _ := setupTestRedis(t, WithPersister(nil))
		if svc.persister != nil {
			t.Error("expected nil persister when nil is passed")
		}
	})
}

// --- Create ---

func TestCreate(t *testing.T) {
	const (
		appName = "test_create_app"
		userID  = "test_create_user"
	)

	svc, rdb := setupTestRedis(t, WithTTL(30*time.Second))
	t.Cleanup(func() {
		cleanupTestKeys(t, rdb,
			fmt.Sprintf("session:%s:*", appName),
			fmt.Sprintf("events:%s:*", appName),
		)
	})

	ctx := context.Background()

	t.Run("auto-generated session ID", func(t *testing.T) {
		resp, err := svc.Create(ctx, &session.CreateRequest{
			AppName: appName,
			UserID:  userID,
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}
		if resp.Session.ID() == "" {
			t.Error("expected non-empty auto-generated session ID")
		}
		if resp.Session.AppName() != appName {
			t.Errorf("expected appName %q, got %q", appName, resp.Session.AppName())
		}
		if resp.Session.UserID() != userID {
			t.Errorf("expected userID %q, got %q", userID, resp.Session.UserID())
		}
	})

	t.Run("caller-supplied session ID", func(t *testing.T) {
		resp, err := svc.Create(ctx, &session.CreateRequest{
			AppName:   appName,
			UserID:    userID,
			SessionID: "custom-id-123",
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}
		if resp.Session.ID() != "custom-id-123" {
			t.Errorf("expected session ID 'custom-id-123', got %q", resp.Session.ID())
		}
	})

	t.Run("session stored in Redis", func(t *testing.T) {
		resp, err := svc.Create(ctx, &session.CreateRequest{
			AppName:   appName,
			UserID:    userID,
			SessionID: "stored-check",
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		key := buildSessionKey(appName, userID, resp.Session.ID())
		exists, err := rdb.Exists(ctx, key).Result()
		if err != nil {
			t.Fatalf("Exists check failed: %v", err)
		}
		if exists != 1 {
			t.Error("expected session key to exist in Redis")
		}
	})

	t.Run("session added to index set", func(t *testing.T) {
		sessionID := "index-check"
		_, err := svc.Create(ctx, &session.CreateRequest{
			AppName:   appName,
			UserID:    userID,
			SessionID: sessionID,
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		indexKey := buildSessionIndexKey(appName, userID)
		isMember, err := rdb.SIsMember(ctx, indexKey, sessionID).Result()
		if err != nil {
			t.Fatalf("SIsMember check failed: %v", err)
		}
		if !isMember {
			t.Error("expected session ID to be in index set")
		}
	})

	t.Run("session has TTL set", func(t *testing.T) {
		_, err := svc.Create(ctx, &session.CreateRequest{
			AppName:   appName,
			UserID:    userID,
			SessionID: "ttl-check",
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		key := buildSessionKey(appName, userID, "ttl-check")
		ttl, err := rdb.TTL(ctx, key).Result()
		if err != nil {
			t.Fatalf("TTL check failed: %v", err)
		}
		if ttl <= 0 {
			t.Errorf("expected positive TTL, got: %v", ttl)
		}
		if ttl > 30*time.Second {
			t.Errorf("expected TTL <= 30s, got: %v", ttl)
		}
	})

	t.Run("create with initial state", func(t *testing.T) {
		state := map[string]any{"key1": "value1", "key2": float64(42)}
		resp, err := svc.Create(ctx, &session.CreateRequest{
			AppName:   appName,
			UserID:    userID,
			SessionID: "state-check",
			State:     state,
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		val, err := resp.Session.State().Get("key1")
		if err != nil {
			t.Fatalf("State.Get failed: %v", err)
		}
		if val != "value1" {
			t.Errorf("expected state key1='value1', got %v", val)
		}
	})
}

// --- Get ---

func TestGet(t *testing.T) {
	const (
		appName = "test_get_app"
		userID  = "test_get_user"
	)

	svc, rdb := setupTestRedis(t, WithTTL(30*time.Second))
	t.Cleanup(func() {
		cleanupTestKeys(t, rdb,
			fmt.Sprintf("session:%s:*", appName),
			fmt.Sprintf("events:%s:*", appName),
		)
	})

	ctx := context.Background()

	t.Run("get existing session", func(t *testing.T) {
		_, err := svc.Create(ctx, &session.CreateRequest{
			AppName:   appName,
			UserID:    userID,
			SessionID: "get-exist",
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		resp, err := svc.Get(ctx, &session.GetRequest{
			AppName:   appName,
			UserID:    userID,
			SessionID: "get-exist",
		})
		if err != nil {
			t.Fatalf("Get failed: %v", err)
		}
		if resp.Session.ID() != "get-exist" {
			t.Errorf("expected session ID 'get-exist', got %q", resp.Session.ID())
		}
		if resp.Session.AppName() != appName {
			t.Errorf("expected appName %q, got %q", appName, resp.Session.AppName())
		}
	})

	t.Run("get non-existent session returns ErrSessionNotFound", func(t *testing.T) {
		_, err := svc.Get(ctx, &session.GetRequest{
			AppName:   appName,
			UserID:    userID,
			SessionID: "non-existent",
		})
		if err == nil {
			t.Fatal("expected error for non-existent session")
		}
		if !errorContains(err, ErrSessionNotFound.Error()) {
			t.Errorf("expected ErrSessionNotFound, got: %v", err)
		}
	})

	t.Run("get session with events", func(t *testing.T) {
		sessionID := "get-with-events"
		createResp, err := svc.Create(ctx, &session.CreateRequest{
			AppName:   appName,
			UserID:    userID,
			SessionID: sessionID,
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		// Append 3 events
		for i := range 3 {
			evt := &session.Event{
				ID:     fmt.Sprintf("evt-%d", i),
				Author: "user",
			}
			if err := svc.AppendEvent(ctx, createResp.Session, evt); err != nil {
				t.Fatalf("AppendEvent %d failed: %v", i, err)
			}
		}

		resp, err := svc.Get(ctx, &session.GetRequest{
			AppName:   appName,
			UserID:    userID,
			SessionID: sessionID,
		})
		if err != nil {
			t.Fatalf("Get failed: %v", err)
		}

		eventCount := 0
		for range resp.Session.Events().All() {
			eventCount++
		}
		if eventCount != 3 {
			t.Errorf("expected 3 events, got %d", eventCount)
		}
	})

	t.Run("get session with NumRecentEvents filter", func(t *testing.T) {
		sessionID := "get-recent-events"
		createResp, err := svc.Create(ctx, &session.CreateRequest{
			AppName:   appName,
			UserID:    userID,
			SessionID: sessionID,
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		for i := range 5 {
			evt := &session.Event{ID: fmt.Sprintf("evt-recent-%d", i), Author: "user"}
			if err := svc.AppendEvent(ctx, createResp.Session, evt); err != nil {
				t.Fatalf("AppendEvent %d failed: %v", i, err)
			}
		}

		resp, err := svc.Get(ctx, &session.GetRequest{
			AppName:         appName,
			UserID:          userID,
			SessionID:       sessionID,
			NumRecentEvents: 2,
		})
		if err != nil {
			t.Fatalf("Get failed: %v", err)
		}

		// Use Len() to check the filtered cache before All() triggers a Redis refresh.
		// All() re-reads events from Redis and bypasses the NumRecentEvents filter,
		// which is applied only during Get(). Len() reads the initially cached slice.
		eventCount := resp.Session.Events().Len()
		if eventCount != 2 {
			t.Errorf("expected 2 recent events, got %d", eventCount)
		}
	})
}

// --- List ---

func TestList(t *testing.T) {
	const (
		appName = "test_list_app"
		userID  = "test_list_user"
	)

	svc, rdb := setupTestRedis(t, WithTTL(30*time.Second))
	t.Cleanup(func() {
		cleanupTestKeys(t, rdb,
			fmt.Sprintf("session:%s:*", appName),
			fmt.Sprintf("events:%s:*", appName),
		)
	})

	ctx := context.Background()

	t.Run("list empty returns nil sessions", func(t *testing.T) {
		resp, err := svc.List(ctx, &session.ListRequest{
			AppName: appName,
			UserID:  "no-sessions-user",
		})
		if err != nil {
			t.Fatalf("List failed: %v", err)
		}
		if resp.Sessions != nil {
			t.Errorf("expected nil sessions, got %d", len(resp.Sessions))
		}
	})

	t.Run("list returns created sessions", func(t *testing.T) {
		for i := range 3 {
			_, err := svc.Create(ctx, &session.CreateRequest{
				AppName:   appName,
				UserID:    userID,
				SessionID: fmt.Sprintf("list-sess-%d", i),
			})
			if err != nil {
				t.Fatalf("Create %d failed: %v", i, err)
			}
		}

		resp, err := svc.List(ctx, &session.ListRequest{
			AppName: appName,
			UserID:  userID,
		})
		if err != nil {
			t.Fatalf("List failed: %v", err)
		}
		if len(resp.Sessions) != 3 {
			t.Errorf("expected 3 sessions, got %d", len(resp.Sessions))
		}
	})

	t.Run("list with stale sessions returns only live ones", func(t *testing.T) {
		staleUser := "stale-user"
		// Create 2 sessions
		for i := range 2 {
			_, err := svc.Create(ctx, &session.CreateRequest{
				AppName:   appName,
				UserID:    staleUser,
				SessionID: fmt.Sprintf("stale-sess-%d", i),
			})
			if err != nil {
				t.Fatalf("Create %d failed: %v", i, err)
			}
		}

		// Manually delete one session's data key (simulate TTL expiry) but leave it in the index
		expiredKey := buildSessionKey(appName, staleUser, "stale-sess-0")
		if err := rdb.Del(ctx, expiredKey).Err(); err != nil {
			t.Fatalf("Del failed: %v", err)
		}

		resp, err := svc.List(ctx, &session.ListRequest{
			AppName: appName,
			UserID:  staleUser,
		})
		if err != nil {
			t.Fatalf("List failed: %v", err)
		}
		if len(resp.Sessions) != 1 {
			t.Errorf("expected 1 live session, got %d", len(resp.Sessions))
		}
		if resp.Sessions[0].ID() != "stale-sess-1" {
			t.Errorf("expected live session 'stale-sess-1', got %q", resp.Sessions[0].ID())
		}
	})

	t.Run("list with all sessions expired returns empty", func(t *testing.T) {
		expUser := "all-expired-user"
		for i := range 2 {
			_, err := svc.Create(ctx, &session.CreateRequest{
				AppName:   appName,
				UserID:    expUser,
				SessionID: fmt.Sprintf("exp-sess-%d", i),
			})
			if err != nil {
				t.Fatalf("Create %d failed: %v", i, err)
			}
		}

		// Delete all session data keys
		for i := range 2 {
			key := buildSessionKey(appName, expUser, fmt.Sprintf("exp-sess-%d", i))
			rdb.Del(ctx, key)
		}

		resp, err := svc.List(ctx, &session.ListRequest{
			AppName: appName,
			UserID:  expUser,
		})
		if err != nil {
			t.Fatalf("List failed: %v", err)
		}
		if len(resp.Sessions) != 0 {
			t.Errorf("expected 0 sessions, got %d", len(resp.Sessions))
		}
	})

	t.Run("user isolation in list", func(t *testing.T) {
		userA := "list-user-a"
		userB := "list-user-b"

		_, err := svc.Create(ctx, &session.CreateRequest{
			AppName: appName, UserID: userA, SessionID: "iso-a",
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}
		_, err = svc.Create(ctx, &session.CreateRequest{
			AppName: appName, UserID: userB, SessionID: "iso-b",
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		resp, err := svc.List(ctx, &session.ListRequest{AppName: appName, UserID: userA})
		if err != nil {
			t.Fatalf("List failed: %v", err)
		}
		if len(resp.Sessions) != 1 {
			t.Errorf("expected 1 session for user A, got %d", len(resp.Sessions))
		}
	})
}

// --- Stale Session Cleanup ---

func TestListStaleCleanup(t *testing.T) {
	const (
		appName = "test_stale_app"
		userID  = "test_stale_user"
	)

	svc, rdb := setupTestRedis(t, WithTTL(30*time.Second))
	t.Cleanup(func() {
		cleanupTestKeys(t, rdb,
			fmt.Sprintf("session:%s:*", appName),
			fmt.Sprintf("events:%s:*", appName),
		)
	})

	ctx := context.Background()

	t.Run("stale IDs removed from index after List", func(t *testing.T) {
		// Create 3 sessions
		for i := range 3 {
			_, err := svc.Create(ctx, &session.CreateRequest{
				AppName:   appName,
				UserID:    userID,
				SessionID: fmt.Sprintf("cleanup-sess-%d", i),
			})
			if err != nil {
				t.Fatalf("Create %d failed: %v", i, err)
			}
		}

		// Delete session data for 2 of them (simulate TTL expiry)
		rdb.Del(ctx, buildSessionKey(appName, userID, "cleanup-sess-0"))
		rdb.Del(ctx, buildSessionKey(appName, userID, "cleanup-sess-1"))

		indexKey := buildSessionIndexKey(appName, userID)

		// Verify index has all 3 before List
		membersBefore, _ := rdb.SMembers(ctx, indexKey).Result()
		if len(membersBefore) != 3 {
			t.Fatalf("expected 3 members in index before List, got %d", len(membersBefore))
		}

		// Call List (triggers async cleanup)
		resp, err := svc.List(ctx, &session.ListRequest{AppName: appName, UserID: userID})
		if err != nil {
			t.Fatalf("List failed: %v", err)
		}
		if len(resp.Sessions) != 1 {
			t.Errorf("expected 1 live session, got %d", len(resp.Sessions))
		}

		// Wait for async goroutine to complete
		time.Sleep(500 * time.Millisecond)

		// Verify stale IDs were removed from index
		membersAfter, _ := rdb.SMembers(ctx, indexKey).Result()
		if len(membersAfter) != 1 {
			t.Errorf("expected 1 member in index after cleanup, got %d: %v", len(membersAfter), membersAfter)
		}
	})

	t.Run("cleanup does not remove concurrently re-created session", func(t *testing.T) {
		raceUser := "race-user"

		// Create a session
		_, err := svc.Create(ctx, &session.CreateRequest{
			AppName:   appName,
			UserID:    raceUser,
			SessionID: "race-sess",
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		// Delete the data key to simulate TTL expiry
		rdb.Del(ctx, buildSessionKey(appName, raceUser, "race-sess"))

		// Now call cleanStaleSessionIDs directly but re-create the session first
		// (simulating the race: session was re-created between pipeline GET and cleanup)
		_, err = svc.Create(ctx, &session.CreateRequest{
			AppName:   appName,
			UserID:    raceUser,
			SessionID: "race-sess",
		})
		if err != nil {
			t.Fatalf("Re-create failed: %v", err)
		}

		// Now run the cleanup with "race-sess" in the stale list
		svc.cleanStaleSessionIDs(ctx, appName, raceUser, []string{"race-sess"})

		// The Lua script should NOT have removed "race-sess" because its key now exists
		indexKey := buildSessionIndexKey(appName, raceUser)
		isMember, _ := rdb.SIsMember(ctx, indexKey, "race-sess").Result()
		if !isMember {
			t.Error("cleanup incorrectly removed a re-created session from the index")
		}
	})

	t.Run("cleanup with truly expired sessions removes them", func(t *testing.T) {
		expUser := "expire-clean-user"

		// Create sessions
		for i := range 2 {
			_, err := svc.Create(ctx, &session.CreateRequest{
				AppName:   appName,
				UserID:    expUser,
				SessionID: fmt.Sprintf("exp-clean-%d", i),
			})
			if err != nil {
				t.Fatalf("Create %d failed: %v", i, err)
			}
		}

		// Delete both data keys
		rdb.Del(ctx, buildSessionKey(appName, expUser, "exp-clean-0"))
		rdb.Del(ctx, buildSessionKey(appName, expUser, "exp-clean-1"))

		// Run cleanup directly
		svc.cleanStaleSessionIDs(ctx, appName, expUser, []string{"exp-clean-0", "exp-clean-1"})

		// Verify both removed from index
		indexKey := buildSessionIndexKey(appName, expUser)
		members, _ := rdb.SMembers(ctx, indexKey).Result()
		if len(members) != 0 {
			t.Errorf("expected 0 members after cleanup, got %d: %v", len(members), members)
		}
	})
}

// --- Delete ---

func TestDelete(t *testing.T) {
	const (
		appName = "test_delete_app"
		userID  = "test_delete_user"
	)

	svc, rdb := setupTestRedis(t, WithTTL(30*time.Second))
	t.Cleanup(func() {
		cleanupTestKeys(t, rdb,
			fmt.Sprintf("session:%s:*", appName),
			fmt.Sprintf("events:%s:*", appName),
		)
	})

	ctx := context.Background()

	t.Run("delete existing session", func(t *testing.T) {
		createResp, err := svc.Create(ctx, &session.CreateRequest{
			AppName: appName, UserID: userID, SessionID: "del-sess",
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		// Append an event to ensure events key exists
		evt := &session.Event{ID: "del-evt", Author: "user"}
		if err := svc.AppendEvent(ctx, createResp.Session, evt); err != nil {
			t.Fatalf("AppendEvent failed: %v", err)
		}

		err = svc.Delete(ctx, &session.DeleteRequest{
			AppName: appName, UserID: userID, SessionID: "del-sess",
		})
		if err != nil {
			t.Fatalf("Delete failed: %v", err)
		}

		// Verify session key deleted
		exists, _ := rdb.Exists(ctx, buildSessionKey(appName, userID, "del-sess")).Result()
		if exists != 0 {
			t.Error("expected session key to be deleted")
		}

		// Verify events key deleted
		exists, _ = rdb.Exists(ctx, buildEventsKey(appName, userID, "del-sess")).Result()
		if exists != 0 {
			t.Error("expected events key to be deleted")
		}

		// Verify removed from index
		isMember, _ := rdb.SIsMember(ctx, buildSessionIndexKey(appName, userID), "del-sess").Result()
		if isMember {
			t.Error("expected session to be removed from index")
		}
	})

	t.Run("delete non-existent session does not error", func(t *testing.T) {
		err := svc.Delete(ctx, &session.DeleteRequest{
			AppName: appName, UserID: userID, SessionID: "non-existent",
		})
		if err != nil {
			t.Fatalf("Delete of non-existent session should not error: %v", err)
		}
	})

	t.Run("get after delete returns not found", func(t *testing.T) {
		_, err := svc.Create(ctx, &session.CreateRequest{
			AppName: appName, UserID: userID, SessionID: "del-then-get",
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		err = svc.Delete(ctx, &session.DeleteRequest{
			AppName: appName, UserID: userID, SessionID: "del-then-get",
		})
		if err != nil {
			t.Fatalf("Delete failed: %v", err)
		}

		_, err = svc.Get(ctx, &session.GetRequest{
			AppName: appName, UserID: userID, SessionID: "del-then-get",
		})
		if err == nil {
			t.Fatal("expected error after deleting session")
		}
		if !errorContains(err, ErrSessionNotFound.Error()) {
			t.Errorf("expected ErrSessionNotFound, got: %v", err)
		}
	})
}

// --- AppendEvent ---

func TestAppendEvent(t *testing.T) {
	const (
		appName = "test_event_app"
		userID  = "test_event_user"
	)

	svc, rdb := setupTestRedis(t, WithTTL(30*time.Second))
	t.Cleanup(func() {
		cleanupTestKeys(t, rdb,
			fmt.Sprintf("session:%s:*", appName),
			fmt.Sprintf("events:%s:*", appName),
		)
	})

	ctx := context.Background()

	t.Run("nil session returns error", func(t *testing.T) {
		err := svc.AppendEvent(ctx, nil, &session.Event{ID: "evt"})
		if err == nil {
			t.Fatal("expected error for nil session")
		}
		if err != ErrNilSession {
			t.Errorf("expected ErrNilSession, got: %v", err)
		}
	})

	t.Run("event appended and retrievable", func(t *testing.T) {
		createResp, err := svc.Create(ctx, &session.CreateRequest{
			AppName: appName, UserID: userID, SessionID: "evt-append",
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		evt := &session.Event{ID: "evt-1", Author: "user"}
		if err := svc.AppendEvent(ctx, createResp.Session, evt); err != nil {
			t.Fatalf("AppendEvent failed: %v", err)
		}

		// Verify event is in events list
		evKey := buildEventsKey(appName, userID, "evt-append")
		length, err := rdb.LLen(ctx, evKey).Result()
		if err != nil {
			t.Fatalf("LLen failed: %v", err)
		}
		if length != 1 {
			t.Errorf("expected 1 event in list, got %d", length)
		}
	})

	t.Run("event gets timestamp and ID assigned", func(t *testing.T) {
		createResp, err := svc.Create(ctx, &session.CreateRequest{
			AppName: appName, UserID: userID, SessionID: "evt-timestamp",
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		evt := &session.Event{Author: "user"} // No ID or Timestamp
		before := time.Now()
		if err := svc.AppendEvent(ctx, createResp.Session, evt); err != nil {
			t.Fatalf("AppendEvent failed: %v", err)
		}

		if evt.ID == "" {
			t.Error("expected auto-generated event ID")
		}
		if evt.Timestamp.Before(before) {
			t.Error("expected event timestamp to be set")
		}
	})

	t.Run("multiple events preserve order", func(t *testing.T) {
		createResp, err := svc.Create(ctx, &session.CreateRequest{
			AppName: appName, UserID: userID, SessionID: "evt-order",
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		for i := range 5 {
			evt := &session.Event{ID: fmt.Sprintf("order-%d", i), Author: "user"}
			if err := svc.AppendEvent(ctx, createResp.Session, evt); err != nil {
				t.Fatalf("AppendEvent %d failed: %v", i, err)
			}
		}

		// Read events from Redis and verify order
		evKey := buildEventsKey(appName, userID, "evt-order")
		eventData, err := rdb.LRange(ctx, evKey, 0, -1).Result()
		if err != nil {
			t.Fatalf("LRange failed: %v", err)
		}
		if len(eventData) != 5 {
			t.Fatalf("expected 5 events, got %d", len(eventData))
		}

		for i, ed := range eventData {
			var evt session.Event
			if err := sonic.Unmarshal([]byte(ed), &evt); err != nil {
				t.Fatalf("unmarshal event %d failed: %v", i, err)
			}
			expectedID := fmt.Sprintf("order-%d", i)
			if evt.ID != expectedID {
				t.Errorf("event %d: expected ID %q, got %q", i, expectedID, evt.ID)
			}
		}
	})

	t.Run("append event updates session LastUpdateTime", func(t *testing.T) {
		createResp, err := svc.Create(ctx, &session.CreateRequest{
			AppName: appName, UserID: userID, SessionID: "evt-update-time",
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		originalTime := createResp.Session.LastUpdateTime()

		time.Sleep(10 * time.Millisecond)

		evt := &session.Event{ID: "time-evt", Author: "user"}
		if err := svc.AppendEvent(ctx, createResp.Session, evt); err != nil {
			t.Fatalf("AppendEvent failed: %v", err)
		}

		// Re-get and check
		getResp, err := svc.Get(ctx, &session.GetRequest{
			AppName: appName, UserID: userID, SessionID: "evt-update-time",
		})
		if err != nil {
			t.Fatalf("Get failed: %v", err)
		}

		if !getResp.Session.LastUpdateTime().After(originalTime) {
			t.Error("expected LastUpdateTime to be updated after AppendEvent")
		}
	})
}

func TestAppendEventTTL(t *testing.T) {
	const (
		appName = "test_event_ttl_app"
		userID  = "test_event_ttl_user"
	)

	svc, rdb := setupTestRedis(t, WithTTL(30*time.Second))
	t.Cleanup(func() {
		cleanupTestKeys(t, rdb,
			fmt.Sprintf("session:%s:*", appName),
			fmt.Sprintf("events:%s:*", appName),
		)
	})

	ctx := context.Background()

	t.Run("events key has TTL set", func(t *testing.T) {
		createResp, err := svc.Create(ctx, &session.CreateRequest{
			AppName: appName, UserID: userID, SessionID: "evt-ttl-check",
		})
		if err != nil {
			t.Fatalf("Create failed: %v", err)
		}

		evt := &session.Event{ID: "ttl-evt", Author: "user"}
		if err := svc.AppendEvent(ctx, createResp.Session, evt); err != nil {
			t.Fatalf("AppendEvent failed: %v", err)
		}

		evKey := buildEventsKey(appName, userID, "evt-ttl-check")
		ttl, err := rdb.TTL(ctx, evKey).Result()
		if err != nil {
			t.Fatalf("TTL check failed: %v", err)
		}
		if ttl <= 0 {
			t.Errorf("expected positive TTL on events key, got: %v", ttl)
		}
	})
}

// --- AppendEvent: Index TTL Refresh ---

func TestAppendEventRefreshesIndexTTL(t *testing.T) {
	const (
		appName = "test_idx_ttl_app"
		userID  = "test_idx_ttl_user"
	)

	svc, rdb := setupTestRedis(t, WithTTL(30*time.Second))
	t.Cleanup(func() {
		cleanupTestKeys(t, rdb,
			fmt.Sprintf("session:%s:*", appName),
			fmt.Sprintf("events:%s:*", appName),
		)
	})

	ctx := context.Background()

	createResp, err := svc.Create(ctx, &session.CreateRequest{
		AppName: appName, UserID: userID, SessionID: "idx-ttl-sess",
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	indexKey := buildSessionIndexKey(appName, userID)

	// Get initial TTL of the index key
	ttlBefore, err := rdb.TTL(ctx, indexKey).Result()
	if err != nil {
		t.Fatalf("TTL check failed: %v", err)
	}
	if ttlBefore <= 0 {
		t.Fatalf("expected positive initial TTL, got: %v", ttlBefore)
	}

	// Wait a bit so TTL decreases
	time.Sleep(2 * time.Second)

	ttlAfterWait, err := rdb.TTL(ctx, indexKey).Result()
	if err != nil {
		t.Fatalf("TTL check failed: %v", err)
	}
	if ttlAfterWait >= ttlBefore {
		t.Fatalf("expected TTL to decrease after waiting, before=%v after=%v", ttlBefore, ttlAfterWait)
	}

	// Append event — this should refresh the index key TTL
	evt := &session.Event{ID: "refresh-evt", Author: "user"}
	if err := svc.AppendEvent(ctx, createResp.Session, evt); err != nil {
		t.Fatalf("AppendEvent failed: %v", err)
	}

	ttlAfterAppend, err := rdb.TTL(ctx, indexKey).Result()
	if err != nil {
		t.Fatalf("TTL check failed: %v", err)
	}

	// After refresh, TTL should be greater than the decreased value
	if ttlAfterAppend <= ttlAfterWait {
		t.Errorf("expected index TTL to be refreshed after AppendEvent, before=%v after=%v",
			ttlAfterWait, ttlAfterAppend)
	}
}

// --- cleanStaleSessionIDs (Lua script) ---

func TestCleanStaleSessionIDs(t *testing.T) {
	const (
		appName = "test_lua_app"
		userID  = "test_lua_user"
	)

	svc, rdb := setupTestRedis(t, WithTTL(30*time.Second))
	t.Cleanup(func() {
		cleanupTestKeys(t, rdb,
			fmt.Sprintf("session:%s:*", appName),
			fmt.Sprintf("events:%s:*", appName),
		)
	})

	ctx := context.Background()

	t.Run("removes only truly expired IDs", func(t *testing.T) {
		// Create 3 sessions
		for i := range 3 {
			_, err := svc.Create(ctx, &session.CreateRequest{
				AppName: appName, UserID: userID,
				SessionID: fmt.Sprintf("lua-sess-%d", i),
			})
			if err != nil {
				t.Fatalf("Create %d failed: %v", i, err)
			}
		}

		// Delete data for sess-0 (expired), keep sess-1 and sess-2 (alive)
		rdb.Del(ctx, buildSessionKey(appName, userID, "lua-sess-0"))

		// Call cleanup with all 3 in stale list
		svc.cleanStaleSessionIDs(ctx, appName, userID,
			[]string{"lua-sess-0", "lua-sess-1", "lua-sess-2"})

		indexKey := buildSessionIndexKey(appName, userID)

		// sess-0 should be removed (truly expired)
		isMember, _ := rdb.SIsMember(ctx, indexKey, "lua-sess-0").Result()
		if isMember {
			t.Error("expected lua-sess-0 to be removed from index")
		}

		// sess-1 and sess-2 should still be in index (alive)
		for _, id := range []string{"lua-sess-1", "lua-sess-2"} {
			isMember, _ = rdb.SIsMember(ctx, indexKey, id).Result()
			if !isMember {
				t.Errorf("expected %s to still be in index", id)
			}
		}
	})

	t.Run("empty stale list is a no-op", func(t *testing.T) {
		// Should not panic or error
		svc.cleanStaleSessionIDs(ctx, appName, "empty-user", []string{})
	})
}

// --- Integration: Full Lifecycle ---

func TestSessionLifecycle(t *testing.T) {
	const (
		appName = "test_lifecycle_app"
		userID  = "test_lifecycle_user"
	)

	svc, rdb := setupTestRedis(t, WithTTL(30*time.Second))
	t.Cleanup(func() {
		cleanupTestKeys(t, rdb,
			fmt.Sprintf("session:%s:*", appName),
			fmt.Sprintf("events:%s:*", appName),
		)
	})

	ctx := context.Background()

	// 1. Create
	createResp, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: "lifecycle-sess",
		State:     map[string]any{"count": float64(0)},
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// 2. Get
	getResp, err := svc.Get(ctx, &session.GetRequest{
		AppName: appName, UserID: userID, SessionID: "lifecycle-sess",
	})
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if getResp.Session.ID() != "lifecycle-sess" {
		t.Errorf("unexpected session ID: %q", getResp.Session.ID())
	}

	// 3. AppendEvent
	evt := &session.Event{ID: "lc-evt-1", Author: "user"}
	if err := svc.AppendEvent(ctx, createResp.Session, evt); err != nil {
		t.Fatalf("AppendEvent failed: %v", err)
	}

	// 4. List
	listResp, err := svc.List(ctx, &session.ListRequest{
		AppName: appName, UserID: userID,
	})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	if len(listResp.Sessions) != 1 {
		t.Errorf("expected 1 session in list, got %d", len(listResp.Sessions))
	}

	// 5. Get with events
	getResp, err = svc.Get(ctx, &session.GetRequest{
		AppName: appName, UserID: userID, SessionID: "lifecycle-sess",
	})
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	eventCount := 0
	for range getResp.Session.Events().All() {
		eventCount++
	}
	if eventCount != 1 {
		t.Errorf("expected 1 event, got %d", eventCount)
	}

	// 6. Delete
	err = svc.Delete(ctx, &session.DeleteRequest{
		AppName: appName, UserID: userID, SessionID: "lifecycle-sess",
	})
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	// 7. Verify deleted
	_, err = svc.Get(ctx, &session.GetRequest{
		AppName: appName, UserID: userID, SessionID: "lifecycle-sess",
	})
	if err == nil {
		t.Fatal("expected error after delete")
	}

	listResp, err = svc.List(ctx, &session.ListRequest{
		AppName: appName, UserID: userID,
	})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	if len(listResp.Sessions) != 0 {
		t.Errorf("expected 0 sessions after delete, got %d", len(listResp.Sessions))
	}
}

// errorContains checks if the error message contains the given substring.
func errorContains(err error, substr string) bool {
	if err == nil {
		return false
	}
	return containsString(err.Error(), substr)
}

func containsString(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
