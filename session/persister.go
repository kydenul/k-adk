// Package session provides interfaces and utilities for session persistence.
package session

import (
	"context"

	"google.golang.org/adk/session"
)

// Persister defines the interface for optional long-term session persistence.
// When configured, sessions and events are automatically synced to the persister.
// This interface is implemented by postgres.SessionPersister and can be used
// with redis.RedisSessionService via WithPersister option.
type Persister interface {
	// PersistSession saves or updates a session.
	PersistSession(ctx context.Context, sess session.Session) error

	// PersistEvent saves a single event (real-time sync).
	PersistEvent(ctx context.Context, sess session.Session, evt *session.Event) error

	// DeleteSession removes a session and all its events.
	DeleteSession(ctx context.Context, appName, userID, sessionID string) error

	// Close closes the persister and releases resources.
	Close() error
}
