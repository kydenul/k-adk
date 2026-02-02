package session

import (
	"time"

	"google.golang.org/adk/session"
)

// storableSession is the JSON-serializable representation of a session.
type storableSession struct {
	ID             string         `json:"id"`
	AppName        string         `json:"app_name"`
	UserID         string         `json:"user_id"`
	State          map[string]any `json:"state"`
	LastUpdateTime time.Time      `json:"last_update_time"`
}

var _ session.Session = (*redisSession)(nil)

// redisSession implements the session.Session interface.
type redisSession struct {
	id             string
	appName        string
	userID         string
	state          *redisState
	events         *redisEvents
	lastUpdateTime time.Time
}

func (s *redisSession) ID() string                { return s.id }
func (s *redisSession) AppName() string           { return s.appName }
func (s *redisSession) UserID() string            { return s.userID }
func (s *redisSession) State() session.State      { return s.state }
func (s *redisSession) Events() session.Events    { return s.events }
func (s *redisSession) LastUpdateTime() time.Time { return s.lastUpdateTime }

func (s *redisSession) toStorable() storableSession {
	return storableSession{
		ID:             s.id,
		AppName:        s.appName,
		UserID:         s.userID,
		State:          s.state.toMap(),
		LastUpdateTime: s.lastUpdateTime,
	}
}
