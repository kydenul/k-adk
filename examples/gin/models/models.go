package models

import (
	"maps"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// ============================================================================
// Request/Response models
// ============================================================================

// RunAgentRequest
type RunAgentRequest struct {
	AppName    string         `json:"appName"`
	UserID     string         `json:"userId"`
	SessionID  string         `json:"sessionId"`
	NewMessage genai.Content  `json:"newMessage"`
	Streaming  bool           `json:"streaming,omitempty"`
	StateDelta map[string]any `json:"stateDelta,omitempty"`
}

// Event
type Event struct {
	ID                 string                   `json:"id"`
	Time               int64                    `json:"time"`
	InvocationID       string                   `json:"invocationId"`
	Branch             string                   `json:"branch"`
	Author             string                   `json:"author"`
	Partial            bool                     `json:"partial"`
	LongRunningToolIDs []string                 `json:"longRunningToolIds"`
	Content            *genai.Content           `json:"content"`
	GroundingMetadata  *genai.GroundingMetadata `json:"groundingMetadata"`
	TurnComplete       bool                     `json:"turnComplete"`
	Interrupted        bool                     `json:"interrupted"`
	ErrorCode          string                   `json:"errorCode"`
	ErrorMessage       string                   `json:"errorMessage"`
	Actions            EventActions             `json:"actions"`
}

// EventActions represents actions performed during an event.
type EventActions struct {
	StateDelta        map[string]any   `json:"stateDelta"`
	ArtifactDelta     map[string]int64 `json:"artifactDelta"`
	SkipSummarization bool             `json:"skipSummarization,omitempty"`
}

// Session represents a session in the API response.
type Session struct {
	ID        string         `json:"id"`
	AppName   string         `json:"appName"`
	UserID    string         `json:"userId"`
	UpdatedAt int64          `json:"lastUpdateTime"`
	Events    []Event        `json:"events"`
	State     map[string]any `json:"state"`
}

// CreateSessionRequest is the request body for creating a session.
type CreateSessionRequest struct {
	State  map[string]any `json:"state,omitempty"`
	Events []Event        `json:"events,omitempty"`
}

// ============================================================================
// Conversion functions
// ============================================================================

// fromSessionEvent converts a session.Event to an API Event.
func FromSessionEvent(e *session.Event) Event {
	return Event{
		ID:                 e.ID,
		Time:               e.Timestamp.Unix(),
		InvocationID:       e.InvocationID,
		Branch:             e.Branch,
		Author:             e.Author,
		Partial:            e.Partial,
		LongRunningToolIDs: e.LongRunningToolIDs,
		Content:            e.Content,
		GroundingMetadata:  e.GroundingMetadata,
		TurnComplete:       e.TurnComplete,
		Interrupted:        e.Interrupted,
		ErrorCode:          e.ErrorCode,
		ErrorMessage:       e.ErrorMessage,
		Actions: EventActions{
			StateDelta:    e.Actions.StateDelta,
			ArtifactDelta: e.Actions.ArtifactDelta,
		},
	}
}

// fromSession converts a session.Session to an API Session.
func FromSession(s session.Session) Session {
	state := maps.Collect(s.State().All())

	events := make([]Event, 0, len(state))
	for e := range s.Events().All() {
		events = append(events, FromSessionEvent(e))
	}

	return Session{
		ID:        s.ID(),
		AppName:   s.AppName(),
		UserID:    s.UserID(),
		UpdatedAt: s.LastUpdateTime().Unix(),
		Events:    events,
		State:     state,
	}
}

// toSessionEvent converts an API Event to a session.Event.
func ToSessionEvent(e Event) *session.Event {
	return &session.Event{
		ID:                 e.ID,
		Timestamp:          time.Unix(e.Time, 0),
		InvocationID:       e.InvocationID,
		Branch:             e.Branch,
		Author:             e.Author,
		LongRunningToolIDs: e.LongRunningToolIDs,
		LLMResponse: model.LLMResponse{
			Content:           e.Content,
			GroundingMetadata: e.GroundingMetadata,
			Partial:           e.Partial,
			TurnComplete:      e.TurnComplete,
			Interrupted:       e.Interrupted,
			ErrorCode:         e.ErrorCode,
			ErrorMessage:      e.ErrorMessage,
		},
		Actions: session.EventActions{
			StateDelta:    e.Actions.StateDelta,
			ArtifactDelta: e.Actions.ArtifactDelta,
		},
	}
}
