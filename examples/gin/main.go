// Package main provides an example ADK agent using Gin framework as HTTP server.
// The API format is compatible with the built-in ADK REST API.
package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/bytedance/sonic"
	"github.com/gin-gonic/gin"
	"github.com/kydenul/k-adk/examples/gin/middleware"
	"github.com/kydenul/k-adk/examples/gin/models"
	ksess "github.com/kydenul/k-adk/session/redis"
	"github.com/kydenul/log"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

const (
	defaultAppName         = "gin_agent"
	defaultRedisSessionTTL = 10 * time.Minute
)

var Logger log.Logger

// init initializes the global logger by loading configuration from config.yaml.
// Panics if the configuration file cannot be loaded.
func init() {
	opt, err := log.LoadFromFile(filepath.Join(".", "config.yaml"))
	if err != nil {
		panic(fmt.Sprintf("Failed to load log config from file: %v", err))
	}
	Logger = log.NewLog(opt)
}

// Server holds the dependencies for HTTP handlers.
type Server struct {
	agentLoader    agent.Loader
	sessionService session.Service
}

// ============================================================================
// Server implementation
// ============================================================================

// NewServer creates a new Server with the given agent loader and session service.
func NewServer(agentLoader agent.Loader, sessionService session.Service) *Server {
	return &Server{
		agentLoader:    agentLoader,
		sessionService: sessionService,
	}
}

// handleRun handles the /run endpoint (compatible with ADK REST API).
// POST /run
// Request: RunAgentRequest
// Response: []Event
func (s *Server) handleRun(c *gin.Context) {
	var req models.RunAgentRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "appName, userId, and sessionId are required"})
		return
	}

	ctx := c.Request.Context()

	// Validate session exists
	_, err := s.sessionService.Get(ctx, &session.GetRequest{
		AppName:   req.AppName,
		UserID:    req.UserID,
		SessionID: req.SessionID,
	})
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("session not found: %v", err)})
		return
	}

	// Load agent
	curAgent, err := s.agentLoader.LoadAgent(req.AppName)
	if err != nil {
		c.JSON(
			http.StatusInternalServerError,
			gin.H{"error": fmt.Sprintf("failed to load agent: %v", err)},
		)
		return
	}

	// Create runner
	r, err := runner.New(runner.Config{
		AppName:        req.AppName,
		Agent:          curAgent,
		SessionService: s.sessionService,
	})
	if err != nil {
		c.JSON(
			http.StatusInternalServerError,
			gin.H{"error": fmt.Sprintf("failed to create runner: %v", err)},
		)
		return
	}

	// Determine streaming mode
	streamingMode := agent.StreamingModeNone
	if req.Streaming {
		streamingMode = agent.StreamingModeSSE
	}

	// Run and collect events
	var events []models.Event
	for event, err := range r.Run(
		ctx, req.UserID, req.SessionID, &req.NewMessage, agent.RunConfig{StreamingMode: streamingMode}) {
		if err != nil {
			c.JSON(
				http.StatusInternalServerError,
				gin.H{"error": fmt.Sprintf("runner error: %v", err)},
			)
			return
		}
		events = append(events, models.FromSessionEvent(event))
	}

	c.JSON(http.StatusOK, events)
}

// handleRunSSE handles the /run_sse endpoint with Server-Sent Events.
// POST /run_sse
// Request: RunAgentRequest
// Response: SSE stream of Event objects
func (s *Server) handleRunSSE(c *gin.Context) {
	var req models.RunAgentRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "appName, userId, and sessionId are required"})
		return
	}

	ctx := c.Request.Context()

	// Validate session exists
	_, err := s.sessionService.Get(ctx, &session.GetRequest{
		AppName:   req.AppName,
		UserID:    req.UserID,
		SessionID: req.SessionID,
	})
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("session not found: %v", err)})
		return
	}

	// Load agent
	curAgent, err := s.agentLoader.LoadAgent(req.AppName)
	if err != nil {
		c.JSON(
			http.StatusInternalServerError,
			gin.H{"error": fmt.Sprintf("failed to load agent: %v", err)},
		)
		return
	}

	// Create runner
	r, err := runner.New(runner.Config{
		AppName:        req.AppName,
		Agent:          curAgent,
		SessionService: s.sessionService,
	})
	if err != nil {
		c.JSON(
			http.StatusInternalServerError,
			gin.H{"error": fmt.Sprintf("failed to create runner: %v", err)},
		)
		return
	}

	// Set SSE headers (same as built-in ADK)
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Status(http.StatusOK)

	// Run with streaming
	for event, err := range r.Run(
		ctx, req.UserID, req.SessionID, &req.NewMessage, agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			_, _ = fmt.Fprintf(c.Writer, "Error while running agent: %v\n", err)
			c.Writer.Flush()
			continue
		}

		// Write SSE format: "data: {json}\n\n"
		eventJSON, _ := sonic.Marshal(models.FromSessionEvent(event))
		_, _ = fmt.Fprintf(c.Writer, "data: %s\n\n", eventJSON)
		c.Writer.Flush()
	}
}

// handleCreateSession handles session creation.
// POST /apps/:app_name/users/:user_id/sessions
// POST /apps/:app_name/users/:user_id/sessions/:session_id
func (s *Server) handleCreateSession(c *gin.Context) {
	appName := c.Param("app_name")
	userID := c.Param("user_id")
	sessionID := c.Param("session_id") // optional

	if appName == "" || userID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "app_name and user_id are required"})
		return
	}

	var req models.CreateSessionRequest
	if c.Request.ContentLength > 0 {
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
	}

	resp, err := s.sessionService.Create(c.Request.Context(), &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		State:     req.State,
	})
	if err != nil {
		c.JSON(
			http.StatusInternalServerError,
			gin.H{"error": fmt.Sprintf("failed to create session: %v", err)},
		)
		return
	}

	// Append initial events if provided
	for _, event := range req.Events {
		if err := s.sessionService.AppendEvent(
			c.Request.Context(),
			resp.Session,
			models.ToSessionEvent(event),
		); err != nil {
			c.JSON(
				http.StatusInternalServerError,
				gin.H{"error": fmt.Sprintf("failed to append event: %v", err)},
			)
			return
		}
	}

	c.JSON(http.StatusOK, models.FromSession(resp.Session))
}

// handleGetSession retrieves a specific session.
// GET /apps/:app_name/users/:user_id/sessions/:session_id
func (s *Server) handleGetSession(c *gin.Context) {
	appName := c.Param("app_name")
	userID := c.Param("user_id")
	sessionID := c.Param("session_id")

	if appName == "" || userID == "" || sessionID == "" {
		c.JSON(
			http.StatusBadRequest,
			gin.H{"error": "app_name, user_id, and session_id are required"},
		)
		return
	}

	resp, err := s.sessionService.Get(c.Request.Context(), &session.GetRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		c.JSON(
			http.StatusInternalServerError,
			gin.H{"error": fmt.Sprintf("failed to get session: %v", err)},
		)
		return
	}

	c.JSON(http.StatusOK, models.FromSession(resp.Session))
}

// handleListSessions lists all sessions for a user.
// GET /apps/:app_name/users/:user_id/sessions
func (s *Server) handleListSessions(c *gin.Context) {
	appName := c.Param("app_name")
	userID := c.Param("user_id")

	if appName == "" || userID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "app_name and user_id are required"})
		return
	}

	resp, err := s.sessionService.List(c.Request.Context(), &session.ListRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		c.JSON(
			http.StatusInternalServerError,
			gin.H{"error": fmt.Sprintf("failed to list sessions: %v", err)},
		)
		return
	}

	sessions := make([]models.Session, 0, len(resp.Sessions))
	for _, sess := range resp.Sessions {
		sessions = append(sessions, models.FromSession(sess))
	}

	c.JSON(http.StatusOK, sessions)
}

// handleDeleteSession deletes a specific session.
// DELETE /apps/:app_name/users/:user_id/sessions/:session_id
func (s *Server) handleDeleteSession(c *gin.Context) {
	appName := c.Param("app_name")
	userID := c.Param("user_id")
	sessionID := c.Param("session_id")

	if appName == "" || userID == "" || sessionID == "" {
		c.JSON(
			http.StatusBadRequest,
			gin.H{"error": "app_name, user_id, and session_id are required"},
		)
		return
	}

	err := s.sessionService.Delete(c.Request.Context(), &session.DeleteRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		c.JSON(
			http.StatusInternalServerError,
			gin.H{"error": fmt.Sprintf("failed to delete session: %v", err)},
		)
		return
	}

	c.JSON(http.StatusOK, nil)
}

// handleListApps lists all available apps/agents.
// GET /list-apps
func (s *Server) handleListApps(c *gin.Context) {
	agents := s.agentLoader.ListAgents()
	c.JSON(http.StatusOK, agents)
}

// handleHealth handles the /health endpoint.
func (s *Server) handleHealth(c *gin.Context) {
	log.Debugf("Health check: %v", c.Request.Form)

	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

// ============================================================================
// Tool definitions
// ============================================================================

// WeatherInput is the input for the get_weather tool.
type WeatherInput struct {
	City string `json:"city" jsonschema:"The city name to get weather for"`
}

// WeatherOutput is the output for the get_weather tool.
type WeatherOutput struct {
	Weather string `json:"weather" jsonschema:"The weather description"`
}

// getWeather is a simple tool that returns simulated weather data.
func getWeather(_ tool.Context, input WeatherInput) (WeatherOutput, error) {
	return WeatherOutput{
		Weather: fmt.Sprintf("The weather in %s is sunny with a temperature of 25°C.", input.City),
	}, nil
}

// ============================================================================
// Main
// ============================================================================

func main() {
	// Create context that cancels on interrupt signal (Ctrl+C)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Create Gemini model
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Create weather tool
	getWeatherTool, err := functiontool.New(functiontool.Config{
		Name:        "get_weather",
		Description: "Get the current weather for a city",
	}, getWeather)
	if err != nil {
		log.Fatalf("Failed to create weather tool: %v", err)
	}

	// Create session services
	rdb, err := ksess.NewRedisClient(ksess.LoadRedisConfigFromFile(
		filepath.Join(".", "config.yaml"), "Production"))
	if err != nil {
		log.Fatalf("Failed to create redis client: %v", err)
	}

	sessionService, err := ksess.NewRedisSessionService(rdb, defaultRedisSessionTTL, Logger)
	if err != nil {
		log.Fatalf("Failed to create session service: %v", err)
	}
	// Create LLMAgent
	a, err := llmagent.New(llmagent.Config{
		Name:        defaultAppName,
		Model:       model,
		Description: "A helpful assistant powered by Gin and ADK.",
		Instruction: `You are a helpful assistant. You can help users with various tasks.
When asked about weather, use the get_weather tool to get current weather information.
Always be polite and helpful.`,
		Tools: []tool.Tool{getWeatherTool},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Create agent loader
	agentLoader := agent.NewSingleLoader(a)

	// Create server
	server := NewServer(agentLoader, sessionService)

	// Setup Gin router
	r := gin.Default()

	// Add middleware
	r.Use(
		middleware.Recovery(),
		middleware.Logger(),
		middleware.CROS(),
	)

	// ========================================================================
	// Routes
	// ========================================================================

	// Health check
	r.GET("/health", server.handleHealth)

	// Runtime API
	r.POST("/run", server.handleRun)
	r.OPTIONS("/run", func(c *gin.Context) { c.Status(http.StatusNoContent) })
	r.POST("/run_sse", server.handleRunSSE)
	r.OPTIONS("/run_sse", func(c *gin.Context) { c.Status(http.StatusNoContent) })

	// Apps API
	r.GET("/list-apps", server.handleListApps)

	// Sessions API
	r.GET("/apps/:app_name/users/:user_id/sessions", server.handleListSessions)
	r.POST("/apps/:app_name/users/:user_id/sessions", server.handleCreateSession)
	r.GET("/apps/:app_name/users/:user_id/sessions/:session_id", server.handleGetSession)
	r.POST("/apps/:app_name/users/:user_id/sessions/:session_id", server.handleCreateSession)
	r.DELETE("/apps/:app_name/users/:user_id/sessions/:session_id", server.handleDeleteSession)
	r.OPTIONS(
		"/apps/:app_name/users/:user_id/sessions/:session_id",
		func(c *gin.Context) { c.Status(http.StatusNoContent) },
	)

	// Start server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      r,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in goroutine
	go func() {
		log.Infof("Starting Gin ADK server on port %s", port)
		log.Infof("API Endpoints (compatible with ADK REST API):")
		log.Infof("  GET    /health")
		log.Infof("  GET    /list-apps")
		log.Infof("  POST   /run")
		log.Infof("  POST   /run_sse")
		log.Infof("  GET    /apps/:app_name/users/:user_id/sessions")
		log.Infof("  POST   /apps/:app_name/users/:user_id/sessions")
		log.Infof("  GET    /apps/:app_name/users/:user_id/sessions/:session_id")
		log.Infof("  POST   /apps/:app_name/users/:user_id/sessions/:session_id")
		log.Infof("  DELETE /apps/:app_name/users/:user_id/sessions/:session_id")

		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for interrupt signal
	<-ctx.Done()
	log.Info("Shutting down server...")

	// Graceful shutdown with timeout
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Infof("Server forced to shutdown: %v", err)
	}

	log.Infoln("Server stopped")
}
