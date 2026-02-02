// Package main demonstrates a web-based multi-agent system using Google ADK
// with Redis session management and multiple specialized agents.
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
	"github.com/kydenul/log"
	"github.com/spf13/viper"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
	"google.golang.org/genai"

	"github.com/kydenul/k-adk/examples/web/agents"
	ksess "github.com/kydenul/k-adk/session/redis"
)

// TTL defines the session expiration time in Redis.
const TTL = 10 * time.Minute

// logger is the global logger instance used throughout the application.
var logger log.Logger

// init initializes the global logger by loading configuration from config.yaml.
// Panics if the configuration file cannot be loaded.
func init() {
	opt, err := log.LoadFromFile(filepath.Join(".", "config.yaml"))
	if err != nil {
		panic(fmt.Sprintf("Failed to load log config from file: %v", err))
	}
	logger = log.NewLog(opt)
}

// redisConfig loads Redis configuration from config.yaml using viper.
// It reads the "RedisProd" section and returns the parsed RedisConfig.
// Fatally exits if configuration loading fails.
func redisConfig() *ksess.RedisConfig {
	viper.AddConfigPath(".")
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	if err := viper.ReadInConfig(); err != nil {
		log.Fatalf("Failed to read config file: %v", err)
	}

	redisConfig := &ksess.RedisConfig{}
	if err := viper.UnmarshalKey("RedisProd", redisConfig); err != nil {
		log.Fatalf("Failed to unmarshal Redis config: %v", err)
	}

	log.Info(redisConfig)

	return redisConfig
}

// saveReportfunc is an AfterModelCallback that persists LLM response content parts
// to the artifact service. Each content part is saved with a unique UUID.
// Returns the original response and error unchanged after saving artifacts.
func saveReportfunc(
	ctx agent.CallbackContext,
	llmResponse *model.LLMResponse,
	llmResponseError error,
) (*model.LLMResponse, error) {
	if llmResponse == nil || llmResponse.Content == nil || llmResponseError != nil {
		return llmResponse, llmResponseError
	}
	for _, part := range llmResponse.Content.Parts {
		_, err := ctx.Artifacts().Save(ctx, uuid.NewString(), part)
		if err != nil {
			return nil, err
		}
	}
	return llmResponse, llmResponseError
}

// main initializes and launches a multi-agent web application.
// It sets up:
//   - Gemini model as the LLM backend
//   - Redis-based session management for conversation persistence
//   - Multiple agents: weather/time agent (root), LLM auditor, and image generator
//   - In-memory artifact storage for saving generated content
func main() {
	ctx := context.Background()
	apiKey := os.Getenv("GOOGLE_API_KEY")

	// Initialize the Gemini model for LLM operations
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{APIKey: apiKey})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Set up Redis client for session persistence
	rdb, err := ksess.NewRedisClient(redisConfig())
	if err != nil {
		log.Fatalf("Failed to create redis client: %v", err)
	}

	// Create Redis-backed session service with TTL expiration
	sessRedisService, err := ksess.NewRedisSessionService(rdb, TTL, logger)
	if err != nil {
		log.Fatalf("Failed to create session redis service: %v", sessRedisService)
	}

	// Create the root agent with weather/time capabilities and Google Search tool
	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "weather_time_agent",
		Model:       model,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "I can answer your questions about the time and weather in a city.",
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},

		AfterModelCallbacks: []llmagent.AfterModelCallback{saveReportfunc},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Initialize specialized sub-agents
	llmAuditor := agents.LLMAuditorAgent(ctx, model)
	imageGeneratorAgent := agents.ImageGeneratorAgent(ctx, model)

	// Create a multi-loader to manage all agents
	agentLoader, err := agent.NewMultiLoader(rootAgent, llmAuditor, imageGeneratorAgent)
	if err != nil {
		log.Fatalf("Failed to create multi loader: %v", err)
	}

	// Use in-memory artifact storage for generated content
	artifactService := artifact.InMemoryService()

	// Configure the launcher with all services
	config := &launcher.Config{
		SessionService:  sessRedisService,
		ArtifactService: artifactService,
		AgentLoader:     agentLoader,
	}

	// Execute the full launcher (includes web server)
	l := full.NewLauncher()
	if err := l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
