package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/kydenul/k-adk/genai/openai"
	rsess "github.com/kydenul/k-adk/session/redis"
	"github.com/kydenul/log"
	"github.com/spf13/viper"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// global logger
var logger log.Logger

const (
	appName = "K-ADK"
	userID  = "kydenul"
)

func init() {
	opt, err := log.LoadFromFile(filepath.Join(".", "config.yaml"))
	if err != nil {
		panic(fmt.Sprintf("Failed to load log config from file: %v", err))
	}
	logger = log.NewLog(opt)
}

//nolint:funlen
func main() {
	ctx := context.Background()

	model := openai.New(openai.Config{
		ModelName: "tngtech/deepseek-r1t2-chimera:free",
		APIKey:    os.Getenv("OPENROUTER_API_KEY"),
		BaseURL:   "https://openrouter.ai/api/v1",

		Logger: logger,
	})

	rdb, err := rsess.NewRedisClient(redisConfig())
	if err != nil {
		log.Fatalf("Failed to create Redis client: %v", err)
	}
	defer func() { _ = rdb.Close() }()

	sessSvr, err := rsess.NewRedisSessionService(rdb, 24*time.Hour, logger)
	if err != nil {
		log.Fatalf("Failed to create Redis session service: %v", err)
	}

	resp, err := sessSvr.Create(ctx, &session.CreateRequest{
		AppName: appName,
		UserID:  userID,
		State:   map[string]any{"conversation_started": time.Now().Format(time.RFC3339)},
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}

	log.Infof("Created session: %s", resp.Session.ID())

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "session_agent",
		Model:       model,
		Description: "An agent with session-based memory.",
		Instruction: `You are a helpful assistant. You remember everything discussed in the current conversation.
The conversation history is maintained automatically through the session.
Be conversational and reference previous parts of the conversation when relevant.`,
		Toolsets: []tool.Toolset{},
	})
	if err != nil {
		log.Fatalf("Failed to create root agent: %v", err)
	}

	runnr, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          rootAgent,
		SessionService: sessSvr,
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	// Example multi-turn conversation
	conversations := []string{
		"Hello! My name is Alice and I'm working on a Go project.",
		"What's my name?",
		"What programming language am I using?",
	}

	for i, userInput := range conversations {
		fmt.Printf("\n=== Turn %d ===\n", i+1)
		fmt.Printf("User: %s\n", userInput)

		response := runAgent(ctx, runnr, resp.Session.ID(), userInput)
		log.Infof("Agent: %s", response)
	}

	// Demonstrate session state access
	log.Infoln("\n=== Session State ===")
	updatedSess, err := sessSvr.Get(ctx, &session.GetRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: resp.Session.ID(),
	})
	if err != nil {
		log.Infof("Failed to get session: %v", err)
	} else {
		log.Infof("Session ID: %s\n", updatedSess.Session.ID())
		log.Infof("Events count: %d\n", updatedSess.Session.Events().Len())

		// Enhancement 1: Display all events
		log.Infoln("\n=== Session Events ===")
		i := 0
		for evt := range updatedSess.Session.Events().All() {
			log.Infof("Event[%d]: ID=%s, Author=%s, Time=%s",
				i, evt.ID, evt.Author, evt.Timestamp.Format(time.RFC3339))
			if evt.Content != nil && len(evt.Content.Parts) > 0 {
				// Show first 100 chars of content
				text := evt.Content.Parts[0].Text
				if len(text) > 100 {
					text = text[:100] + "..."
				}
				log.Infof("  Content: %s", text)
			}
			i++
		}

		log.Infoln("\n=== Session State ===")
		for k, v := range updatedSess.Session.State().All() {
			log.Infof("State[%s] = %v\n", k, v)
		}
	}

	// List all sessions for this user
	fmt.Println("\n=== User Sessions ===")
	listResp, err := sessSvr.List(ctx, &session.ListRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		log.Infof("Failed to list sessions: %v", err)
	} else {
		for _, s := range listResp.Sessions {
			fmt.Printf(
				"- Session: %s (last updated: %s)\n",
				s.ID(),
				s.LastUpdateTime().Format(time.RFC3339),
			)
		}
	}

	// Enhancement 2: Verify session persistence by creating a new runner
	// and continuing the conversation
	log.Infoln("\n=== Testing Session Persistence ===")
	log.Infof("Creating new runner instance to test session recovery...")

	// Create a new agent instance
	newAgent, err := llmagent.New(llmagent.Config{
		Name:        "session_agent_new",
		Model:       model,
		Description: "A new agent instance with session-based memory.",
		Instruction: `You are a helpful assistant. You remember everything discussed in the current conversation.
The conversation history is maintained automatically through the session.
Be conversational and reference previous parts of the conversation when relevant.`,
		Toolsets: []tool.Toolset{},
	})
	if err != nil {
		log.Fatalf("Failed to create new agent: %v", err)
	}

	// Create a new runner instance
	newRunner, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          newAgent,
		SessionService: sessSvr,
	})
	if err != nil {
		log.Fatalf("Failed to create new runner: %v", err)
	}

	// Continue conversation with the same session ID
	log.Infof("Continuing conversation with session: %s", resp.Session.ID())
	persistenceTestQuestions := []string{
		"Can you summarize what we've discussed so far?",
		"What's the first thing I told you?",
	}

	for i, question := range persistenceTestQuestions {
		fmt.Printf("\n=== Persistence Test %d ===\n", i+1)
		fmt.Printf("User: %s\n", question)

		response := runAgent(ctx, newRunner, resp.Session.ID(), question)
		log.Infof("Agent: %s", response)
	}

	// Verify final event count
	log.Infoln("\n=== Final Session State ===")
	finalSess, err := sessSvr.Get(ctx, &session.GetRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: resp.Session.ID(),
	})
	if err != nil {
		log.Errorf("Failed to get final session: %v", err)
	} else {
		log.Infof(
			"Total events after persistence test: %d (expected: 10 = 3 initial + 2 persistence test * 2)",
			finalSess.Session.Events().Len(),
		)
	}
}

func redisConfig() *rsess.RedisConfig {
	viper.AddConfigPath(".")
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	if err := viper.ReadInConfig(); err != nil {
		log.Fatalf("Failed to read config file: %v", err)
	}

	redisConfig := &rsess.RedisConfig{}
	if err := viper.UnmarshalKey("RedisProd", redisConfig); err != nil {
		log.Fatalf("Failed to unmarshal Redis config: %v", err)
	}

	log.Info(redisConfig)

	return redisConfig
}

func runAgent(ctx context.Context, runnr *runner.Runner, sessionID, input string) string {
	userMsg := genai.NewContentFromText(input, genai.RoleUser)

	var respText strings.Builder
	for event, err := range runnr.Run(
		ctx, userID, sessionID, userMsg, agent.RunConfig{}) {
		if err != nil {
			log.Errorf("Failed to run agent: %v", err)
			break
		}

		if event.ErrorCode != "" {
			log.Errorf("Agent error: %s - %s", event.ErrorCode, event.ErrorMessage)
			break
		}

		if event.Content != nil && len(event.Content.Parts) > 0 {
			respText.WriteString(event.Content.Parts[0].Text)
		}
	}

	return respText.String()
}
