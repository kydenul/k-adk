package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/kydenul/k-adk/genai/openai"
	"github.com/kydenul/log"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// global logger
var logger log.Logger

func init() {
	opt, err := log.LoadFromFile(filepath.Join(".", "config.yaml"))
	if err != nil {
		panic(fmt.Sprintf("Failed to load log config from file: %v", err))
	}
	logger = log.NewLog(opt)
}

func main() {
	ctx := context.Background()

	// NOTE: 1. create the openai model
	modelConfig := openai.Config{
		ModelName: "tngtech/deepseek-r1t2-chimera:free",
		APIKey:    os.Getenv("OPENROUTER_API_KEY"),
		BaseURL:   "https://openrouter.ai/api/v1",

		Logger: logger,
	}

	log.Debugf("Model Config: %+v", modelConfig)

	model := openai.New(modelConfig)

	// NOTE: 2. create an kAgent with the openai model
	kAgent, err := llmagent.New(llmagent.Config{
		Name:        "Assistant",
		Model:       model,
		Description: "A helpful assistant",
		Instruction: "You are a helpful assistant. Be concise.",
	})
	if err != nil {
		log.Fatalf("failed to create agent: %v", err)
	}

	// NOTE: 3. Standard ADK Setup: session service + runner
	sessionSrv := session.InMemoryService()
	sessionResp, err := sessionSrv.Create(ctx, &session.CreateRequest{
		AppName: "example",
		UserID:  "User_1",
	})
	if err != nil {
		log.Fatalf("failed to create session: %v", err)
	}

	runner, err := runner.New(runner.Config{
		AppName:        "example",
		Agent:          kAgent,
		SessionService: sessionSrv,
	})
	if err != nil {
		log.Fatalf("failed to create runner: %v", err)
	}

	// NOTE: 4. Send a message and get streaming response
	userMsg := genai.NewContentFromText("What is the capital of France?", genai.RoleUser)

	for event, err := range runner.Run(ctx, "User_1", sessionResp.Session.ID(), userMsg, agent.RunConfig{
		StreamingMode: agent.StreamingModeSSE,
	}) {
		if err != nil {
			log.Fatalf("failed to run: %v", err)
		}

		if event.Content != nil && len(event.Content.Parts) > 0 {
			log.Info(event.Content.Parts[0].Text)
		}
	}

	log.Info("Done")
}
