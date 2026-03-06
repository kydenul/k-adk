# K-ADK

[![Go Version](https://img.shields.io/badge/Go-1.26+-blue.svg)](https://go.dev/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

K-ADK is a Go library that provides adapter implementations for integrating [Google's Agent Development Kit (ADK)](https://google.golang.org/adk) with **OpenAI** and **Anthropic** LLM APIs. It enables developers to use Google's ADK framework for building AI agents while choosing their preferred LLM provider.

## Features

- **OpenAI Adapter** - Full support for OpenAI API and compatible providers (Ollama, vLLM, OpenRouter, etc.)
- **Anthropic Adapter** - Native Claude API support with extended thinking and automatic message history repair
- **Multi-Modal Support** - Images, audio (wav/mp3), PDF documents, and text files across both adapters
- **ContextGuard Plugin** - Automatic context window management with token-threshold and sliding-window compaction strategies
- **Memory Toolset** - Agent-facing tools for searching, saving, updating, and deleting long-term memories
- **Redis Session Service** - Persistent session management with Redis backend
- **PostgreSQL Session Persister** - Hybrid Redis + PostgreSQL session persistence for durability
- **PostgreSQL Memory Service** - Long-term memory storage with pgvector semantic search, plus CRUD operations
- **Streaming Support** - Real-time streaming responses via Go 1.23+ iterators
- **Tool Calling** - Full function/tool calling support with automatic ID normalization
- **Custom HTTP Headers** - Both adapters support custom headers for proxy/auth scenarios
- **Gin REST API** - Production-ready HTTP server example with ADK-compatible REST API

## Installation

```bash
go get github.com/kydenul/k-adk
```

## Quick Start

### Using OpenAI Adapter

```go
package main

import (
    "context"
    "os"

    "github.com/kydenul/k-adk/genai/openai"
    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/runner"
    "google.golang.org/adk/session"
    "google.golang.org/genai"
)

func main() {
    ctx := context.Background()

    // Create the OpenAI model
    model := openai.New(openai.Config{
        ModelName: "gpt-4o",
        APIKey:    os.Getenv("OPENAI_API_KEY"),
        // BaseURL: "http://localhost:11434/v1", // Optional: for Ollama/vLLM
    })

    // Create an agent with the model
    agent, _ := llmagent.New(llmagent.Config{
        Name:        "Assistant",
        Model:       model,
        Description: "A helpful assistant",
        Instruction: "You are a helpful assistant. Be concise.",
    })

    // Standard ADK setup
    sessionSrv := session.InMemoryService()
    sess, _ := sessionSrv.Create(ctx, &session.CreateRequest{
        AppName: "myapp",
        UserID:  "user1",
    })

    runner, _ := runner.New(runner.Config{
        AppName:        "myapp",
        Agent:          agent,
        SessionService: sessionSrv,
    })

    // Run with streaming
    userMsg := genai.NewContentFromText("Hello!", genai.RoleUser)
    for event, err := range runner.Run(ctx, "user1", sess.Session.ID(), userMsg, agent.RunConfig{}) {
        if err != nil {
            panic(err)
        }
        if event.Content != nil {
            fmt.Print(event.Content.Parts[0].Text)
        }
    }
}
```

### Using Anthropic Adapter

```go
import "github.com/kydenul/k-adk/genai/anthropic"

model := anthropic.New(anthropic.Config{
    ModelName: "claude-sonnet-4-20250514",
    APIKey:    os.Getenv("ANTHROPIC_API_KEY"),
})

// With extended thinking enabled
model := anthropic.New(anthropic.Config{
    ModelName:            "claude-sonnet-4-20250514",
    APIKey:               os.Getenv("ANTHROPIC_API_KEY"),
    MaxOutputTokens:      16000,
    ThinkingBudgetTokens: 10000,
})
```

### Using with OpenAI-Compatible Providers

```go
// OpenRouter
model := openai.New(openai.Config{
    ModelName: "anthropic/claude-3.5-sonnet",
    APIKey:    os.Getenv("OPENROUTER_API_KEY"),
    BaseURL:   "https://openrouter.ai/api/v1",
})

// Ollama
model := openai.New(openai.Config{
    ModelName: "llama3:8b",
    BaseURL:   "http://localhost:11434/v1",
})

// vLLM
model := openai.New(openai.Config{
    ModelName: "meta-llama/Llama-3-8b-chat-hf",
    BaseURL:   "http://localhost:8000/v1",
})
```

## Services

### Redis Session Service

Persistent session management with automatic TTL expiration:

```go
import (
    ksess "github.com/kydenul/k-adk/session/redis"
    "time"
)

// Create Redis client
rdb, _ := ksess.NewRedisClient(&ksess.RedisConfig{
    Addr:     "localhost:6379",
    Password: "",
    DB:       0,
})

// Create session service with 7-day TTL (recommended)
sessionSrv, _ := ksess.NewRedisSessionService(rdb,
    ksess.WithTTL(7 * 24 * time.Hour),
    ksess.WithLogger(logger),
)

// Use with ADK runner
runner, _ := runner.New(runner.Config{
    AppName:        "myapp",
    Agent:          agent,
    SessionService: sessionSrv,
})
```

> **⚠️ Important: Redis is the sole read source**
>
> All read operations (`Get`, `List`) only query Redis. The optional PostgreSQL persister is **write-only** — it archives session and event data for durability, auditing, or feeding the memory service, but is never used to restore expired sessions.
>
> Once a session's Redis TTL expires, it becomes inaccessible through the session service, even if the data still exists in PostgreSQL. **We recommend setting the TTL to at least 7 days** (`7 * 24 * time.Hour`) to keep sessions available for a reasonable window.

### PostgreSQL Session Persister (Hybrid Storage)

For production deployments requiring data durability, use the hybrid Redis + PostgreSQL architecture. Redis serves as the fast primary cache while PostgreSQL provides long-term persistence:

```go
import (
    ksess "github.com/kydenul/k-adk/session/redis"
    pg "github.com/kydenul/k-adk/session/postgres"
    "time"
)

// Create PostgreSQL client
pgClient, _ := pg.NewClient(ctx, &pg.Config{
    ConnStr:      "postgres://user:pass@localhost:5432/dbname?sslmode=disable",
    MaxOpenConns: 25,
    MaxIdleConns: 10,
    ShardCount:   8, // Events are sharded across tables for scalability
})

// Create session persister (handles async persistence)
pgPersister, _ := pg.NewSessionPersister(ctx, pgClient)

// Create Redis client
rdb, _ := ksess.NewRedisClient(&ksess.RedisConfig{
    Addr: "localhost:6379",
})

// Create hybrid session service
sessionSrv, _ := ksess.NewRedisSessionService(
    rdb,
    ksess.WithTTL(7 * 24 * time.Hour), // Recommended: 7-day Redis TTL
    ksess.WithLogger(logger),
    ksess.WithPersister(pgPersister), // Enable PostgreSQL persistence
)

// Use with ADK runner - sessions automatically sync to PostgreSQL
runner, _ := runner.New(runner.Config{
    AppName:        "myapp",
    Agent:          agent,
    SessionService: sessionSrv,
})
```

**Key Features:**
- **Async Persistence**: Events are queued and persisted asynchronously (configurable buffer size)
- **Sharded Events**: Events table is sharded by user_id hash for horizontal scalability
- **Automatic Schema**: Tables and indexes are created automatically on startup
- **Graceful Fallback**: If async queue is full, falls back to synchronous persistence

### PostgreSQL Memory Service

Long-term memory with semantic search using pgvector:

```go
import (
    memory "github.com/kydenul/k-adk/memory/postgres"
)

// Create memory service with optional embedding model
memorySrv, _ := memory.NewPostgresMemoryService(ctx, memory.PgMemSvrConfig{
    ConnStr:        "postgres://user:pass@localhost:5432/dbname?sslmode=disable",
    EmbeddingModel: myEmbeddingModel, // Optional: for semantic search
    Logger:         logger,
})

// Add session to memory
memorySrv.AddSession(ctx, session)

// Search memories
results, _ := memorySrv.Search(ctx, &memory.SearchRequest{
    AppName: "myapp",
    UserID:  "user1",
    Query:   "What did we discuss about Go?",
})
```

> **⚠️ Important: Memory requires manual persistence**
>
> The ADK `runner.Run()` only writes events to `session.Service` — it does **not** call `memory.Service.AddSession()` automatically.
>
> **Why:** In ADK's design, Session is the real-time conversation store (managed by the runner), while Memory is a cross-session knowledge layer intended for developer-controlled ingestion. The runner has no knowledge of when or which sessions should be committed to long-term memory.
>
> **How to handle:** Call `AddSession` yourself after the runner finishes execution:
>
> ```go
> // After runner.Run() completes, re-fetch the session (now with latest events)
> // and persist it to memory
> resp, _ := sessionService.Get(ctx, &session.GetRequest{
>     AppName: appName, UserID: userID, SessionID: sessionID,
> })
> memoryService.AddSession(ctx, resp.Session)
> ```
>
> See `examples/gin/main.go` for a complete working example.

## Plugins & Tools

### ContextGuard Plugin

Automatic context window management that prevents token limit overflow. Supports two compaction strategies:

- **Threshold Strategy** (default): Monitors estimated token count and triggers summarization when approaching the model's context window limit.
- **Sliding Window Strategy**: Triggers summarization after a configurable number of conversation turns.

```go
import (
    "github.com/kydenul/k-adk/plugin/contextguard"
)

// Create a model registry (provides context window metadata)
registry := contextguard.NewCrushRegistry()

// Create the guard
guard := contextguard.New(registry)

// Register agents with their preferred strategy
guard.Add("my_agent", model) // Default: threshold strategy
guard.Add("chat_agent", model, contextguard.WithSlidingWindow(30)) // Sliding window: 30 turns
guard.Add("custom_agent", model, contextguard.WithMaxTokens(64000)) // Manual token limit

// Pass to runner
runner, _ := runner.New(runner.Config{
    AppName:        "myapp",
    Agent:          agent,
    SessionService: sessionSrv,
    PluginConfig:   guard.PluginConfig(),
})
```

**How it works:**
- Hooks into ADK's `BeforeModel` callback to check and compact context before each LLM call
- Uses `AfterModel` callback to calibrate heuristic token estimates against real usage metadata
- Compaction generates a summary of older messages using the agent's own LLM, then replaces them with the summary
- Summaries are stored in session state and re-injected on subsequent calls

### Memory Toolset

Provides ADK-compatible tools that agents can use to interact with long-term memory during conversations:

```go
import (
    memtools "github.com/kydenul/k-adk/tools/memory"
)

// Create the toolset (works with any memorytypes.MemoryService implementation)
memoryToolset, _ := memtools.NewToolset(memtools.ToolsetConfig{
    MemoryService: memorySrv, // e.g., PostgresMemoryService
    AppName:       "myapp",
})

// Register with agent
agent, _ := llmagent.New(llmagent.Config{
    Name:     "Assistant",
    Model:    model,
    Toolsets: []tool.Toolset{memoryToolset},
})
```

**Available tools:**

| Tool | Description |
|------|-------------|
| `search_memory` | Search long-term memory with natural language queries |
| `save_to_memory` | Save information to long-term memory with optional category |
| `update_memory` | Update an existing memory entry by ID |
| `delete_memory` | Delete a memory entry by ID |

> `update_memory` and `delete_memory` are automatically available when the underlying memory service implements `ExtendedMemoryService` (e.g., `PostgresMemoryService`). Disable them with `DisableExtendedTools: true`.

## Architecture

```
google.golang.org/adk/model.LLM (interface)
           │
           ├── genai/openai/    → Uses github.com/openai/openai-go/v3
           └── genai/anthropic/ → Uses github.com/anthropics/anthropic-sdk-go

google.golang.org/adk/session.Service (interface)
           │
           └── session/redis/   → Uses github.com/redis/go-redis/v9
                    │
                    └── (optional) session/postgres/ → Long-term persistence

google.golang.org/adk/memory.Service (interface)
           │
           └── memory/postgres/ → Uses github.com/lib/pq + pgvector

google.golang.org/adk/plugin.Plugin
           │
           └── plugin/contextguard/ → Context window management

google.golang.org/adk/tool.Toolset (interface)
           │
           └── tools/memory/ → Agent-facing memory tools
```

### Hybrid Session Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ADK Runner                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              RedisSessionService                            │
│  ┌─────────────────┐    ┌─────────────────────────────┐    │
│  │  Redis Client   │    │  Persister (optional)       │    │
│  │  (Primary)      │───▶│  PostgreSQL SessionPersister│    │
│  │  - Fast R/W     │    │  - Async queue (1000 ops)   │    │
│  │  - TTL-based    │    │  - Sharded events tables    │    │
│  └─────────────────┘    └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│       Redis         │       │     PostgreSQL      │
│  - Session cache    │       │  - sessions table   │
│  - Events list      │       │  - session_events_N │
│  - Auto-expire      │       │  (N = shard count)  │
└─────────────────────┘       └─────────────────────┘
```

### Key Implementation Details

- **Streaming**: Uses Go 1.23+ `iter.Seq2[*model.LLMResponse, error]` for streaming responses
- **Multi-Modal**: Both adapters support images (JPEG, PNG, GIF, WebP), PDFs, and text files; OpenAI adapter additionally supports audio (WAV, MP3)
- **Extended Thinking**: Anthropic adapter supports extended thinking via `ThinkingBudgetTokens` config
- **Tool Call ID Normalization**:
  - OpenAI: 40-character limit with SHA256 hashing for longer IDs
  - Anthropic: Regex sanitization for `[a-zA-Z0-9_-]` pattern compliance
- **Message History Repair**: Anthropic adapter includes `repairMessageHistory()` to fix sequences where `tool_use` blocks lack matching `tool_result`

## Project Structure

```
k-adk/
├── genai/
│   ├── openai/              # OpenAI adapter implementation
│   │   ├── openai.go        # Main adapter (model.LLM interface)
│   │   ├── openai_test.go   # Adapter unit tests
│   │   └── base.go          # Conversion utilities (images, audio, PDF, text)
│   └── anthropic/           # Anthropic adapter implementation
│       ├── anthropic.go     # Main adapter (model.LLM interface)
│       ├── anthropic_test.go# Adapter unit tests
│       └── base.go          # Conversion utilities
├── session/
│   ├── persister.go         # Persister interface for long-term storage
│   ├── redis/               # Redis session service
│   │   ├── service.go       # session.Service implementation
│   │   ├── session.go       # Session struct
│   │   ├── state.go         # State management
│   │   └── events.go        # Event handling
│   └── postgres/            # PostgreSQL session persister
│       ├── client.go        # PostgreSQL client with connection pool
│       └── persister.go     # Async session/event persistence
├── memory/
│   ├── types/               # Memory service interfaces
│   │   └── types.go         # MemoryService, ExtendedMemoryService interfaces
│   └── postgres/            # PostgreSQL memory service
│       ├── memory.go        # memory.Service + ExtendedMemoryService implementation
│       └── embedding.go     # Embedding utilities
├── plugin/
│   └── contextguard/        # Context window management plugin
│       ├── contextguard.go  # Plugin entry point and configuration
│       ├── model_registry.go        # ModelRegistry interface
│       ├── model_registry_crush.go  # Built-in model registry
│       ├── compaction_strategy_threshold.go     # Token-threshold strategy
│       ├── compaction_strategy_sliding_window.go # Sliding-window strategy
│       └── compaction_utils.go      # Summarization and token estimation
├── tools/
│   └── memory/              # Agent-facing memory tools
│       └── toolset.go       # search, save, update, delete memory tools
├── internal/
│   └── discard_log/         # No-op logger implementation
└── examples/
    ├── openai-cli/          # CLI example with OpenAI
    ├── session/             # Redis session example
    ├── persist/             # Redis + PostgreSQL hybrid persistence demo
    ├── web/                 # Multi-agent web example
    └── gin/                 # Gin REST API server (ADK-compatible)
```

## Configuration

### OpenAI Config

| Field | Type | Description |
|-------|------|-------------|
| `ModelName` | string | Model to use (e.g., "gpt-4o", "qwen3:8b") |
| `APIKey` | string | API key (falls back to `OPENAI_API_KEY` env var) |
| `BaseURL` | string | API endpoint (falls back to `OPENAI_API_BASE` then OpenAI default) |
| `HTTPOptions` | HTTPOptions | Custom HTTP headers for every request |
| `Logger` | log.Logger | Optional logger instance |

### Anthropic Config

| Field | Type | Description |
|-------|------|-------------|
| `ModelName` | string | Model to use (e.g., "claude-sonnet-4-20250514") |
| `APIKey` | string | API key (falls back to `ANTHROPIC_API_KEY` env var) |
| `BaseURL` | string | Optional custom API endpoint |
| `HTTPOptions` | HTTPOptions | Custom HTTP headers for every request |
| `MaxOutputTokens` | int64 | Default cap for output tokens (default: 4096) |
| `ThinkingBudgetTokens` | int64 | Enables extended thinking with the given budget (0 = disabled) |
| `Logger` | log.Logger | Optional logger instance |

### Redis Session Config

| Field | Type | Description |
|-------|------|-------------|
| `Addr` | string | Redis address (e.g., "localhost:6379") |
| `Password` | string | Redis password |
| `DB` | int | Redis database number |

### PostgreSQL Session Persister Config

| Field | Type | Description |
|-------|------|-------------|
| `ConnStr` | string | PostgreSQL connection string |
| `MaxOpenConns` | int | Maximum open connections (default: 25) |
| `MaxIdleConns` | int | Maximum idle connections (default: 10) |
| `ConnMaxIdleTime` | duration | Max idle time per connection (default: 10m) |
| `ConnMaxLifetime` | duration | Max lifetime per connection (default: 30m) |
| `ShardCount` | int | Number of event table shards, must be power of 2 (default: 8) |
| `Logger` | log.Logger | Optional logger instance |

**Persister Options:**

| Option | Description |
|--------|-------------|
| `WithAsyncBufferSize(n)` | Set async queue size (default: 1000, set 0 for sync mode) |

## Build Commands

| Command | Description |
|---------|-------------|
| `make` or `make build` | Full build: clean + tidy + fumpt + lint + check |
| `make compile` | Quick build without linting |
| `make test` | Run all tests with verbose output |
| `make test-cover` | Run tests with coverage report |
| `make lint` | Run golangci-lint |
| `make fumpt` | Format code with gofumpt |
| `make tidy` | Tidy and verify go modules |
| `make clean` | Clean build cache |

Debug build: `DEBUG=true make build`

## Examples

### CLI Example

```bash
cd examples/openai-cli
export OPENROUTER_API_KEY="your-key"
go run main.go
```

### Redis Session Example

```bash
cd examples/session
# Configure Redis in config.yaml
go run main.go
```

### Multi-Agent Web Example

```bash
cd examples/web
export GOOGLE_API_KEY="your-key"
# Configure Redis in config.yaml
go run main.go web --port 8080
```

### Hybrid Persistence Example

Demonstrates Redis + PostgreSQL hybrid session storage with automatic persistence:

```bash
cd examples/persist
# Configure Redis and PostgreSQL in config.yaml
go run main.go demo   # Run persistence validation demo
go run main.go serve  # Start web server with hybrid storage
```

The demo mode validates:
- Session creation and event persistence
- Data verification in both Redis and PostgreSQL
- Simulated Redis expiration with PostgreSQL data survival
- Multiple session handling

### Gin REST API Example

A production-ready HTTP server using Gin framework with ADK-compatible REST API:

```bash
cd examples/gin
export GOOGLE_API_KEY="your-key"
# Configure Redis in config.yaml
go run main.go
```

The server provides the following endpoints (compatible with built-in ADK REST API):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/list-apps` | GET | List available agents |
| `/run` | POST | Run agent (non-streaming) |
| `/run_sse` | POST | Run agent (SSE streaming) |
| `/apps/{app_name}/users/{user_id}/sessions` | GET/POST | List/Create sessions |
| `/apps/{app_name}/users/{user_id}/sessions/{session_id}` | GET/POST/DELETE | Session operations |

Example usage:

```bash
# Create a session
curl -X POST http://localhost:8080/apps/gin_agent/users/user1/sessions

# Run agent
curl -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{
    "appName": "gin_agent",
    "userId": "user1",
    "sessionId": "YOUR_SESSION_ID",
    "newMessage": {
      "role": "user",
      "parts": [{"text": "What is the weather in Tokyo?"}]
    }
  }'
```

## Requirements

- Go 1.26+
- For Redis session: Redis server
- For PostgreSQL session persister: PostgreSQL 12+
- For PostgreSQL memory: PostgreSQL with pgvector extension

## Dependencies

- [google.golang.org/adk](https://google.golang.org/adk) - Google Agent Development Kit
- [github.com/openai/openai-go/v3](https://github.com/openai/openai-go) - Official OpenAI Go SDK
- [github.com/anthropics/anthropic-sdk-go](https://github.com/anthropics/anthropic-sdk-go) - Official Anthropic Go SDK
- [github.com/redis/go-redis/v9](https://github.com/redis/go-redis) - Redis client
- [github.com/lib/pq](https://github.com/lib/pq) - PostgreSQL driver

## License

MIT License

## Contributing

Contributions are welcome! Please ensure your code passes linting:

```bash
make lint
make test
```
