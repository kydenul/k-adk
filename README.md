# K-ADK

[![Go Version](https://img.shields.io/badge/Go-1.25+-blue.svg)](https://go.dev/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

K-ADK is a Go library that provides adapter implementations for integrating [Google's Agent Development Kit (ADK)](https://google.golang.org/adk) with **OpenAI** and **Anthropic** LLM APIs. It enables developers to use Google's ADK framework for building AI agents while choosing their preferred LLM provider.

## Features

- **OpenAI Adapter** - Full support for OpenAI API and compatible providers (Ollama, vLLM, OpenRouter, etc.)
- **Anthropic Adapter** - Native Claude API support with automatic message history repair
- **Redis Session Service** - Persistent session management with Redis backend
- **PostgreSQL Session Persister** - Hybrid Redis + PostgreSQL session persistence for durability
- **PostgreSQL Memory Service** - Long-term memory storage with pgvector semantic search
- **Streaming Support** - Real-time streaming responses via Go 1.23+ iterators
- **Tool Calling** - Full function/tool calling support with automatic ID normalization
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

// Create session service with 24-hour TTL
sessionSrv, _ := ksess.NewRedisSessionService(rdb, 24*time.Hour, logger)

// Use with ADK runner
runner, _ := runner.New(runner.Config{
    AppName:        "myapp",
    Agent:          agent,
    SessionService: sessionSrv,
})
```

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
    10*time.Minute, // Redis TTL
    logger,
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
- **Tool Call ID Normalization**:
  - OpenAI: 40-character limit with SHA256 hashing for longer IDs
  - Anthropic: Regex sanitization for `[a-zA-Z0-9_-]` pattern compliance
- **Message History Repair**: Anthropic adapter includes `repairMessageHistory()` to fix sequences where `tool_use` blocks lack matching `tool_result`

## Project Structure

```
k-adk/
├── genai/
│   ├── openai/           # OpenAI adapter implementation
│   │   ├── openai.go     # Main adapter (model.LLM interface)
│   │   └── base.go       # Conversion utilities
│   └── anthropic/        # Anthropic adapter implementation
│       ├── anthropic.go  # Main adapter (model.LLM interface)
│       └── base.go       # Conversion utilities
├── session/
│   ├── persister.go      # Persister interface for long-term storage
│   ├── redis/            # Redis session service
│   │   ├── service.go    # session.Service implementation
│   │   ├── session.go    # Session struct
│   │   ├── state.go      # State management
│   │   └── events.go     # Event handling
│   └── postgres/         # PostgreSQL session persister
│       ├── client.go     # PostgreSQL client with connection pool
│       └── persister.go  # Async session/event persistence
├── memory/
│   └── postgres/         # PostgreSQL memory service
│       ├── memory.go     # memory.Service implementation
│       └── embedding.go  # Embedding utilities
├── internal/
│   └── discard_log/      # No-op logger implementation
└── examples/
    ├── openai-cli/       # CLI example with OpenAI
    ├── session/          # Redis session example
    ├── persist/          # Redis + PostgreSQL hybrid persistence demo
    ├── web/              # Multi-agent web example
    └── gin/              # Gin REST API server (ADK-compatible)
```

## Configuration

### OpenAI Config

| Field | Type | Description |
|-------|------|-------------|
| `ModelName` | string | Model to use (e.g., "gpt-4o", "qwen3:8b") |
| `APIKey` | string | API key (falls back to `OPENAI_API_KEY` env var) |
| `BaseURL` | string | API endpoint (falls back to `OPENAI_API_BASE` then OpenAI default) |
| `Logger` | log.Logger | Optional logger instance |

### Anthropic Config

| Field | Type | Description |
|-------|------|-------------|
| `ModelName` | string | Model to use (e.g., "claude-sonnet-4-20250514") |
| `APIKey` | string | API key (falls back to `ANTHROPIC_API_KEY` env var) |
| `BaseURL` | string | Optional custom API endpoint |
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

- Go 1.25+
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
