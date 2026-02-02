
# Gin + ADK Example

This example demonstrates how to use ADK with the [Gin](https://github.com/gin-gonic/gin) web framework. The API is **fully compatible** with the built-in ADK REST API, so you can use the same client code or ADK Web UI.

## API Compatibility

This example implements the same API format as the built-in ADK REST API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/list-apps` | GET | List available agents |
| `/run` | POST | Run agent (non-streaming) |
| `/run_sse` | POST | Run agent (SSE streaming) |
| `/apps/{app_name}/users/{user_id}/sessions` | GET | List sessions |
| `/apps/{app_name}/users/{user_id}/sessions` | POST | Create session |
| `/apps/{app_name}/users/{user_id}/sessions/{session_id}` | GET | Get session |
| `/apps/{app_name}/users/{user_id}/sessions/{session_id}` | POST | Create session with ID |
| `/apps/{app_name}/users/{user_id}/sessions/{session_id}` | DELETE | Delete session |

## Prerequisites

Set your Google API key:

```bash
export GOOGLE_API_KEY=your_api_key_here
```

## Running the Server

```bash
go run ./examples/gin
```

The server starts on port 8080 by default. Set `PORT` environment variable to change it.

## Usage Examples

### 1. Create a Session

```bash
curl -X POST http://localhost:8080/apps/gin_agent/users/user1/sessions
```

Response:

```json
{
  "id": "abc123",
  "appName": "gin_agent",
  "userId": "user1",
  "lastUpdateTime": 1234567890,
  "events": [],
  "state": {}
}
```

### 2. Run Agent (Non-Streaming)

```bash
curl -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{
    "appName": "gin_agent",
    "userId": "user1",
    "sessionId": "abc123",
    "newMessage": {
      "role": "user",
      "parts": [{"text": "What is the weather in Tokyo?"}]
    }
  }'
```

Response (array of events):

```json
[
  {
    "id": "evt1",
    "time": 1234567890,
    "author": "gin_agent",
    "content": {
      "role": "model",
      "parts": [{"text": "The weather in Tokyo is sunny with a temperature of 25°C."}]
    },
    "turnComplete": true,
    ...
  }
]
```

### 3. Run Agent (SSE Streaming)

```bash
curl -X POST http://localhost:8080/run_sse \
  -H "Content-Type: application/json" \
  -d '{
    "appName": "gin_agent",
    "userId": "user1",
    "sessionId": "58af61bc-1bce-4a77-96ee-f5ac17ad3cbb",
    "newMessage": {
      "role": "user",
      "parts": [{"text": "Tell me about the weather in Paris"}]
    }
  }'
```

Response (SSE stream):

```
data: {"id":"evt1","time":1234567890,"partial":true,"content":{"parts":[{"text":"The"}]},...}

data: {"id":"evt1","time":1234567890,"partial":true,"content":{"parts":[{"text":" weather"}]},...}

data: {"id":"evt1","time":1234567890,"turnComplete":true,"content":{"parts":[{"text":"The weather in Paris is sunny with a temperature of 25°C."}]},...}

```

### 4. List Sessions

```bash
curl http://localhost:8080/apps/gin_agent/users/user1/sessions
```

### 5. Get Session Details

```bash
curl http://localhost:8080/apps/gin_agent/users/user1/sessions/abc123
```

### 6. Delete Session

```bash
curl -X DELETE http://localhost:8080/apps/gin_agent/users/user1/sessions/abc123
```

## Request/Response Format

### RunAgentRequest

```json
{
  "appName": "string",       // Required: Agent/app name
  "userId": "string",        // Required: User ID
  "sessionId": "string",     // Required: Session ID
  "newMessage": {            // Required: User message (genai.Content format)
    "role": "user",
    "parts": [{"text": "message"}]
  },
  "streaming": false,        // Optional: Enable streaming in /run endpoint
  "stateDelta": {}           // Optional: State changes
}
```

### Event Response

```json
{
  "id": "string",
  "time": 1234567890,
  "invocationId": "string",
  "branch": "string",
  "author": "string",
  "partial": false,
  "content": {
    "role": "model",
    "parts": [{"text": "response"}]
  },
  "turnComplete": true,
  "actions": {
    "stateDelta": {},
    "artifactDelta": {}
  }
}
```

## Code Structure

- **Request/Response Models**: Compatible with `server/adkrest/internal/models`
- **Conversion Functions**: `fromSessionEvent`, `fromSession`, `toSessionEvent`
- **Handlers**: Direct mapping to ADK REST API controllers
- **Tool Example**: `get_weather` using `functiontool.New()`

## Customization

### Adding Custom Tools

```go
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

func main() {
    ...
    getWeatherTool, err := functiontool.New(functiontool.Config{
        Name:        "get_weather",
        Description: "Get the current weather for a city",
    }, getWeather)

    if err != nil {
        log.Fatalf("Failed to create weather tool: %v", err)
    }
    ...
}
```
