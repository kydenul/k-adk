package anthropic

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// --- New / Config ---

func TestNew(t *testing.T) {
	t.Run("creates model with minimal config", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514"})
		if m.Name() != "claude-sonnet-4-20250514" {
			t.Errorf("expected model name 'claude-sonnet-4-20250514', got %q", m.Name())
		}
		if m.client == nil {
			t.Fatal("expected non-nil client")
		}
	})

	t.Run("nil logger defaults to discard log", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514"})
		if m.Logger == nil {
			t.Fatal("expected non-nil logger")
		}
	})

	t.Run("MaxOutputTokens stored on model", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514", MaxOutputTokens: 8192})
		if m.maxOutputTokens != 8192 {
			t.Errorf("expected maxOutputTokens=8192, got %d", m.maxOutputTokens)
		}
	})

	t.Run("ThinkingBudgetTokens stored on model", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514", ThinkingBudgetTokens: 2048})
		if m.thinkingBudgetTokens != 2048 {
			t.Errorf("expected thinkingBudgetTokens=2048, got %d", m.thinkingBudgetTokens)
		}
	})

	t.Run("zero MaxOutputTokens and ThinkingBudgetTokens", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514"})
		if m.maxOutputTokens != 0 {
			t.Errorf("expected maxOutputTokens=0, got %d", m.maxOutputTokens)
		}
		if m.thinkingBudgetTokens != 0 {
			t.Errorf("expected thinkingBudgetTokens=0, got %d", m.thinkingBudgetTokens)
		}
	})

	t.Run("accepts HTTPOptions with headers", func(t *testing.T) {
		headers := http.Header{}
		headers.Set("X-Custom-Header", "test-value")
		headers.Add("X-Multi", "val1")
		headers.Add("X-Multi", "val2")

		m := New(Config{
			ModelName: "claude-sonnet-4-20250514",
			HTTPOptions: HTTPOptions{
				Headers: headers,
			},
		})
		if m.client == nil {
			t.Fatal("expected non-nil client")
		}
	})

	t.Run("accepts empty HTTPOptions", func(t *testing.T) {
		m := New(Config{
			ModelName:   "claude-sonnet-4-20250514",
			HTTPOptions: HTTPOptions{},
		})
		if m.client == nil {
			t.Fatal("expected non-nil client")
		}
	})
}

// --- buildMessageParams ---

func TestBuildMessageParams(t *testing.T) {
	t.Run("default maxTokens is 4096", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514"})
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			},
		}

		params, err := m.buildMessageParams(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if params.MaxTokens != 4096 {
			t.Errorf("expected default MaxTokens=4096, got %d", params.MaxTokens)
		}
	})

	t.Run("config MaxOutputTokens overrides default", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514", MaxOutputTokens: 8192})
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			},
		}

		params, err := m.buildMessageParams(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if params.MaxTokens != 8192 {
			t.Errorf("expected MaxTokens=8192, got %d", params.MaxTokens)
		}
	})

	t.Run("request MaxOutputTokens overrides config", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514", MaxOutputTokens: 8192})
		req := &model.LLMRequest{
			Config: &genai.GenerateContentConfig{
				MaxOutputTokens: 2048,
			},
			Contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			},
		}

		params, err := m.buildMessageParams(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if params.MaxTokens != 2048 {
			t.Errorf("expected MaxTokens=2048, got %d", params.MaxTokens)
		}
	})

	t.Run("thinking budget applied when set", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514", ThinkingBudgetTokens: 4096})
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			},
		}

		params, err := m.buildMessageParams(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if params.Thinking.OfEnabled == nil {
			t.Fatal("expected Thinking.OfEnabled to be set")
		}
		if params.Thinking.OfEnabled.BudgetTokens != 4096 {
			t.Errorf("expected BudgetTokens=4096, got %d", params.Thinking.OfEnabled.BudgetTokens)
		}
	})

	t.Run("thinking not applied when zero", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514"})
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			},
		}

		params, err := m.buildMessageParams(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if params.Thinking.OfEnabled != nil {
			t.Fatal("expected Thinking.OfEnabled to be nil when budget is zero")
		}
	})

	t.Run("system instruction included", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514"})
		req := &model.LLMRequest{
			Config: &genai.GenerateContentConfig{
				SystemInstruction: &genai.Content{
					Parts: []*genai.Part{{Text: "You are a helper"}},
				},
			},
			Contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			},
		}

		params, err := m.buildMessageParams(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(params.System) == 0 {
			t.Fatal("expected system instruction to be set")
		}
		if params.System[0].Text != "You are a helper" {
			t.Errorf("expected system text 'You are a helper', got %q", params.System[0].Text)
		}
	})
}

func TestBuildMessageParamsConfig(t *testing.T) {
	t.Run("temperature and top_p applied", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514"})
		temp := float32(0.7)
		topP := float32(0.9)
		req := &model.LLMRequest{
			Config: &genai.GenerateContentConfig{
				Temperature: &temp,
				TopP:        &topP,
			},
			Contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			},
		}

		params, err := m.buildMessageParams(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !params.Temperature.Valid() {
			t.Fatal("expected temperature to be set")
		}
		if !params.TopP.Valid() {
			t.Fatal("expected top_p to be set")
		}
	})

	t.Run("stop sequences applied", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514"})
		req := &model.LLMRequest{
			Config: &genai.GenerateContentConfig{
				StopSequences: []string{"END", "STOP"},
			},
			Contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			},
		}

		params, err := m.buildMessageParams(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(params.StopSequences) != 2 {
			t.Fatalf("expected 2 stop sequences, got %d", len(params.StopSequences))
		}
	})

	t.Run("nil config is ok", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514"})
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			},
		}

		params, err := m.buildMessageParams(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(params.Messages) != 1 {
			t.Fatalf("expected 1 message, got %d", len(params.Messages))
		}
	})

	t.Run("returns error on unsupported inline data", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514"})
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{InlineData: &genai.Blob{MIMEType: "video/avi", Data: []byte("data")}},
					},
				},
			},
		}

		_, err := m.buildMessageParams(req)
		if err == nil {
			t.Fatal("expected error for unsupported MIME type")
		}
	})

	t.Run("tools converted", func(t *testing.T) {
		m := New(Config{ModelName: "claude-sonnet-4-20250514"})
		req := &model.LLMRequest{
			Config: &genai.GenerateContentConfig{
				Tools: []*genai.Tool{
					{
						FunctionDeclarations: []*genai.FunctionDeclaration{
							{Name: "test_fn", Description: "test function"},
						},
					},
				},
			},
			Contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			},
		}

		params, err := m.buildMessageParams(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(params.Tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(params.Tools))
		}
	})
}

// --- convertContentToMessage: Media Support ---

func TestConvertContentToMessage_ImageTypes(t *testing.T) {
	imageTypes := []string{
		"image/jpeg",
		"image/jpg",
		"image/png",
		"image/gif",
		"image/webp",
	}

	for _, mime := range imageTypes {
		t.Run(mime, func(t *testing.T) {
			content := &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{MIMEType: mime, Data: []byte("fake-image-data")}},
				},
			}

			msg, err := convertContentToMessage(content)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if msg == nil {
				t.Fatal("expected non-nil message")
			}
			if len(msg.Content) != 1 {
				t.Fatalf("expected 1 content block, got %d", len(msg.Content))
			}
			if msg.Content[0].OfImage == nil {
				t.Fatal("expected OfImage content block")
			}
		})
	}
}

func TestConvertContentToMessage_PDF(t *testing.T) {
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("fake-pdf-data")}},
		},
	}

	msg, err := convertContentToMessage(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if msg == nil {
		t.Fatal("expected non-nil message")
	}
	if len(msg.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(msg.Content))
	}
	if msg.Content[0].OfDocument == nil {
		t.Fatal("expected OfDocument content block for PDF")
	}
}

func TestConvertContentToMessage_TextFiles(t *testing.T) {
	textTypes := []string{
		"text/plain",
		"text/csv",
		"text/html",
		"text/markdown",
	}

	for _, mime := range textTypes {
		t.Run(mime, func(t *testing.T) {
			content := &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{MIMEType: mime, Data: []byte("text file content")}},
				},
			}

			msg, err := convertContentToMessage(content)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if msg == nil {
				t.Fatal("expected non-nil message")
			}
			if len(msg.Content) != 1 {
				t.Fatalf("expected 1 content block, got %d", len(msg.Content))
			}
			if msg.Content[0].OfDocument == nil {
				t.Fatal("expected OfDocument content block for text file")
			}
		})
	}
}

func TestConvertContentToMessage_PlainTextUsesRawData(t *testing.T) {
	rawText := "Hello, this is raw text content!"
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{InlineData: &genai.Blob{MIMEType: "text/plain", Data: []byte(rawText)}},
		},
	}

	msg, err := convertContentToMessage(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if msg == nil {
		t.Fatal("expected non-nil message")
	}

	doc := msg.Content[0].OfDocument
	if doc == nil {
		t.Fatal("expected OfDocument")
	}
	// PlainTextSourceParam uses raw text (not base64)
	if doc.Source.OfText == nil {
		t.Fatal("expected OfText source for plain text")
	}
	if doc.Source.OfText.Data != rawText {
		t.Errorf("expected Data=%q, got %q", rawText, doc.Source.OfText.Data)
	}
}

func TestConvertContentToMessage_UnsupportedMIME(t *testing.T) {
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{InlineData: &genai.Blob{MIMEType: "video/mp4", Data: []byte("fake")}},
		},
	}

	_, err := convertContentToMessage(content)
	if err == nil {
		t.Fatal("expected error for unsupported MIME type")
	}
	if !strings.Contains(err.Error(), "unsupported MIME type") {
		t.Errorf("expected 'unsupported MIME type' in error, got: %v", err)
	}
}

func TestConvertContentToMessage_MixedParts(t *testing.T) {
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: "Look at this image"},
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("png-data")}},
		},
	}

	msg, err := convertContentToMessage(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if msg == nil {
		t.Fatal("expected non-nil message")
	}
	if len(msg.Content) != 2 {
		t.Fatalf("expected 2 content blocks (text + image), got %d", len(msg.Content))
	}
}

func TestConvertContentToMessage_TextOnly(t *testing.T) {
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: "Hello world"},
		},
	}

	msg, err := convertContentToMessage(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if msg == nil {
		t.Fatal("expected non-nil message")
	}
	if len(msg.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(msg.Content))
	}
}

func TestConvertContentToMessage_FunctionCallAndResponse(t *testing.T) {
	t.Run("function call", func(t *testing.T) {
		content := &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						ID:   "call_123",
						Name: "get_weather",
						Args: map[string]any{"city": "NYC"},
					},
				},
			},
		}

		msg, err := convertContentToMessage(content)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if msg == nil {
			t.Fatal("expected non-nil message")
		}
		if msg.Role != anthropic.MessageParamRoleAssistant {
			t.Errorf("expected assistant role, got %v", msg.Role)
		}
		if len(msg.Content) != 1 {
			t.Fatalf("expected 1 content block, got %d", len(msg.Content))
		}
		if msg.Content[0].OfToolUse == nil {
			t.Fatal("expected OfToolUse block")
		}
		if msg.Content[0].OfToolUse.Name != "get_weather" {
			t.Errorf("expected name 'get_weather', got %q", msg.Content[0].OfToolUse.Name)
		}
	})

	t.Run("function response", func(t *testing.T) {
		content := &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{
					FunctionResponse: &genai.FunctionResponse{
						ID:       "call_123",
						Name:     "get_weather",
						Response: map[string]any{"temp": "72F"},
					},
				},
			},
		}

		msg, err := convertContentToMessage(content)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if msg == nil {
			t.Fatal("expected non-nil message")
		}
		if msg.Content[0].OfToolResult == nil {
			t.Fatal("expected OfToolResult block")
		}
	})
}

func TestConvertContentToMessage_EmptyParts(t *testing.T) {
	content := &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{},
	}

	msg, err := convertContentToMessage(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if msg != nil {
		t.Fatal("expected nil message for empty parts")
	}
}

// --- convertTools ---

func TestConvertTools(t *testing.T) {
	t.Run("nil tool skipped", func(t *testing.T) {
		tools := convertTools([]*genai.Tool{nil})
		if len(tools) != 0 {
			t.Errorf("expected 0 tools, got %d", len(tools))
		}
	})

	t.Run("function declaration converted", func(t *testing.T) {
		tools := convertTools([]*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:        "get_weather",
						Description: "Get weather for a city",
						Parameters: &genai.Schema{
							Type: genai.TypeObject,
							Properties: map[string]*genai.Schema{
								"city": {Type: genai.TypeString},
							},
							Required: []string{"city"},
						},
					},
				},
			},
		})
		if len(tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(tools))
		}
		if tools[0].OfTool == nil {
			t.Fatal("expected OfTool")
		}
		if tools[0].OfTool.Name != "get_weather" {
			t.Errorf("expected name 'get_weather', got %q", tools[0].OfTool.Name)
		}
	})
}

func TestConvertToolsParams(t *testing.T) {
	t.Run("map params with required []string", func(t *testing.T) {
		tools := convertTools([]*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:        "test_fn",
						Description: "test",
						ParametersJsonSchema: map[string]any{
							"type": "object",
							"properties": map[string]any{
								"name": map[string]any{"type": "string"},
							},
							"required": []string{"name"},
						},
					},
				},
			},
		})
		if len(tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(tools))
		}
		schema := tools[0].OfTool.InputSchema
		if len(schema.Required) != 1 || schema.Required[0] != "name" {
			t.Errorf("expected required=[name], got %v", schema.Required)
		}
	})

	t.Run("map params with required []interface{} from JSON unmarshal", func(t *testing.T) {
		// Simulate what happens after JSON round-trip: []interface{} instead of []string
		params := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string"},
			},
			"required": []any{"query"},
		}
		tools := convertTools([]*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:                 "search",
						Description:          "search fn",
						ParametersJsonSchema: params,
					},
				},
			},
		})
		if len(tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(tools))
		}
		schema := tools[0].OfTool.InputSchema
		if len(schema.Required) != 1 || schema.Required[0] != "query" {
			t.Errorf("expected required=[query], got %v", schema.Required)
		}
	})

	t.Run("struct params via JSON round-trip", func(t *testing.T) {
		type testSchema struct {
			Type       string         `json:"type"`
			Properties map[string]any `json:"properties"`
			Required   []string       `json:"required"`
		}
		params := testSchema{
			Type: "object",
			Properties: map[string]any{
				"x": map[string]any{"type": "number"},
			},
			Required: []string{"x"},
		}
		tools := convertTools([]*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:                 "compute",
						Description:          "compute fn",
						ParametersJsonSchema: params,
					},
				},
			},
		})
		if len(tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(tools))
		}
		schema := tools[0].OfTool.InputSchema
		if schema.Type != "object" {
			t.Errorf("expected type=object, got %q", schema.Type)
		}
		// After JSON round-trip, required arrives as []interface{} and gets converted
		if len(schema.Required) != 1 || schema.Required[0] != "x" {
			t.Errorf("expected required=[x], got %v", schema.Required)
		}
	})

	t.Run("nil params results in empty schema", func(t *testing.T) {
		tools := convertTools([]*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{Name: "no_params", Description: "no params fn"},
				},
			},
		})
		if len(tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(tools))
		}
		schema := tools[0].OfTool.InputSchema
		if schema.Type != "object" {
			t.Errorf("expected type=object, got %q", schema.Type)
		}
	})

	t.Run("ParametersJsonSchema preferred over Parameters", func(t *testing.T) {
		jsonSchema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string"},
			},
			"required": []string{"query"},
		}
		tools := convertTools([]*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:                 "search",
						Description:          "search fn",
						ParametersJsonSchema: jsonSchema,
						Parameters: &genai.Schema{
							Type: genai.TypeObject,
						},
					},
				},
			},
		})
		if len(tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(tools))
		}
		schema := tools[0].OfTool.InputSchema
		if schema.Properties == nil {
			t.Fatal("expected properties from ParametersJsonSchema")
		}
	})
}

// --- toStringSlice ---

func TestToStringSlice(t *testing.T) {
	t.Run("nil returns nil", func(t *testing.T) {
		result := toStringSlice(nil)
		if result != nil {
			t.Errorf("expected nil, got %v", result)
		}
	})

	t.Run("[]string passthrough", func(t *testing.T) {
		input := []string{"a", "b", "c"}
		result := toStringSlice(input)
		if len(result) != 3 {
			t.Fatalf("expected 3 elements, got %d", len(result))
		}
		for i, v := range input {
			if result[i] != v {
				t.Errorf("expected %q at index %d, got %q", v, i, result[i])
			}
		}
	})

	t.Run("[]any with strings", func(t *testing.T) {
		input := []any{"x", "y"}
		result := toStringSlice(input)
		if len(result) != 2 {
			t.Fatalf("expected 2 elements, got %d", len(result))
		}
		if result[0] != "x" || result[1] != "y" {
			t.Errorf("expected [x y], got %v", result)
		}
	})

	t.Run("[]any with non-strings skipped", func(t *testing.T) {
		input := []any{"a", 42, "b", true}
		result := toStringSlice(input)
		if len(result) != 2 {
			t.Fatalf("expected 2 string elements, got %d", len(result))
		}
		if result[0] != "a" || result[1] != "b" {
			t.Errorf("expected [a b], got %v", result)
		}
	})

	t.Run("unsupported type returns nil", func(t *testing.T) {
		result := toStringSlice(42)
		if result != nil {
			t.Errorf("expected nil for unsupported type, got %v", result)
		}
	})

	t.Run("empty []any returns empty slice", func(t *testing.T) {
		result := toStringSlice([]any{})
		if result == nil {
			t.Fatal("expected non-nil empty slice")
		}
		if len(result) != 0 {
			t.Errorf("expected 0 elements, got %d", len(result))
		}
	})
}

// --- convertResponse ---

func TestConvertResponse(t *testing.T) {
	t.Run("text response converted", func(t *testing.T) {
		// Build a message with JSON to simulate text block
		msgJSON := `{
			"id": "msg_test",
			"type": "message",
			"role": "assistant",
			"model": "claude-sonnet-4-20250514",
			"content": [{"type": "text", "text": "Hello!"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`
		var msg anthropic.Message
		if err := json.Unmarshal([]byte(msgJSON), &msg); err != nil {
			t.Fatalf("failed to unmarshal test message: %v", err)
		}

		resp := convertResponse(&msg)
		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			t.Fatal("expected content with parts")
		}
		if resp.Content.Parts[0].Text != "Hello!" {
			t.Errorf("expected text 'Hello!', got %q", resp.Content.Parts[0].Text)
		}
		if !resp.TurnComplete {
			t.Error("expected TurnComplete to be true")
		}
		if resp.UsageMetadata == nil {
			t.Fatal("expected non-nil usage metadata")
		}
		if resp.UsageMetadata.PromptTokenCount != 10 {
			t.Errorf("expected prompt tokens 10, got %d", resp.UsageMetadata.PromptTokenCount)
		}
		if resp.UsageMetadata.CandidatesTokenCount != 5 {
			t.Errorf("expected candidates tokens 5, got %d", resp.UsageMetadata.CandidatesTokenCount)
		}
	})

	t.Run("tool use response converted", func(t *testing.T) {
		msgJSON := `{
			"id": "msg_test",
			"type": "message",
			"role": "assistant",
			"model": "claude-sonnet-4-20250514",
			"content": [{
				"type": "tool_use",
				"id": "toolu_abc",
				"name": "get_weather",
				"input": {"city": "NYC"}
			}],
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 10, "output_tokens": 20}
		}`
		var msg anthropic.Message
		if err := json.Unmarshal([]byte(msgJSON), &msg); err != nil {
			t.Fatalf("failed to unmarshal test message: %v", err)
		}

		resp := convertResponse(&msg)
		if len(resp.Content.Parts) != 1 {
			t.Fatalf("expected 1 part, got %d", len(resp.Content.Parts))
		}
		fc := resp.Content.Parts[0].FunctionCall
		if fc == nil {
			t.Fatal("expected FunctionCall")
		}
		if fc.Name != "get_weather" {
			t.Errorf("expected function name 'get_weather', got %q", fc.Name)
		}
		if fc.ID != "toolu_abc" {
			t.Errorf("expected ID 'toolu_abc', got %q", fc.ID)
		}
	})

	t.Run("empty message returns empty parts", func(t *testing.T) {
		resp := convertResponse(&anthropic.Message{})
		if resp.Content == nil {
			t.Fatal("expected non-nil content")
		}
		if len(resp.Content.Parts) != 0 {
			t.Errorf("expected 0 parts, got %d", len(resp.Content.Parts))
		}
	})
}

// --- Helper functions ---

func TestConvertRoleToAnthropic(t *testing.T) {
	tests := []struct {
		input    string
		expected anthropic.MessageParamRole
	}{
		{"user", anthropic.MessageParamRoleUser},
		{"model", anthropic.MessageParamRoleAssistant},
		{"unknown", anthropic.MessageParamRoleUser},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			result := convertRoleToAnthropic(tc.input)
			if result != tc.expected {
				t.Errorf("convertRoleToAnthropic(%q) = %v, want %v", tc.input, result, tc.expected)
			}
		})
	}
}

func TestConvertStopReason(t *testing.T) {
	tests := []struct {
		input    anthropic.StopReason
		expected genai.FinishReason
	}{
		{anthropic.StopReasonEndTurn, genai.FinishReasonStop},
		{anthropic.StopReasonMaxTokens, genai.FinishReasonMaxTokens},
		{anthropic.StopReasonStopSequence, genai.FinishReasonStop},
		{anthropic.StopReasonToolUse, genai.FinishReasonStop},
		{"unknown", genai.FinishReasonUnspecified},
	}

	for _, tc := range tests {
		t.Run(string(tc.input), func(t *testing.T) {
			result := convertStopReason(tc.input)
			if result != tc.expected {
				t.Errorf("convertStopReason(%q) = %v, want %v", tc.input, result, tc.expected)
			}
		})
	}
}

func TestExtractTextFromContent(t *testing.T) {
	t.Run("nil content", func(t *testing.T) {
		result := extractTextFromContent(nil)
		if result != "" {
			t.Errorf("expected empty string, got %q", result)
		}
	})

	t.Run("single text part", func(t *testing.T) {
		result := extractTextFromContent(&genai.Content{
			Parts: []*genai.Part{{Text: "hello"}},
		})
		if result != "hello" {
			t.Errorf("expected 'hello', got %q", result)
		}
	})

	t.Run("multiple text parts joined", func(t *testing.T) {
		result := extractTextFromContent(&genai.Content{
			Parts: []*genai.Part{{Text: "hello"}, {Text: "world"}},
		})
		if result != "hello\nworld" {
			t.Errorf("expected 'hello\\nworld', got %q", result)
		}
	})

	t.Run("non-text parts ignored", func(t *testing.T) {
		result := extractTextFromContent(&genai.Content{
			Parts: []*genai.Part{
				{Text: "hello"},
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("data")}},
				{Text: "world"},
			},
		})
		if result != "hello\nworld" {
			t.Errorf("expected 'hello\\nworld', got %q", result)
		}
	})
}

func TestSanitizeToolID(t *testing.T) {
	t.Run("valid ID unchanged", func(t *testing.T) {
		id := "toolu_abc123_def-456"
		result := sanitizeToolID(id)
		if result != id {
			t.Errorf("expected %q, got %q", id, result)
		}
	})

	t.Run("invalid chars produce hash", func(t *testing.T) {
		id := "invalid id with spaces!"
		result := sanitizeToolID(id)
		if result == id {
			t.Fatal("expected sanitized ID to differ from original")
		}
		if !strings.HasPrefix(result, "toolu_") {
			t.Errorf("expected 'toolu_' prefix, got %q", result)
		}
	})

	t.Run("deterministic", func(t *testing.T) {
		id := "special@chars#here"
		r1 := sanitizeToolID(id)
		r2 := sanitizeToolID(id)
		if r1 != r2 {
			t.Error("expected deterministic sanitization")
		}
	})
}

func TestConvertToolInputToRaw(t *testing.T) {
	t.Run("nil returns empty object", func(t *testing.T) {
		result := convertToolInputToRaw(nil)
		if string(result) != "{}" {
			t.Errorf("expected '{}', got %q", string(result))
		}
	})

	t.Run("map converted to JSON", func(t *testing.T) {
		input := map[string]any{"key": "value"}
		result := convertToolInputToRaw(input)
		var m map[string]any
		if err := json.Unmarshal(result, &m); err != nil {
			t.Fatalf("invalid JSON: %v", err)
		}
		if m["key"] != "value" {
			t.Errorf("expected key=value, got %v", m)
		}
	})

	t.Run("json.RawMessage passthrough", func(t *testing.T) {
		raw := json.RawMessage(`{"x":1}`)
		result := convertToolInputToRaw(raw)
		if string(result) != `{"x":1}` {
			t.Errorf("expected raw passthrough, got %q", string(result))
		}
	})
}

func TestConvertToolInput(t *testing.T) {
	t.Run("nil returns empty map", func(t *testing.T) {
		result := convertToolInput(nil)
		if len(result) != 0 {
			t.Errorf("expected empty map, got %v", result)
		}
	})

	t.Run("map passthrough", func(t *testing.T) {
		input := map[string]any{"key": "value"}
		result := convertToolInput(input)
		if result["key"] != "value" {
			t.Errorf("expected key=value, got %v", result)
		}
	})

	t.Run("json.RawMessage unmarshaled", func(t *testing.T) {
		raw := json.RawMessage(`{"city":"NYC"}`)
		result := convertToolInput(raw)
		if result["city"] != "NYC" {
			t.Errorf("expected city=NYC, got %v", result)
		}
	})
}

// --- repairMessageHistory ---

func TestRepairMessageHistory(t *testing.T) {
	t.Run("empty messages", func(t *testing.T) {
		result := repairMessageHistory(nil)
		if len(result) != 0 {
			t.Errorf("expected 0 messages, got %d", len(result))
		}
	})

	t.Run("no tool use passes through", func(t *testing.T) {
		messages := []anthropic.MessageParam{
			{
				Role:    anthropic.MessageParamRoleUser,
				Content: []anthropic.ContentBlockParamUnion{anthropic.NewTextBlock("hello")},
			},
			{
				Role:    anthropic.MessageParamRoleAssistant,
				Content: []anthropic.ContentBlockParamUnion{anthropic.NewTextBlock("hi")},
			},
		}
		result := repairMessageHistory(messages)
		if len(result) != 2 {
			t.Errorf("expected 2 messages, got %d", len(result))
		}
	})

	t.Run("orphaned tool_use removed", func(t *testing.T) {
		messages := []anthropic.MessageParam{
			{
				Role: anthropic.MessageParamRoleAssistant,
				Content: []anthropic.ContentBlockParamUnion{
					anthropic.NewTextBlock("thinking..."),
					{OfToolUse: &anthropic.ToolUseBlockParam{
						ID:    "call_1",
						Name:  "test",
						Input: json.RawMessage(`{}`),
					}},
				},
			},
			// No tool_result message follows
			{
				Role:    anthropic.MessageParamRoleUser,
				Content: []anthropic.ContentBlockParamUnion{anthropic.NewTextBlock("next message")},
			},
		}
		result := repairMessageHistory(messages)

		// The assistant message should still have the text block but not the tool_use
		found := false
		for _, msg := range result {
			if msg.Role == anthropic.MessageParamRoleAssistant {
				found = true
				for _, block := range msg.Content {
					if block.OfToolUse != nil {
						t.Error("expected orphaned tool_use to be removed")
					}
				}
			}
		}
		// It's acceptable if the assistant message was removed entirely
		// (it only had text + orphaned tool_use, text part remains)
		_ = found
	})

	t.Run("matched tool_use and tool_result preserved", func(t *testing.T) {
		messages := []anthropic.MessageParam{
			{
				Role: anthropic.MessageParamRoleAssistant,
				Content: []anthropic.ContentBlockParamUnion{
					{OfToolUse: &anthropic.ToolUseBlockParam{
						ID:    "call_1",
						Name:  "test",
						Input: json.RawMessage(`{}`),
					}},
				},
			},
			{
				Role: anthropic.MessageParamRoleUser,
				Content: []anthropic.ContentBlockParamUnion{
					anthropic.NewToolResultBlock("call_1", "result", false),
				},
			},
		}
		result := repairMessageHistory(messages)
		if len(result) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(result))
		}

		// Tool use should be preserved in first message
		hasToolUse := false
		for _, block := range result[0].Content {
			if block.OfToolUse != nil {
				hasToolUse = true
			}
		}
		if !hasToolUse {
			t.Error("expected matched tool_use to be preserved")
		}
	})
}

// --- HTTPOptions struct ---

func TestHTTPOptions(t *testing.T) {
	t.Run("zero value has nil headers", func(t *testing.T) {
		opts := HTTPOptions{}
		if opts.Headers != nil {
			t.Error("expected nil headers for zero value")
		}
	})

	t.Run("headers can be set", func(t *testing.T) {
		headers := http.Header{}
		headers.Set("Authorization", "Bearer test")
		opts := HTTPOptions{Headers: headers}
		if opts.Headers.Get("Authorization") != "Bearer test" {
			t.Error("expected header value")
		}
	})
}

// --- Model interface compliance ---

func TestModelInterface(t *testing.T) {
	m := New(Config{ModelName: "claude-sonnet-4-20250514"})
	var _ model.LLM = m

	if m.Name() != "claude-sonnet-4-20250514" {
		t.Errorf("expected name 'claude-sonnet-4-20250514', got %q", m.Name())
	}
}

// --- JSON serialization of messages ---

func TestMessageSerialization(t *testing.T) {
	t.Run("message with image serializes", func(t *testing.T) {
		content := &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{Text: "describe this"},
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("png")}},
			},
		}

		msg, err := convertContentToMessage(content)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("failed to marshal message: %v", err)
		}
		if len(data) == 0 {
			t.Fatal("expected non-empty JSON")
		}
	})

	t.Run("message with PDF serializes", func(t *testing.T) {
		content := &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("pdf")}},
			},
		}

		msg, err := convertContentToMessage(content)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("failed to marshal message: %v", err)
		}
		if len(data) == 0 {
			t.Fatal("expected non-empty JSON")
		}
	})

	t.Run("message with text file serializes", func(t *testing.T) {
		content := &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{InlineData: &genai.Blob{MIMEType: "text/plain", Data: []byte("hello")}},
			},
		}

		msg, err := convertContentToMessage(content)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("failed to marshal message: %v", err)
		}
		if len(data) == 0 {
			t.Fatal("expected non-empty JSON")
		}
	})
}
