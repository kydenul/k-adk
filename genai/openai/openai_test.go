package openai

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// --- New / Config ---

func TestNew(t *testing.T) {
	t.Run("creates model with minimal config", func(t *testing.T) {
		m := New(Config{ModelName: "gpt-4o"})
		if m.Name() != "gpt-4o" {
			t.Errorf("expected model name 'gpt-4o', got %q", m.Name())
		}
		if m.client == nil {
			t.Fatal("expected non-nil client")
		}
		if m.toolCall == nil {
			t.Fatal("expected initialized toolCall map")
		}
	})

	t.Run("nil logger defaults to discard log", func(t *testing.T) {
		m := New(Config{ModelName: "gpt-4o"})
		if m.Logger == nil {
			t.Fatal("expected non-nil logger")
		}
	})

	t.Run("accepts HTTPOptions with headers", func(t *testing.T) {
		headers := http.Header{}
		headers.Set("X-Custom-Header", "test-value")
		headers.Add("X-Multi", "val1")
		headers.Add("X-Multi", "val2")

		// Should not panic
		m := New(Config{
			ModelName: "gpt-4o",
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
			ModelName:   "gpt-4o",
			HTTPOptions: HTTPOptions{},
		})
		if m.client == nil {
			t.Fatal("expected non-nil client")
		}
	})
}

// --- convertContentToMessages: Media Support ---

func TestConvertContentToMessages_ImageTypes(t *testing.T) {
	m := New(Config{ModelName: "gpt-4o"})

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

			msgs, err := m.convertContentToMessages(content)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(msgs) != 1 {
				t.Fatalf("expected 1 message, got %d", len(msgs))
			}

			msg := msgs[0]
			if msg.OfUser == nil {
				t.Fatal("expected user message")
			}
			parts := msg.OfUser.Content.OfArrayOfContentParts
			if len(parts) != 1 {
				t.Fatalf("expected 1 content part, got %d", len(parts))
			}
			if parts[0].OfImageURL == nil {
				t.Fatal("expected OfImageURL content part")
			}
			if !strings.Contains(parts[0].OfImageURL.ImageURL.URL, mime) {
				t.Errorf("expected URL to contain MIME type %q", mime)
			}
		})
	}
}

func TestConvertContentToMessages_AudioTypes(t *testing.T) {
	m := New(Config{ModelName: "gpt-4o"})

	tests := []struct {
		mime           string
		expectedFormat string
	}{
		{"audio/wav", "wav"},
		{"audio/mp3", "mp3"},
		{"audio/mpeg", "mp3"},
		{"audio/webm", "wav"},
	}

	for _, tc := range tests {
		t.Run(tc.mime, func(t *testing.T) {
			content := &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{MIMEType: tc.mime, Data: []byte("fake-audio-data")}},
				},
			}

			msgs, err := m.convertContentToMessages(content)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(msgs) != 1 {
				t.Fatalf("expected 1 message, got %d", len(msgs))
			}

			msg := msgs[0]
			if msg.OfUser == nil {
				t.Fatal("expected user message")
			}
			parts := msg.OfUser.Content.OfArrayOfContentParts
			if len(parts) != 1 {
				t.Fatalf("expected 1 content part, got %d", len(parts))
			}
			if parts[0].OfInputAudio == nil {
				t.Fatal("expected OfInputAudio content part")
			}
			if parts[0].OfInputAudio.InputAudio.Format != tc.expectedFormat {
				t.Errorf("expected format %q, got %q", tc.expectedFormat, parts[0].OfInputAudio.InputAudio.Format)
			}
		})
	}
}

func TestConvertContentToMessages_FileTypes(t *testing.T) {
	m := New(Config{ModelName: "gpt-4o"})

	fileTypes := []string{
		"application/pdf",
		"text/plain",
		"text/csv",
		"text/html",
	}

	for _, mime := range fileTypes {
		t.Run(mime, func(t *testing.T) {
			content := &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{MIMEType: mime, Data: []byte("fake-file-data")}},
				},
			}

			msgs, err := m.convertContentToMessages(content)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(msgs) != 1 {
				t.Fatalf("expected 1 message, got %d", len(msgs))
			}

			msg := msgs[0]
			if msg.OfUser == nil {
				t.Fatal("expected user message")
			}
			parts := msg.OfUser.Content.OfArrayOfContentParts
			if len(parts) != 1 {
				t.Fatalf("expected 1 content part, got %d", len(parts))
			}
			if parts[0].OfFile == nil {
				t.Fatal("expected OfFile content part")
			}
		})
	}
}

func TestConvertContentToMessages_UnsupportedMIME(t *testing.T) {
	m := New(Config{ModelName: "gpt-4o"})

	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{InlineData: &genai.Blob{MIMEType: "video/mp4", Data: []byte("fake")}},
		},
	}

	_, err := m.convertContentToMessages(content)
	if err == nil {
		t.Fatal("expected error for unsupported MIME type")
	}
	if !strings.Contains(err.Error(), "unsupported MIME type") {
		t.Errorf("expected 'unsupported MIME type' in error, got: %v", err)
	}
}

func TestConvertContentToMessages_MixedParts(t *testing.T) {
	m := New(Config{ModelName: "gpt-4o"})

	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: "Look at this image"},
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("png-data")}},
		},
	}

	msgs, err := m.convertContentToMessages(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}

	msg := msgs[0]
	if msg.OfUser == nil {
		t.Fatal("expected user message")
	}
	parts := msg.OfUser.Content.OfArrayOfContentParts
	if len(parts) != 2 {
		t.Fatalf("expected 2 content parts (text + image), got %d", len(parts))
	}
}

func TestConvertContentToMessages_TextOnly(t *testing.T) {
	m := New(Config{ModelName: "gpt-4o"})

	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: "Hello world"},
		},
	}

	msgs, err := m.convertContentToMessages(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}

	msg := msgs[0]
	if msg.OfUser == nil {
		t.Fatal("expected user message")
	}
	// Text-only should use OfString, not OfArrayOfContentParts
	if !msg.OfUser.Content.OfString.Valid() {
		t.Fatal("expected text-only message to use OfString")
	}
}

func TestConvertContentToMessages_FunctionCallAndResponse(t *testing.T) {
	m := New(Config{ModelName: "gpt-4o"})

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

		msgs, err := m.convertContentToMessages(content)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(msgs) != 1 {
			t.Fatalf("expected 1 message, got %d", len(msgs))
		}
		if msgs[0].OfAssistant == nil {
			t.Fatal("expected assistant message")
		}
		if len(msgs[0].OfAssistant.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call, got %d", len(msgs[0].OfAssistant.ToolCalls))
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

		msgs, err := m.convertContentToMessages(content)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(msgs) != 1 {
			t.Fatalf("expected 1 message, got %d", len(msgs))
		}
		if msgs[0].OfTool == nil {
			t.Fatal("expected tool message")
		}
	})
}

func TestConvertContentToMessages_EmptyParts(t *testing.T) {
	m := New(Config{ModelName: "gpt-4o"})

	content := &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{},
	}

	msgs, err := m.convertContentToMessages(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) != 0 {
		t.Fatalf("expected 0 messages for empty parts, got %d", len(msgs))
	}
}

// --- buildRoleMessage ---

func TestBuildRoleMessage(t *testing.T) {
	t.Run("user with text only", func(t *testing.T) {
		msg := buildRoleMessage("user", []string{"hello"}, nil, nil)
		if msg == nil {
			t.Fatal("expected non-nil message")
		}
		if msg.OfUser == nil {
			t.Fatal("expected user message")
		}
	})

	t.Run("user with media", func(t *testing.T) {
		media := []openai.ChatCompletionContentPartUnionParam{
			openai.TextContentPart("extra"),
		}
		msg := buildRoleMessage("user", []string{"hi"}, media, nil)
		if msg == nil {
			t.Fatal("expected non-nil message")
		}
		if msg.OfUser == nil {
			t.Fatal("expected user message")
		}
	})

	t.Run("model role maps to assistant", func(t *testing.T) {
		msg := buildRoleMessage("model", []string{"response"}, nil, nil)
		if msg == nil {
			t.Fatal("expected non-nil message")
		}
		if msg.OfAssistant == nil {
			t.Fatal("expected assistant message for 'model' role")
		}
	})

	t.Run("system role", func(t *testing.T) {
		msg := buildRoleMessage("system", []string{"instruction"}, nil, nil)
		if msg == nil {
			t.Fatal("expected non-nil message")
		}
		if msg.OfSystem == nil {
			t.Fatal("expected system message")
		}
	})

	t.Run("unknown role returns nil", func(t *testing.T) {
		msg := buildRoleMessage("unknown_role", []string{"text"}, nil, nil)
		if msg != nil {
			t.Fatal("expected nil message for unknown role")
		}
	})
}

// --- buildChatCompletionParameters ---

func TestBuildChatCompletionParameters(t *testing.T) {
	m := New(Config{ModelName: "gpt-4o"})

	t.Run("system instruction included", func(t *testing.T) {
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

		params, err := m.buildChatCompletionParameters(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Should have system message + user message
		if len(params.Messages) < 2 {
			t.Fatalf("expected at least 2 messages, got %d", len(params.Messages))
		}
		if params.Messages[0].OfSystem == nil {
			t.Fatal("expected first message to be system")
		}
	})

	t.Run("nil config is ok", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			},
		}

		params, err := m.buildChatCompletionParameters(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(params.Messages) != 1 {
			t.Fatalf("expected 1 message, got %d", len(params.Messages))
		}
	})

	t.Run("returns error on unsupported inline data", func(t *testing.T) {
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

		_, err := m.buildChatCompletionParameters(req)
		if err == nil {
			t.Fatal("expected error for unsupported MIME type")
		}
	})
}

// --- normalizeToolCallID ---

func TestNormalizeToolCallID(t *testing.T) {
	m := New(Config{ModelName: "gpt-4o"})

	t.Run("short ID unchanged", func(t *testing.T) {
		id := "short-id"
		result := m.normalizeToolCallID(id)
		if result != id {
			t.Errorf("expected %q, got %q", id, result)
		}
	})

	t.Run("exactly 40 chars unchanged", func(t *testing.T) {
		id := strings.Repeat("a", 40)
		result := m.normalizeToolCallID(id)
		if result != id {
			t.Errorf("expected ID unchanged at exactly 40 chars")
		}
	})

	t.Run("long ID is hashed", func(t *testing.T) {
		id := strings.Repeat("a", 41)
		result := m.normalizeToolCallID(id)
		if result == id {
			t.Fatal("expected normalized ID to be different from original")
		}
		if !strings.HasPrefix(result, "tc_") {
			t.Errorf("expected prefix 'tc_', got %q", result)
		}
		if len(result) > maxToolCallIDLength {
			t.Errorf("expected length <= %d, got %d", maxToolCallIDLength, len(result))
		}
	})

	t.Run("same input produces same output", func(t *testing.T) {
		id := strings.Repeat("x", 50)
		r1 := m.normalizeToolCallID(id)
		r2 := m.normalizeToolCallID(id)
		if r1 != r2 {
			t.Error("expected deterministic normalization")
		}
	})
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
	})

	t.Run("ParametersJsonSchema preferred over Parameters", func(t *testing.T) {
		jsonSchema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string"},
			},
		}
		tools := convertTools([]*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:                 "search",
						Description:          "Search",
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
	})
}

// --- convertResponse ---

func TestConvertResponse(t *testing.T) {
	t.Run("nil response returns error", func(t *testing.T) {
		_, err := convertResponse(nil)
		if err == nil {
			t.Fatal("expected error for nil response")
		}
	})

	t.Run("empty choices returns error", func(t *testing.T) {
		_, err := convertResponse(&openai.ChatCompletion{})
		if err != ErrNoChoicesInResponse {
			t.Errorf("expected ErrNoChoicesInResponse, got %v", err)
		}
	})

	t.Run("text response converted", func(t *testing.T) {
		resp, err := convertResponse(&openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: "Hello!",
					},
					FinishReason: "stop",
				},
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			t.Fatal("expected content with parts")
		}
		if resp.Content.Parts[0].Text != "Hello!" {
			t.Errorf("expected text 'Hello!', got %q", resp.Content.Parts[0].Text)
		}
		if !resp.TurnComplete {
			t.Error("expected TurnComplete to be true")
		}
	})

	t.Run("tool call response converted", func(t *testing.T) {
		resp, err := convertResponse(&openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						ToolCalls: []openai.ChatCompletionMessageToolCallUnion{
							{
								ID: "call_abc",
								Function: openai.ChatCompletionMessageFunctionToolCallFunction{
									Name:      "get_weather",
									Arguments: `{"city":"NYC"}`,
								},
							},
						},
					},
					FinishReason: "tool_calls",
				},
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
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
		if fc.Args["city"] != "NYC" {
			t.Errorf("expected city=NYC, got %v", fc.Args["city"])
		}
	})
}

// --- applyGenerationConfig ---

func TestApplyGenerationConfig(t *testing.T) {
	t.Run("temperature", func(t *testing.T) {
		temp := float32(0.7)
		params := openai.ChatCompletionNewParams{}
		applyGenerationConfig(&params, &genai.GenerateContentConfig{
			Temperature: &temp,
		})
		if !params.Temperature.Valid() {
			t.Fatal("expected temperature to be set")
		}
	})

	t.Run("max output tokens", func(t *testing.T) {
		params := openai.ChatCompletionNewParams{}
		applyGenerationConfig(&params, &genai.GenerateContentConfig{
			MaxOutputTokens: 1024,
		})
		if !params.MaxTokens.Valid() {
			t.Fatal("expected MaxTokens to be set")
		}
	})

	t.Run("stop sequences single", func(t *testing.T) {
		params := openai.ChatCompletionNewParams{}
		applyGenerationConfig(&params, &genai.GenerateContentConfig{
			StopSequences: []string{"END"},
		})
		if !params.Stop.OfString.Valid() {
			t.Fatal("expected single stop string")
		}
	})

	t.Run("stop sequences multiple", func(t *testing.T) {
		params := openai.ChatCompletionNewParams{}
		applyGenerationConfig(&params, &genai.GenerateContentConfig{
			StopSequences: []string{"END", "STOP"},
		})
		if len(params.Stop.OfStringArray) != 2 {
			t.Fatalf("expected 2 stop sequences, got %d", len(params.Stop.OfStringArray))
		}
	})

	t.Run("JSON response mode", func(t *testing.T) {
		params := openai.ChatCompletionNewParams{}
		applyGenerationConfig(&params, &genai.GenerateContentConfig{
			ResponseMIMEType: "application/json",
		})
		if params.ResponseFormat.OfJSONObject == nil {
			t.Fatal("expected JSON response format")
		}
	})

	t.Run("tools applied", func(t *testing.T) {
		params := openai.ChatCompletionNewParams{}
		applyGenerationConfig(&params, &genai.GenerateContentConfig{
			Tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{Name: "test_fn", Description: "test"},
					},
				},
			},
		})
		if len(params.Tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(params.Tools))
		}
	})
}

// --- convertInlineDataToPart ---

func TestConvertInlineDataToPart(t *testing.T) {
	t.Run("image/jpeg", func(t *testing.T) {
		part, err := convertInlineDataToPart(&genai.Blob{
			MIMEType: "image/jpeg",
			Data:     []byte("jpeg-data"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if part.OfImageURL == nil {
			t.Fatal("expected OfImageURL")
		}
		if part.OfImageURL.ImageURL.Detail != "auto" {
			t.Errorf("expected detail 'auto', got %q", part.OfImageURL.ImageURL.Detail)
		}
	})

	t.Run("image/png", func(t *testing.T) {
		part, err := convertInlineDataToPart(&genai.Blob{
			MIMEType: "image/png",
			Data:     []byte("png-data"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if part.OfImageURL == nil {
			t.Fatal("expected OfImageURL")
		}
	})

	t.Run("image/gif", func(t *testing.T) {
		part, err := convertInlineDataToPart(&genai.Blob{
			MIMEType: "image/gif",
			Data:     []byte("gif-data"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if part.OfImageURL == nil {
			t.Fatal("expected OfImageURL")
		}
	})

	t.Run("image/webp", func(t *testing.T) {
		part, err := convertInlineDataToPart(&genai.Blob{
			MIMEType: "image/webp",
			Data:     []byte("webp-data"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if part.OfImageURL == nil {
			t.Fatal("expected OfImageURL")
		}
	})
}

func TestConvertInlineDataToPartAudio(t *testing.T) {
	t.Run("audio/wav returns wav format", func(t *testing.T) {
		part, err := convertInlineDataToPart(&genai.Blob{
			MIMEType: "audio/wav",
			Data:     []byte("wav-data"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if part.OfInputAudio == nil {
			t.Fatal("expected OfInputAudio")
		}
		if part.OfInputAudio.InputAudio.Format != "wav" {
			t.Errorf("expected format 'wav', got %q", part.OfInputAudio.InputAudio.Format)
		}
	})

	t.Run("audio/mp3 returns mp3 format", func(t *testing.T) {
		part, err := convertInlineDataToPart(&genai.Blob{
			MIMEType: "audio/mp3",
			Data:     []byte("mp3-data"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if part.OfInputAudio == nil {
			t.Fatal("expected OfInputAudio")
		}
		if part.OfInputAudio.InputAudio.Format != "mp3" {
			t.Errorf("expected format 'mp3', got %q", part.OfInputAudio.InputAudio.Format)
		}
	})

	t.Run("audio/mpeg returns mp3 format", func(t *testing.T) {
		part, err := convertInlineDataToPart(&genai.Blob{
			MIMEType: "audio/mpeg",
			Data:     []byte("mpeg-data"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if part.OfInputAudio == nil {
			t.Fatal("expected OfInputAudio")
		}
		if part.OfInputAudio.InputAudio.Format != "mp3" {
			t.Errorf("expected format 'mp3', got %q", part.OfInputAudio.InputAudio.Format)
		}
	})

	t.Run("audio/webm returns wav format", func(t *testing.T) {
		part, err := convertInlineDataToPart(&genai.Blob{
			MIMEType: "audio/webm",
			Data:     []byte("webm-data"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if part.OfInputAudio == nil {
			t.Fatal("expected OfInputAudio")
		}
		if part.OfInputAudio.InputAudio.Format != "wav" {
			t.Errorf("expected format 'wav', got %q", part.OfInputAudio.InputAudio.Format)
		}
	})
}

func TestConvertInlineDataToPartFiles(t *testing.T) {
	t.Run("application/pdf returns OfFile", func(t *testing.T) {
		part, err := convertInlineDataToPart(&genai.Blob{
			MIMEType: "application/pdf",
			Data:     []byte("pdf-data"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if part.OfFile == nil {
			t.Fatal("expected OfFile")
		}
	})

	t.Run("text/plain returns OfFile", func(t *testing.T) {
		part, err := convertInlineDataToPart(&genai.Blob{
			MIMEType: "text/plain",
			Data:     []byte("some text content"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if part.OfFile == nil {
			t.Fatal("expected OfFile")
		}
	})

	t.Run("text/csv returns OfFile", func(t *testing.T) {
		part, err := convertInlineDataToPart(&genai.Blob{
			MIMEType: "text/csv",
			Data:     []byte("a,b,c\n1,2,3"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if part.OfFile == nil {
			t.Fatal("expected OfFile")
		}
	})

	t.Run("unsupported MIME returns error", func(t *testing.T) {
		_, err := convertInlineDataToPart(&genai.Blob{
			MIMEType: "application/octet-stream",
			Data:     []byte("binary"),
		})
		if err == nil {
			t.Fatal("expected error for unsupported MIME type")
		}
		if !strings.Contains(err.Error(), "unsupported MIME type") {
			t.Errorf("expected 'unsupported MIME type' in error, got: %v", err)
		}
	})

	t.Run("video/mp4 unsupported", func(t *testing.T) {
		_, err := convertInlineDataToPart(&genai.Blob{
			MIMEType: "video/mp4",
			Data:     []byte("video-data"),
		})
		if err == nil {
			t.Fatal("expected error")
		}
	})
}

// --- buildUserMessage ---

func TestBuildUserMessage(t *testing.T) {
	t.Run("text only uses OfString", func(t *testing.T) {
		msg := buildUserMessage([]string{"hello"}, nil)
		if msg == nil || msg.OfUser == nil {
			t.Fatal("expected user message")
		}
		if !msg.OfUser.Content.OfString.Valid() {
			t.Fatal("expected OfString for text-only")
		}
	})

	t.Run("with media uses OfArrayOfContentParts", func(t *testing.T) {
		media := []openai.ChatCompletionContentPartUnionParam{
			{OfImageURL: &openai.ChatCompletionContentPartImageParam{
				ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
					URL: "data:image/png;base64,abc",
				},
			}},
		}
		msg := buildUserMessage([]string{"look"}, media)
		if msg == nil || msg.OfUser == nil {
			t.Fatal("expected user message")
		}
		parts := msg.OfUser.Content.OfArrayOfContentParts
		if len(parts) != 2 {
			t.Fatalf("expected 2 parts (text + image), got %d", len(parts))
		}
	})

	t.Run("media only without text", func(t *testing.T) {
		media := []openai.ChatCompletionContentPartUnionParam{
			{OfImageURL: &openai.ChatCompletionContentPartImageParam{
				ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
					URL: "data:image/png;base64,abc",
				},
			}},
		}
		msg := buildUserMessage(nil, media)
		if msg == nil || msg.OfUser == nil {
			t.Fatal("expected user message")
		}
		parts := msg.OfUser.Content.OfArrayOfContentParts
		if len(parts) != 1 {
			t.Fatalf("expected 1 part (image only), got %d", len(parts))
		}
	})
}

// --- buildAssistantMessage ---

func TestBuildAssistantMessage(t *testing.T) {
	t.Run("text only", func(t *testing.T) {
		msg := buildAssistantMessage([]string{"response"}, nil)
		if msg == nil || msg.OfAssistant == nil {
			t.Fatal("expected assistant message")
		}
		if !msg.OfAssistant.Content.OfString.Valid() {
			t.Fatal("expected OfString content")
		}
	})

	t.Run("with tool calls", func(t *testing.T) {
		toolCalls := []openai.ChatCompletionMessageToolCallUnionParam{
			{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: "call_1",
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      "test",
						Arguments: "{}",
					},
				},
			},
		}
		msg := buildAssistantMessage(nil, toolCalls)
		if msg == nil || msg.OfAssistant == nil {
			t.Fatal("expected assistant message")
		}
		if len(msg.OfAssistant.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call, got %d", len(msg.OfAssistant.ToolCalls))
		}
	})
}

// --- Helper functions ---

func TestConvertRole(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"model", "assistant"},
		{"user", "user"},
		{"system", "system"},
		{"other", "other"},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			result := convertRole(tc.input)
			if result != tc.expected {
				t.Errorf("convertRole(%q) = %q, want %q", tc.input, result, tc.expected)
			}
		})
	}
}

func TestConvertFinishReason(t *testing.T) {
	tests := []struct {
		input    string
		expected genai.FinishReason
	}{
		{"stop", genai.FinishReasonStop},
		{"tool_calls", genai.FinishReasonStop},
		{"function_call", genai.FinishReasonStop},
		{"length", genai.FinishReasonMaxTokens},
		{"content_filter", genai.FinishReasonSafety},
		{"unknown", genai.FinishReasonUnspecified},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			result := convertFinishReason(tc.input)
			if result != tc.expected {
				t.Errorf("convertFinishReason(%q) = %v, want %v", tc.input, result, tc.expected)
			}
		})
	}
}

func TestConvertUsageMetadata(t *testing.T) {
	t.Run("zero tokens returns nil", func(t *testing.T) {
		result := convertUsageMetadata(openai.CompletionUsage{})
		if result != nil {
			t.Fatal("expected nil for zero tokens")
		}
	})

	t.Run("non-zero tokens converted", func(t *testing.T) {
		result := convertUsageMetadata(openai.CompletionUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		})
		if result == nil {
			t.Fatal("expected non-nil metadata")
		}
		if result.PromptTokenCount != 100 {
			t.Errorf("expected prompt tokens 100, got %d", result.PromptTokenCount)
		}
		if result.CandidatesTokenCount != 50 {
			t.Errorf("expected candidates tokens 50, got %d", result.CandidatesTokenCount)
		}
		if result.TotalTokenCount != 150 {
			t.Errorf("expected total tokens 150, got %d", result.TotalTokenCount)
		}
	})
}

func TestExtractText(t *testing.T) {
	t.Run("nil content", func(t *testing.T) {
		result := extractText(nil)
		if result != "" {
			t.Errorf("expected empty string, got %q", result)
		}
	})

	t.Run("single text part", func(t *testing.T) {
		result := extractText(&genai.Content{
			Parts: []*genai.Part{{Text: "hello"}},
		})
		if result != "hello" {
			t.Errorf("expected 'hello', got %q", result)
		}
	})

	t.Run("multiple text parts joined", func(t *testing.T) {
		result := extractText(&genai.Content{
			Parts: []*genai.Part{{Text: "hello"}, {Text: "world"}},
		})
		if result != "hello\nworld" {
			t.Errorf("expected 'hello\\nworld', got %q", result)
		}
	})
}

func TestParseJSONArgs(t *testing.T) {
	t.Run("empty string returns empty map", func(t *testing.T) {
		result := parseJSONArgs("")
		if len(result) != 0 {
			t.Errorf("expected empty map, got %v", result)
		}
	})

	t.Run("valid JSON parsed", func(t *testing.T) {
		result := parseJSONArgs(`{"key":"value"}`)
		if result["key"] != "value" {
			t.Errorf("expected key=value, got %v", result)
		}
	})

	t.Run("invalid JSON returns empty map", func(t *testing.T) {
		result := parseJSONArgs("not-json")
		if len(result) != 0 {
			t.Errorf("expected empty map for invalid JSON, got %v", result)
		}
	})
}

func TestConvertToFunctionParams(t *testing.T) {
	t.Run("nil returns nil", func(t *testing.T) {
		result := convertToFunctionParams(nil)
		if result != nil {
			t.Fatal("expected nil")
		}
	})

	t.Run("map input", func(t *testing.T) {
		input := map[string]any{
			"type":       "object",
			"properties": map[string]any{"name": map[string]any{"type": "string"}},
		}
		result := convertToFunctionParams(input)
		if result == nil {
			t.Fatal("expected non-nil result")
		}
	})

	t.Run("struct input via JSON round-trip", func(t *testing.T) {
		type testSchema struct {
			Type       string         `json:"type"`
			Properties map[string]any `json:"properties"`
		}
		input := testSchema{
			Type:       "object",
			Properties: map[string]any{"x": map[string]any{"type": "string"}},
		}
		result := convertToFunctionParams(input)
		if result == nil {
			t.Fatal("expected non-nil result")
		}
	})
}

func TestEnsureObjectProperties(t *testing.T) {
	t.Run("nil schema does not panic", func(t *testing.T) {
		ensureObjectProperties(nil)
	})

	t.Run("object without properties gets empty properties", func(t *testing.T) {
		schema := map[string]any{"type": "object"}
		ensureObjectProperties(schema)
		if _, ok := schema["properties"]; !ok {
			t.Fatal("expected properties to be added")
		}
	})

	t.Run("nested object gets properties", func(t *testing.T) {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"inner": map[string]any{"type": "object"},
			},
		}
		ensureObjectProperties(schema)
		inner := schema["properties"].(map[string]any)["inner"].(map[string]any)
		if _, ok := inner["properties"]; !ok {
			t.Fatal("expected nested properties to be added")
		}
	})
}

func TestSchemaTypeToString(t *testing.T) {
	tests := []struct {
		input    genai.Type
		expected string
	}{
		{genai.TypeString, "string"},
		{genai.TypeNumber, "number"},
		{genai.TypeInteger, "integer"},
		{genai.TypeBoolean, "boolean"},
		{genai.TypeArray, "array"},
		{genai.TypeObject, "object"},
		{genai.TypeUnspecified, "string"},
	}

	for _, tc := range tests {
		t.Run(tc.expected, func(t *testing.T) {
			result := schemaTypeToString(tc.input)
			if result != tc.expected {
				t.Errorf("expected %q, got %q", tc.expected, result)
			}
		})
	}
}

func TestConvertSchema(t *testing.T) {
	t.Run("nil schema", func(t *testing.T) {
		result, err := convertSchema(nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result["type"] != "object" {
			t.Error("expected type=object for nil schema")
		}
	})

	t.Run("schema with properties", func(t *testing.T) {
		schema := &genai.Schema{
			Type:     genai.TypeObject,
			Required: []string{"name"},
			Properties: map[string]*genai.Schema{
				"name": {Type: genai.TypeString, Description: "The name"},
			},
		}
		result, err := convertSchema(schema)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result["type"] != "object" {
			t.Error("expected type=object")
		}
		props, ok := result["properties"].(map[string]any)
		if !ok {
			t.Fatal("expected properties map")
		}
		if _, ok := props["name"]; !ok {
			t.Fatal("expected 'name' property")
		}
	})

	t.Run("schema with items", func(t *testing.T) {
		schema := &genai.Schema{
			Type:  genai.TypeArray,
			Items: &genai.Schema{Type: genai.TypeString},
		}
		result, err := convertSchema(schema)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if _, ok := result["items"]; !ok {
			t.Fatal("expected items")
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
	m := New(Config{ModelName: "gpt-4o"})
	var _ model.LLM = m

	if m.Name() != "gpt-4o" {
		t.Errorf("expected name 'gpt-4o', got %q", m.Name())
	}
}

// --- JSON serialization of messages ---

func TestMessageSerialization(t *testing.T) {
	t.Run("user message with image serializes", func(t *testing.T) {
		m := New(Config{ModelName: "gpt-4o"})
		content := &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{Text: "describe this"},
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("png")}},
			},
		}

		msgs, err := m.convertContentToMessages(content)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should be serializable to JSON
		data, err := json.Marshal(msgs)
		if err != nil {
			t.Fatalf("failed to marshal messages: %v", err)
		}
		if len(data) == 0 {
			t.Fatal("expected non-empty JSON")
		}
	})

	t.Run("user message with audio serializes", func(t *testing.T) {
		m := New(Config{ModelName: "gpt-4o"})
		content := &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{InlineData: &genai.Blob{MIMEType: "audio/wav", Data: []byte("wav")}},
			},
		}

		msgs, err := m.convertContentToMessages(content)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data, err := json.Marshal(msgs)
		if err != nil {
			t.Fatalf("failed to marshal messages: %v", err)
		}
		if len(data) == 0 {
			t.Fatal("expected non-empty JSON")
		}
	})

	t.Run("user message with file serializes", func(t *testing.T) {
		m := New(Config{ModelName: "gpt-4o"})
		content := &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("pdf")}},
			},
		}

		msgs, err := m.convertContentToMessages(content)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		data, err := json.Marshal(msgs)
		if err != nil {
			t.Fatalf("failed to marshal messages: %v", err)
		}
		if len(data) == 0 {
			t.Fatal("expected non-empty JSON")
		}
	})
}
