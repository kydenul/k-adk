package openai

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"sync"

	discardlog "github.com/kydenul/k-adk/internal/discard_log"
	"github.com/kydenul/log"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

var (
	// Ensure Model implements model.LLM
	_ model.LLM = (*Model)(nil)

	ErrNoChoicesInResponse = errors.New("no choices in OpenAI response")
)

const maxToolCallIDLength = 40

// Model implements the model.LLM of google official adk with the official OpenAI Go SDK.
// It works with OpenAI and compatible providers (Ollame, vLLM, etc.)
type Model struct {
	log.Logger

	client    *openai.Client
	modelName string

	toolCallMtx sync.RWMutex
	toolCall    map[string]string
}

// Config is the configuration for creating an OpenAI model.
type Config struct {
	// ModelName specifies which model to use (e.g., "gpt-4o", "qwen3:8b").
	ModelName string

	// Optional. APIKey for authentication.
	//
	// Falls back to OPENAI_API_KEY environment variable if empty.
	APIKey string

	// Optional. BaseURL for the API endpoint. Use for OpenAI-compatible providers.
	// e.g. "http://localhost:11434/v1" for Ollama.
	//
	// Falls back if empty:
	//	First -> `OPENAI_API_BASE` environment variable
	//	Secoed -> `https://api.openai.com/v1/`
	BaseURL string

	// Optional. Logger for logging. Falls back to `DiscardLog` if nil.
	Logger log.Logger
}

// New creates a new OpenAI model with the specified configuration.
func New(config Config) *Model {
	opts := make([]option.RequestOption, 0, 2)

	if config.Logger == nil {
		config.Logger = discardlog.NewDiscardLog()
	}

	if config.APIKey != "" {
		opts = append(opts, option.WithAPIKey(config.APIKey))
	}

	if config.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(config.BaseURL))
	}

	// Create a new OpenAI client
	client := openai.NewClient(opts...)

	config.Logger.Infof("openai model created: model=%s, baseURL=%s",
		config.ModelName, config.BaseURL)

	// Return the OpenAI model
	return &Model{
		Logger:    config.Logger,
		client:    &client,
		modelName: config.ModelName,

		toolCall: make(map[string]string),
	}
}

// Name returns the name of the model
func (m *Model) Name() string { return m.modelName }

// GenerateContent sends a request to the LLM and returns responses.
// Set stream to `true` for streaming responses, `false` for a single response.
func (m *Model) GenerateContent(
	ctx context.Context,
	req *model.LLMRequest,
	strem bool,
) iter.Seq2[*model.LLMResponse, error] {
	m.Debugf("GenerateContent called: stream=%v, contents=%d", strem, len(req.Contents))

	if strem {
		return m.generateStream(ctx, req)
	}

	return m.generate(ctx, req)
}

// buildChatCompletionParameters converts the LLMRequest to OpenAI API parameters.
func (m *Model) buildChatCompletionParameters(
	req *model.LLMRequest,
) (openai.ChatCompletionNewParams, error) {
	m.Debugf("building chat completion parameters")

	var message []openai.ChatCompletionMessageParamUnion

	// Add system instruction
	if req.Config != nil && req.Config.SystemInstruction != nil {
		if text := extractText(req.Config.SystemInstruction); text != "" {
			message = append(message, openai.SystemMessage(text))
			m.Debugf("added system instruction: length=%d", len(text))
		}
	}

	// Convert conversation messages
	for _, content := range req.Contents {
		msgs, err := m.convertContentToMessages(content)
		if err != nil {
			m.Errorf("failed to convert content to messages: %v", err)
			return openai.ChatCompletionNewParams{}, err
		}

		message = append(message, msgs...)
	}

	m.Debugf("total messages built: %d", len(message))

	params := openai.ChatCompletionNewParams{
		Model:    m.modelName,
		Messages: message,
	}

	// Apply optional configuration
	if req.Config != nil {
		m.Debugf("applying optional generation config")
		applyGenerationConfig(&params, req.Config)
	}

	return params, nil
}

// applyGenerationConfig applies optional generation settings to the request params.
func applyGenerationConfig(
	params *openai.ChatCompletionNewParams,
	cfg *genai.GenerateContentConfig,
) {
	if cfg.Temperature != nil {
		params.Temperature = openai.Float(float64(*cfg.Temperature))
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxTokens = openai.Int(int64(cfg.MaxOutputTokens))
	}
	if cfg.TopP != nil {
		params.TopP = openai.Float(float64(*cfg.TopP))
	}

	// Stop sequences
	if len(cfg.StopSequences) == 1 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{
			OfString: openai.String(cfg.StopSequences[0]),
		}
	} else if len(cfg.StopSequences) > 1 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: cfg.StopSequences,
		}
	}

	// Reasoning effort (for o-series models)
	if cfg.ThinkingConfig != nil {
		params.ReasoningEffort = convertThinkingLevel(cfg.ThinkingConfig.ThinkingLevel)
	}

	// JSON mode
	if cfg.ResponseMIMEType == "application/json" {
		params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &openai.ResponseFormatJSONObjectParam{},
		}
	}

	// Structured output with schema
	if cfg.ResponseSchema != nil {
		if schemaMap, err := convertSchema(cfg.ResponseSchema); err == nil {
			params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
					JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:        "response",
						Description: openai.String(cfg.ResponseSchema.Description),
						Schema:      schemaMap,
						Strict:      openai.Bool(true),
					},
				},
			}
		}
	}

	// Tools
	if len(cfg.Tools) > 0 {
		params.Tools = convertTools(cfg.Tools)
	}
}

// normalizeToolCallID shortens IDs exceeding OpenAI's 40-char limit using a hash.
// The mapping is stored to allow reverse lookup if needed.
func (m *Model) normalizeToolCallID(id string) string {
	if len(id) <= maxToolCallIDLength {
		return id
	}

	hash := sha256.Sum256([]byte(id))
	shortID := "tc_" + hex.EncodeToString(hash[:])[:maxToolCallIDLength-3]

	m.toolCallMtx.Lock()
	m.toolCall[shortID] = id
	m.toolCallMtx.Unlock()

	m.Debugf("normalized tool call ID: original=%s, short=%s", id, shortID)

	return shortID
}

// denormalizeToolCallID restores the original ID from a shortened one.
//
//nolint:unused
func (m *Model) denormalizeToolCallID(shortID string) string {
	m.toolCallMtx.RLock()
	defer m.toolCallMtx.RUnlock()

	if original, exists := m.toolCall[shortID]; exists {
		return original
	}
	return shortID
}

// convertContentToMessages converts a genai.Content into OpenAI message format.
// Handles text, images, function calls, and function responses.
func (m *Model) convertContentToMessages(
	content *genai.Content,
) ([]openai.ChatCompletionMessageParamUnion, error) {
	var messages []openai.ChatCompletionMessageParamUnion
	var textParts []string
	var toolCalls []openai.ChatCompletionMessageToolCallUnionParam
	var imageParts []openai.ChatCompletionContentPartImageParam

	for _, part := range content.Parts {
		switch {
		case part.FunctionResponse != nil:
			// Tool responses become separate messages
			responseJSON, err := json.Marshal(part.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function response: %w", err)
			}
			normalizedID := m.normalizeToolCallID(part.FunctionResponse.ID)
			messages = append(messages, openai.ToolMessage(string(responseJSON), normalizedID))

		case part.FunctionCall != nil:
			// Collect tool calls for assistant message
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function args: %w", err)
			}
			normalizedID := m.normalizeToolCallID(part.FunctionCall.ID)
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: normalizedID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      part.FunctionCall.Name,
						Arguments: string(argsJSON),
					},
				},
			})

		case part.Text != "":
			textParts = append(textParts, part.Text)

		case part.InlineData != nil:
			if img := convertInlineDataToImage(part.InlineData); img != nil {
				imageParts = append(imageParts, *img)
			}
		}
	}

	// Build role-specific message if there's content
	if len(textParts) > 0 || len(imageParts) > 0 || len(toolCalls) > 0 {
		msg := buildRoleMessage(content.Role, textParts, imageParts, toolCalls)
		if msg != nil {
			messages = append(messages, *msg)
		}
	}

	return messages, nil
}

// buildRoleMessage creates the appropriate message type based on role.
func buildRoleMessage(
	role string,
	texts []string,
	images []openai.ChatCompletionContentPartImageParam,
	toolCalls []openai.ChatCompletionMessageToolCallUnionParam,
) *openai.ChatCompletionMessageParamUnion {
	switch convertRole(role) {
	case "user":
		return buildUserMessage(texts, images)
	case "assistant":
		return buildAssistantMessage(texts, toolCalls)
	case "system":
		msg := openai.SystemMessage(joinTexts(texts))
		return &msg
	}
	return nil
}

// convertResponse transforms an OpenAI response into an LLMResponse.
func convertResponse(resp *openai.ChatCompletion) (*model.LLMResponse, error) {
	if resp == nil {
		return nil, errors.New("received nil response from OpenAI API")
	}
	if len(resp.Choices) == 0 {
		return nil, ErrNoChoicesInResponse
	}

	choice := resp.Choices[0]
	content := &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{},
	}

	if choice.Message.Content != "" {
		content.Parts = append(content.Parts, &genai.Part{Text: choice.Message.Content})
	}

	for _, tc := range choice.Message.ToolCalls {
		content.Parts = append(content.Parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   tc.ID,
				Name: tc.Function.Name,
				Args: parseJSONArgs(tc.Function.Arguments),
			},
		})
	}

	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: convertUsageMetadata(resp.Usage),
		FinishReason:  convertFinishReason(choice.FinishReason),
		TurnComplete:  true,
	}, nil
}

// convertTools transforms genai tools into OpenAI function tool format.
func convertTools(genaiTools []*genai.Tool) []openai.ChatCompletionToolUnionParam {
	var tools []openai.ChatCompletionToolUnionParam

	for _, genaiTool := range genaiTools {
		if genaiTool == nil {
			continue
		}

		for _, funcDecl := range genaiTool.FunctionDeclarations {
			params := funcDecl.ParametersJsonSchema
			if params == nil {
				params = funcDecl.Parameters
			}

			tools = append(tools, openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
				Name:        funcDecl.Name,
				Description: openai.String(funcDecl.Description),
				Parameters:  convertToFunctionParams(params),
			}))
		}
	}

	return tools
}

// generate sends a non-streaming request to the LLM and yields a single responses
func (m *Model) generate(
	ctx context.Context,
	req *model.LLMRequest,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.Debugf("starting non-streaming generation")

		params, err := m.buildChatCompletionParameters(req)
		if err != nil {
			m.Errorf("failed to build parameters: %v", err)
			yield(nil, err)
			return
		}

		m.Debugf("sending request to OpenAI API")
		resp, err := m.client.Chat.Completions.New(ctx, params)
		if err != nil {
			m.Errorf("OpenAI API request failed: %v", err)
			yield(nil, err)
			return
		}

		m.Debugf("received response from OpenAI API: choices=%d", len(resp.Choices))

		llmResp, err := convertResponse(resp)
		if err != nil {
			m.Errorf("failed to convert response: %v", err)
			yield(nil, err)
			return
		}

		if llmResp.UsageMetadata != nil {
			m.Infof("generation completed: prompt_tokens=%d, completion_tokens=%d, total_tokens=%d",
				llmResp.UsageMetadata.PromptTokenCount,
				llmResp.UsageMetadata.CandidatesTokenCount,
				llmResp.UsageMetadata.TotalTokenCount)
		}

		yield(llmResp, nil)
	}
}

// generateStream sends a streaming request to the LLM and yields responses when they arrive,
// followed by a final aggregated responses.
func (m *Model) generateStream(
	ctx context.Context,
	req *model.LLMRequest,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.Debugf("starting streaming generation")

		params, err := m.buildChatCompletionParameters(req)
		if err != nil {
			m.Errorf("failed to build chat completion parameters: %v", err)
			yield(nil, err)
			return
		}

		m.Debugf("opening stream to OpenAI API")
		stream := m.client.Chat.Completions.NewStreaming(ctx, params)
		accum := openai.ChatCompletionAccumulator{}

		chunkCount := 0
		for stream.Next() {
			chunk := stream.Current()
			accum.AddChunk(chunk)
			chunkCount++

			if len(chunk.Choices) <= 0 || chunk.Choices[0].Delta.Content == "" {
				continue
			}

			// Yield partial responses as chunks are received
			if !yield(&model.LLMResponse{
				Content: &genai.Content{
					Role:  genai.RoleModel,
					Parts: []*genai.Part{{Text: chunk.Choices[0].Delta.Content}},
				},
				Partial:      true,
				TurnComplete: false,
			}, nil) {
				m.Warnf("streaming response cancelled by caller")
				return
			}
		}

		if err := stream.Err(); err != nil {
			m.Errorf("stream error: %v", err)
			yield(nil, err)
			return
		}

		m.Debugf("stream completed: total_chunks=%d", chunkCount)

		// Build and yield final aggregated response
		finalResp := buildStreamFinalResponse(&accum)
		if finalResp.UsageMetadata != nil {
			m.Infof("stream done: in=%d, out=%d, total=%d",
				finalResp.UsageMetadata.PromptTokenCount,
				finalResp.UsageMetadata.CandidatesTokenCount,
				finalResp.UsageMetadata.TotalTokenCount)
		}

		yield(finalResp, nil)
	}
}

// buildStreamFinalResponse creates the final LLMResponse from accumulated stream chunks.
func buildStreamFinalResponse(
	accum *openai.ChatCompletionAccumulator,
) *model.LLMResponse {
	content := &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{},
	}

	if len(accum.Choices) > 0 {
		choice := accum.Choices[0]

		if choice.Message.Content != "" {
			content.Parts = append(content.Parts,
				&genai.Part{Text: choice.Message.Content})
		}

		for _, tc := range choice.Message.ToolCalls {
			content.Parts = append(content.Parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   tc.ID,
					Name: tc.Function.Name,
					Args: parseJSONArgs(tc.Function.Arguments),
				},
			})
		}
	}

	var finalReason genai.FinishReason
	if len(accum.Choices) > 0 {
		finalReason = convertFinishReason(accum.Choices[0].FinishReason)
	}

	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: convertUsageMetadata(accum.Usage),
		FinishReason:  finalReason,
		Partial:       false,
		TurnComplete:  true,
	}
}
