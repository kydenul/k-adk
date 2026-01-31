package anthropic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/kydenul/log"
	"github.com/spf13/cast"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

var (
	// Ensure Model implements model.LLM
	_ model.LLM = (*Model)(nil)

	ErrNoContentInResponse = errors.New("no content in Anthropic response")
)

// Model implements model.LLM using the official Anthropic Go SDK.
type Model struct {
	log.Logger

	client    *anthropic.Client
	modelName string
}

type Config struct {
	// ModelName specifies which model to use (e.g., "gpt-4o", "qwen3:8b").
	ModelName string

	// Optional. APIKey for authentication.
	//
	// Falls back to ANTHROPIC_API_KEY environment variable if empty.
	APIKey string

	// Optional. BaseURL for the custom API endpoint.
	BaseURL string

	// Optional. Logger for logging. Falls back to `DiscardLog` if nil.
	Logger log.Logger
}

// New creates a new Anthropic model with the specified configuration.
func New(config Config) *Model {
	opts := make([]option.RequestOption, 0, 2)

	if config.Logger == nil {
		config.Logger = NewDiscardLog()
	}

	if config.APIKey != "" {
		opts = append(opts, option.WithAPIKey(config.APIKey))
	}

	if config.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(config.BaseURL))
	}

	// Create a new Anthropic client
	client := anthropic.NewClient(opts...)

	config.Logger.Infof("anthropic model created: model=%s, baseURL=%s",
		config.ModelName, config.BaseURL)

	return &Model{
		Logger:    config.Logger,
		client:    &client,
		modelName: config.ModelName,
	}
}

// Name returns the name of the model
func (m *Model) Name() string { return m.modelName }

// GenerateContent sends the request to Anthropic and returns responses(single or streaming).
func (m *Model) GenerateContent(
	ctx context.Context,
	req *model.LLMRequest,
	stream bool,
) iter.Seq2[*model.LLMResponse, error] {
	m.Debugf("GenerateContent called: stream=%v, contents=%d", stream, len(req.Contents))

	if stream {
		return m.generateStream(ctx, req)
	}

	return m.generate(ctx, req)
}

// generate sends a single request and yields one complete response.
func (m *Model) generate(
	ctx context.Context,
	req *model.LLMRequest,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.Debugf("starting non-streaming generation")

		params, err := m.buildMessageParams(req)
		if err != nil {
			m.Errorf("failed to build parameters: %v", err)
			yield(nil, err)
			return
		}

		m.Debugf("sending request to Anthropic API")
		resp, err := m.client.Messages.New(ctx, params)
		if err != nil {
			m.Errorf("Anthropic API request failed: %v", err)
			yield(nil, err)
			return
		}

		m.Debugf("received response from Anthropic API: content_blocks=%d", len(resp.Content))

		llmResp := convertResponse(resp)
		if llmResp.UsageMetadata != nil {
			m.Infof("generation completed: prompt_tokens=%d, completion_tokens=%d, total_tokens=%d",
				llmResp.UsageMetadata.PromptTokenCount,
				llmResp.UsageMetadata.CandidatesTokenCount,
				llmResp.UsageMetadata.TotalTokenCount)
		}

		yield(llmResp, nil)
	}
}

// generateStream sends a request and yields partial responses as they arrive, then a final complete one.
func (m *Model) generateStream(
	ctx context.Context,
	req *model.LLMRequest,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.Debugf("starting streaming generation")

		params, err := m.buildMessageParams(req)
		if err != nil {
			m.Errorf("failed to build parameters: %v", err)
			yield(nil, err)
			return
		}

		m.Debugf("opening stream to Anthropic API")
		stream := m.client.Messages.NewStreaming(ctx, params)

		message := anthropic.Message{}
		chunkCount := 0

		for stream.Next() {
			event := stream.Current()
			if err := message.Accumulate(event); err != nil {
				m.Errorf("failed to accumulate stream event: %v", err)
				yield(nil, err)
				return
			}
			chunkCount++

			// Yield partial text content
			switch eventVariant := event.AsAny().(type) {
			case anthropic.ContentBlockDeltaEvent:
				switch deltaVariant := eventVariant.Delta.AsAny().(type) {
				case anthropic.TextDelta:
					if deltaVariant.Text != "" {
						part := &genai.Part{Text: deltaVariant.Text}
						llmResp := &model.LLMResponse{
							Content: &genai.Content{
								Role:  genai.RoleModel,
								Parts: []*genai.Part{part},
							},
							Partial:      true,
							TurnComplete: false,
						}
						if !yield(llmResp, nil) {
							m.Warnf("streaming response cancelled by caller")
							return
						}
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			m.Errorf("stream error: %v", err)
			yield(nil, err)
			return
		}

		m.Debugf("stream completed: total_chunks=%d", chunkCount)

		// Build final aggregated response
		llmResp := convertResponse(&message)

		llmResp.Partial = false
		llmResp.TurnComplete = true

		if llmResp.UsageMetadata != nil {
			m.Infof("stream done: in=%d, out=%d, total=%d",
				llmResp.UsageMetadata.PromptTokenCount,
				llmResp.UsageMetadata.CandidatesTokenCount,
				llmResp.UsageMetadata.TotalTokenCount)
		}

		yield(llmResp, nil)
	}
}

// buildMessageParams converts an LLMRequest into Anthropic's API format (system prompt, messages, tools, config).
func (m *Model) buildMessageParams(req *model.LLMRequest) (anthropic.MessageNewParams, error) {
	m.Debugf("building message parameters")

	// Default max tokens (required by Anthropic API)
	maxTokens := int64(4096)
	if req.Config != nil && req.Config.MaxOutputTokens > 0 {
		maxTokens = int64(req.Config.MaxOutputTokens)
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(m.modelName),
		MaxTokens: maxTokens,
	}

	// Add system instruction if present
	if req.Config != nil && req.Config.SystemInstruction != nil {
		systemText := extractTextFromContent(req.Config.SystemInstruction)
		if systemText != "" {
			params.System = []anthropic.TextBlockParam{
				{Text: systemText},
			}
			m.Debugf("added system instruction: length=%d", len(systemText))
		}
	}

	// Convert content messages
	messages := []anthropic.MessageParam{}
	for _, content := range req.Contents {
		msg, err := convertContentToMessage(content)
		if err != nil {
			m.Errorf("failed to convert content to message: %v", err)
			return anthropic.MessageNewParams{}, err
		}
		if msg != nil {
			messages = append(messages, *msg)
		}
	}

	// Repair message history to comply with Anthropic's requirements
	// (each tool_use must have a corresponding tool_result immediately after)
	originalLen := len(messages)
	messages = repairMessageHistory(messages)
	if len(messages) != originalLen {
		m.Debugf("repaired message history: original=%d, repaired=%d", originalLen, len(messages))
	}

	params.Messages = messages
	m.Debugf("total messages built: %d", len(messages))

	// Apply config settings
	if req.Config != nil {
		if req.Config.Temperature != nil {
			params.Temperature = anthropic.Float(float64(*req.Config.Temperature))
		}
		if req.Config.TopP != nil {
			params.TopP = anthropic.Float(float64(*req.Config.TopP))
		}
		if len(req.Config.StopSequences) > 0 {
			params.StopSequences = req.Config.StopSequences
		}

		// Convert tools
		if len(req.Config.Tools) > 0 {
			params.Tools = convertTools(req.Config.Tools)
			m.Debugf("added %d tools", len(params.Tools))
		}
	}

	return params, nil
}

// convertContentToMessage transforms a genai.Content (text, images, tool calls/results) into an Anthropic message.
func convertContentToMessage(content *genai.Content) (*anthropic.MessageParam, error) {
	role := convertRoleToAnthropic(content.Role)

	var blocks []anthropic.ContentBlockParamUnion

	for _, part := range content.Parts {
		if part.Text != "" {
			blocks = append(blocks, anthropic.NewTextBlock(part.Text))
		}

		if part.InlineData != nil {
			mediaType := part.InlineData.MIMEType
			switch mediaType {
			case "image/jpg", "image/jpeg", "image/png", "image/gif", "image/webp":
				base64Data := base64.StdEncoding.EncodeToString(part.InlineData.Data)
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfImage: &anthropic.ImageBlockParam{
						Source: anthropic.ImageBlockParamSourceUnion{
							OfBase64: &anthropic.Base64ImageSourceParam{
								MediaType: anthropic.Base64ImageSourceMediaType(mediaType),
								Data:      base64Data,
							},
						},
					},
				})
			}
		}

		if part.FunctionCall != nil {
			blocks = append(blocks, anthropic.ContentBlockParamUnion{
				OfToolUse: &anthropic.ToolUseBlockParam{
					ID:    sanitizeToolID(part.FunctionCall.ID),
					Name:  part.FunctionCall.Name,
					Input: convertToolInputToRaw(part.FunctionCall.Args),
				},
			})
		}

		if part.FunctionResponse != nil {
			responseJSON, err := json.Marshal(part.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function response: %w", err)
			}
			blocks = append(
				blocks,
				anthropic.NewToolResultBlock(
					sanitizeToolID(part.FunctionResponse.ID),
					string(responseJSON),
					false,
				),
			)
		}
	}

	if len(blocks) == 0 {
		return nil, nil
	}

	return &anthropic.MessageParam{Role: role, Content: blocks}, nil
}

// convertResponse transforms Anthropic's response (text, tool_use blocks, usage) into the generic LLMResponse.
func convertResponse(resp *anthropic.Message) *model.LLMResponse {
	content := &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{},
	}

	// Convert content blocks
	for _, block := range resp.Content {
		switch variant := block.AsAny().(type) {
		case anthropic.TextBlock:
			content.Parts = append(content.Parts, &genai.Part{Text: variant.Text})
		case anthropic.ToolUseBlock:
			content.Parts = append(content.Parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   variant.ID,
					Name: variant.Name,
					Args: convertToolInput(variant.Input),
				},
			})
		}
	}

	// Convert usage metadata
	var usageMetadata *genai.GenerateContentResponseUsageMetadata
	if resp.Usage.InputTokens > 0 || resp.Usage.OutputTokens > 0 {
		usageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     cast.ToInt32(resp.Usage.InputTokens),
			CandidatesTokenCount: cast.ToInt32(resp.Usage.OutputTokens),
			TotalTokenCount:      cast.ToInt32(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		}
	}

	return &model.LLMResponse{
		Content:       content,
		UsageMetadata: usageMetadata,
		FinishReason:  convertStopReason(resp.StopReason),
		TurnComplete:  true,
	}
}

// convertTools transforms genai tool definitions into Anthropic's tool format (name, description, JSON schema).
func convertTools(genaiTools []*genai.Tool) []anthropic.ToolUnionParam {
	var tools []anthropic.ToolUnionParam

	for _, genaiTool := range genaiTools {
		if genaiTool == nil {
			continue
		}

		for _, funcDecl := range genaiTool.FunctionDeclarations {
			params := funcDecl.ParametersJsonSchema
			if params == nil {
				params = funcDecl.Parameters
			}

			var inputSchema anthropic.ToolInputSchemaParam
			// Type is required by Anthropic API, must be "object"
			inputSchema.Type = "object"
			if params != nil {
				if m, ok := params.(map[string]any); ok {
					if props, ok := m["properties"]; ok {
						inputSchema.Properties = props
					}
					if req, ok := m["required"].([]string); ok {
						inputSchema.Required = req
					}
				}
			}

			tools = append(tools, anthropic.ToolUnionParam{
				OfTool: &anthropic.ToolParam{
					Name:        funcDecl.Name,
					Description: anthropic.String(funcDecl.Description),
					InputSchema: inputSchema,
				},
			})
		}
	}

	return tools
}
