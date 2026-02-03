package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/spf13/cast"
	"google.golang.org/genai"
)

// buildUserMessage creates a user message, with multi-part support for images.
func buildUserMessage(
	texts []string,
	images []openai.ChatCompletionContentPartImageParam,
) *openai.ChatCompletionMessageParamUnion {
	if len(images) == 0 {
		msg := openai.UserMessage(joinTexts(texts))
		return &msg
	}

	// Multi-part message with images
	var parts []openai.ChatCompletionContentPartUnionParam
	for _, text := range texts {
		parts = append(parts, openai.ChatCompletionContentPartUnionParam{
			OfText: &openai.ChatCompletionContentPartTextParam{Text: text},
		})
	}
	for _, img := range images {
		parts = append(parts, openai.ChatCompletionContentPartUnionParam{
			OfImageURL: &img,
		})
	}

	return &openai.ChatCompletionMessageParamUnion{
		OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfArrayOfContentParts: parts,
			},
		},
	}
}

// buildAssistantMessage creates an assistant message with optional tool calls.
func buildAssistantMessage(
	texts []string,
	toolCalls []openai.ChatCompletionMessageToolCallUnionParam,
) *openai.ChatCompletionMessageParamUnion {
	msg := openai.ChatCompletionAssistantMessageParam{}

	if len(texts) > 0 {
		msg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
			OfString: openai.String(joinTexts(texts)),
		}
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	return &openai.ChatCompletionMessageParamUnion{OfAssistant: &msg}
}

// convertToFunctionParams converts various parameter types to OpenAI format.
// OpenAI requires object schemas to have a "properties" field, even if empty.
func convertToFunctionParams(params any) shared.FunctionParameters {
	if params == nil {
		return nil
	}

	var m map[string]any

	// Direct map
	if dm, ok := params.(map[string]any); ok {
		m = dm
	} else {
		// Convert via JSON for other types (e.g., *jsonschema.Schema)
		jsonBytes, err := json.Marshal(params)
		if err != nil {
			return nil
		}
		if json.Unmarshal(jsonBytes, &m) != nil {
			return nil
		}
	}

	// OpenAI requires "properties" for object types
	ensureObjectProperties(m)

	return shared.FunctionParameters(m)
}

// ensureObjectProperties recursively ensures all object schemas have a properties field.
func ensureObjectProperties(schema map[string]any) {
	if schema == nil {
		return
	}

	// If type is "object" and no properties, add empty properties
	if t, ok := schema["type"].(string); ok && t == "object" {
		if _, hasProps := schema["properties"]; !hasProps {
			schema["properties"] = map[string]any{}
		}
	}

	// Recursively process nested properties
	if props, ok := schema["properties"].(map[string]any); ok {
		for _, prop := range props {
			if propMap, ok := prop.(map[string]any); ok {
				ensureObjectProperties(propMap)
			}
		}
	}

	// Process array items
	if items, ok := schema["items"].(map[string]any); ok {
		ensureObjectProperties(items)
	}
}

// convertSchema recursively converts a genai.Schema to OpenAI JSON schema format.
func convertSchema(schema *genai.Schema) (map[string]any, error) {
	if schema == nil {
		return map[string]any{"type": "object", "properties": map[string]any{}}, nil
	}

	result := make(map[string]any)

	if schema.Type != genai.TypeUnspecified {
		result["type"] = schemaTypeToString(schema.Type)
	}
	if schema.Description != "" {
		result["description"] = schema.Description
	}
	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}
	if len(schema.Enum) > 0 {
		result["enum"] = schema.Enum
	}

	if len(schema.Properties) > 0 {
		props := make(map[string]any)
		for name, propSchema := range schema.Properties {
			converted, err := convertSchema(propSchema)
			if err != nil {
				return nil, err
			}
			props[name] = converted
		}
		result["properties"] = props
	}

	if schema.Items != nil {
		items, err := convertSchema(schema.Items)
		if err != nil {
			return nil, err
		}
		result["items"] = items
	}

	return result, nil
}

// convertInlineDataToImage converts inline image data to OpenAI format.
func convertInlineDataToImage(data *genai.Blob) *openai.ChatCompletionContentPartImageParam {
	supportedTypes := map[string]bool{
		"image/jpg":  true,
		"image/jpeg": true,
		"image/png":  true,
		"image/gif":  true,
		"image/webp": true,
	}

	if !supportedTypes[data.MIMEType] {
		return nil
	}

	return &openai.ChatCompletionContentPartImageParam{
		ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
			URL: fmt.Sprintf(
				"data:%s;base64,%s",
				data.MIMEType,
				base64.StdEncoding.EncodeToString(data.Data),
			),
			Detail: "auto",
		},
	}
}

// convertUsageMetadata converts OpenAI usage stats to genai format.
func convertUsageMetadata(
	usage openai.CompletionUsage,
) *genai.GenerateContentResponseUsageMetadata {
	if usage.TotalTokens == 0 {
		return nil
	}

	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     cast.ToInt32(usage.PromptTokens),
		CandidatesTokenCount: cast.ToInt32(usage.CompletionTokens),
		TotalTokenCount:      cast.ToInt32(usage.TotalTokens),
	}
}

// convertRole maps genai roles to OpenAI roles.
func convertRole(role string) string {
	if role == "model" {
		return "assistant"
	}

	return role // "user" and "system" are the same
}

// convertFinishReason maps OpenAI finish reasons to genai format.
func convertFinishReason(reason string) genai.FinishReason {
	switch reason {
	case "stop", "tool_calls", "function_call":
		return genai.FinishReasonStop

	case "length":
		return genai.FinishReasonMaxTokens

	case "content_filter":
		return genai.FinishReasonSafety

	default:
		return genai.FinishReasonUnspecified
	}
}

// convertThinkingLevel maps genai thinking levels to OpenAI reasoning effort.
func convertThinkingLevel(level genai.ThinkingLevel) shared.ReasoningEffort {
	switch level {
	case genai.ThinkingLevelLow:
		return shared.ReasoningEffortLow

	case genai.ThinkingLevelHigh:
		return shared.ReasoningEffortHigh

	default:
		return shared.ReasoningEffortMedium
	}
}

// schemaTypeToString converts genai.Type to JSON schema type string.
func schemaTypeToString(t genai.Type) string {
	types := map[genai.Type]string{
		genai.TypeString:  "string",
		genai.TypeNumber:  "number",
		genai.TypeInteger: "integer",
		genai.TypeBoolean: "boolean",
		genai.TypeArray:   "array",
		genai.TypeObject:  "object",
	}

	if s, ok := types[t]; ok {
		return s
	}

	return "string"
}

// extractText extracts all text parts from a Content and joins them.
func extractText(content *genai.Content) string {
	if content == nil {
		return ""
	}

	var texts []string
	for _, part := range content.Parts {
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}

	return joinTexts(texts)
}

// joinTexts joins multiple text strings with newlines.
func joinTexts(texts []string) string { return strings.Join(texts, "\n") }

// parseJSONArgs parses a JSON string into a map. Returns empty map on error.
func parseJSONArgs(argsJSON string) map[string]any {
	if argsJSON == "" {
		return make(map[string]any)
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return make(map[string]any)
	}

	return args
}
