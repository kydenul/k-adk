package anthropic

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"regexp"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

var (
	// emptyJSONObject is the JSON representation of an empty object.
	emptyJSONObject = json.RawMessage(`{}`)

	// anthropicToolIDPattern matches valid Anthropic tool_use IDs: ^[a-zA-Z0-9_-]+$
	anthropicToolIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)
)

// convertRoleToAnthropic maps "user"/"model" to Anthropic's role enum (user/assistant).
func convertRoleToAnthropic(role string) anthropic.MessageParamRole {
	switch role {
	case "user":
		return anthropic.MessageParamRoleUser

	case "model":
		return anthropic.MessageParamRoleAssistant

	default:
		return anthropic.MessageParamRoleUser
	}
}

// convertStopReason maps Anthropic's stop reasons (end_turn, max_tokens, tool_use) to genai.FinishReason.
func convertStopReason(reason anthropic.StopReason) genai.FinishReason {
	switch reason {
	case anthropic.StopReasonEndTurn:
		return genai.FinishReasonStop

	case anthropic.StopReasonMaxTokens:
		return genai.FinishReasonMaxTokens

	case anthropic.StopReasonStopSequence:
		return genai.FinishReasonStop

	case anthropic.StopReasonToolUse:
		return genai.FinishReasonStop

	default:
		return genai.FinishReasonUnspecified
	}
}

// convertToolInputToRaw converts tool input to json.RawMessage for sending to Anthropic API.
// Handles nil values and nil maps inside interfaces by returning "{}".
func convertToolInputToRaw(input any) json.RawMessage {
	if input == nil {
		return emptyJSONObject
	}

	// If already json.RawMessage, use directly
	if raw, ok := input.(json.RawMessage); ok && len(raw) > 0 {
		return raw
	}

	// Marshal to JSON (handles nil maps inside interface correctly)
	data, err := json.Marshal(input)
	if err != nil || len(data) == 0 || string(data) == "null" {
		return emptyJSONObject
	}
	return data
}

// convertToolInput converts tool input to map[string]any for storing in genai.FunctionCall.Args.
// Used when receiving tool_use blocks from Anthropic responses.
func convertToolInput(input any) map[string]any {
	if input == nil {
		return map[string]any{}
	}

	if m, ok := input.(map[string]any); ok {
		return m
	}

	// Get JSON bytes: use directly if json.RawMessage, otherwise marshal
	var data []byte
	if raw, ok := input.(json.RawMessage); ok {
		data = raw
	} else {
		var err error
		if data, err = json.Marshal(input); err != nil {
			return map[string]any{}
		}
	}

	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		return map[string]any{}
	}
	return result
}

// extractTextFromContent concatenates all text parts from a genai.Content with newlines.
func extractTextFromContent(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var texts []string
	for _, part := range content.Parts {
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// sanitizeToolID replaces invalid tool IDs (chars outside [a-zA-Z0-9_-]) with a SHA256-based valid ID.
func sanitizeToolID(id string) string {
	if anthropicToolIDPattern.MatchString(id) {
		return id
	}

	// Generate a valid ID from the original using SHA256
	hash := sha256.Sum256([]byte(id))
	return "toolu_" + hex.EncodeToString(hash[:16])
}

// repairMessageHistory removes orphaned tool_use blocks (those without a matching tool_result in the next message).
func repairMessageHistory(messages []anthropic.MessageParam) []anthropic.MessageParam {
	if len(messages) == 0 {
		return messages
	}

	result := make([]anthropic.MessageParam, 0, len(messages))

	for i := 0; i < len(messages); i++ {
		msg := messages[i]

		// Check if this assistant message has tool_use blocks
		if msg.Role == anthropic.MessageParamRoleAssistant {
			toolUseIDs := extractToolUseIDs(msg)

			if len(toolUseIDs) > 0 {
				// Check if next message is a user message with matching tool_results
				if i+1 < len(messages) && messages[i+1].Role == anthropic.MessageParamRoleUser {
					toolResultIDs := extractToolResultIDs(messages[i+1])

					// Find which tool_use IDs have matching tool_results
					matchedIDs := make(map[string]bool)
					for _, id := range toolResultIDs {
						matchedIDs[id] = true
					}

					// Filter out unmatched tool_use blocks from this message
					filteredMsg := filterToolUse(msg, matchedIDs)
					if hasContent(filteredMsg) {
						result = append(result, filteredMsg)
					}
					continue
				}

				// No following user message with tool_results - remove all tool_use blocks
				filteredMsg := filterToolUse(msg, nil)
				if hasContent(filteredMsg) {
					result = append(result, filteredMsg)
				}
				continue
			}
		}

		result = append(result, msg)
	}

	return result
}

// extractToolUseIDs returns all tool_use IDs from an assistant message.
func extractToolUseIDs(msg anthropic.MessageParam) []string {
	var ids []string
	for _, block := range msg.Content {
		if block.OfToolUse != nil {
			ids = append(ids, block.OfToolUse.ID)
		}
	}
	return ids
}

// extractToolResultIDs returns all tool_result IDs from a user message.
func extractToolResultIDs(msg anthropic.MessageParam) []string {
	var ids []string
	for _, block := range msg.Content {
		if block.OfToolResult != nil {
			ids = append(ids, block.OfToolResult.ToolUseID)
		}
	}
	return ids
}

// filterToolUse keeps tool_use blocks whose IDs are in allowedIDs. If allowedIDs is nil, removes all tool_use.
func filterToolUse(msg anthropic.MessageParam, allowedIDs map[string]bool) anthropic.MessageParam {
	var filteredBlocks []anthropic.ContentBlockParamUnion
	for _, block := range msg.Content {
		if block.OfToolUse != nil {
			if allowedIDs != nil && allowedIDs[block.OfToolUse.ID] {
				filteredBlocks = append(filteredBlocks, block)
			}
			continue
		}
		filteredBlocks = append(filteredBlocks, block)
	}
	return anthropic.MessageParam{Role: msg.Role, Content: filteredBlocks}
}

// hasContent returns true if the message has at least one content block.
func hasContent(msg anthropic.MessageParam) bool {
	return len(msg.Content) > 0
}
