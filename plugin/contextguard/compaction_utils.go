package contextguard

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

const summarizeSystemPrompt = "You are summarizing a conversation " +
	"to preserve context for continuing later.\n\n" +
	"Critical: This summary will be the ONLY context " +
	"available when the conversation resumes. " +
	"Assume all previous messages will be lost. " +
	"Be thorough.\n\n" +
	"Required sections:\n\n" +
	"## Current State\n\n" +
	"- What was being discussed or worked on " +
	"(exact user request if applicable)\n" +
	"- Current progress and what has been completed\n" +
	"- What was being addressed right now " +
	"(incomplete work or open thread)\n" +
	"- What remains to be done or answered " +
	"(specific, not vague)\n\n" +
	"## Key Information\n\n" +
	"- Facts, data, and specific details mentioned " +
	"(names, dates, numbers, URLs, identifiers)\n" +
	"- User preferences, instructions, and constraints " +
	"stated during the conversation\n" +
	"- Definitions, terminology, or domain knowledge " +
	"established\n" +
	"- Any external resources, references, or sources " +
	"mentioned\n\n" +
	"## Context & Decisions\n\n" +
	"- Decisions made during the conversation and why\n" +
	"- Alternatives that were considered and discarded " +
	"(and why)\n" +
	"- Assumptions made\n" +
	"- Important clarifications or corrections " +
	"that occurred\n" +
	"- Any blockers, risks, or open questions " +
	"identified\n\n" +
	"## Exact Next Steps\n\n" +
	"Be specific. Don't write \"continue with the task\"" +
	" — write exactly what should happen next, " +
	"with enough detail that someone reading only " +
	"this summary can pick up without asking " +
	"questions.\n\n" +
	"Tone: Write as if briefing a colleague taking " +
	"over mid-conversation. Include everything they " +
	"would need to continue without asking questions. " +
	"Write in the same language as the conversation." +
	"\n\n" +
	"Length: A dynamic word limit will be appended " +
	"to this prompt at runtime based on the model's " +
	"buffer size. Within that limit, err on the side " +
	"of too much detail rather than too little. " +
	"Critical context is worth the tokens."

// --- Session state helpers ---

func loadSummary(ctx agent.CallbackContext) string {
	key := stateKeyPrefixSummary + ctx.AgentName()
	val, err := ctx.State().Get(key)
	if err != nil {
		return ""
	}
	s, _ := val.(string)
	return s
}

func persistSummary(ctx agent.CallbackContext, summary string, tokenCount int) {
	keySummary := stateKeyPrefixSummary + ctx.AgentName()
	keySummarizedAt := stateKeyPrefixSummarizedAt + ctx.AgentName()
	if err := ctx.State().Set(keySummary, summary); err != nil {
		slog.Warn("ContextGuard: failed to persist summary", "error", err)
	}
	if err := ctx.State().Set(keySummarizedAt, tokenCount); err != nil {
		slog.Warn("ContextGuard: failed to persist token count", "error", err)
	}
}

func loadContentsAtCompaction(ctx agent.CallbackContext) int {
	key := stateKeyPrefixContentsAtCompaction + ctx.AgentName()
	val, err := ctx.State().Get(key)
	if err != nil {
		return 0
	}
	if val == nil {
		return 0
	}
	switch v := val.(type) {
	case int:
		return v
	case float64:
		return int(v)
	}
	return 0
}

func persistContentsAtCompaction(ctx agent.CallbackContext, count int) {
	key := stateKeyPrefixContentsAtCompaction + ctx.AgentName()
	if err := ctx.State().Set(key, count); err != nil {
		slog.Warn("ContextGuard: failed to persist contents count", "error", err)
	}
}

func persistRealTokens(ctx agent.CallbackContext, tokens int) {
	key := stateKeyPrefixRealTokens + ctx.AgentName()
	if err := ctx.State().Set(key, tokens); err != nil {
		slog.Warn("ContextGuard: failed to persist real token count", "error", err)
	}
}

func loadRealTokens(ctx agent.CallbackContext) int {
	key := stateKeyPrefixRealTokens + ctx.AgentName()
	val, err := ctx.State().Get(key)
	if err != nil {
		return 0
	}
	if val == nil {
		return 0
	}
	switch v := val.(type) {
	case int:
		return v
	case float64:
		return int(v)
	}
	return 0
}

func persistLastHeuristic(ctx agent.CallbackContext, tokens int) {
	key := stateKeyPrefixLastHeuristic + ctx.AgentName()
	if err := ctx.State().Set(key, tokens); err != nil {
		slog.Warn("ContextGuard: failed to persist last heuristic", "error", err)
	}
}

func loadLastHeuristic(ctx agent.CallbackContext) int {
	key := stateKeyPrefixLastHeuristic + ctx.AgentName()
	val, err := ctx.State().Get(key)
	if err != nil {
		return 0
	}
	if val == nil {
		return 0
	}
	switch v := val.(type) {
	case int:
		return v
	case float64:
		return int(v)
	}
	return 0
}

func resetCalibration(ctx agent.CallbackContext) {
	keyReal := stateKeyPrefixRealTokens + ctx.AgentName()
	keyHeuristic := stateKeyPrefixLastHeuristic + ctx.AgentName()
	if err := ctx.State().Set(keyReal, 0); err != nil {
		slog.Warn("ContextGuard: failed to reset real tokens", "error", err)
	}
	if err := ctx.State().Set(keyHeuristic, 0); err != nil {
		slog.Warn("ContextGuard: failed to reset last heuristic", "error", err)
	}
}

func truncateForSummarizer(contents []*genai.Content, contextWindow int) []*genai.Content {
	budget := int(float64(contextWindow) * 0.80)
	total := estimateContentTokens(contents)
	if total <= budget {
		return contents
	}

	for len(contents) > 2 && estimateContentTokens(contents) > budget {
		contents = contents[1:]
	}
	return contents
}

func tokenCount(ctx agent.CallbackContext, req *model.LLMRequest) int {
	currentHeuristic := estimateTokens(req)
	realTokens := loadRealTokens(ctx)

	if realTokens <= 0 {
		result := int(float64(currentHeuristic) * defaultHeuristicCorrectionFactor)
		slog.Debug("ContextGuard [tokenCount]: no calibration data, using default factor",
			"agent", ctx.AgentName(),
			"heuristic", currentHeuristic,
			"factor", defaultHeuristicCorrectionFactor,
			"result", result,
		)
		return result
	}

	lastHeuristic := loadLastHeuristic(ctx)
	var calibrated int
	var correction float64
	if lastHeuristic > 0 {
		correction = float64(realTokens) / float64(lastHeuristic)
		if correction < 1.0 {
			correction = 1.0
		}
		if correction > maxCorrectionFactor {
			correction = maxCorrectionFactor
		}
		calibrated = int(float64(currentHeuristic) * correction)
	} else {
		correction = defaultHeuristicCorrectionFactor
		calibrated = int(float64(currentHeuristic) * correction)
	}

	result := calibrated
	if realTokens > calibrated {
		result = realTokens
	}

	slog.Debug("ContextGuard [tokenCount]: calibrated estimate",
		"agent", ctx.AgentName(),
		"heuristic", currentHeuristic,
		"realTokens", realTokens,
		"lastHeuristic", lastHeuristic,
		"correction", fmt.Sprintf("%.2f", correction),
		"calibrated", calibrated,
		"result", result,
	)
	return result
}

// --- Todo state helpers ---

// TodoItem represents a single task tracked in session state.
type TodoItem struct {
	Content    string `json:"content"`
	Status     string `json:"status"`
	ActiveForm string `json:"active_form,omitempty"`
}

func loadTodos(ctx agent.CallbackContext) []TodoItem {
	val, err := ctx.State().Get("todos")
	if err != nil || val == nil {
		return nil
	}

	switch v := val.(type) {
	case []TodoItem:
		return v
	case []any:
		var items []TodoItem
		for _, raw := range v {
			m, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			item := TodoItem{}
			if c, ok := m["content"].(string); ok {
				item.Content = c
			}
			if s, ok := m["status"].(string); ok {
				item.Status = s
			}
			if a, ok := m["active_form"].(string); ok {
				item.ActiveForm = a
			}
			if item.Content != "" {
				items = append(items, item)
			}
		}
		return items
	}
	return nil
}

// --- Summarization ---

func summarize(
	ctx context.Context,
	llm model.LLM,
	contents []*genai.Content,
	previousSummary string,
	bufferTokens int,
	todos []TodoItem,
) (string, error) {
	maxOutputTokens := int32(float64(bufferTokens) * 0.50)
	maxWords := int(float64(maxOutputTokens) * 0.75)

	systemPrompt := summarizeSystemPrompt + fmt.Sprintf(
		"\n\nKeep the summary under %d words.",
		maxWords,
	)
	userPrompt := buildSummarizePrompt(contents, previousSummary, todos)

	req := &model.LLMRequest{
		Model: llm.Name(),
		Contents: []*genai.Content{
			{
				Role:  "user",
				Parts: []*genai.Part{{Text: userPrompt}},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: systemPrompt}},
			},
			MaxOutputTokens: maxOutputTokens,
		},
	}

	var sb strings.Builder
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return "", fmt.Errorf("summarization LLM call failed: %w", err)
		}
		if resp != nil && resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part != nil && part.Text != "" {
					sb.WriteString(part.Text)
				}
			}
		}
	}

	result := sb.String()
	if result == "" {
		return buildFallbackSummary(contents, previousSummary), nil
	}

	return result, nil
}

func buildSummarizePrompt(
	contents []*genai.Content,
	previousSummary string,
	todos []TodoItem,
) string {
	var sb strings.Builder
	sb.WriteString("Provide a detailed summary of the following conversation.")
	sb.WriteString("\n\n")

	if previousSummary != "" {
		sb.WriteString("[Previous summary for context]\n")
		sb.WriteString(previousSummary)
		sb.WriteString("\n[End previous summary]\n\n")
		sb.WriteString(
			"Incorporate the previous summary into your new summary, updating any information that has changed.\n\n",
		)
	}

	sb.WriteString("[Conversation to summarize]\n")

	for _, content := range contents {
		if content == nil {
			continue
		}
		role := content.Role
		if role == "" {
			role = "unknown"
		}
		for _, part := range content.Parts {
			if part == nil {
				continue
			}
			if part.Text != "" {
				sb.WriteString(role)
				sb.WriteString(": ")
				sb.WriteString(part.Text)
				sb.WriteString("\n")
			}
			if part.FunctionCall != nil {
				sb.WriteString(role)
				sb.WriteString(": [called tool: ")
				sb.WriteString(part.FunctionCall.Name)
				sb.WriteString("]\n")
			}
			if part.FunctionResponse != nil {
				sb.WriteString(role)
				sb.WriteString(": [tool ")
				sb.WriteString(part.FunctionResponse.Name)
				sb.WriteString(" returned a result]\n")
			}
		}
	}
	sb.WriteString("[End of conversation]\n")

	if len(todos) > 0 {
		sb.WriteString("\n[Current todo list]\n")
		for _, t := range todos {
			fmt.Fprintf(&sb, "- [%s] %s\n", t.Status, t.Content)
		}
		sb.WriteString("[End todo list]\n\n")
		sb.WriteString(
			"Include these tasks and their statuses in your summary under a dedicated \"## Todo List\" section. ",
		)
		sb.WriteString("Instruct the resuming assistant to restore them ")
		sb.WriteString("using the `todos` tool to continue tracking progress.\n")
	}

	return sb.String()
}

func buildFallbackSummary(contents []*genai.Content, previousSummary string) string {
	var sb strings.Builder
	if previousSummary != "" {
		sb.WriteString(previousSummary)
		sb.WriteString("\n\n---\n\n")
	}
	for _, content := range contents {
		if content == nil {
			continue
		}
		for _, part := range content.Parts {
			if part != nil && part.Text != "" {
				role := content.Role
				if role == "" {
					role = "unknown"
				}
				sb.WriteString(role)
				sb.WriteString(": ")
				if len(part.Text) > 200 {
					sb.WriteString(part.Text[:200])
					sb.WriteString("...")
				} else {
					sb.WriteString(part.Text)
				}
				sb.WriteString("\n")
			}
		}
	}
	return sb.String()
}

// --- Token estimation ---

func estimatePartTokens(part *genai.Part) int {
	if part == nil {
		return 0
	}
	total := 0
	if part.Text != "" {
		total += len(part.Text) / 4
	}
	if part.FunctionCall != nil {
		total += len(part.FunctionCall.Name) / 4
		for k, v := range part.FunctionCall.Args {
			total += len(k) / 4
			total += len(fmt.Sprintf("%v", v)) / 4
		}
	}
	if part.FunctionResponse != nil {
		total += len(part.FunctionResponse.Name) / 4
		total += len(fmt.Sprintf("%v", part.FunctionResponse.Response)) / 4
	}
	if part.InlineData != nil {
		total += len(part.InlineData.MIMEType) / 4
		total += len(part.InlineData.Data) / 4
	}
	return total
}

func estimateTokens(req *model.LLMRequest) int {
	total := estimateContentTokens(req.Contents)
	if req.Config != nil {
		if req.Config.SystemInstruction != nil {
			for _, part := range req.Config.SystemInstruction.Parts {
				total += estimatePartTokens(part)
			}
		}
		total += estimateToolTokens(req.Config.Tools)
	}
	return total
}

func estimateContentTokens(contents []*genai.Content) int {
	total := 0
	for _, content := range contents {
		if content == nil {
			continue
		}
		for _, part := range content.Parts {
			total += estimatePartTokens(part)
		}
	}
	return total
}

func estimateToolTokens(tools []*genai.Tool) int {
	total := 0
	for _, t := range tools {
		if t == nil {
			continue
		}
		for _, fd := range t.FunctionDeclarations {
			if fd == nil {
				continue
			}
			total += len(fd.Name) / 4
			total += len(fd.Description) / 4
			if fd.ParametersJsonSchema != nil {
				data, err := json.Marshal(fd.ParametersJsonSchema)
				if err == nil {
					total += len(data) / 4
				}
			} else if fd.Parameters != nil {
				data, err := json.Marshal(fd.Parameters)
				if err == nil {
					total += len(data) / 4
				}
			}
		}
	}
	return total
}

func computeBuffer(contextWindow int) int {
	if contextWindow >= largeContextWindowThreshold {
		return largeContextWindowBuffer
	}
	return int(float64(contextWindow) * smallContextWindowRatio)
}

// --- Content splitting and summary injection ---

func safeSplitIndex(contents []*genai.Content, idx int) int {
	if idx <= 0 || idx >= len(contents) {
		return idx
	}

	origIdx := idx

	idx = walkBackToPairBoundary(contents, idx)

	if idx <= 0 {
		idx = walkForwardToPairBoundary(contents, origIdx)
	}

	if idx <= 0 {
		idx = 1
	}
	if idx >= len(contents) {
		idx = len(contents) - 1
	}

	return idx
}

func walkBackToPairBoundary(contents []*genai.Content, idx int) int {
	for idx > 0 {
		c := contents[idx]
		if c == nil {
			break
		}

		if c.Role == "user" && contentHasFunctionResponse(c) {
			idx--
			continue
		}

		if c.Role == "model" && contentHasFunctionCall(c) {
			idx--
			continue
		}

		break
	}
	return idx
}

func walkForwardToPairBoundary(contents []*genai.Content, idx int) int {
	for idx < len(contents) {
		c := contents[idx]
		if c == nil {
			break
		}

		if c.Role == "model" && contentHasFunctionCall(c) {
			idx++
			continue
		}

		if c.Role == "user" && contentHasFunctionResponse(c) {
			idx++
			break
		}

		break
	}
	return idx
}

func contentHasFunctionResponse(c *genai.Content) bool {
	for _, part := range c.Parts {
		if part != nil && part.FunctionResponse != nil {
			return true
		}
	}
	return false
}

func contentHasFunctionCall(c *genai.Content) bool {
	for _, part := range c.Parts {
		if part != nil && part.FunctionCall != nil {
			return true
		}
	}
	return false
}

func injectSummary(req *model.LLMRequest, summary string, contentsAtCompaction int) {
	summaryText := fmt.Sprintf(
		"[Previous conversation summary]\n%s\n[End of summary — conversation continues below]",
		summary,
	)

	if len(req.Contents) > 0 && req.Contents[0] != nil &&
		req.Contents[0].Role == "user" && len(req.Contents[0].Parts) > 0 {
		first := req.Contents[0]
		if first.Parts[0] != nil && first.Parts[0].Text != "" &&
			strings.HasPrefix(first.Parts[0].Text, "[Previous conversation summary]") {
			return
		}
	}

	summaryContent := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: summaryText},
		},
	}

	if contentsAtCompaction > 0 && contentsAtCompaction <= len(req.Contents) {
		newContents := req.Contents[contentsAtCompaction:]
		req.Contents = append([]*genai.Content{summaryContent}, newContents...)
	} else {
		req.Contents = append([]*genai.Content{summaryContent}, req.Contents...)
	}
}

func replaceSummary(req *model.LLMRequest, summary string, recentContents []*genai.Content) {
	summaryContent := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{
				Text: fmt.Sprintf(
					"[Previous conversation summary]\n%s\n[End of summary — conversation continues below]",
					summary,
				),
			},
		},
	}
	req.Contents = append([]*genai.Content{summaryContent}, recentContents...)
}

func injectContinuation(req *model.LLMRequest, userContent *genai.Content) {
	var text string
	if userContent != nil {
		for _, part := range userContent.Parts {
			if part != nil && part.Text != "" {
				text = part.Text
				break
			}
		}
	}

	var msg string
	if text != "" {
		msg = fmt.Sprintf(
			"[System: The conversation was compacted because it exceeded the context window. "+
				"The summary above contains all prior context. The user's current request is: `%s`. "+
				"Continue working on this request without asking the user to repeat anything.]", text)
	} else {
		msg = "[System: The conversation was compacted because it exceeded the context window. " +
			"The summary above contains all prior context. " +
			"Continue working without asking the user to repeat anything.]"
	}

	req.Contents = append(req.Contents, &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{{Text: msg}},
	})
}
