package contextguard

import (
	"log/slog"
	"sync"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

type thresholdStrategy struct {
	registry  ModelRegistry
	llm       model.LLM
	maxTokens int
	mu        sync.Mutex
}

func newThresholdStrategy(registry ModelRegistry, llm model.LLM, maxTokens int) *thresholdStrategy {
	return &thresholdStrategy{
		registry:  registry,
		llm:       llm,
		maxTokens: maxTokens,
	}
}

func (s *thresholdStrategy) Name() string {
	return StrategyThreshold
}

func (s *thresholdStrategy) Compact(ctx agent.CallbackContext, req *model.LLMRequest) error {
	var contextWindow int
	if s.maxTokens > 0 {
		contextWindow = s.maxTokens
	} else {
		contextWindow = s.registry.ContextWindow(req.Model)
	}
	buffer := computeBuffer(contextWindow)
	threshold := contextWindow - buffer

	existingSummary := loadSummary(ctx)
	contentsAtLastCompaction := loadContentsAtCompaction(ctx)
	totalSessionContents := len(req.Contents)
	if existingSummary != "" {
		injectSummary(req, existingSummary, contentsAtLastCompaction)
	}

	totalTokens := tokenCount(ctx, req)
	if totalTokens < threshold {
		return nil
	}

	slog.Info("ContextGuard [threshold]: threshold exceeded, summarizing",
		"agent", ctx.AgentName(),
		"session", ctx.SessionID(),
		"tokens", totalTokens,
		"threshold", threshold,
		"contextWindow", contextWindow,
		"buffer", buffer,
		"maxSummaryWords", int(float64(buffer)*0.50*0.75),
	)

	s.mu.Lock()
	defer s.mu.Unlock()

	userContent := ctx.UserContent()
	todos := loadTodos(ctx)

	contentsForSummary := truncateForSummarizer(req.Contents, contextWindow)

	summary, err := summarize(ctx, s.llm, contentsForSummary, existingSummary, buffer, todos)
	if err != nil {
		slog.Warn("ContextGuard [threshold]: summarization failed, using fallback",
			"agent", ctx.AgentName(),
			"session", ctx.SessionID(),
			"error", err,
		)
		summary = buildFallbackSummary(contentsForSummary, existingSummary)
	}

	persistSummary(ctx, summary, totalTokens)
	persistContentsAtCompaction(ctx, totalSessionContents)
	replaceSummary(req, summary, nil)
	injectContinuation(req, userContent)

	resetCalibration(ctx)

	newTokens := estimateTokens(req)

	slog.Info("ContextGuard [threshold]: compaction completed",
		"agent", ctx.AgentName(),
		"session", ctx.SessionID(),
		"oldMessages", len(req.Contents),
		"newTokenEstimate", newTokens,
		"threshold", threshold,
	)

	return nil
}
