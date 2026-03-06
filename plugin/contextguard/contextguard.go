package contextguard

import (
	"github.com/kydenul/log"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
)

const (
	// StrategyThreshold selects the token-threshold strategy.
	StrategyThreshold = "threshold"

	// StrategySlidingWindow selects the sliding-window strategy.
	StrategySlidingWindow = "sliding_window"
)

const (
	stateKeyPrefixSummary              = "__context_guard_summary_"
	stateKeyPrefixSummarizedAt         = "__context_guard_summarized_at_"
	stateKeyPrefixContentsAtCompaction = "__context_guard_contents_at_compaction_"
	stateKeyPrefixRealTokens           = "__context_guard_real_tokens_"
	stateKeyPrefixLastHeuristic        = "__context_guard_last_heuristic_"

	largeContextWindowThreshold = 200_000
	largeContextWindowBuffer    = 20_000
	smallContextWindowRatio     = 0.20

	maxCompactionAttempts = 3

	defaultHeuristicCorrectionFactor = 2.5
	maxCorrectionFactor              = 5.0
)

const defaultMaxTurns = 20

// Strategy defines how a compaction algorithm decides whether and how to
// compact conversation history before an LLM call.
type Strategy interface {
	Name() string
	Compact(ctx agent.CallbackContext, req *model.LLMRequest) error
}

// AgentOption configures per-agent behavior when calling Add.
type AgentOption func(*agentConfig)

type agentConfig struct {
	strategy  string
	maxTurns  int
	maxTokens int
}

// WithSlidingWindow selects the sliding-window strategy with the given
// maximum number of Content entries before compaction.
func WithSlidingWindow(maxTurns int) AgentOption {
	return func(c *agentConfig) {
		c.strategy = StrategySlidingWindow
		c.maxTurns = maxTurns
	}
}

// WithMaxTokens sets a manual context window size override (in tokens).
func WithMaxTokens(maxTokens int) AgentOption {
	return func(c *agentConfig) { c.maxTokens = maxTokens }
}

// ContextGuard accumulates per-agent strategies and produces a single
// runner.PluginConfig.
type ContextGuard struct {
	registry   ModelRegistry
	strategies map[string]Strategy
}

// New creates a ContextGuard backed by the given ModelRegistry.
func New(registry ModelRegistry) *ContextGuard {
	return &ContextGuard{
		registry:   registry,
		strategies: make(map[string]Strategy),
	}
}

// Add registers an agent with its LLM for summarization.
func (g *ContextGuard) Add(agentID string, llm model.LLM, opts ...AgentOption) {
	cfg := &agentConfig{
		strategy: StrategyThreshold,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	switch cfg.strategy {
	case StrategySlidingWindow:
		maxTurns := cfg.maxTurns
		if maxTurns <= 0 {
			maxTurns = defaultMaxTurns
		}
		g.strategies[agentID] = newSlidingWindowStrategy(g.registry, llm, maxTurns)
	default:
		g.strategies[agentID] = newThresholdStrategy(g.registry, llm, cfg.maxTokens)
	}

	log.Info("ContextGuard: strategy configured", "agent", agentID,
		"strategy", g.strategies[agentID].Name())
}

// PluginConfig returns a runner.PluginConfig ready to pass to the ADK runner.
func (g *ContextGuard) PluginConfig() runner.PluginConfig {
	guard := &contextGuardPlugin{strategies: g.strategies}

	p, _ := plugin.New(plugin.Config{
		Name:                "context_guard",
		BeforeModelCallback: llmagent.BeforeModelCallback(guard.beforeModel),
		AfterModelCallback:  llmagent.AfterModelCallback(guard.afterModel),
	})

	return runner.PluginConfig{Plugins: []*plugin.Plugin{p}}
}

type contextGuardPlugin struct {
	strategies map[string]Strategy
}

func (g *contextGuardPlugin) beforeModel(
	ctx agent.CallbackContext,
	req *model.LLMRequest,
) (*model.LLMResponse, error) {
	if req == nil || len(req.Contents) == 0 {
		return nil, nil
	}

	strategy, ok := g.strategies[ctx.AgentName()]
	if !ok {
		return nil, nil
	}

	if err := strategy.Compact(ctx, req); err != nil {
		log.Warn("ContextGuard: compaction failed, passing through",
			"agent", ctx.AgentName(),
			"strategy", strategy.Name(),
			"error", err,
		)
	}

	persistLastHeuristic(ctx, estimateTokens(req))

	return nil, nil
}

func (g *contextGuardPlugin) afterModel(
	ctx agent.CallbackContext,
	resp *model.LLMResponse,
	_ error,
) (*model.LLMResponse, error) {
	if resp == nil || resp.Partial {
		return nil, nil
	}
	if resp.UsageMetadata == nil {
		return nil, nil
	}
	if _, ok := g.strategies[ctx.AgentName()]; !ok {
		return nil, nil
	}
	promptTokens := int(resp.UsageMetadata.PromptTokenCount)
	if promptTokens > 0 {
		persistRealTokens(ctx, promptTokens)
	}
	return nil, nil
}
