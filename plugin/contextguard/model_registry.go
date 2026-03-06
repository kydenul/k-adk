package contextguard

// ModelRegistry provides model metadata needed by the ContextGuard plugin.
type ModelRegistry interface {
	// ContextWindow returns the maximum context window size (in tokens) for
	// the given model ID. If the model is unknown, a reasonable default
	// should be returned (e.g. 128000).
	ContextWindow(modelID string) int

	// DefaultMaxTokens returns the default maximum output tokens for the
	// given model ID. If the model is unknown, a reasonable default should
	// be returned (e.g. 4096).
	DefaultMaxTokens(modelID string) int
}
