package memory

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"

	"github.com/bytedance/sonic"
	discardlog "github.com/kydenul/k-adk/internal/discard_log"
	"github.com/kydenul/log"
	"github.com/spf13/cast"
)

var errNoEmbedding = errors.New("no embedding returned")

// OpenAICompatibleEmbedding implements EmbeddingModel using the OpenAI embeddings API format.
// This is the de facto standard supported by: OpenAI, Ollama (/v1), Azure OpenAI, vLLM, LocalAI, LiteLLM, etc.
type OpenAICompatibleEmbedding struct {
	log.Logger

	// e.g., "https://api.openai.com/v1", "http://localhost:11434/v1"
	BaseURL string

	// Optional. not required for local models
	APIKey string

	// e.g., "text-embedding-3-small", "nomic-embed-text"
	Model string

	// embedding dimension, auto-detected if 0
	dim atomic.Int32

	// HTTPClient allows customizing the HTTP client used for requests.
	// If nil, http.DefaultClient is used.
	HTTPClient *http.Client
}

// EmbeddingConfig holds configuration for creating an OpenAICompatibleEmbedding.
type EmbeddingConfig struct {
	BaseURL string
	APIKey  string
	Model   string
	// optional, will be auto-detected on first call if 0
	Dimension int32

	// HTTPClient allows customizing the HTTP client used for requests.
	// Useful for testing with mock servers.
	HTTPClient *http.Client

	// Optional. Logger for logging. Falls back to DiscardLog if nil.
	Logger log.Logger
}

// NewOpenAICompatibleEmbedding creates a new embedding model using OpenAI-compatible API.
// Works with OpenAI, Ollama, vLLM, LocalAI, LiteLLM, Azure OpenAI, etc.
func NewOpenAICompatibleEmbedding(cfg EmbeddingConfig) *OpenAICompatibleEmbedding {
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	if cfg.Logger == nil {
		cfg.Logger = discardlog.NewDiscardLog()
	}

	emb := &OpenAICompatibleEmbedding{
		Logger:     cfg.Logger,
		BaseURL:    strings.TrimSuffix(cfg.BaseURL, "/"),
		APIKey:     cfg.APIKey,
		Model:      cfg.Model,
		HTTPClient: httpClient,
	}
	emb.dim.Store(cfg.Dimension)

	cfg.Logger.Infof("embedding model created: model=%s, baseURL=%s, dimension=%d",
		cfg.Model, cfg.BaseURL, cfg.Dimension)

	return emb
}

// Dimension returns the embedding dimension.
// Returns 0 if not yet known (will be auto-detected on first Embed call).
func (e *OpenAICompatibleEmbedding) Dimension() int { return cast.ToInt(e.dim.Load()) }

// Embed generates an embedding vector for the given text.
func (e *OpenAICompatibleEmbedding) Embed(ctx context.Context, text string) ([]float32, error) {
	e.Debugf("generating embedding: model=%s, text_length=%d", e.Model, len(text))

	reqBody := map[string]any{
		"model": e.Model,
		"input": text,
	}

	jsonBody, err := sonic.Marshal(reqBody)
	if err != nil {
		e.Errorf("failed to marshal embedding request: %v", err)
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST",
		e.BaseURL+"/embeddings", bytes.NewReader(jsonBody))
	if err != nil {
		e.Errorf("failed to create embedding request: %v", err)
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if e.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.APIKey)
	}

	e.Debugf("sending embedding request to %s/embeddings", e.BaseURL)
	resp, err := e.HTTPClient.Do(req)
	if err != nil {
		e.Errorf("embedding API call failed: %v", err)
		return nil, fmt.Errorf("failed to call embedding API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		e.Errorf("embedding API returned error: status=%d, body=%s", resp.StatusCode, string(body))
		return nil, fmt.Errorf("embedding API returned status %d: %s",
			resp.StatusCode, string(body))
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		e.Errorf("failed to read embedding response body: %v", err)
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	var result embeddingResponse
	if err := sonic.Unmarshal(respBody, &result); err != nil {
		e.Errorf("failed to decode embedding response: %v", err)
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Data) == 0 {
		e.Errorf("no embedding returned from API")
		return nil, errNoEmbedding
	}

	embedding := result.Data[0].Embedding

	// Auto-detect dimension on first successful call (thread-safe using CAS)
	if len(embedding) > 0 && e.dim.Load() == 0 {
		e.dim.CompareAndSwap(0, cast.ToInt32(len(embedding)))
		e.Infof("auto-detected embedding dimension: %d", len(embedding))
	}

	e.Debugf("embedding generated successfully: dimension=%d, prompt_tokens=%d",
		len(embedding), result.Usage.PromptTokens)

	return embedding, nil
}

// embeddingResponse represents the OpenAI embeddings API response format.
type embeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// Ensure interface is implemented
var _ EmbeddingModel = (*OpenAICompatibleEmbedding)(nil)
