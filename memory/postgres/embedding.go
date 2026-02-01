package postgres

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
	"github.com/spf13/cast"
)

var errNoEmbedding = errors.New("no embedding returned")

// OpenAICompatibleEmbedding implements EmbeddingModel using the OpenAI embeddings API format.
// This is the de facto standard supported by: OpenAI, Ollama (/v1), Azure OpenAI, vLLM, LocalAI, LiteLLM, etc.
type OpenAICompatibleEmbedding struct {
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

type Config struct {
	BaseURL string
	APIKey  string
	Model   string
	// optional, will be auto-detected on first call if 0
	Dimension int32

	// HTTPClient allows customizing the HTTP client used for requests.
	// Useful for testing with mock servers.
	HTTPClient *http.Client
}

// NewOpenAICompatibleEmbedding creates a new embedding model using OpenAI-compatible API.
// Works with OpenAI, Ollama, vLLM, LocalAI, LiteLLM, Azure OpenAI, etc.
func NewOpenAICompatibleEmbedding(cfg Config) *OpenAICompatibleEmbedding {
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	emb := &OpenAICompatibleEmbedding{
		BaseURL:    strings.TrimSuffix(cfg.BaseURL, "/"),
		APIKey:     cfg.APIKey,
		Model:      cfg.Model,
		HTTPClient: httpClient,
	}
	emb.dim.Store(cfg.Dimension)
	return emb
}

// Dimension returns the embedding dimension.
// Returns 0 if not yet known (will be auto-detected on first Embed call).
func (e *OpenAICompatibleEmbedding) Dimension() int { return cast.ToInt(e.dim.Load()) }

// Embed generates an embedding vector for the given text.
func (e *OpenAICompatibleEmbedding) Embed(ctx context.Context, text string) ([]float32, error) {
	reqBody := map[string]any{
		"model": e.Model,
		"input": text,
	}

	jsonBody, err := sonic.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST",
		e.BaseURL+"/embeddings", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if e.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.APIKey)
	}

	resp, err := e.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call embedding API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding API returned status %d: %s",
			resp.StatusCode, string(body))
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	var result embeddingResponse
	if err := sonic.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Data) == 0 {
		return nil, errNoEmbedding
	}

	embedding := result.Data[0].Embedding

	// Auto-detect dimension on first successful call (thread-safe using CAS)
	if len(embedding) > 0 && e.dim.Load() == 0 {
		e.dim.CompareAndSwap(0, cast.ToInt32(len(embedding)))
	}

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
