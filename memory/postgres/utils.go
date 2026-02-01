package postgres

import (
	"fmt"
	"strings"

	"google.golang.org/genai"
)

// extractTextFromContent extracts text from a genai.Content.
func extractTextFromContent(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var parts []string
	for _, part := range content.Parts {
		if part.Text != "" {
			parts = append(parts, part.Text)
		}
	}
	return strings.TrimSpace(strings.Join(parts, " "))
}

// vectorToString converts a float32 slice to PostgreSQL vector format.
func vectorToString(v []float32) string {
	if len(v) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("[")
	for i, f := range v {
		if i > 0 {
			sb.WriteString(",")
		}
		fmt.Fprintf(&sb, "%f", f)
	}
	sb.WriteString("]")
	return sb.String()
}
