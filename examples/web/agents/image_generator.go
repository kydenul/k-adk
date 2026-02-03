// Package agents provides agent implementations for the web example application.
package agents

import (
	"context"

	"github.com/kydenul/log"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/loadartifactstool"
	"google.golang.org/genai"
)

// generatorImageInput defines the input parameters for the image generation tool.
type generatorImageInput struct {
	// Prompt is the text description of the image to generate.
	Prompt string `json:"prompt"`
	// Filename is the name under which the generated image will be saved.
	Filename string `json:"filename"`
}

// generatorImageResult contains the result of an image generation operation.
type generatorImageResult struct {
	// Filename is the name of the saved image file.
	Filename string `json:"filename"`
	// Status indicates the result of the operation ("success" or "fail").
	Status string `json:"status"`
}

// generatorImage is a tool function that generates an image using Google's Imagen model
// and saves it to the artifact service. It uses Vertex AI backend with Imagen 3.0.
func generatorImage(ctx tool.Context, input generatorImageInput) (generatorImageResult, error) {
	// Create a new genai client configured for Vertex AI backend.
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Project:  "",
		Location: "",
		Backend:  genai.BackendVertexAI,
	})
	if err != nil {
		//nolint:nilerr
		return generatorImageResult{
			Status: "fail",
		}, nil
	}

	// Call the Imagen 3.0 model to generate a single image from the prompt.
	response, err := client.Models.GenerateImages(
		ctx,
		"gemini-2.5-flash-image",
		input.Prompt,
		&genai.GenerateImagesConfig{NumberOfImages: 1},
	)
	if err != nil {
		//nolint:nilerr
		return generatorImageResult{
			Status: "fail",
		}, nil
	}

	// Save the generated image bytes to the artifact service as a PNG file.
	_, err = ctx.Artifacts().Save(ctx, input.Filename,
		genai.NewPartFromBytes(response.GeneratedImages[0].Image.ImageBytes, "image/png"))
	if err != nil {
		//nolint:nilerr
		return generatorImageResult{
			Status: "fail",
		}, nil
	}

	return generatorImageResult{
		Filename: input.Filename,
		Status:   "success",
	}, nil
}

// ImageGeneratorAgent creates an AI agent capable of generating images based on user prompts.
// The agent uses Google's Imagen model via Vertex AI and can save generated images
// to the artifact service for later retrieval.
func ImageGeneratorAgent(_ context.Context, model model.LLM) agent.Agent {
	// Create the image generation function tool that wraps the generatorImage function.
	geneImgtool, err := functiontool.New(functiontool.Config{
		Name:        "generate_image",
		Description: "Generates image and saves in artifact service.",
	}, generatorImage)
	if err != nil {
		log.Fatalf("Failed to create function tool: %v", err)
	}

	// Create the LLM agent with image generation and artifact loading capabilities.
	imageGeneratorAgent, err := llmagent.New(llmagent.Config{
		Name:        "image_generator",
		Model:       model,
		Description: "Agent to generate pictures, answers questions about it and saves it locally if asked.",
		Instruction: "You are an agent whose job is to generate or edit an image based on the user's prompt.",
		Tools: []tool.Tool{
			geneImgtool,             // Tool for generating new images
			loadartifactstool.New(), // Tool for loading previously saved artifacts
		},
	})
	if err != nil {
		log.Fatalf("Failed to create image generator agent: %v", err)
	}

	return imageGeneratorAgent
}
