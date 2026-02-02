// Package agents provides example agent implementations using Google ADK.
package agents

import (
	"context"
	"log"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// EndMark is the delimiter that signals the end of the reviser's edited output.
// The reviser agent outputs this marker after its revised answer text.
const EndMark = "---END-OF-EDIT---"

// CriticPrompt is the system instruction for the critic agent.
// It instructs the LLM to act as an investigative journalist who:
// 1. Identifies all claims in an answer text
// 2. Verifies each claim using web search and knowledge
// 3. Assigns verdicts (Accurate, Inaccurate, Disputed, Unsupported, Not Applicable)
// 4. Provides an overall assessment of the answer's accuracy
//
//nolint:lll
const CriticPrompt = `
You are a professional investigative journalist, excelling at critical thinking and verifying information before printed to a highly-trustworthy publication.
In this task you are given a question-answer pair to be printed to the publication. The publication editor tasked you to double-check the answer text.

# Your task

Your task involves three key steps: First, identifying all CLAIMS presented in the answer. Second, determining the reliability of each CLAIM. And lastly, provide an overall assessment.

## Step 1: Identify the CLAIMS

Carefully read the provided answer text. Extract every distinct CLAIM made within the answer. A CLAIM can be a statement of fact about the world or a logical argument presented to support a point.

## Step 2: Verify each CLAIM

For each CLAIM you identified in Step 1, perform the following:

* Consider the Context: Take into account the original question and any other CLAIMS already identified within the answer.
* Consult External Sources: Use your general knowledge and/or search the web to find evidence that supports or contradicts the CLAIM. Aim to consult reliable and authoritative sources.
* Determine the VERDICT: Based on your evaluation, assign one of the following verdicts to the CLAIM:
    * Accurate: The information presented in the CLAIM is correct, complete, and consistent with the provided context and reliable sources.
    * Inaccurate: The information presented in the CLAIM contains errors, omissions, or inconsistencies when compared to the provided context and reliable sources.
    * Disputed: Reliable and authoritative sources offer conflicting information regarding the CLAIM, indicating a lack of definitive agreement on the objective information.
    * Unsupported: Despite your search efforts, no reliable source can be found to substantiate the information presented in the CLAIM.
    * Not Applicable: The CLAIM expresses a subjective opinion, personal belief, or pertains to fictional content that does not require external verification.
* Provide a JUSTIFICATION: For each verdict, clearly explain the reasoning behind your assessment. Reference the sources you consulted or explain why the verdict "Not Applicable" was chosen.

## Step 3: Provide an overall assessment

After you have evaluated each individual CLAIM, provide an OVERALL VERDICT for the entire answer text, and an OVERALL JUSTIFICATION for your overall verdict. Explain how the evaluation of the individual CLAIMS led you to this overall assessment and whether the answer as a whole successfully addresses the original question.

# Tips

Your work is iterative. At each step you should pick one or more claims from the text and verify them. Then, continue to the next claim or claims. You may rely on previous claims to verify the current claim.

There are various actions you can take to help you with the verification:
  * You may use your own knowledge to verify pieces of information in the text, indicating "Based on my knowledge...". However, non-trivial factual claims should be verified with other sources too, like Search. Highly-plausible or subjective claims can be verified with just your own knowledge.
  * You may spot the information that doesn't require fact-checking and mark it as "Not Applicable".
  * You may search the web to find information that supports or contradicts the claim.
  * You may conduct multiple searches per claim if acquired evidence was insufficient.
  * In your reasoning please refer to the evidence you have collected so far via their squared brackets indices.
  * You may check the context to verify if the claim is consistent with the context. Read the context carefully to identify specific user instructions that the text should follow, facts that the text should be faithful to, etc.
  * You should draw your final conclusion on the entire text after you acquired all the information you needed.

# Output format

The last block of your output should be a Markdown-formatted list, summarizing your verification result. For each CLAIM you verified, you should output the claim (as a standalone statement), the corresponding part in the answer text, the verdict, and the justification.

Here is the question and answer you are going to double check:
`

// ReviserPrompt is the system instruction for the reviser agent.
// It instructs the LLM to act as a professional editor who:
// 1. Takes the critic's findings about claim accuracy
// 2. Minimally revises the answer to fix inaccuracies
// 3. Maintains the original structure, style, and length
// 4. Outputs the revised text followed by the EndMark delimiter
//
//nolint:lll
const ReviserPrompt = `
You are a professional editor working for a highly-trustworthy publication.
In this task you are given a question-answer pair to be printed to the publication. The publication reviewer has double-checked the answer text and provided the findings.
Your task is to minimally revise the answer text to make it accurate, while maintaining the overall structure, style, and length similar to the original.

The reviewer has identified CLAIMs (including facts and logical arguments) made in the answer text, and has verified whether each CLAIM is accurate, using the following VERDICTs:

    * Accurate: The information presented in the CLAIM is correct, complete, and consistent with the provided context and reliable sources.
    * Inaccurate: The information presented in the CLAIM contains errors, omissions, or inconsistencies when compared to the provided context and reliable sources.
    * Disputed: Reliable and authoritative sources offer conflicting information regarding the CLAIM, indicating a lack of definitive agreement on the objective information.
    * Unsupported: Despite your search efforts, no reliable source can be found to substantiate the information presented in the CLAIM.
    * Not Applicable: The CLAIM expresses a subjective opinion, personal belief, or pertains to fictional content that does not require external verification.

Editing guidelines for each type of claim:

  * Accurate claims: There is no need to edit them.
  * Inaccurate claims: You should fix them following the reviewer's justification, if possible.
  * Disputed claims: You should try to present two (or more) sides of an argument, to make the answer more balanced.
  * Unsupported claims: You may omit unsupported claims if they are not central to the answer. Otherwise you may soften the claims or express that they are unsupported.
  * Not applicable claims: There is no need to edit them.

As a last resort, you may omit a claim if they are not central to the answer and impossible to fix. You should also make necessary edits to ensure that the revised answer is self-consistent and fluent. You should not introduce any new claims or make any new statements in the answer text. Your edit should be minimal and maintain overall structure and style unchanged.

Output format:

  * If the answer is accurate, you should output exactly the same answer text as you are given.
  * If the answer is inaccurate, disputed, or unsupported, then you should output your revised answer text.
  * After the answer, output a line of "---END-OF-EDIT---" and stop.

Here are some examples of the task:

=== Example 1 ===

Question: Who was the first president of the US?

Answer: George Washington was the first president of the United States.

Findings:

  * Claim 1: George Washington was the first president of the United States.
      * Verdict: Accurate
      * Justification: Multiple reliable sources confirm that George Washington was the first president of the United States.
  * Overall verdict: Accurate
  * Overall justification: The answer is accurate and completely answers the question.

Your expected response:

George Washington was the first president of the United States.
---END-OF-EDIT---

=== Example 2 ===

Question: What is the shape of the sun?

Answer: The sun is cube-shaped and very hot.

Findings:

  * Claim 1: The sun is cube-shaped.
      * Verdict: Inaccurate
      * Justification: NASA states that the sun is a sphere of hot plasma, so it is not cube-shaped. It is a sphere.
  * Claim 2: The sun is very hot.
      * Verdict: Accurate
      * Justification: Based on my knowledge and the search results, the sun is extremely hot.
  * Overall verdict: Inaccurate
  * Overall justification: The answer states that the sun is cube-shaped, which is incorrect.

Your expected response:

The sun is sphere-shaped and very hot.
---END-OF-EDIT---

Here are the question-answer pair and the reviewer-provided findings:
`

// afterCriticCB is a callback executed after the critic agent's model call.
// It processes the grounding metadata from the LLM response to extract
// references (URLs, titles, text snippets) that were used during fact-checking.
// These references are appended to the response content for transparency.
// Finally, it consolidates all text parts into a single part for cleaner output.
func afterCriticCB(
	_ agent.CallbackContext,
	resp *model.LLMResponse,
	_ error,
) (*model.LLMResponse, error) {
	// Early return if response or grounding metadata is missing
	if resp == nil || resp.Content == nil || resp.Content.Parts == nil ||
		resp.GroundingMetadata == nil {
		return resp, nil
	}

	// Extract references from grounding chunks (web searches, retrieved context)
	references := []string{}
	for _, chunk := range resp.GroundingMetadata.GroundingChunks {
		var title, uri, text string
		// Handle retrieved context (e.g., from RAG systems)
		if chunk.RetrievedContext != nil {
			title = chunk.RetrievedContext.Title
			uri = chunk.RetrievedContext.URI
			text = chunk.RetrievedContext.Text
		} else if chunk.Web != nil {
			// Handle web search results
			title = chunk.Web.Title
			uri = chunk.Web.URI
		}

		// Build reference string from available parts
		reference := ""
		parts := []string{}
		if title != "" {
			parts = append(parts, title)
		}
		if uri != "" {
			parts = append(parts, uri)
		}
		if text != "" {
			parts = append(parts, text)
		}

		if len(parts) > 0 {
			reference = "* " + strings.Join(parts, ": ") + "\n"
		}
		if reference != "" {
			references = append(references, reference)
		}
	}

	// Append references section if any were found
	if len(references) > 0 {
		reference := "\n\nReference:\n\n" + strings.Join(references, "")
		resp.Content.Parts = append(resp.Content.Parts, &genai.Part{Text: reference})
	}

	// Consolidate all text parts into a single part for cleaner output
	textParts := []string{}
	for _, part := range resp.Content.Parts {
		if part.Text != "" {
			textParts = append(textParts, part.Text)
		}
	}
	text := strings.Join(textParts, "")
	if text != "" {
		resp.Content.Parts = []*genai.Part{{Text: text}}
	}
	return resp, nil
}

// afterReviserCB is a callback executed after the reviser agent's model call.
// It strips the EndMark delimiter and any content after it from the response,
// leaving only the clean revised answer text.
func afterReviserCB(
	_ agent.CallbackContext,
	resp *model.LLMResponse,
	_ error,
) (*model.LLMResponse, error) {
	if resp == nil || resp.Content == nil || resp.Content.Parts == nil {
		return resp, nil
	}

	// Find and remove the EndMark delimiter and everything after it
	for idx, part := range resp.Content.Parts {
		if strings.Contains(part.Text, EndMark) {
			// Truncate parts list to exclude parts after the marker
			resp.Content.Parts = resp.Content.Parts[:idx]
			// Remove the marker from the current part
			part.Text = strings.SplitN(part.Text, EndMark, 1)[0]
		}
	}

	return resp, nil
}

// LLMAuditorAgent creates a sequential agent pipeline for auditing LLM-generated answers.
// The pipeline consists of two stages:
//  1. Critic Agent: Analyzes the answer, identifies claims, verifies them using
//     web search and knowledge, and provides verdicts with justifications.
//  2. Reviser Agent: Takes the critic's findings and minimally edits the answer
//     to fix any inaccuracies while maintaining the original style.
//
// This implements a "self-correction" pattern where LLM output is verified and
// improved through a multi-agent workflow.
func LLMAuditorAgent(_ context.Context, model model.LLM) agent.Agent {
	// Create the critic agent that analyzes and verifies claims
	criticAgent, err := llmagent.New(llmagent.Config{
		Model:               model,
		Name:                "critic_agent",
		Instruction:         CriticPrompt,
		AfterModelCallbacks: []llmagent.AfterModelCallback{afterCriticCB},
	})
	if err != nil {
		log.Fatalf("Failed to create critic agent: %v", err)
	}

	// Create the reviser agent that fixes identified issues
	reviserAgent, err := llmagent.New(llmagent.Config{
		Model:               model,
		Name:                "reviser_agent",
		Instruction:         ReviserPrompt,
		AfterModelCallbacks: []llmagent.AfterModelCallback{afterReviserCB},
	})
	if err != nil {
		log.Fatalf("Failed to create reviser agent: %v", err)
	}

	// Create the root sequential agent that orchestrates the pipeline
	// The critic runs first, then the reviser processes the findings
	rootAgent, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:        "llm_auditor",
			Description: "Evaluates LLM-generated answers.",
			SubAgents: []agent.Agent{
				criticAgent,
				reviserAgent,
			},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create root agent: %v", err)
	}

	return rootAgent
}
