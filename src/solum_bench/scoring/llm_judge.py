"""LLM-as-judge scoring — uses a frontier model to evaluate responses."""

from __future__ import annotations

import json
import re

from solum_bench.providers.base import ModelProvider, ModelResponse
from solum_bench.questions.schema import Question
from solum_bench.scoring.base import ScoreResult

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a regenerative agriculture AI benchmark.
Your task is to score an AI model's response to a question about soil science, farming practices,
ecology, or related topics.

You must be rigorous and fair. Score based on the rubric provided, not on style or length.
A shorter, accurate response should score higher than a longer, vague one.

Return your evaluation as JSON with this exact structure:
{
  "scores": {
    "dimension_name": {
      "score": <integer 0-10>,
      "explanation": "<brief justification>"
    }
  }
}

Only use the dimensions specified in the rubric. Score each from 0 to 10."""

DEFAULT_DIMENSIONS = [
    "factual_accuracy",
    "reasoning_depth",
    "specificity",
    "completeness",
    "practical_actionability",
]


class LLMJudgeScorer:
    """Use a frontier LLM to evaluate response quality against a rubric."""

    def __init__(self, judge_provider: ModelProvider):
        self.judge = judge_provider

    def score(self, question: Question, response: ModelResponse) -> list[ScoreResult]:
        config = question.scoring_config
        dimensions = config.dimensions or DEFAULT_DIMENSIONS
        rubric = config.rubric or _build_default_rubric(question, dimensions)

        eval_prompt = _build_eval_prompt(question, response.text, rubric, dimensions)

        judge_response = self.judge.generate(
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": eval_prompt}],
            max_tokens=2048,
            temperature=0.0,
        )

        return _parse_judge_response(judge_response.text, dimensions)

    def score_batch(self, questions_and_responses: list[tuple[Question, ModelResponse]]) -> list[list[ScoreResult]]:
        """Score multiple responses. Currently sequential; async batch in Phase 8."""
        return [self.score(q, r) for q, r in questions_and_responses]


def _build_eval_prompt(question: Question, response_text: str, rubric: str, dimensions: list[str]) -> str:
    """Build the evaluation prompt for the judge."""
    # Reconstruct the original question
    question_text = question.user_prompt
    if question.system_prompt:
        question_text = f"[System: {question.system_prompt}]\n\n{question_text}"

    return f"""## Question Asked
{question_text}

## Response Being Evaluated
{response_text}

## Evaluation Rubric
{rubric}

## Dimensions to Score
Score each of the following dimensions from 0 to 10:
{chr(10).join(f'- **{d}**' for d in dimensions)}

Return your scores as JSON."""


def _build_default_rubric(question: Question, dimensions: list[str]) -> str:
    """Build a default rubric based on question type and dimensions."""
    rubric_parts = ["Score the response on these criteria:"]

    dim_descriptions = {
        "factual_accuracy": "Are the facts correct? No errors in soil science, ecology, or agricultural knowledge.",
        "reasoning_depth": "Does the response explain WHY, not just WHAT? Look for causal reasoning, mechanistic explanations, and multi-step logic.",
        "specificity": "Does it use specific numbers, species names, and localized data? Or is it vague and generic?",
        "completeness": "Does it address all aspects of the question? Soil, climate, species, practices, and their interactions?",
        "practical_actionability": "Could a farmer actually implement these recommendations? Are timing, rates, and methods provided?",
        "regional_adaptation": "Is the advice appropriate for the specific region? Does it account for local climate, soils, and species?",
        "systems_thinking": "Does it consider the whole ecosystem — soil biology, water cycling, biodiversity, and management interactions?",
        "coherence": "Is the response well-organized, internally consistent, and logically structured?",
        "data_grounding": "Does it reference the actual data provided, rather than making up numbers or ignoring context?",
        "context_retention": "In multi-turn context: does it build on prior turns without contradiction or repetition?",
    }

    for dim in dimensions:
        desc = dim_descriptions.get(dim, f"Evaluate the quality of '{dim}'.")
        rubric_parts.append(f"- **{dim}**: {desc}")

    return "\n".join(rubric_parts)


def _parse_judge_response(text: str, dimensions: list[str]) -> list[ScoreResult]:
    """Parse the judge's JSON response into ScoreResults."""
    # Extract JSON from response (may be wrapped in markdown code blocks)
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return [
            ScoreResult(
                dimension=dim,
                score=0.0,
                explanation="Failed to parse judge response",
                details={"raw_judge_output": text[:500]},
            )
            for dim in dimensions
        ]

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return [
            ScoreResult(
                dimension=dim,
                score=0.0,
                explanation="Invalid JSON from judge",
                details={"raw_judge_output": text[:500]},
            )
            for dim in dimensions
        ]

    scores_data = parsed.get("scores", parsed)  # Handle both {"scores": {...}} and flat dict
    results = []

    for dim in dimensions:
        if dim in scores_data:
            entry = scores_data[dim]
            if isinstance(entry, dict):
                raw_score = entry.get("score", 0)
                explanation = entry.get("explanation", "")
            else:
                raw_score = entry
                explanation = ""
            # Normalize from 0-10 to 0-1
            results.append(
                ScoreResult(
                    dimension=dim,
                    score=float(raw_score) / 10.0,
                    details={"raw_score": raw_score},
                    explanation=explanation,
                )
            )
        else:
            results.append(
                ScoreResult(
                    dimension=dim,
                    score=0.0,
                    explanation=f"Dimension '{dim}' not found in judge output",
                )
            )

    return results
