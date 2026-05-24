"""Exact match and keyword-based scoring for factual questions."""

from __future__ import annotations

import re

from solum_bench.providers.base import ModelResponse
from solum_bench.questions.schema import Question
from solum_bench.scoring.base import ScoreResult


class ExactMatchScorer:
    """Score by exact match against ground truth. Works for MCQ and numeric answers."""

    def score(self, question: Question, response: ModelResponse) -> list[ScoreResult]:
        config = question.scoring_config
        if not config.ground_truth:
            return [ScoreResult(dimension="factual_accuracy", score=0.0, explanation="No ground truth defined")]

        ground_truth = config.ground_truth.strip().upper()
        response_text = response.text.strip()

        # For MCQ: extract the letter answer
        if question.question_type == "mcq":
            match = _extract_mcq_answer(response_text)
            is_correct = match == ground_truth if match else False
        else:
            # For numeric/text answers: check containment or close match
            tolerance = config.tolerance
            if tolerance is not None:
                is_correct = _numeric_match(response_text, config.ground_truth, tolerance)
            else:
                is_correct = ground_truth.lower() in response_text.lower()

        return [
            ScoreResult(
                dimension="factual_accuracy",
                score=1.0 if is_correct else 0.0,
                details={"ground_truth": config.ground_truth, "extracted_answer": match if question.question_type == "mcq" else response_text[:100]},
                explanation=f"{'Correct' if is_correct else 'Incorrect'}: expected '{config.ground_truth}'",
            )
        ]


class KeywordRubricScorer:
    """Score by required keyword presence and rubric criteria checklist."""

    def score(self, question: Question, response: ModelResponse) -> list[ScoreResult]:
        config = question.scoring_config
        results = []
        text_lower = response.text.lower()

        # Keyword presence
        if config.required_keywords:
            found = [kw for kw in config.required_keywords if kw.lower() in text_lower]
            keyword_score = len(found) / len(config.required_keywords)
            results.append(
                ScoreResult(
                    dimension="completeness",
                    score=keyword_score,
                    details={"found": found, "missing": [kw for kw in config.required_keywords if kw.lower() not in text_lower]},
                    explanation=f"Keywords: {len(found)}/{len(config.required_keywords)} present",
                )
            )

        # Rubric criteria — heuristic: check if criterion description keywords appear in response
        if config.rubric_criteria:
            total_points = sum(c.max_points for c in config.rubric_criteria)
            earned_points = 0
            criteria_details = []

            for criterion in config.rubric_criteria:
                # Extract key concepts from the criterion description
                key_terms = _extract_key_terms(criterion.description)
                matches = sum(1 for t in key_terms if t.lower() in text_lower)
                if key_terms:
                    coverage = min(matches / max(len(key_terms) // 2, 1), 1.0)
                else:
                    coverage = 0.0
                points = round(coverage * criterion.max_points)
                earned_points += points
                criteria_details.append({"name": criterion.name, "points": points, "max": criterion.max_points})

            rubric_score = earned_points / total_points if total_points > 0 else 0.0
            results.append(
                ScoreResult(
                    dimension="factual_accuracy",
                    score=rubric_score,
                    details={"criteria": criteria_details, "total_earned": earned_points, "total_possible": total_points},
                    explanation=f"Rubric: {earned_points}/{total_points} points",
                )
            )

        return results


def _extract_mcq_answer(text: str) -> str | None:
    """Extract the MCQ answer letter from a response."""
    # Look for common patterns: "A", "A)", "Answer: A", "The answer is A"
    patterns = [
        r"(?:the\s+)?answer\s+is\s+([A-D])",
        r"^([A-D])\b",
        r"\b([A-D])\)",
        r"\*\*([A-D])\*\*",
        r"(?:correct\s+answer|option)\s*(?:is\s+)?([A-D])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    # Last resort: find any standalone letter
    match = re.search(r"\b([A-D])\b", text)
    return match.group(1).upper() if match else None


def _numeric_match(text: str, ground_truth: str, tolerance: float) -> bool:
    """Check if a numeric value in text matches ground truth within tolerance."""
    try:
        expected = float(ground_truth)
    except ValueError:
        return False

    numbers = re.findall(r"[\-]?\d+\.?\d*", text)
    for n in numbers:
        try:
            if abs(float(n) - expected) <= tolerance:
                return True
        except ValueError:
            continue
    return False


def _extract_key_terms(description: str) -> list[str]:
    """Extract meaningful terms from a rubric criterion description."""
    # Remove common stop words and short words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                  "has", "have", "had", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "can", "shall", "that", "this", "these",
                  "those", "it", "its", "of", "in", "to", "for", "with", "on", "at",
                  "by", "from", "as", "or", "and", "not", "no", "but", "if", "so",
                  "than", "too", "very", "just", "about", "also", "into", "over"}
    words = re.findall(r"[a-z]+", description.lower())
    return [w for w in words if w not in stop_words and len(w) > 2]
