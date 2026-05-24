"""Numeric grounding scorer — checks if model uses numbers from provided context data."""

from __future__ import annotations

import re

from solum_bench.providers.base import ModelResponse
from solum_bench.questions.schema import Question
from solum_bench.scoring.base import ScoreResult


class NumericGroundingScorer:
    """Score how well the response grounds itself in numeric data from the context."""

    def score(self, question: Question, response: ModelResponse) -> list[ScoreResult]:
        results = []

        # Data grounding: compare numbers in context vs response
        context_text = question.context_data or ""
        if context_text:
            context_numbers = set(re.findall(r"\d+\.\d+", context_text))
            response_numbers = set(re.findall(r"\d+\.\d+", response.text))

            if context_numbers:
                overlap = context_numbers & response_numbers
                # Cap denominator to avoid penalizing for large context datasets
                denom = min(len(context_numbers), 10)
                grounding_score = min(len(overlap) / denom, 1.0)
            else:
                grounding_score = 0.5  # Neutral if no numbers in context

            results.append(
                ScoreResult(
                    dimension="data_grounding",
                    score=grounding_score,
                    details={
                        "context_numbers": len(context_numbers),
                        "response_numbers": len(response_numbers),
                        "overlap": len(overlap) if context_numbers else 0,
                    },
                    explanation=f"Response cites {len(overlap) if context_numbers else 0} of {len(context_numbers)} context values",
                )
            )

        # Citation fidelity: check expected_values if specified
        config = question.scoring_config
        if config.expected_values:
            matches = 0
            checked = 0
            details = {}
            for key, expected_val in config.expected_values.items():
                checked += 1
                tolerance = config.tolerance or 0.5
                # Search for the expected value in the response
                found = _find_value_in_text(response.text, expected_val, tolerance)
                details[key] = {"expected": expected_val, "found": found}
                if found:
                    matches += 1

            fidelity_score = matches / checked if checked > 0 else 0.0
            results.append(
                ScoreResult(
                    dimension="citation_fidelity",
                    score=fidelity_score,
                    details=details,
                    explanation=f"Matched {matches}/{checked} expected values",
                )
            )

        return results


def _find_value_in_text(text: str, expected: float, tolerance: float) -> bool:
    """Check if a numeric value appears in text within tolerance."""
    numbers = re.findall(r"[\-]?\d+\.?\d*", text)
    for n in numbers:
        try:
            if abs(float(n) - expected) <= tolerance:
                return True
        except ValueError:
            continue
    return False
