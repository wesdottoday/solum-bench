"""Rubric-based checklist scoring for open-ended responses."""

from __future__ import annotations

import re

from solum_bench.providers.base import ModelResponse
from solum_bench.questions.schema import Question
from solum_bench.scoring.base import ScoreResult


class RubricScorer:
    """Score using rubric criteria checklists. More structured than keyword matching."""

    # Aspect keyword sets for completeness checking
    ASPECT_KEYWORDS = {
        "soil": {"soil", "clay", "loam", "sand", "silt", "ph", "organic matter", "humus", "texture", "cec",
                 "compaction", "aggregate", "bulk density", "horizon"},
        "climate": {"temperature", "precipitation", "rainfall", "frost", "climate", "season", "humidity",
                    "drought", "moisture", "growing season", "hardiness zone"},
        "species": {"species", "plant", "crop", "tree", "grass", "legume", "cover crop", "pollinator",
                    "bee", "bird", "earthworm", "fungi", "mycorrhiz"},
        "recommendation": {"recommend", "suggest", "consider", "plant", "grow", "apply", "manage",
                          "practice", "implement", "transition"},
        "reasoning": {"because", "since", "therefore", "due to", "as a result", "this indicates",
                     "suggesting that", "which means", "given that", "combined with"},
    }

    def score(self, question: Question, response: ModelResponse) -> list[ScoreResult]:
        results = []
        text_lower = response.text.lower()

        # Completeness: how many aspects are covered
        aspects_found = {}
        for aspect, keywords in self.ASPECT_KEYWORDS.items():
            aspects_found[aspect] = any(kw in text_lower for kw in keywords)

        covered = sum(1 for v in aspects_found.values() if v)
        completeness = covered / len(self.ASPECT_KEYWORDS)
        results.append(
            ScoreResult(
                dimension="completeness",
                score=completeness,
                details={"aspects": aspects_found},
                explanation=f"Covers {covered}/{len(self.ASPECT_KEYWORDS)} aspects",
            )
        )

        # Reasoning depth: causal markers
        causal_markers = [
            "because", "since", "therefore", "which means", "as a result",
            "this indicates", "suggesting that", "due to", "combined with",
            "this makes", "ideal for", "well-suited", "poorly suited",
            "limiting factor", "the combination of", "given that",
        ]
        hits = sum(1 for marker in causal_markers if marker in text_lower)
        reasoning = min(hits / 5.0, 1.0)
        results.append(
            ScoreResult(
                dimension="reasoning_depth",
                score=reasoning,
                details={"causal_markers_found": hits},
                explanation=f"{hits} causal markers found (5+ = full score)",
            )
        )

        # Specificity: numeric values used
        numbers_in_response = set(re.findall(r"\d+\.?\d*", response.text))
        specificity = min(len(numbers_in_response) / 5.0, 1.0)
        results.append(
            ScoreResult(
                dimension="specificity",
                score=specificity,
                details={"numeric_values_count": len(numbers_in_response)},
                explanation=f"{len(numbers_in_response)} numeric values (5+ = full score)",
            )
        )

        return results
