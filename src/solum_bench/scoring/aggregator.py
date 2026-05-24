"""Score aggregation across dimensions, questions, and models."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from solum_bench.config import EVAL_DIMENSIONS
from solum_bench.providers.base import ModelResponse
from solum_bench.questions.schema import Question
from solum_bench.scoring.base import ScoreResult, Scorer
from solum_bench.scoring.factual import ExactMatchScorer, KeywordRubricScorer
from solum_bench.scoring.numeric_grounding import NumericGroundingScorer
from solum_bench.scoring.rubric import RubricScorer
from solum_bench.scoring.safety import SafetyScorer


@dataclass
class QuestionResult:
    """Complete scoring result for a single question."""

    question_id: str
    model_name: str
    response: ModelResponse
    scores: list[ScoreResult] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.normalized for s in self.scores) / len(self.scores)

    @property
    def safety_passed(self) -> bool:
        for s in self.scores:
            if s.dimension == "safety":
                return s.score > 0.0
        return True  # No safety score = pass

    def score_by_dimension(self, dimension: str) -> float | None:
        for s in self.scores:
            if s.dimension == dimension:
                return s.normalized
        return None

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "model": self.model_name,
            "composite": round(self.composite_score, 4),
            "safety_passed": self.safety_passed,
            "scores": {s.dimension: {"score": round(s.normalized, 4), "explanation": s.explanation} for s in self.scores},
            "response_text": self.response.text,
            "input_tokens": self.response.input_tokens,
            "output_tokens": self.response.output_tokens,
            "latency_ms": round(self.response.latency_ms, 1),
        }


@dataclass
class ModelSummary:
    """Aggregated scores for a model across all questions."""

    model_name: str
    question_results: list[QuestionResult] = field(default_factory=list)

    @property
    def total_questions(self) -> int:
        return len(self.question_results)

    @property
    def overall_composite(self) -> float:
        if not self.question_results:
            return 0.0
        return sum(r.composite_score for r in self.question_results) / len(self.question_results)

    @property
    def safety_failure_rate(self) -> float:
        safety_questions = [r for r in self.question_results if r.score_by_dimension("safety") is not None]
        if not safety_questions:
            return 0.0
        failures = sum(1 for r in safety_questions if not r.safety_passed)
        return failures / len(safety_questions)

    def dimension_average(self, dimension: str) -> float | None:
        scores = [r.score_by_dimension(dimension) for r in self.question_results]
        valid = [s for s in scores if s is not None]
        return sum(valid) / len(valid) if valid else None

    def category_average(self, category: str) -> float | None:
        results = [r for r in self.question_results if _get_category(r.question_id) == category]
        if not results:
            return None
        return sum(r.composite_score for r in results) / len(results)

    def dimension_breakdown(self) -> dict[str, float]:
        breakdown = {}
        for dim in EVAL_DIMENSIONS:
            avg = self.dimension_average(dim)
            if avg is not None:
                breakdown[dim] = round(avg, 4)
        return breakdown

    @property
    def total_tokens(self) -> int:
        return sum(r.response.total_tokens for r in self.question_results)

    @property
    def total_latency_ms(self) -> float:
        return sum(r.response.latency_ms for r in self.question_results)

    @property
    def avg_latency_ms(self) -> float:
        if not self.question_results:
            return 0.0
        return self.total_latency_ms / len(self.question_results)


class ScoringPipeline:
    """Orchestrate scoring across multiple methods for a question."""

    def __init__(self, llm_judge_scorer=None):
        self.exact_match = ExactMatchScorer()
        self.keyword_rubric = KeywordRubricScorer()
        self.rubric = RubricScorer()
        self.numeric_grounding = NumericGroundingScorer()
        self.safety = SafetyScorer()
        self.llm_judge = llm_judge_scorer

    def score_question(self, question: Question, response: ModelResponse) -> list[ScoreResult]:
        """Score a response using the method specified in the question."""
        method = question.scoring_method

        if method == "exact_match":
            return self.exact_match.score(question, response)

        elif method == "keyword_rubric":
            return self.keyword_rubric.score(question, response)

        elif method == "numeric_grounding":
            results = self.numeric_grounding.score(question, response)
            if self.llm_judge:
                results.extend(self.llm_judge.score(question, response))
            return results

        elif method == "safety_check":
            return self.safety.score(question, response)

        elif method == "llm_judge":
            if self.llm_judge:
                return self.llm_judge.score(question, response)
            # Fallback to rubric heuristics if no judge available
            return self.rubric.score(question, response)

        elif method == "composite":
            return self._score_composite(question, response)

        else:
            return self.rubric.score(question, response)

    def _score_composite(self, question: Question, response: ModelResponse) -> list[ScoreResult]:
        """Run multiple scoring methods and combine results."""
        config = question.scoring_config
        methods = config.methods or ["keyword_rubric", "llm_judge"]

        all_results = []
        for method in methods:
            if method == "exact_match":
                all_results.extend(self.exact_match.score(question, response))
            elif method == "keyword_rubric":
                all_results.extend(self.keyword_rubric.score(question, response))
            elif method == "numeric_grounding":
                all_results.extend(self.numeric_grounding.score(question, response))
            elif method == "safety_check":
                all_results.extend(self.safety.score(question, response))
            elif method == "llm_judge" and self.llm_judge:
                all_results.extend(self.llm_judge.score(question, response))
            elif method == "rubric":
                all_results.extend(self.rubric.score(question, response))

        return all_results


def _get_category(question_id: str) -> str:
    """Extract category from question ID (e.g., 'factual_soil_0001' -> 'factual')."""
    parts = question_id.split("_")
    return parts[0] if parts else "unknown"
