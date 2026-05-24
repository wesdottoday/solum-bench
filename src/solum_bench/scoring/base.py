"""Base protocol and data types for scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from solum_bench.providers.base import ModelResponse
from solum_bench.questions.schema import Question


@dataclass
class ScoreResult:
    """Result from scoring a single dimension of a response."""

    dimension: str
    score: float  # 0.0 to 1.0 normalized
    max_score: float = 1.0
    details: dict = field(default_factory=dict)
    explanation: str = ""

    @property
    def normalized(self) -> float:
        return self.score / self.max_score if self.max_score > 0 else 0.0


class Scorer(Protocol):
    """Protocol for scoring a model response against a question."""

    def score(self, question: Question, response: ModelResponse) -> list[ScoreResult]: ...
