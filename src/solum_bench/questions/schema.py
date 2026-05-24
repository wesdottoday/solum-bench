"""Question schema with Pydantic validation."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from solum_bench.config import (
    SUPPORTED_CATEGORIES,
    SUPPORTED_DIFFICULTIES,
    SUPPORTED_QUESTION_TYPES,
    SUPPORTED_SCORING_METHODS,
)


class Turn(BaseModel):
    """A single turn in a conversation."""

    role: str = Field(pattern=r"^(user|assistant|system)$")
    content: str | None = None  # None for assistant turns the model must generate


class RubricCriterion(BaseModel):
    """A single criterion in a rubric-based scoring config."""

    name: str
    description: str
    max_points: int = Field(ge=1, le=10)


class ScoringConfig(BaseModel):
    """Flexible scoring configuration. Fields used depend on scoring_method."""

    # exact_match
    ground_truth: str | None = None
    tolerance: float | None = None  # For numeric exact_match

    # keyword_rubric
    required_keywords: list[str] | None = None
    rubric_criteria: list[RubricCriterion] | None = None

    # numeric_grounding
    expected_values: dict[str, float] | None = None

    # safety_check
    banned_terms: list[str] | None = None
    must_include_warnings: list[str] | None = None
    must_not_recommend: list[str] | None = None

    # llm_judge
    rubric: str | None = None
    dimensions: list[str] | None = None

    # composite
    methods: list[str] | None = None
    weights: list[float] | None = None

    model_config = {"extra": "allow"}


class Question(BaseModel):
    """A single benchmark question with metadata and scoring specification."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    version: str = Field(default="v1")
    category: str
    subcategory: str
    question_type: str
    difficulty: str
    regions: list[str] = Field(default_factory=lambda: ["global"])

    system_prompt: str = ""
    turns: list[Turn] = Field(min_length=1)

    # MCQ fields
    choices: list[str] | None = None
    correct_answer: str | None = None

    # Data-grounded questions
    context_data: str | None = None

    # Scoring
    scoring_method: str
    scoring_config: ScoringConfig = Field(default_factory=ScoringConfig)

    # Metadata
    source_notes: str = ""
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_question(self) -> Question:
        if self.category not in SUPPORTED_CATEGORIES:
            raise ValueError(f"Unknown category '{self.category}'. Must be one of: {SUPPORTED_CATEGORIES}")
        if self.question_type not in SUPPORTED_QUESTION_TYPES:
            raise ValueError(f"Unknown question_type '{self.question_type}'. Must be one of: {SUPPORTED_QUESTION_TYPES}")
        if self.difficulty not in SUPPORTED_DIFFICULTIES:
            raise ValueError(f"Unknown difficulty '{self.difficulty}'. Must be one of: {SUPPORTED_DIFFICULTIES}")
        if self.scoring_method not in SUPPORTED_SCORING_METHODS:
            raise ValueError(
                f"Unknown scoring_method '{self.scoring_method}'. Must be one of: {SUPPORTED_SCORING_METHODS}"
            )

        # MCQ must have choices and correct_answer
        if self.question_type == "mcq":
            if not self.choices:
                raise ValueError("MCQ questions must have 'choices'")
            if not self.correct_answer:
                raise ValueError("MCQ questions must have 'correct_answer'")

        # exact_match needs ground_truth
        if self.scoring_method == "exact_match" and not self.scoring_config.ground_truth:
            raise ValueError("exact_match scoring requires 'ground_truth' in scoring_config")

        # composite needs methods
        if self.scoring_method == "composite":
            if not self.scoring_config.methods:
                raise ValueError("composite scoring requires 'methods' in scoring_config")

        return self

    @property
    def user_prompt(self) -> str:
        """Extract the first user turn's content, with context_data prepended if present."""
        for turn in self.turns:
            if turn.role == "user" and turn.content:
                if self.context_data:
                    return f"{self.context_data}\n\n{turn.content}"
                return turn.content
        return ""

    @property
    def is_multi_turn(self) -> bool:
        user_turns = [t for t in self.turns if t.role == "user"]
        return len(user_turns) > 1


class QuestionBank(BaseModel):
    """Metadata for a collection of questions loaded from a YAML file."""

    metadata: dict = Field(default_factory=dict)
    questions: list[Question]
