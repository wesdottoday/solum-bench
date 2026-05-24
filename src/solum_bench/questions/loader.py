"""Load and filter questions from the YAML question bank."""

from __future__ import annotations

import random
from pathlib import Path

import yaml

from solum_bench.questions.schema import Question


def load_questions_from_file(path: Path) -> list[Question]:
    """Load questions from a single YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    if not data or "questions" not in data:
        return []

    file_meta = data.get("metadata", {})
    questions = []
    for q_data in data["questions"]:
        # Inherit category/subcategory from file metadata if not set per-question
        if "category" not in q_data and "category" in file_meta:
            q_data["category"] = file_meta["category"]
        if "subcategory" not in q_data and "subcategory" in file_meta:
            q_data["subcategory"] = file_meta["subcategory"]
        questions.append(Question.model_validate(q_data))
    return questions


def load_question_bank(bank_dir: Path) -> list[Question]:
    """Load all questions from a question bank directory."""
    questions = []
    for yaml_file in sorted(bank_dir.glob("*.yaml")):
        if yaml_file.name == "manifest.yaml":
            continue
        questions.extend(load_questions_from_file(yaml_file))
    return questions


def filter_questions(
    questions: list[Question],
    *,
    categories: list[str] | None = None,
    subcategories: list[str] | None = None,
    question_types: list[str] | None = None,
    difficulties: list[str] | None = None,
    regions: list[str] | None = None,
    tags: list[str] | None = None,
) -> list[Question]:
    """Filter questions by various criteria. All filters are AND-combined."""
    filtered = questions

    if categories:
        filtered = [q for q in filtered if q.category in categories]
    if subcategories:
        filtered = [q for q in filtered if q.subcategory in subcategories]
    if question_types:
        filtered = [q for q in filtered if q.question_type in question_types]
    if difficulties:
        filtered = [q for q in filtered if q.difficulty in difficulties]
    if regions:
        filtered = [q for q in filtered if any(r in q.regions for r in regions) or "global" in q.regions]
    if tags:
        tag_set = set(tags)
        filtered = [q for q in filtered if tag_set & set(q.tags)]

    return filtered


def sample_questions(
    questions: list[Question],
    max_questions: int | None = None,
    seed: int = 42,
) -> list[Question]:
    """Sample a subset of questions with a fixed seed for reproducibility."""
    if max_questions is None or max_questions >= len(questions):
        return questions

    rng = random.Random(seed)
    return rng.sample(questions, max_questions)
