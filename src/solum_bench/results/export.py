"""Export results for external LLM judging (e.g., via Claude Code)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from solum_bench.config import EVAL_DIMENSIONS
from solum_bench.questions.loader import load_question_bank
from solum_bench.questions.schema import Question
from solum_bench.results.storage import load_question_results
from solum_bench.scoring.base import ScoreResult
from solum_bench.scoring.aggregator import QuestionResult


def export_for_judging(
    run_dir: Path,
    output_path: Path,
    question_bank_dir: Path | None = None,
    fmt: str = "csv",
):
    """Export un-judged responses to CSV or JSONL for external evaluation.

    The exported file contains the question, the model's response, and a rubric
    so that a human or LLM (e.g., Claude Code) can score each response.
    """
    # Load questions for rubric context
    questions_by_id: dict[str, Question] = {}
    if question_bank_dir:
        for q in load_question_bank(question_bank_dir):
            questions_by_id[q.id] = q

    # Discover all model results
    rows = []
    for results_file in sorted(run_dir.glob("*_results.jsonl")):
        model_name = results_file.stem.replace("_results", "")
        results = load_question_results(run_dir, model_name)

        for result in results:
            question = questions_by_id.get(result.question_id)
            row = _build_export_row(result, question)
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        _write_csv(rows, output_path)
    else:
        _write_jsonl(rows, output_path)

    return len(rows)


def import_scores(
    scores_path: Path,
    run_dir: Path,
    fmt: str = "csv",
):
    """Import externally-produced judge scores and merge into results.

    Expected CSV columns: question_id, model, dimension, score (0-10), explanation
    Expected JSONL: {"question_id": ..., "model": ..., "scores": {"dim": {"score": N, "explanation": "..."}}}
    """
    if fmt == "csv":
        imported = _read_csv_scores(scores_path)
    else:
        imported = _read_jsonl_scores(scores_path)

    # Write merged scores as a new file
    output_file = run_dir / "judge_scores.jsonl"
    with open(output_file, "w") as f:
        for entry in imported:
            f.write(json.dumps(entry) + "\n")

    return len(imported)


def _build_export_row(result: QuestionResult, question: Question | None) -> dict:
    """Build a single row for export."""
    row = {
        "question_id": result.question_id,
        "model": result.model_name,
        "question_text": "",
        "system_prompt": "",
        "response_text": result.response.text,
        "category": "",
        "subcategory": "",
        "scoring_method": "",
        "rubric": "",
        "dimensions_to_score": "",
    }

    if question:
        row["question_text"] = question.user_prompt
        row["system_prompt"] = question.system_prompt
        row["category"] = question.category
        row["subcategory"] = question.subcategory
        row["scoring_method"] = question.scoring_method

        # Build rubric guidance for the judge
        rubric_parts = []
        if question.scoring_config.rubric:
            rubric_parts.append(question.scoring_config.rubric)
        if question.scoring_config.rubric_criteria:
            for c in question.scoring_config.rubric_criteria:
                rubric_parts.append(f"- {c.name}: {c.description} (max {c.max_points} pts)")
        row["rubric"] = "\n".join(rubric_parts) if rubric_parts else "Score on factual accuracy, reasoning depth, specificity, completeness, and practical actionability."

        dims = question.scoring_config.dimensions or [
            "factual_accuracy", "reasoning_depth", "specificity",
            "completeness", "practical_actionability",
        ]
        row["dimensions_to_score"] = ",".join(dims)

    return row


def _write_csv(rows: list[dict], output_path: Path):
    """Write export rows to CSV."""
    if not rows:
        return

    fieldnames = [
        "question_id", "model", "category", "subcategory",
        "question_text", "system_prompt", "response_text",
        "scoring_method", "rubric", "dimensions_to_score",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(rows: list[dict], output_path: Path):
    """Write export rows to JSONL."""
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_csv_scores(path: Path) -> list[dict]:
    """Read scores from CSV. Expected: question_id, model, dimension, score (0-10), explanation."""
    entries: dict[str, dict] = {}

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['question_id']}::{row['model']}"
            if key not in entries:
                entries[key] = {
                    "question_id": row["question_id"],
                    "model": row["model"],
                    "scores": {},
                }
            entries[key]["scores"][row["dimension"]] = {
                "score": float(row["score"]) / 10.0,  # Normalize 0-10 to 0-1
                "explanation": row.get("explanation", ""),
            }

    return list(entries.values())


def _read_jsonl_scores(path: Path) -> list[dict]:
    """Read scores from JSONL."""
    entries = []
    for line in path.read_text().strip().split("\n"):
        if line:
            data = json.loads(line)
            # Normalize scores from 0-10 to 0-1 if needed
            for dim, score_data in data.get("scores", {}).items():
                if isinstance(score_data, dict) and score_data.get("score", 0) > 1.0:
                    score_data["score"] = score_data["score"] / 10.0
            entries.append(data)
    return entries
