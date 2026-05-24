"""JSONL-based result storage."""

from __future__ import annotations

import json
from pathlib import Path

from solum_bench.scoring.aggregator import ModelSummary, QuestionResult


def save_question_result(result: QuestionResult, output_dir: Path):
    """Append a single question result to the model's JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{result.model_name}_results.jsonl"
    with open(file_path, "a") as f:
        f.write(json.dumps(result.to_dict()) + "\n")


def load_question_results(output_dir: Path, model_name: str) -> list[QuestionResult]:
    """Load all question results for a model from JSONL."""
    from solum_bench.providers.base import ModelResponse
    from solum_bench.scoring.base import ScoreResult

    file_path = output_dir / f"{model_name}_results.jsonl"
    if not file_path.exists():
        return []

    results = []
    for line in file_path.read_text().strip().split("\n"):
        if not line:
            continue
        data = json.loads(line)

        response = ModelResponse(
            text=data.get("response_text", ""),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            latency_ms=data.get("latency_ms", 0),
            finish_reason="loaded",
        )

        scores = []
        for dim, score_data in data.get("scores", {}).items():
            scores.append(
                ScoreResult(
                    dimension=dim,
                    score=score_data["score"],
                    explanation=score_data.get("explanation", ""),
                )
            )

        results.append(
            QuestionResult(
                question_id=data["question_id"],
                model_name=data["model"],
                response=response,
                scores=scores,
            )
        )

    return results


def load_model_summary(output_dir: Path, model_name: str) -> ModelSummary:
    """Load a full model summary from stored results."""
    results = load_question_results(output_dir, model_name)
    return ModelSummary(model_name=model_name, question_results=results)


def save_model_summary(summary: ModelSummary, output_dir: Path):
    """Save aggregated model summary as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{summary.model_name}_summary.json"

    data = {
        "model": summary.model_name,
        "total_questions": summary.total_questions,
        "overall_composite": round(summary.overall_composite, 4),
        "safety_failure_rate": round(summary.safety_failure_rate, 4),
        "dimension_breakdown": summary.dimension_breakdown(),
        "total_tokens": summary.total_tokens,
        "avg_latency_ms": round(summary.avg_latency_ms, 1),
    }
    file_path.write_text(json.dumps(data, indent=2))
