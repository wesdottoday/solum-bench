"""Cross-model comparison logic."""

from __future__ import annotations

import json
from pathlib import Path

from solum_bench.scoring.aggregator import ModelSummary
from solum_bench.results.storage import load_model_summary


def compare_runs(run_dirs: list[Path]) -> list[ModelSummary]:
    """Load model summaries from multiple run directories for comparison."""
    summaries = []

    for run_dir in run_dirs:
        # Find all *_results.jsonl files to discover model names
        for results_file in sorted(run_dir.glob("*_results.jsonl")):
            model_name = results_file.stem.replace("_results", "")
            summary = load_model_summary(run_dir, model_name)
            if summary.total_questions > 0:
                summaries.append(summary)

    return summaries


def find_discriminative_questions(summaries: list[ModelSummary], top_n: int = 10) -> list[dict]:
    """Find questions where models disagree most (highest score variance)."""
    if len(summaries) < 2:
        return []

    # Collect scores by question_id across all models
    question_scores: dict[str, list[tuple[str, float]]] = {}
    for summary in summaries:
        for result in summary.question_results:
            qid = result.question_id
            if qid not in question_scores:
                question_scores[qid] = []
            question_scores[qid].append((summary.model_name, result.composite_score))

    # Calculate variance for each question
    discriminative = []
    for qid, model_scores in question_scores.items():
        if len(model_scores) < 2:
            continue
        scores = [s for _, s in model_scores]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        discriminative.append({
            "question_id": qid,
            "variance": variance,
            "spread": max(scores) - min(scores),
            "model_scores": {m: round(s, 3) for m, s in model_scores},
        })

    discriminative.sort(key=lambda x: x["variance"], reverse=True)
    return discriminative[:top_n]
