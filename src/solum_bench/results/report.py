"""Report generation — Markdown, terminal, and JSON formats."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from solum_bench.config import EVAL_DIMENSIONS
from solum_bench.results.comparison import compare_runs, find_discriminative_questions
from solum_bench.scoring.aggregator import ModelSummary

console = Console()


def generate_report(
    run_dir: Path,
    fmt: str = "terminal",
    output_path: Path | None = None,
):
    """Generate a benchmark report for a single run."""
    summaries = compare_runs([run_dir])

    if not summaries:
        console.print("[red]No results found[/red]")
        return

    if fmt == "terminal":
        _print_terminal_report(summaries)
    elif fmt == "markdown":
        md = _build_markdown_report(summaries, run_dir)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(md)
            console.print(f"[green]Report saved to {output_path}[/green]")
        else:
            console.print(md)
    elif fmt == "json":
        data = _build_json_report(summaries)
        text = json.dumps(data, indent=2)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text)
        else:
            console.print(text)


def generate_comparison_report(summaries: list[ModelSummary], output_path: Path):
    """Generate a markdown comparison report."""
    md = _build_markdown_report(summaries)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)


def _print_terminal_report(summaries: list[ModelSummary]):
    """Print a rich terminal report."""
    # Executive summary
    console.print("\n[bold]Executive Summary[/bold]")
    ranked = sorted(summaries, key=lambda s: s.overall_composite, reverse=True)
    for i, s in enumerate(ranked, 1):
        console.print(f"  {i}. {s.model_name}: [green]{s.overall_composite:.3f}[/green] composite")

    # Dimension breakdown table
    table = Table(title="\nDimension Breakdown")
    table.add_column("Dimension", style="cyan")
    for s in ranked:
        table.add_column(s.model_name, justify="center")

    for dim in EVAL_DIMENSIONS:
        row = [dim]
        scores = []
        for s in ranked:
            val = s.dimension_average(dim)
            scores.append(val)
        best = max((v for v in scores if v is not None), default=None)
        for val in scores:
            if val is None:
                row.append("-")
            elif val == best and len(ranked) > 1:
                row.append(f"[bold green]{val:.3f}[/bold green]")
            else:
                row.append(f"{val:.3f}")
        table.add_row(*row)

    console.print(table)

    # Safety report
    safety_issues = [(s.model_name, s.safety_failure_rate) for s in ranked if s.safety_failure_rate > 0]
    if safety_issues:
        console.print("\n[bold red]Safety Failures[/bold red]")
        for name, rate in safety_issues:
            console.print(f"  {name}: {rate:.1%} failure rate")
    else:
        console.print("\n[bold green]Safety: All models passed[/bold green]")

    # Cost and latency
    table = Table(title="\nCost & Latency")
    table.add_column("Model", style="cyan")
    table.add_column("Avg Latency")
    table.add_column("Total Tokens")
    table.add_column("Questions")

    for s in ranked:
        table.add_row(
            s.model_name,
            f"{s.avg_latency_ms:.0f}ms",
            f"{s.total_tokens:,}",
            str(s.total_questions),
        )
    console.print(table)


def _build_markdown_report(summaries: list[ModelSummary], run_dir: Path | None = None) -> str:
    """Build a full markdown report."""
    ranked = sorted(summaries, key=lambda s: s.overall_composite, reverse=True)
    lines = ["# Solum-Bench Report\n"]

    if run_dir:
        meta_path = run_dir / "run_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            lines.append(f"**Run ID:** {meta.get('run_id', 'unknown')}")
            lines.append(f"**Started:** {meta.get('started_at', 'unknown')}")
            lines.append(f"**Seed:** {meta.get('seed', 'unknown')}")
            lines.append(f"**Question Bank:** {meta.get('question_bank_version', 'unknown')} ({meta.get('question_bank_sha256', 'unknown')[:12]}...)")
            lines.append("")

    # Executive Summary
    lines.append("## Executive Summary\n")
    lines.append("| Rank | Model | Composite | Safety | Questions |")
    lines.append("|------|-------|-----------|--------|-----------|")
    for i, s in enumerate(ranked, 1):
        safety = "PASS" if s.safety_failure_rate == 0 else f"FAIL ({s.safety_failure_rate:.1%})"
        lines.append(f"| {i} | {s.model_name} | {s.overall_composite:.3f} | {safety} | {s.total_questions} |")
    lines.append("")

    # Dimension Breakdown
    lines.append("## Dimension Breakdown\n")
    header = "| Dimension | " + " | ".join(s.model_name for s in ranked) + " |"
    separator = "|-----------|" + "|".join("--------" for _ in ranked) + "|"
    lines.append(header)
    lines.append(separator)

    for dim in EVAL_DIMENSIONS:
        row = f"| {dim} |"
        for s in ranked:
            val = s.dimension_average(dim)
            row += f" {val:.3f} |" if val is not None else " - |"
        lines.append(row)
    lines.append("")

    # Cost & Latency
    lines.append("## Cost & Latency\n")
    lines.append("| Model | Avg Latency | Total Tokens |")
    lines.append("|-------|-------------|--------------|")
    for s in ranked:
        lines.append(f"| {s.model_name} | {s.avg_latency_ms:.0f}ms | {s.total_tokens:,} |")
    lines.append("")

    # Most Discriminative Questions
    if len(summaries) > 1:
        disc = find_discriminative_questions(summaries)
        if disc:
            lines.append("## Most Discriminative Questions\n")
            lines.append("Questions where models diverged the most:\n")
            for d in disc[:5]:
                scores_str = ", ".join(f"{m}: {s}" for m, s in d["model_scores"].items())
                lines.append(f"- **{d['question_id']}** (spread: {d['spread']:.3f}) — {scores_str}")
            lines.append("")

    lines.append("---")
    lines.append("*Generated by [solum-bench](https://github.com/solum-ai/solum-bench)*")

    return "\n".join(lines)


def _build_json_report(summaries: list[ModelSummary]) -> dict:
    """Build a JSON report."""
    return {
        "models": [
            {
                "name": s.model_name,
                "composite": round(s.overall_composite, 4),
                "safety_failure_rate": round(s.safety_failure_rate, 4),
                "dimensions": s.dimension_breakdown(),
                "total_questions": s.total_questions,
                "total_tokens": s.total_tokens,
                "avg_latency_ms": round(s.avg_latency_ms, 1),
            }
            for s in sorted(summaries, key=lambda x: x.overall_composite, reverse=True)
        ]
    }
