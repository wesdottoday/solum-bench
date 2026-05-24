"""CLI entry point for solum-bench."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from solum_bench.config import DEFAULT_QUESTION_BANK, DEFAULT_RESULTS_DIR

console = Console()


@click.group()
@click.version_option()
def main():
    """solum-bench: Comprehensive benchmark for regenerative agriculture LLMs."""
    pass


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path), help="Model config YAML")
@click.option("--questions", "bank_dir", default=None, type=click.Path(exists=True, path_type=Path), help="Question bank directory")
@click.option("--output", "output_dir", default=None, type=click.Path(path_type=Path), help="Output directory for results")
@click.option("--categories", default=None, help="Comma-separated categories to include")
@click.option("--regions", default=None, help="Comma-separated regions to include")
@click.option("--difficulties", default=None, help="Comma-separated difficulties to include")
@click.option("--max-questions", default=None, type=int, help="Max questions to run")
@click.option("--seed", default=42, type=int, help="Random seed for sampling")
@click.option("--resume/--no-resume", default=False, help="Resume from checkpoint")
@click.option("--judge-model", default=None, help="Model name for LLM-as-judge scoring")
@click.option("--skip-llm-judge", is_flag=True, help="Skip LLM-as-judge scoring (fast scorers only)")
def run(config_path, bank_dir, output_dir, categories, regions, difficulties, max_questions, seed, resume, judge_model, skip_llm_judge):
    """Run the benchmark against configured models."""
    from datetime import datetime

    from solum_bench.providers.registry import load_providers_from_config
    from solum_bench.questions.loader import filter_questions, load_question_bank, sample_questions
    from solum_bench.runner.engine import run_benchmark
    from solum_bench.scoring.aggregator import ScoringPipeline

    bank_dir = bank_dir or DEFAULT_QUESTION_BANK
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = DEFAULT_RESULTS_DIR / f"run_{timestamp}"

    # Load questions
    console.print(f"[bold]Loading questions from {bank_dir}[/bold]")
    questions = load_question_bank(bank_dir)
    console.print(f"  Loaded {len(questions)} questions")

    # Apply filters
    cat_list = categories.split(",") if categories else None
    reg_list = regions.split(",") if regions else None
    diff_list = difficulties.split(",") if difficulties else None
    questions = filter_questions(questions, categories=cat_list, regions=reg_list, difficulties=diff_list)
    if cat_list or reg_list or diff_list:
        console.print(f"  After filtering: {len(questions)} questions")

    questions = sample_questions(questions, max_questions, seed)
    if max_questions:
        console.print(f"  Sampled: {len(questions)} questions")

    # Load providers
    console.print(f"\n[bold]Loading models from {config_path}[/bold]")
    providers = load_providers_from_config(config_path)
    console.print(f"  Models: {', '.join(p.name for p in providers)}")

    # Set up scoring pipeline
    llm_judge_scorer = None
    if not skip_llm_judge and judge_model:
        from solum_bench.providers.registry import create_provider
        from solum_bench.scoring.llm_judge import LLMJudgeScorer

        # Try to find the judge in the loaded providers, or create a new one
        judge_provider = None
        for p in providers:
            if p.name == judge_model:
                judge_provider = p
                break
        if judge_provider is None:
            # Assume it's an Anthropic model by default
            judge_provider = create_provider({"provider": "anthropic", "model_id": judge_model, "name": judge_model})
        llm_judge_scorer = LLMJudgeScorer(judge_provider)
        console.print(f"  Judge model: {judge_model}")

    pipeline = ScoringPipeline(llm_judge_scorer=llm_judge_scorer)

    # Run benchmark
    console.print(f"\n[bold green]Starting benchmark run[/bold green]")
    console.print(f"  Output: {output_dir}")

    summaries = run_benchmark(
        providers=providers,
        questions=questions,
        output_dir=output_dir,
        question_bank_dir=bank_dir,
        scoring_pipeline=pipeline,
        seed=seed,
        resume=resume,
        judge_model_name=judge_model,
    )

    # Print comparison table
    if len(summaries) > 1:
        _print_comparison(summaries)

    console.print(f"\n[bold green]Benchmark complete.[/bold green] Results saved to {output_dir}")


@main.command()
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True, path_type=Path), help="Results directory to score")
@click.option("--judge-model", default=None, help="Model name for LLM-as-judge scoring")
@click.option("--skip-llm-judge", is_flag=True, help="Only run fast deterministic scorers")
def score(run_dir, judge_model, skip_llm_judge):
    """Re-score existing results with different judges or rubrics."""
    console.print(f"[bold]Scoring results in {run_dir}[/bold]")
    console.print("[yellow]Score command: implementation in Phase 6[/yellow]")


@main.command()
@click.option("--runs", required=True, multiple=True, type=click.Path(exists=True, path_type=Path), help="Result directories to compare")
@click.option("--output", default=None, type=click.Path(path_type=Path), help="Output report path")
def compare(runs, output):
    """Compare results across multiple benchmark runs."""
    from solum_bench.results.comparison import compare_runs
    from solum_bench.results.report import generate_comparison_report

    summaries = compare_runs([Path(r) for r in runs])

    if output:
        generate_comparison_report(summaries, Path(output))
        console.print(f"[green]Report saved to {output}[/green]")
    else:
        _print_comparison(summaries)


@main.command()
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True, path_type=Path), help="Results directory")
@click.option("--format", "fmt", default="terminal", type=click.Choice(["terminal", "markdown", "json"]))
@click.option("--output", default=None, type=click.Path(path_type=Path), help="Output file path")
def report(run_dir, fmt, output):
    """Generate a benchmark report."""
    from solum_bench.results.report import generate_report

    generate_report(run_dir, fmt=fmt, output_path=Path(output) if output else None)


@main.command()
@click.option("--questions", "bank_dir", default=None, type=click.Path(exists=True, path_type=Path))
def validate(bank_dir):
    """Validate the question bank integrity."""
    from solum_bench.questions.validator import validate_bank

    bank_dir = Path(bank_dir) if bank_dir else DEFAULT_QUESTION_BANK
    console.print(f"[bold]Validating {bank_dir}[/bold]")

    errors = validate_bank(bank_dir)
    if errors:
        for e in errors:
            console.print(f"  [red]ERROR:[/red] {e}")
        raise SystemExit(1)
    else:
        from solum_bench.questions.loader import load_question_bank

        questions = load_question_bank(bank_dir)
        console.print(f"  [green]VALID[/green] — {len(questions)} questions loaded successfully")


@main.command("export-for-judging")
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True, path_type=Path), help="Results directory")
@click.option("--questions", "bank_dir", default=None, type=click.Path(exists=True, path_type=Path), help="Question bank directory")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path), help="Output file path (.csv or .jsonl)")
@click.option("--format", "fmt", default="csv", type=click.Choice(["csv", "jsonl"]))
def export_for_judging(run_dir, bank_dir, output_path, fmt):
    """Export responses for external LLM judging (e.g., via Claude Code).

    Produces a CSV or JSONL file containing each question, the model's response,
    and a scoring rubric. Feed this to Claude Code or another evaluator, then
    import the scores back with 'import-scores'.
    """
    from solum_bench.results.export import export_for_judging as do_export

    bank_dir = Path(bank_dir) if bank_dir else DEFAULT_QUESTION_BANK
    count = do_export(run_dir, Path(output_path), question_bank_dir=bank_dir, fmt=fmt)
    console.print(f"[green]Exported {count} responses to {output_path}[/green]")
    console.print(f"\nNext steps:")
    console.print(f"  1. Open {output_path} in Claude Code")
    console.print(f"  2. Ask Claude to score each response on the listed dimensions (0-10)")
    console.print(f"  3. Save scores as CSV with columns: question_id, model, dimension, score, explanation")
    console.print(f"  4. Import: solum-bench import-scores --scores <scores.csv> --run {run_dir}")


@main.command("import-scores")
@click.option("--scores", "scores_path", required=True, type=click.Path(exists=True, path_type=Path), help="Scored CSV or JSONL")
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True, path_type=Path), help="Results directory to merge into")
@click.option("--format", "fmt", default="csv", type=click.Choice(["csv", "jsonl"]))
def import_scores(scores_path, run_dir, fmt):
    """Import externally-produced judge scores into a benchmark run.

    Expected CSV columns: question_id, model, dimension, score (0-10), explanation
    """
    from solum_bench.results.export import import_scores as do_import

    count = do_import(Path(scores_path), run_dir, fmt=fmt)
    console.print(f"[green]Imported scores for {count} question-model pairs into {run_dir}[/green]")


@main.command("list-questions")
@click.option("--questions", "bank_dir", default=None, type=click.Path(exists=True, path_type=Path))
@click.option("--category", default=None, help="Filter by category")
@click.option("--show-stats", is_flag=True, help="Show statistics by category/type/difficulty")
def list_questions(bank_dir, category, show_stats):
    """List questions in the question bank."""
    from collections import Counter

    from solum_bench.questions.loader import filter_questions, load_question_bank

    bank_dir = Path(bank_dir) if bank_dir else DEFAULT_QUESTION_BANK
    questions = load_question_bank(bank_dir)

    if category:
        questions = filter_questions(questions, categories=[category])

    if show_stats:
        _print_question_stats(questions)
    else:
        table = Table(title=f"Questions ({len(questions)} total)")
        table.add_column("ID", style="cyan")
        table.add_column("Type")
        table.add_column("Difficulty")
        table.add_column("Category")
        table.add_column("Scoring")
        for q in questions:
            table.add_row(q.id, q.question_type, q.difficulty, f"{q.category}/{q.subcategory}", q.scoring_method)
        console.print(table)


def _print_question_stats(questions):
    """Print statistics about the question bank."""
    from collections import Counter

    table = Table(title="Question Bank Statistics")
    table.add_column("Dimension", style="cyan")
    table.add_column("Breakdown", style="green")

    cats = Counter(q.category for q in questions)
    table.add_row("Categories", ", ".join(f"{k}: {v}" for k, v in sorted(cats.items())))

    types = Counter(q.question_type for q in questions)
    table.add_row("Types", ", ".join(f"{k}: {v}" for k, v in sorted(types.items())))

    diffs = Counter(q.difficulty for q in questions)
    table.add_row("Difficulties", ", ".join(f"{k}: {v}" for k, v in sorted(diffs.items())))

    methods = Counter(q.scoring_method for q in questions)
    table.add_row("Scoring", ", ".join(f"{k}: {v}" for k, v in sorted(methods.items())))

    console.print(table)
    console.print(f"\n[bold]Total: {len(questions)} questions[/bold]")


def _print_comparison(summaries):
    """Print a comparison table across models."""
    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan bold")
    table.add_column("Composite", style="green")
    table.add_column("Safety Fail", style="red")
    table.add_column("Avg Latency", style="yellow")
    table.add_column("Tokens", style="dim")

    for s in sorted(summaries, key=lambda x: x.overall_composite, reverse=True):
        table.add_row(
            s.model_name,
            f"{s.overall_composite:.3f}",
            f"{s.safety_failure_rate:.1%}",
            f"{s.avg_latency_ms:.0f}ms",
            f"{s.total_tokens:,}",
        )

    console.print(table)
