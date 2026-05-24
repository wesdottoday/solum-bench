"""Main benchmark execution engine."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

from solum_bench.providers.base import ModelProvider, ModelResponse
from solum_bench.questions.schema import Question
from solum_bench.results.storage import save_question_result, save_model_summary
from solum_bench.runner.session import BenchmarkSession
from solum_bench.scoring.aggregator import ModelSummary, QuestionResult, ScoringPipeline

console = Console()


def run_benchmark(
    providers: list[ModelProvider],
    questions: list[Question],
    output_dir: Path,
    question_bank_dir: Path,
    scoring_pipeline: ScoringPipeline,
    seed: int = 42,
    resume: bool = False,
    judge_model_name: str | None = None,
) -> list[ModelSummary]:
    """Run the full benchmark: generate responses and score them."""
    session = BenchmarkSession(output_dir)

    if resume:
        session.load_checkpoint()
        console.print(f"[yellow]Resuming from checkpoint ({len(session._completed_ids)} completed)[/yellow]")
    else:
        session.initialize(
            models=[p.name for p in providers],
            question_bank_dir=question_bank_dir,
            total_questions=len(questions),
            seed=seed,
            judge_model=judge_model_name,
        )

    summaries = []

    for provider in providers:
        console.print(f"\n[bold blue]Model: {provider.name}[/bold blue]")
        summary = _run_model(provider, questions, output_dir, scoring_pipeline, session)
        summaries.append(summary)
        save_model_summary(summary, output_dir)
        _print_model_summary(summary)

    session.finalize()
    return summaries


def _run_model(
    provider: ModelProvider,
    questions: list[Question],
    output_dir: Path,
    scoring_pipeline: ScoringPipeline,
    session: BenchmarkSession,
) -> ModelSummary:
    """Run all questions against a single model."""
    summary = ModelSummary(model_name=provider.name)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Running {provider.name}", total=len(questions))

        for question in questions:
            if session.is_completed(provider.name, question.id):
                progress.update(task, advance=1)
                continue

            try:
                response = _generate_response(provider, question)
                scores = scoring_pipeline.score_question(question, response)

                result = QuestionResult(
                    question_id=question.id,
                    model_name=provider.name,
                    response=response,
                    scores=scores,
                )

                save_question_result(result, output_dir)
                session.mark_completed(provider.name, question.id)
                summary.question_results.append(result)

            except Exception as e:
                console.print(f"[red]Error on {question.id}: {e}[/red]")
                # Create a failed result
                error_response = ModelResponse(
                    text="", input_tokens=0, output_tokens=0,
                    latency_ms=0, finish_reason="error",
                    raw_response={"error": str(e)},
                )
                result = QuestionResult(
                    question_id=question.id,
                    model_name=provider.name,
                    response=error_response,
                    scores=[],
                )
                save_question_result(result, output_dir)
                session.mark_completed(provider.name, question.id)
                summary.question_results.append(result)

            progress.update(task, advance=1)

    return summary


def _generate_response(provider: ModelProvider, question: Question) -> ModelResponse:
    """Generate a response for a single-turn or multi-turn question."""
    if question.is_multi_turn:
        return _generate_multi_turn(provider, question)

    messages = [{"role": "user", "content": question.user_prompt}]
    return provider.generate(
        system=question.system_prompt,
        messages=messages,
    )


def _generate_multi_turn(provider: ModelProvider, question: Question) -> ModelResponse:
    """Handle multi-turn conversations by generating each assistant turn."""
    messages = []
    last_response = None

    for turn in question.turns:
        if turn.role == "user":
            content = turn.content or ""
            if question.context_data and not messages:
                content = f"{question.context_data}\n\n{content}"
            messages.append({"role": "user", "content": content})
        elif turn.role == "assistant" and turn.content is None:
            # Generate this turn
            last_response = provider.generate(
                system=question.system_prompt,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": last_response.text})
        elif turn.role == "assistant" and turn.content:
            messages.append({"role": "assistant", "content": turn.content})

    if last_response is None:
        raise ValueError(f"No assistant turns to generate in question {question.id}")

    return last_response


def _print_model_summary(summary: ModelSummary):
    """Print a quick summary table for a model."""
    table = Table(title=f"Results: {summary.model_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Questions", str(summary.total_questions))
    table.add_row("Overall Composite", f"{summary.overall_composite:.3f}")
    table.add_row("Safety Failure Rate", f"{summary.safety_failure_rate:.1%}")
    table.add_row("Avg Latency", f"{summary.avg_latency_ms:.0f}ms")
    table.add_row("Total Tokens", f"{summary.total_tokens:,}")

    dims = summary.dimension_breakdown()
    for dim, score in dims.items():
        table.add_row(f"  {dim}", f"{score:.3f}")

    console.print(table)
