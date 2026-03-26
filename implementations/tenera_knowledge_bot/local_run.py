"""Run a few Tenera Knowledge Bot questions locally with optional Langfuse tracing."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import click
from aieng.agent_evals.configs import Configs, TeneraRetrievalConfig
from aieng.agent_evals.evaluation.trace import flush_traces
from aieng.agent_evals.langfuse import init_tracing
from aieng.agent_evals.logging_config import setup_logging
from aieng.agent_evals.tenera_knowledge_bot import BotResponse, TeneraKnowledgeBot
from dotenv import load_dotenv


load_dotenv(verbose=True)
setup_logging(level=logging.INFO, show_time=True, show_path=False)
logger = logging.getLogger(__name__)


def _build_retrieval_config(backend: str | None) -> TeneraRetrievalConfig | None:
    configs = Configs()  # type: ignore[call-arg]
    retrieval_config = configs.tenera_retrieval
    if retrieval_config is None and backend is None:
        return None
    resolved = retrieval_config or TeneraRetrievalConfig()
    if backend is None:
        return resolved
    return resolved.model_copy(update={"backend": backend})


def _load_questions(question_values: tuple[str, ...], questions_file: str | None) -> list[str]:
    questions = [question.strip() for question in question_values if question.strip()]
    if questions_file:
        file_questions = [
            line.strip()
            for line in Path(questions_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        questions.extend(file_questions)
    return questions


def _print_result(question: str, response: BotResponse) -> None:
    click.echo()
    click.echo("=" * 80)
    click.echo(f"Question: {question}")
    click.echo("-" * 80)

    if response.tool_calls:
        click.echo(f"\nTool calls ({len(response.tool_calls)}):")
        for tc in response.tool_calls:
            args_str = str(tc.get("args", {}))
            if len(args_str) > 120:
                args_str = args_str[:120] + "..."
            click.echo(f"  - {tc.get('name', 'unknown')}({args_str})")

    click.echo(f"\nAnswer: {response.answer}")

    if response.citations:
        click.echo("\nCitations:")
        for c in response.citations:
            click.echo(f"  - [{c.chapter}] {c.section_label} {c.title} ({c.chunk_id})")
            excerpt = c.excerpt
            if len(excerpt) > 120:
                excerpt = excerpt[:120] + "..."
            click.echo(f"    Excerpt: {excerpt}")

    if response.caveats:
        click.echo("\nCaveats:")
        for caveat in response.caveats:
            click.echo(f"  - {caveat}")

    usage = response.token_usage
    click.echo(
        f"\nDuration: {response.total_duration_ms}ms | "
        f"Tool calls: {len(response.tool_calls)} | "
        f"Tokens: {usage.total_tokens} (prompt {usage.total_prompt_tokens}, completion {usage.total_completion_tokens})"
    )


async def _run_questions(
    *,
    questions: list[str],
    output_path: str | None,
    retrieval_backend: str | None,
    agent_timeout: int,
    log_trace: bool,
) -> None:
    tracing_enabled = False
    if log_trace:
        tracing_enabled = init_tracing()
        if tracing_enabled:
            logger.info("Langfuse tracing enabled")

    retrieval_config = _build_retrieval_config(retrieval_backend)
    bot = TeneraKnowledgeBot(
        retrieval_config=retrieval_config,
        timeout_sec=agent_timeout,
    )

    output_file = Path(output_path) if output_path else None
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        for index, question in enumerate(questions, start=1):
            bot.reset()
            response = await bot.answer_async(question)

            if not response.text:
                click.echo(f"\nQuestion {index} produced no result.")
                continue

            _print_result(question, response)

            if output_file:
                record = {"question": question, "result": response.model_dump(mode="json")}
                with output_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        await bot.close()
        if tracing_enabled:
            flush_traces()
            logger.info("Traces flushed to Langfuse")


@click.command()
@click.option(
    "--question",
    "questions",
    multiple=True,
    help="Question to run locally. Can be provided multiple times.",
)
@click.option(
    "--questions-file",
    default=None,
    help="Path to a text file containing one question per line.",
)
@click.option(
    "--output-path",
    default=None,
    help="Optional JSONL file to append local results to.",
)
@click.option(
    "--retrieval-backend",
    type=click.Choice(["sqlite", "http", "postgres"]),
    default=None,
    help="Override the configured retrieval backend.",
)
@click.option(
    "--agent-timeout",
    default=180,
    type=int,
    help="Timeout in seconds for each local run.",
)
@click.option(
    "--log-trace",
    is_flag=True,
    default=False,
    help="Enable Langfuse tracing for this run.",
)
def cli(
    questions: tuple[str, ...],
    questions_file: str | None,
    output_path: str | None,
    retrieval_backend: str | None,
    agent_timeout: int,
    log_trace: bool,
) -> None:
    """Run a few local questions against the Tenera Knowledge Bot."""
    loaded_questions = _load_questions(questions, questions_file)
    if not loaded_questions:
        raise click.BadParameter("Provide at least one --question or a --questions-file.")

    asyncio.run(
        _run_questions(
            questions=loaded_questions,
            output_path=output_path,
            retrieval_backend=retrieval_backend,
            agent_timeout=agent_timeout,
            log_trace=log_trace,
        )
    )


if __name__ == "__main__":
    cli()
