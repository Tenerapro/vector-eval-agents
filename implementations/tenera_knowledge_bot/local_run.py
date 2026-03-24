"""Run a few Tenera Knowledge Bot questions locally without Langfuse evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import click
from aieng.agent_evals.configs import Configs, TeneraRetrievalConfig
from aieng.agent_evals.tenera_knowledge_bot import TeneraKnowledgeBotTask, create_tenera_knowledge_bot_agent
from dotenv import load_dotenv


load_dotenv(verbose=True)


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


def _print_result(question: str, result: dict[str, Any]) -> None:
    click.echo()
    click.echo("=" * 80)
    click.echo(f"Question: {question}")
    click.echo("-" * 80)
    click.echo(f"Answer: {result.get('answer', '')}")

    citations = result.get("citations", [])
    if citations:
        click.echo("\nCitations:")
        for citation in citations:
            click.echo(
                f"- [{citation.get('chapter')}] {citation.get('section_label')} "
                f"{citation.get('title')} ({citation.get('chunk_id')})"
            )
            click.echo(f"  Excerpt: {citation.get('excerpt')}")

    caveats = result.get("caveats", [])
    if caveats:
        click.echo("\nCaveats:")
        for caveat in caveats:
            click.echo(f"- {caveat}")


async def _run_questions(
    *,
    questions: list[str],
    output_path: str | None,
    retrieval_backend: str | None,
    agent_timeout: int,
) -> None:
    retrieval_config = _build_retrieval_config(retrieval_backend)
    agent = create_tenera_knowledge_bot_agent(
        timeout_sec=agent_timeout,
        retrieval_config=retrieval_config,
        enable_tracing=False,
    )
    task = TeneraKnowledgeBotTask(agent=agent)

    output_file = Path(output_path) if output_path else None
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        for index, question in enumerate(questions, start=1):
            item = {
                "input": question,
                "metadata": {"id": f"local-{index}"},
            }
            result = await task(item=item)
            if result is None:
                click.echo(f"\nQuestion {index} produced no result.")
                continue

            _print_result(question, result)

            if output_file:
                record = {"question": question, "result": result}
                with output_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        await task.close()


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
def cli(
    questions: tuple[str, ...],
    questions_file: str | None,
    output_path: str | None,
    retrieval_backend: str | None,
    agent_timeout: int,
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
        )
    )


if __name__ == "__main__":
    cli()
