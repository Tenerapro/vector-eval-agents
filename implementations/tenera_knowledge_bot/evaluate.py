"""Evaluate the Tenera Knowledge Bot against a Langfuse dataset."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable

import click
from aieng.agent_evals.configs import Configs, TeneraRetrievalConfig
from aieng.agent_evals.evaluation import TraceWaitConfig, run_experiment, run_experiment_with_trace_evals
from aieng.agent_evals.evaluation.graders import (
    create_llm_as_judge_evaluator,
    create_trace_groundedness_evaluator,
)
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.tenera_knowledge_bot import TeneraKnowledgeBotTask, create_tenera_knowledge_bot_agent
from dotenv import load_dotenv
from langfuse.experiment import Evaluation


load_dotenv(verbose=True)


DEFAULT_DATASET_NAME = "TeneraKnowledgeBotEval"
DEFAULT_EXPERIMENT_NAME = "Tenera Knowledge Bot Evaluation"
DEFAULT_RUBRIC_PATH = "implementations/tenera_knowledge_bot/rubrics/answer_quality.md"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _build_retrieval_config(backend: str | None) -> TeneraRetrievalConfig | None:
    configs = Configs()  # type: ignore[call-arg]
    retrieval_config = configs.tenera_retrieval
    if retrieval_config is None and backend is None:
        return None
    resolved = retrieval_config or TeneraRetrievalConfig()
    if backend is None:
        return resolved
    return resolved.model_copy(update={"backend": backend})


def deterministic_retrieval_evaluator(
    *,
    input: Any,  # noqa: A002, ARG001
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> list[Evaluation]:
    """Grade retrieval quality without another model call."""
    output_dict = output if isinstance(output, dict) else {}
    citations = output_dict.get("citations", [])
    expected_chapter = (metadata or {}).get("chapter")
    supporting_text = str(expected_output.get("supporting_text", "")) if isinstance(expected_output, dict) else ""

    normalized_supporting = _normalize_text(supporting_text)
    chapter_hit = any(citation.get("chapter") == expected_chapter for citation in citations if isinstance(citation, dict))
    citation_overlap = any(
        (
            _normalize_text(str(citation.get("excerpt", ""))) in normalized_supporting
            or normalized_supporting in _normalize_text(str(citation.get("excerpt", "")))
        )
        for citation in citations
        if isinstance(citation, dict) and citation.get("excerpt")
    )
    has_citations = len(citations) > 0

    return [
        Evaluation(name="retrieval_has_citations", value=int(has_citations)),
        Evaluation(name="retrieval_expected_chapter_hit", value=int(chapter_hit)),
        Evaluation(name="retrieval_supporting_text_overlap", value=int(citation_overlap)),
    ]


@click.command()
@click.option("--dataset-name", default=DEFAULT_DATASET_NAME, help="Name of the Langfuse dataset to evaluate.")
@click.option("--experiment-name", default=DEFAULT_EXPERIMENT_NAME, help="Name for this evaluation run.")
@click.option("--agent-timeout", default=180, type=int, help="Timeout in seconds for each bot run.")
@click.option("--max-concurrency", default=5, type=int, help="Maximum concurrent agent runs.")
@click.option("--max-trace-wait-time", default=300, type=int, help="Maximum time to wait for Langfuse traces.")
@click.option(
    "--retrieval-backend",
    type=click.Choice(["sqlite", "http", "postgres"]),
    default=None,
    help="Override the configured retrieval backend for this evaluation.",
)
@click.option(
    "--enable-trace-evals/--no-enable-trace-evals",
    default=False,
    help="Whether to run the optional trace-level evaluation pass.",
)
def cli(
    dataset_name: str,
    experiment_name: str,
    agent_timeout: int,
    max_concurrency: int,
    max_trace_wait_time: int,
    retrieval_backend: str | None,
    enable_trace_evals: bool,
) -> None:
    """Run an offline evaluation for the Tenera Knowledge Bot."""
    asyncio.run(
        evaluate(
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            agent_timeout=agent_timeout,
            max_concurrency=max_concurrency,
            max_trace_wait_time=max_trace_wait_time,
            retrieval_backend=retrieval_backend,
            enable_trace_evals=enable_trace_evals,
        )
    )


async def evaluate(
    *,
    dataset_name: str,
    experiment_name: str,
    agent_timeout: int,
    max_concurrency: int,
    max_trace_wait_time: int,
    retrieval_backend: str | None = None,
    enable_trace_evals: bool = False,
    agent_factory: Callable[..., Any] = create_tenera_knowledge_bot_agent,
) -> None:
    """Execute the Langfuse experiment and upload scores."""
    retrieval_config = _build_retrieval_config(retrieval_backend)
    llm_judge = create_llm_as_judge_evaluator(
        name="tenera_answer_quality",
        rubric_markdown=DEFAULT_RUBRIC_PATH,
        model_config=LLMRequestConfig(temperature=0.0),
    )
    agent = agent_factory(timeout_sec=agent_timeout, retrieval_config=retrieval_config)
    task = TeneraKnowledgeBotTask(agent=agent)

    try:
        evaluators = [deterministic_retrieval_evaluator, llm_judge]
        if enable_trace_evals:
            trace_groundedness = create_trace_groundedness_evaluator(
                model_config=LLMRequestConfig(temperature=0.0),
            )
            run_experiment_with_trace_evals(
                dataset_name=dataset_name,
                name=experiment_name,
                task=task,
                evaluators=evaluators,
                trace_evaluators=[trace_groundedness],
                max_concurrency=max_concurrency,
                trace_wait=TraceWaitConfig(max_wait_sec=max_trace_wait_time),
            )
        else:
            run_experiment(
                dataset_name=dataset_name,
                name=experiment_name,
                task=task,
                evaluators=evaluators,
                max_concurrency=max_concurrency,
            )
    finally:
        await task.close()


if __name__ == "__main__":
    cli()
