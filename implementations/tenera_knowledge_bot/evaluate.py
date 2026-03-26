"""Evaluate the Tenera Knowledge Bot against a Langfuse dataset."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import click
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.configs import Configs, TeneraRetrievalConfig
from aieng.agent_evals.evaluation import TraceWaitConfig, run_experiment, run_experiment_with_trace_evals
from aieng.agent_evals.langfuse import init_tracing
from aieng.agent_evals.evaluation.graders import (
    create_llm_as_judge_evaluator,
    create_trace_groundedness_evaluator,
)
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.types import EvaluationResult
from aieng.agent_evals.logging_config import setup_logging
from aieng.agent_evals.tenera_knowledge_bot import TeneraKnowledgeBot
from dotenv import load_dotenv
from langfuse.experiment import Evaluation, ExperimentResult


load_dotenv(verbose=True)
setup_logging(level=logging.INFO, show_time=True, show_path=False)
logger = logging.getLogger(__name__)


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
@click.option("--max-concurrency", default=1, type=int, help="Maximum concurrent agent runs.")
@click.option("--max-trace-wait-time", default=300, type=int, help="Maximum time to wait for Langfuse traces.")
@click.option(
    "--retrieval-backend",
    type=click.Choice(["sqlite", "http", "postgres"]),
    default=None,
    help="Override the configured retrieval backend for this evaluation.",
)
@click.option(
    "--enable-trace-evals/--no-enable-trace-evals",
    default=True,
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
) -> None:
    """Execute the Langfuse experiment and upload scores."""
    client_manager = AsyncClientManager.get_instance()
    llm_judge = create_llm_as_judge_evaluator(
        name="tenera_answer_quality",
        rubric_markdown=DEFAULT_RUBRIC_PATH,
        model_config=LLMRequestConfig(temperature=0.0),
    )
    # Enable OTEL tracing so GoogleADKInstrumentor creates child spans under
    # the Langfuse experiment trace (Langfuse SDK uses OTEL internally, so
    # ADK tool call spans automatically nest via OTEL context propagation).
    init_tracing()
    retrieval_config = _build_retrieval_config(retrieval_backend)

    async def agent_task(*, item: Any, **kwargs: Any) -> dict[str, Any] | None:  # noqa: ARG001
        item_input = item.input if hasattr(item, "input") else item.get("input")
        question = item_input if isinstance(item_input, str) else json.dumps(item_input, ensure_ascii=False, indent=2)
        logger.info("Running agent on: %s...", question[:80])

        bot = TeneraKnowledgeBot(
            retrieval_config=retrieval_config,
            timeout_sec=agent_timeout,
        )
        try:
            response = await bot.answer_async(question)
            if not response.text:
                metadata = item.metadata if hasattr(item, "metadata") else item.get("metadata", {})
                item_id = metadata.get("id") if metadata else "unknown"
                logger.warning("No final response produced for item=%s", item_id)
                return None

            result = {
                "answer": response.answer,
                "citations": [c.model_dump() for c in response.citations],
                "caveats": response.caveats,
            }
            client_manager.langfuse_client.update_current_span(
                metadata={
                    **result,
                    "execution_trace": [step.model_dump() for step in response.execution_trace],
                    "tool_calls": response.tool_calls,
                    "tool_responses": response.tool_responses,
                    "reasoning_chain": response.reasoning_chain,
                    "total_duration_ms": response.total_duration_ms,
                    "token_usage": response.token_usage.model_dump(),
                }
            )
            logger.info("Agent completed: %d chars, %d tool calls", len(response.text), len(response.tool_calls))
            return result
        finally:
            await bot.close()

    try:
        logger.info("Starting experiment '%s' on dataset '%s'", experiment_name, dataset_name)
        logger.info("Max concurrency: %d, Trace evals: %s", max_concurrency, "enabled" if enable_trace_evals else "disabled")

        evaluators = [deterministic_retrieval_evaluator, llm_judge]
        result: ExperimentResult | EvaluationResult
        if enable_trace_evals:
            trace_groundedness = create_trace_groundedness_evaluator(
                model_config=LLMRequestConfig(temperature=0.0),
            )
            result = run_experiment_with_trace_evals(
                dataset_name=dataset_name,
                name=experiment_name,
                task=agent_task,
                evaluators=evaluators,
                trace_evaluators=[trace_groundedness],
                max_concurrency=max_concurrency,
                trace_wait=TraceWaitConfig(max_wait_sec=max_trace_wait_time),
            )
        else:
            result = run_experiment(
                dataset_name=dataset_name,
                name=experiment_name,
                task=agent_task,
                evaluators=evaluators,
                max_concurrency=max_concurrency,
            )

        logger.info("Experiment complete!")
        if isinstance(result, EvaluationResult):
            logger.info("Results: %s", result.experiment)
            if result.trace_evaluations:
                trace_evals = result.trace_evaluations
                logger.info(
                    "Trace evaluations: %d traces, %d skipped, %d failed",
                    len(trace_evals.evaluations_by_trace_id),
                    len(trace_evals.skipped_trace_ids),
                    len(trace_evals.failed_trace_ids),
                )
        else:
            logger.info("Results: %s", result)

    finally:
        logger.info("Closing client manager and flushing data...")
        try:
            await client_manager.close()
            await asyncio.sleep(0.1)
            logger.info("Cleanup complete")
        except Exception as e:
            logger.warning("Cleanup warning: %s", e)


if __name__ == "__main__":
    cli()
