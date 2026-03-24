"""Evaluate the Tenera Knowledge Bot against a Langfuse dataset."""

import asyncio
import logging

import click
from aieng.agent_evals.evaluation import TraceWaitConfig, run_experiment_with_trace_evals
from aieng.agent_evals.evaluation.graders import (
    create_llm_as_judge_evaluator,
    create_trace_groundedness_evaluator,
)
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.tenera_knowledge_bot import TeneraKnowledgeBotTask, create_tenera_knowledge_bot_agent
from dotenv import load_dotenv


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DATASET_NAME = "TeneraKnowledgeBotEval"
DEFAULT_EXPERIMENT_NAME = "Tenera Knowledge Bot Evaluation"
DEFAULT_RUBRIC_PATH = "implementations/tenera_knowledge_bot/rubrics/answer_quality.md"


@click.command()
@click.option("--dataset-name", default=DEFAULT_DATASET_NAME, help="Name of the Langfuse dataset to evaluate.")
@click.option("--experiment-name", default=DEFAULT_EXPERIMENT_NAME, help="Name for this evaluation run.")
@click.option("--agent-timeout", default=180, type=int, help="Timeout in seconds for each bot run.")
@click.option("--max-concurrency", default=5, type=int, help="Maximum concurrent agent runs.")
@click.option("--max-trace-wait-time", default=300, type=int, help="Maximum time to wait for Langfuse traces.")
def cli(
    dataset_name: str,
    experiment_name: str,
    agent_timeout: int,
    max_concurrency: int,
    max_trace_wait_time: int,
) -> None:
    """Run an offline evaluation for the Tenera Knowledge Bot."""
    asyncio.run(
        evaluate(
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            agent_timeout=agent_timeout,
            max_concurrency=max_concurrency,
            max_trace_wait_time=max_trace_wait_time,
        )
    )


async def evaluate(
    *,
    dataset_name: str,
    experiment_name: str,
    agent_timeout: int,
    max_concurrency: int,
    max_trace_wait_time: int,
) -> None:
    """Execute the Langfuse experiment and upload scores."""
    llm_judge = create_llm_as_judge_evaluator(
        name="tenera_answer_quality",
        rubric_markdown=DEFAULT_RUBRIC_PATH,
        model_config=LLMRequestConfig(temperature=0.0),
    )
    trace_groundedness = create_trace_groundedness_evaluator(
        model_config=LLMRequestConfig(temperature=0.0),
    )

    agent = create_tenera_knowledge_bot_agent(timeout_sec=agent_timeout)
    task = TeneraKnowledgeBotTask(agent=agent)

    try:
        result = run_experiment_with_trace_evals(
            dataset_name=dataset_name,
            name=experiment_name,
            task=task,
            evaluators=[llm_judge],
            trace_evaluators=[trace_groundedness],
            max_concurrency=max_concurrency,
            trace_wait=TraceWaitConfig(max_wait_sec=max_trace_wait_time),
        )
        logger.info("Evaluation complete: %s", result.experiment)
    finally:
        await task.close()


if __name__ == "__main__":
    cli()
