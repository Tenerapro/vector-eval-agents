"""Upload the Tenera Knowledge Bot evaluation dataset to Langfuse."""

import asyncio

import click
from aieng.agent_evals.langfuse import upload_dataset_to_langfuse


DEFAULT_EVALUATION_DATASET_PATH = "implementations/tenera_knowledge_bot/data/tenera_knowledge_eval.jsonl"
DEFAULT_EVALUATION_DATASET_NAME = "TeneraKnowledgeBotEval"


@click.command()
@click.option("--dataset-path", default=DEFAULT_EVALUATION_DATASET_PATH, help="Path to the local JSONL dataset.")
@click.option("--dataset-name", default=DEFAULT_EVALUATION_DATASET_NAME, help="Name of the Langfuse dataset.")
def cli(dataset_path: str, dataset_name: str) -> None:
    """Upload a local Tenera Knowledge Bot evaluation dataset."""
    asyncio.run(upload_dataset_to_langfuse(dataset_path, dataset_name))


if __name__ == "__main__":
    cli()
