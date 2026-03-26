"""Upload the Tenera Knowledge Bot evaluation dataset to Langfuse."""

import asyncio
import csv
import json
import tempfile
from pathlib import Path

import click
from aieng.agent_evals.langfuse import upload_dataset_to_langfuse


DEFAULT_EVALUATION_DATASET_PATH = "implementations/tenera_knowledge_bot/data/fl_building_code_test_set.csv"
DEFAULT_EVALUATION_DATASET_NAME = "TeneraKnowledgeBotEval"


async def upload_csv_dataset(dataset_path: str, dataset_name: str, limit: int | None = None) -> None:
    """Convert the source CSV test set into Langfuse dataset rows and upload it."""
    source_path = Path(dataset_path)
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".jsonl", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        for index, row in enumerate(rows, start=1):
            record = {
                "input": row["question"],
                "expected_output": {
                    "answer": row["answer"],
                    "supporting_text": row["ground_truth"],
                },
                "metadata": {
                    "id": f"fl-building-code-{index}",
                    "chapter": row["chapter"],
                },
            }
            temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        await upload_dataset_to_langfuse(str(temp_path), dataset_name)
    finally:
        temp_path.unlink(missing_ok=True)


@click.command()
@click.option("--dataset-path", default=DEFAULT_EVALUATION_DATASET_PATH, help="Path to the source CSV dataset.")
@click.option("--dataset-name", default=DEFAULT_EVALUATION_DATASET_NAME, help="Name of the Langfuse dataset.")
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Optional limit for the number of CSV rows to upload from the top of the file.",
)
def cli(dataset_path: str, dataset_name: str, limit: int | None) -> None:
    """Upload the CSV-derived Tenera Knowledge Bot evaluation dataset."""
    asyncio.run(upload_csv_dataset(dataset_path, dataset_name, limit=limit))


if __name__ == "__main__":
    cli()
