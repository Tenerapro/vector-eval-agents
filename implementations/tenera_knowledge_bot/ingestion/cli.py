"""CLI for the one-off Tenera knowledge-base ingestion pipeline."""

from pathlib import Path

import click
from dotenv import load_dotenv

from .chunker import chunk_sections
from .indexer import index_knowledge_base_sync
from .parser import parse_source_file


load_dotenv(verbose=True)


DEFAULT_SOURCE_PATH = "implementations/tenera_knowledge_bot/data/Florida_Building_Codes_Combined_2023.txt"
DEFAULT_DB_PATH = "implementations/tenera_knowledge_bot/data/tenera_knowledge.db"


@click.command()
@click.option("--source-path", default=DEFAULT_SOURCE_PATH, help="Path to the source text file.")
@click.option("--db-path", default=DEFAULT_DB_PATH, help="Path to the output SQLite knowledge base.")
@click.option("--max-chars", default=1600, type=int, help="Maximum characters per chunk.")
def cli(source_path: str, db_path: str, max_chars: int) -> None:
    """Build the local Tenera knowledge base from the source text."""
    sections = parse_source_file(Path(source_path))
    chunks = chunk_sections(sections, max_chars=max_chars)
    index_knowledge_base_sync(db_path=db_path, sections=sections, chunks=chunks)
    click.echo(f"Indexed {len(sections)} sections and {len(chunks)} chunks into {db_path}")


if __name__ == "__main__":
    cli()
