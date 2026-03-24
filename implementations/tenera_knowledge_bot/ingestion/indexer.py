"""Index parsed chunks into a SQLite knowledge base."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

from langfuse.openai import AsyncOpenAI

from aieng.agent_evals.configs import Configs

from .chunker import ChunkRecord
from .parser import ParsedSection


def _candidate_models(model: str, base_url: str) -> list[str]:
    if model.startswith("@"):
        raise ValueError(
            "Embedding model '@...' is not supported by the current ingestion pipeline. "
            "Set TENERA_RETRIEVAL__EMBEDDING_MODEL to a real provider model such as "
            "'gemini-embedding-001' or another OpenAI-compatible embedding model."
        )
    if "generativelanguage.googleapis.com" in base_url and not model.startswith("models/"):
        return [model, f"models/{model}"]
    return [model]


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP TABLE IF EXISTS documents;
        DROP TABLE IF EXISTS sections;
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS chunk_embeddings;
        DROP TABLE IF EXISTS chunks_fts;

        CREATE TABLE documents (
            document_id TEXT PRIMARY KEY,
            source_file TEXT NOT NULL,
            code_family TEXT NOT NULL,
            chapter TEXT NOT NULL,
            source_slug TEXT NOT NULL
        );

        CREATE TABLE sections (
            section_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chapter TEXT NOT NULL,
            section_label TEXT NOT NULL,
            title TEXT NOT NULL,
            source_slug TEXT NOT NULL,
            source_file TEXT NOT NULL,
            text TEXT NOT NULL
        );

        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            section_id TEXT NOT NULL,
            chapter TEXT NOT NULL,
            section_label TEXT NOT NULL,
            title TEXT NOT NULL,
            source_file TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL
        );

        CREATE TABLE chunk_embeddings (
            chunk_id TEXT PRIMARY KEY,
            embedding_json TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            text,
            chapter,
            section_label,
            title
        );
        """
    )


async def _embed_texts(texts: list[str], model: str, base_url: str, api_key: str) -> list[list[float]]:
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    try:
        last_error: Exception | None = None
        for candidate_model in _candidate_models(model, base_url):
            try:
                response = await client.embeddings.create(model=candidate_model, input=texts)
                return [list(item.embedding) for item in response.data]
            except Exception as exc:
                last_error = exc
                if "unexpected model name format" not in str(exc):
                    raise
        assert last_error is not None
        raise last_error
    finally:
        await client.close()


async def index_knowledge_base(
    *,
    db_path: str | Path,
    sections: list[ParsedSection],
    chunks: list[ChunkRecord],
) -> None:
    """Build the SQLite knowledge base from parsed sections and chunks."""
    configs = Configs()  # type: ignore[call-arg]
    retrieval_config = configs.tenera_retrieval
    embedding_model = (
        retrieval_config.embedding_model
        if retrieval_config is not None
        else configs.embedding_model_name
    )
    embedding_base_url = (
        retrieval_config.embedding_base_url
        if retrieval_config is not None and retrieval_config.embedding_base_url
        else configs.embedding_base_url or configs.openai_base_url
    )
    embedding_api_key = (
        retrieval_config.embedding_api_key.get_secret_value()
        if retrieval_config is not None and retrieval_config.embedding_api_key
        else (
            configs.embedding_api_key.get_secret_value()
            if configs.embedding_api_key
            else configs.openai_api_key.get_secret_value()
        )
    )

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    try:
        _create_schema(connection)

        seen_documents: set[str] = set()
        for section in sections:
            document_id = f"{section.source_slug}:{section.chapter}"
            if document_id not in seen_documents:
                connection.execute(
                    """
                    INSERT INTO documents (document_id, source_file, code_family, chapter, source_slug)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (document_id, section.source_file, section.code_family, section.chapter, section.source_slug),
                )
                seen_documents.add(document_id)

            connection.execute(
                """
                INSERT INTO sections (section_id, document_id, chapter, section_label, title, source_slug, source_file, text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    section.section_id,
                    document_id,
                    section.chapter,
                    section.section_label,
                    section.title,
                    section.source_slug,
                    section.source_file,
                    section.text,
                ),
            )

        batch_size = 32
        for start in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start : start + batch_size]
            embeddings = await _embed_texts(
                [chunk.text for chunk in chunk_batch],
                model=embedding_model,
                base_url=embedding_base_url,
                api_key=embedding_api_key,
            )

            for chunk, embedding in zip(chunk_batch, embeddings, strict=False):
                connection.execute(
                    """
                    INSERT INTO chunks (chunk_id, section_id, chapter, section_label, title, source_file, chunk_index, text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.section_id,
                        chunk.chapter,
                        chunk.section_label,
                        chunk.title,
                        chunk.source_file,
                        chunk.chunk_index,
                        chunk.text,
                    ),
                )
                connection.execute(
                    "INSERT INTO chunk_embeddings (chunk_id, embedding_json) VALUES (?, ?)",
                    (chunk.chunk_id, json.dumps(embedding)),
                )
                connection.execute(
                    "INSERT INTO chunks_fts (chunk_id, text, chapter, section_label, title) VALUES (?, ?, ?, ?, ?)",
                    (chunk.chunk_id, chunk.text, chunk.chapter, chunk.section_label, chunk.title),
                )

        connection.commit()
    finally:
        connection.close()


def index_knowledge_base_sync(*, db_path: str | Path, sections: list[ParsedSection], chunks: list[ChunkRecord]) -> None:
    """Synchronous wrapper for CLI usage."""
    asyncio.run(index_knowledge_base(db_path=db_path, sections=sections, chunks=chunks))
