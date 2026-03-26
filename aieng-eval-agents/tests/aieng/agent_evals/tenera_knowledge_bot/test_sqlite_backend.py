"""Tests for the SQLite retrieval backend performance helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aieng.agent_evals.configs import TeneraRetrievalConfig
from aieng.agent_evals.retrieval.backends import sqlite_backend


def _build_test_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
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
        connection.executemany(
            """
            INSERT INTO chunks (chunk_id, section_id, chapter, section_label, title, source_file, chunk_index, text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("chunk-a", "sec-a", "Chapter A", "101", "Title A", "source.txt", 0, "alpha text"),
                ("chunk-b", "sec-b", "Chapter B", "202", "Title B", "source.txt", 0, "beta text"),
            ],
        )
        connection.executemany(
            "INSERT INTO chunk_embeddings (chunk_id, embedding_json) VALUES (?, ?)",
            [
                ("chunk-a", "[1.0, 0.0]"),
                ("chunk-b", "[0.0, 1.0]"),
            ],
        )
        connection.executemany(
            "INSERT INTO chunks_fts (chunk_id, text, chapter, section_label, title) VALUES (?, ?, ?, ?, ?)",
            [
                ("chunk-a", "alpha text", "Chapter A", "101", "Title A"),
                ("chunk-b", "beta text", "Chapter B", "202", "Title B"),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def _fake_configs() -> SimpleNamespace:
    return SimpleNamespace(
        embedding_api_key=None,
        openai_api_key=SimpleNamespace(get_secret_value=lambda: "test-key"),
        embedding_base_url=None,
        openai_base_url="https://example.test/v1",
        embedding_model_name="test-embedding-model",
    )


def test_connect_read_only_uses_sqlite_uri_mode_ro(tmp_path):
    db_path = tmp_path / "kb.db"
    db_path.touch()

    mock_connection = MagicMock()
    with patch("aieng.agent_evals.retrieval.backends.sqlite_backend.sqlite3.connect", return_value=mock_connection) as mock_connect:
        connection = sqlite_backend._connect_read_only(str(db_path))

    assert connection is mock_connection
    connect_args = mock_connect.call_args
    assert connect_args.args[0].startswith(f"file:{db_path.resolve()}?mode=ro")
    assert connect_args.kwargs["uri"] is True
    assert mock_connection.execute.call_args_list[0].args[0] == "PRAGMA query_only = ON"
    assert mock_connection.execute.call_args_list[1].args[0] == "PRAGMA busy_timeout = 5000"


def test_vector_retriever_reuses_cached_embedding_rows_across_instances(tmp_path):
    db_path = tmp_path / "kb.db"
    _build_test_db(db_path)
    sqlite_backend._load_embedding_rows.cache_clear()

    real_connect = sqlite_backend._connect_read_only
    connect_calls = 0

    def counting_connect(kb_path: str):
        nonlocal connect_calls
        connect_calls += 1
        return real_connect(kb_path)

    settings = TeneraRetrievalConfig(kb_path=str(db_path))
    with (
        patch("aieng.agent_evals.retrieval.backends.sqlite_backend.Configs", side_effect=_fake_configs),
        patch("aieng.agent_evals.retrieval.backends.sqlite_backend._connect_read_only", side_effect=counting_connect),
    ):
        sqlite_backend.SqliteVectorRetriever(str(db_path), settings)
        sqlite_backend.SqliteVectorRetriever(str(db_path), settings)

    assert connect_calls == 1


def test_vector_search_uses_cached_rows_after_init_and_respects_filters(tmp_path):
    db_path = tmp_path / "kb.db"
    _build_test_db(db_path)
    sqlite_backend._load_embedding_rows.cache_clear()

    settings = TeneraRetrievalConfig(kb_path=str(db_path))
    with patch("aieng.agent_evals.retrieval.backends.sqlite_backend.Configs", side_effect=_fake_configs):
        retriever = sqlite_backend.SqliteVectorRetriever(str(db_path), settings)

    with (
        patch.object(retriever, "_embed_query", return_value=[1.0, 0.0]),
        patch("aieng.agent_evals.retrieval.backends.sqlite_backend._connect_read_only", side_effect=AssertionError("search should not reopen sqlite")),
    ):
        unfiltered = retriever.search("alpha", top_k=2)
        filtered = retriever.search(
            "alpha",
            top_k=2,
            filters=SimpleNamespace(chapter="Chapter B", section_label=None),
        )

    assert [result.chunk_id for result in unfiltered] == ["chunk-a", "chunk-b"]
    assert [result.chunk_id for result in filtered] == ["chunk-b"]
