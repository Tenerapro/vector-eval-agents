"""SQLite FTS5 plus local embedding retrieval backend."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path

import httpx

from aieng.agent_evals.configs import Configs, TeneraRetrievalConfig

from .. import register_backend
from ..types import ChunkCitation, ChunkStore, LexicalRetriever, RetrievalFilters, RetrievalResult, VectorRetriever


def _row_to_result(row: sqlite3.Row, method: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=row["chunk_id"],
        text=row["text"],
        score=float(row["score"]),
        metadata=ChunkCitation(
            section_id=row["section_id"],
            chapter=row["chapter"],
            section_label=row["section_label"],
            title=row["title"],
            source_file=row["source_file"],
            chunk_index=int(row["chunk_index"]),
        ),
        retrieval_method=method,  # type: ignore[arg-type]
    )


def _apply_filters_sql(filters: RetrievalFilters | None) -> tuple[str, list[str]]:
    clauses: list[str] = []
    params: list[str] = []
    if filters and filters.chapter:
        clauses.append("c.chapter = ?")
        params.append(filters.chapter)
    if filters and filters.section_label:
        clauses.append("c.section_label = ?")
        params.append(filters.section_label)
    if not clauses:
        return "", params
    return f" AND {' AND '.join(clauses)}", params


def _escape_fts_query(query: str) -> str:
    """Convert arbitrary user text into a conservative FTS5 query.

    SQLite FTS5 treats punctuation like ``-`` and reserved operators as syntax.
    We tokenize to alphanumeric-ish terms and quote each token so natural-language
    questions behave like keyword search instead of raising parse errors.
    """
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-_/]*", query)
    if not tokens:
        return '""'
    return " ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot_product = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _candidate_models(model: str, base_url: str) -> list[str]:
    if model.startswith("@"):
        raise ValueError(
            "Embedding model '@...' is not supported by the current retrieval pipeline. "
            "Set TENERA_RETRIEVAL__EMBEDDING_MODEL to a provider model such as "
            "'gemini-embedding-001' or another OpenAI-compatible embedding model."
        )
    if "generativelanguage.googleapis.com" in base_url and not model.startswith("models/"):
        return [model, f"models/{model}"]
    return [model]


class SqliteChunkStore(ChunkStore):
    """Chunk-store access backed by SQLite."""

    def __init__(self, kb_path: str) -> None:
        self._kb_path = kb_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._kb_path)
        connection.row_factory = sqlite3.Row
        return connection

    def get_chunk(self, chunk_id: str) -> RetrievalResult | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT chunk_id, section_id, chapter, section_label, title, source_file, chunk_index, text, 1.0 AS score
                FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        return _row_to_result(row, "hybrid") if row else None

    def get_section_chunks(self, section_id: str) -> list[RetrievalResult]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, section_id, chapter, section_label, title, source_file, chunk_index, text, 1.0 AS score
                FROM chunks
                WHERE section_id = ?
                ORDER BY chunk_index ASC
                """,
                (section_id,),
            ).fetchall()
        return [_row_to_result(row, "hybrid") for row in rows]


class SqliteFtsLexicalRetriever(LexicalRetriever):
    """Lexical retrieval backed by SQLite FTS5."""

    def __init__(self, kb_path: str) -> None:
        self._kb_path = kb_path

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        filter_sql, filter_params = _apply_filters_sql(filters)
        safe_query = _escape_fts_query(query)
        connection = sqlite3.connect(self._kb_path)
        connection.row_factory = sqlite3.Row
        try:
            sql = f"""
                SELECT
                    c.chunk_id,
                    c.section_id,
                    c.chapter,
                    c.section_label,
                    c.title,
                    c.source_file,
                    c.chunk_index,
                    c.text,
                    -bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                WHERE chunks_fts MATCH ?
                {filter_sql}
                ORDER BY bm25(chunks_fts)
                LIMIT ?
            """
            try:
                rows = connection.execute(sql, [safe_query, *filter_params, top_k]).fetchall()
            except sqlite3.OperationalError:
                # Fall back to a looser OR query if the stricter tokenized search
                # still hits an FTS parsing edge case.
                fallback_query = " OR ".join(part for part in safe_query.split() if part)
                rows = connection.execute(sql, [fallback_query, *filter_params, top_k]).fetchall()
        finally:
            connection.close()
        return [_row_to_result(row, "lexical") for row in rows]


class SqliteVectorRetriever(VectorRetriever):
    """Vector retrieval backed by SQLite-stored embeddings."""

    def __init__(self, kb_path: str, settings: TeneraRetrievalConfig) -> None:
        self._kb_path = kb_path
        self._settings = settings
        self._configs = Configs()  # type: ignore[call-arg]

    def _embed_query(self, query: str) -> list[float]:
        api_key = (
            self._settings.embedding_api_key.get_secret_value()
            if self._settings.embedding_api_key
            else (
                self._configs.embedding_api_key.get_secret_value()
                if self._configs.embedding_api_key
                else self._configs.openai_api_key.get_secret_value()
            )
        )
        base_url = self._settings.embedding_base_url or self._configs.embedding_base_url or self._configs.openai_base_url
        model = self._settings.embedding_model or self._configs.embedding_model_name
        last_error: Exception | None = None
        for candidate_model in _candidate_models(model, base_url):
            try:
                response = httpx.post(
                    f"{base_url.rstrip('/')}/embeddings",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": candidate_model, "input": [query]},
                    timeout=30.0,
                )
                response.raise_for_status()
                payload = response.json()
                return list(payload["data"][0]["embedding"])
            except Exception as exc:
                last_error = exc
                if "unexpected model name format" not in str(exc):
                    raise
        assert last_error is not None
        raise last_error

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._kb_path)
        connection.row_factory = sqlite3.Row
        return connection

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        query_embedding = self._embed_query(query)
        filter_sql, filter_params = _apply_filters_sql(filters)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    c.chunk_id,
                    c.section_id,
                    c.chapter,
                    c.section_label,
                    c.title,
                    c.source_file,
                    c.chunk_index,
                    c.text,
                    e.embedding_json
                FROM chunk_embeddings e
                JOIN chunks c ON c.chunk_id = e.chunk_id
                WHERE 1 = 1
                {filter_sql}
                """,
                filter_params,
            ).fetchall()

        scored_results: list[RetrievalResult] = []
        for row in rows:
            similarity = _cosine_similarity(query_embedding, json.loads(row["embedding_json"]))
            scored_results.append(
                RetrievalResult(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    score=similarity,
                    metadata=ChunkCitation(
                        section_id=row["section_id"],
                        chapter=row["chapter"],
                        section_label=row["section_label"],
                        title=row["title"],
                        source_file=row["source_file"],
                        chunk_index=int(row["chunk_index"]),
                    ),
                    retrieval_method="vector",
                )
            )

        scored_results.sort(key=lambda item: item.score, reverse=True)
        return scored_results[:top_k]


class SqliteRetrieverBackend:
    """Backend bundle for local SQLite retrieval."""

    def __init__(self, settings: TeneraRetrievalConfig) -> None:
        kb_path = settings.kb_path or "implementations/tenera_knowledge_bot/data/tenera_knowledge.db"
        if not Path(kb_path).exists():
            raise FileNotFoundError(f"SQLite knowledge base not found at '{kb_path}'. Run the ingestion CLI first.")
        self._store = SqliteChunkStore(kb_path)
        self._lexical = SqliteFtsLexicalRetriever(kb_path)
        self._vector = SqliteVectorRetriever(kb_path, settings=settings)

    @classmethod
    def from_config(cls, settings: TeneraRetrievalConfig) -> "SqliteRetrieverBackend":
        return cls(settings)

    def lexical_retriever(self) -> LexicalRetriever:
        return self._lexical

    def vector_retriever(self) -> VectorRetriever:
        return self._vector

    def chunk_store(self) -> ChunkStore:
        return self._store


register_backend("sqlite", SqliteRetrieverBackend)
