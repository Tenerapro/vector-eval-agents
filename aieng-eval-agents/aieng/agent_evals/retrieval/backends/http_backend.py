"""HTTP retrieval backend adapters."""

from __future__ import annotations

from dataclasses import asdict

import httpx

from aieng.agent_evals.configs import TeneraRetrievalConfig

from .. import register_backend
from ..types import ChunkStore, LexicalRetriever, RetrievalFilters, RetrievalResult, VectorRetriever


class _HttpClientMixin:
    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(headers=headers, timeout=30.0)


class HttpLexicalRetriever(_HttpClientMixin):
    """Lexical retrieval over HTTP."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        response = self._client.post(
            f"{self._base_url}/search/lexical",
            json={"query": query, "top_k": top_k, "filters": asdict(filters) if filters else None},
        )
        response.raise_for_status()
        return [RetrievalResult.model_validate(item) for item in response.json().get("results", [])]


class HttpVectorRetriever(_HttpClientMixin):
    """Vector retrieval over HTTP."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        response = self._client.post(
            f"{self._base_url}/search/vector",
            json={"query": query, "top_k": top_k, "filters": asdict(filters) if filters else None},
        )
        response.raise_for_status()
        return [RetrievalResult.model_validate(item) for item in response.json().get("results", [])]


class HttpChunkStore(_HttpClientMixin):
    """Chunk-store access over HTTP."""

    def get_chunk(self, chunk_id: str) -> RetrievalResult | None:
        response = self._client.get(f"{self._base_url}/chunks/{chunk_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return RetrievalResult.model_validate(response.json())

    def get_section_chunks(self, section_id: str) -> list[RetrievalResult]:
        response = self._client.get(f"{self._base_url}/sections/{section_id}/chunks")
        response.raise_for_status()
        return [RetrievalResult.model_validate(item) for item in response.json().get("results", [])]


class HttpRetrieverBackend:
    """Backend bundle for remote retrieval services."""

    def __init__(self, settings: TeneraRetrievalConfig) -> None:
        if not settings.http_base_url:
            raise ValueError("TENERA_RETRIEVAL__HTTP_BASE_URL is required for the http backend.")
        api_key = settings.http_api_key.get_secret_value() if settings.http_api_key else None
        self._lexical = HttpLexicalRetriever(settings.http_base_url, api_key=api_key)
        self._vector = HttpVectorRetriever(settings.http_base_url, api_key=api_key)
        self._store = HttpChunkStore(settings.http_base_url, api_key=api_key)

    @classmethod
    def from_config(cls, settings: TeneraRetrievalConfig) -> "HttpRetrieverBackend":
        return cls(settings)

    def lexical_retriever(self) -> LexicalRetriever:
        return self._lexical

    def vector_retriever(self) -> VectorRetriever:
        return self._vector

    def chunk_store(self) -> ChunkStore:
        return self._store


register_backend("http", HttpRetrieverBackend)
