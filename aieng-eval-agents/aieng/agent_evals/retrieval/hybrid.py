"""Hybrid retriever implementation."""

from __future__ import annotations

from .fusion import FusionStrategy
from .types import LexicalRetriever, RetrievalFilters, RetrievalResult, VectorRetriever


class HybridRetriever:
    """Compose lexical and vector retrievers behind one interface."""

    def __init__(
        self,
        lexical: LexicalRetriever,
        vector: VectorRetriever,
        *,
        fusion: FusionStrategy,
    ) -> None:
        self._lexical = lexical
        self._vector = vector
        self._fusion = fusion

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        """Run both retrievers and merge the results."""
        lexical_results = self._lexical.search(query, top_k=top_k, filters=filters)
        vector_results = self._vector.search(query, top_k=top_k, filters=filters)
        return self._fusion.fuse(lexical_results, vector_results, top_k=top_k)
