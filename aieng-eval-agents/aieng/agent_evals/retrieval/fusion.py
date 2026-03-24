"""Fusion strategies for hybrid retrieval."""

from typing import Protocol

from .types import RetrievalResult


class FusionStrategy(Protocol):
    """Protocol for merging lexical and vector results."""

    def fuse(
        self,
        lexical: list[RetrievalResult],
        vector: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Fuse two ranked result lists into one."""


class RRFFusion:
    """Reciprocal rank fusion."""

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        lexical: list[RetrievalResult],
        vector: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        chosen: dict[str, RetrievalResult] = {}

        for rank, result in enumerate(lexical, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (self.k + rank)
            chosen.setdefault(result.chunk_id, result)

        for rank, result in enumerate(vector, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (self.k + rank)
            chosen.setdefault(result.chunk_id, result)

        merged = sorted(chosen.values(), key=lambda item: scores.get(item.chunk_id, 0.0), reverse=True)
        return [
            item.model_copy(update={"score": scores[item.chunk_id], "retrieval_method": "hybrid"})
            for item in merged[:top_k]
        ]


class WeightedFusion:
    """Weighted score fusion."""

    def __init__(self, lexical_weight: float = 0.4, vector_weight: float = 0.6) -> None:
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight

    def fuse(
        self,
        lexical: list[RetrievalResult],
        vector: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        chosen: dict[str, RetrievalResult] = {}

        for result in lexical:
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + (self.lexical_weight * result.score)
            chosen.setdefault(result.chunk_id, result)

        for result in vector:
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + (self.vector_weight * result.score)
            chosen.setdefault(result.chunk_id, result)

        merged = sorted(chosen.values(), key=lambda item: scores.get(item.chunk_id, 0.0), reverse=True)
        return [
            item.model_copy(update={"score": scores[item.chunk_id], "retrieval_method": "hybrid"})
            for item in merged[:top_k]
        ]


__all__ = ["FusionStrategy", "RRFFusion", "WeightedFusion"]
