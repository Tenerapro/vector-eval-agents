"""Shared types and protocols for retrieval backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class ChunkCitation(BaseModel):
    """Citation metadata for a retrieved chunk."""

    section_id: str
    chapter: str
    section_label: str
    title: str
    source_file: str
    chunk_index: int


class RetrievalResult(BaseModel):
    """Normalized retrieval result returned by all backends."""

    chunk_id: str
    text: str
    score: float
    metadata: ChunkCitation
    retrieval_method: Literal["lexical", "vector", "hybrid"]


@dataclass(frozen=True)
class RetrievalFilters:
    """Backend-agnostic retrieval filters."""

    chapter: str | None = None
    section_label: str | None = None


class LexicalRetriever(Protocol):
    """Protocol for lexical retrieval."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        """Search using lexical signals."""


class VectorRetriever(Protocol):
    """Protocol for vector retrieval."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        """Search using vector similarity."""


class ChunkStore(Protocol):
    """Protocol for direct chunk and section access."""

    def get_chunk(self, chunk_id: str) -> RetrievalResult | None:
        """Fetch one chunk by ID."""

    def get_section_chunks(self, section_id: str) -> list[RetrievalResult]:
        """Fetch all chunks in one section."""


class AnswerCitation(BaseModel):
    """Citation included in the final agent response."""

    chunk_id: str = Field(description="Retrieved chunk identifier.")
    chapter: str = Field(description="Chapter or appendix label.")
    section_label: str = Field(description="Section identifier.")
    title: str = Field(description="Section title.")
    excerpt: str = Field(description="Quoted or paraphrased excerpt from the retrieved chunk.")
    retrieval_method: Literal["lexical", "vector", "hybrid"] = Field(
        description="How the cited chunk was retrieved."
    )


__all__ = [
    "ChunkCitation",
    "RetrievalResult",
    "RetrievalFilters",
    "LexicalRetriever",
    "VectorRetriever",
    "ChunkStore",
    "AnswerCitation",
]
