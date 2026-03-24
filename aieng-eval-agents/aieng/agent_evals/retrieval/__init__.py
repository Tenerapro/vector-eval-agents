"""Backend-swappable retrieval layer for knowledge agents."""

from __future__ import annotations

from typing import Protocol

from aieng.agent_evals.configs import Configs, TeneraRetrievalConfig

from .fusion import RRFFusion, WeightedFusion
from .hybrid import HybridRetriever
from .types import ChunkStore, LexicalRetriever, VectorRetriever


class RetrieverBackend(Protocol):
    """Factory contract for retrieval backend bundles."""

    @classmethod
    def from_config(cls, settings: TeneraRetrievalConfig) -> "RetrieverBackend":
        """Build the backend from config."""

    def lexical_retriever(self) -> LexicalRetriever:
        """Return the lexical retriever."""

    def vector_retriever(self) -> VectorRetriever:
        """Return the vector retriever."""

    def chunk_store(self) -> ChunkStore:
        """Return the chunk store."""


_BACKEND_REGISTRY: dict[str, type[RetrieverBackend]] = {}


def register_backend(name: str, cls: type[RetrieverBackend]) -> None:
    """Register a retrieval backend."""
    _BACKEND_REGISTRY[name] = cls


def create_retriever(settings: TeneraRetrievalConfig | None = None) -> tuple[HybridRetriever, ChunkStore]:
    """Create the configured hybrid retriever and chunk store."""
    resolved_settings = settings or Configs().tenera_retrieval or TeneraRetrievalConfig()
    backend_cls = _BACKEND_REGISTRY[resolved_settings.backend]
    backend = backend_cls.from_config(resolved_settings)

    fusion = RRFFusion()
    if resolved_settings.fusion_mode == "weighted":
        fusion = WeightedFusion(
            lexical_weight=resolved_settings.lexical_weight,
            vector_weight=resolved_settings.vector_weight,
        )

    hybrid = HybridRetriever(
        lexical=backend.lexical_retriever(),
        vector=backend.vector_retriever(),
        fusion=fusion,
    )
    return hybrid, backend.chunk_store()


# Import backends so they self-register.
from .backends import http_backend as _http_backend  # noqa: F401,E402
from .backends import postgres_backend as _postgres_backend  # noqa: F401,E402
from .backends import sqlite_backend as _sqlite_backend  # noqa: F401,E402


__all__ = [
    "RetrieverBackend",
    "register_backend",
    "create_retriever",
    "HybridRetriever",
    "ChunkStore",
    "LexicalRetriever",
    "VectorRetriever",
]
