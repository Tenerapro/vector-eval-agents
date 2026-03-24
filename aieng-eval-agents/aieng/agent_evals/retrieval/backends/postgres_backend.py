"""Placeholder PostgreSQL retrieval backend."""

from __future__ import annotations

from aieng.agent_evals.configs import TeneraRetrievalConfig

from .. import register_backend


class PostgresRetrieverBackend:
    """Future backend bundle for PostgreSQL lexical/vector retrieval."""

    @classmethod
    def from_config(cls, settings: TeneraRetrievalConfig) -> "PostgresRetrieverBackend":
        raise NotImplementedError("The postgres retrieval backend is not implemented yet.")


register_backend("postgres", PostgresRetrieverBackend)
