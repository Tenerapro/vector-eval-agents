"""Agent-facing retrieval tool builders."""

from __future__ import annotations

from .hybrid import HybridRetriever
from .types import ChunkStore, RetrievalFilters, RetrievalResult


def _format_result(result: RetrievalResult) -> str:
    citation = result.metadata
    return (
        f"- chunk_id: {result.chunk_id}\n"
        f"  chapter: {citation.chapter}\n"
        f"  section_label: {citation.section_label}\n"
        f"  title: {citation.title}\n"
        f"  retrieval_method: {result.retrieval_method}\n"
        f"  score: {result.score:.6f}\n"
        f"  text: {result.text}"
    )


def make_retrieval_tools(retriever: HybridRetriever, store: ChunkStore) -> list:
    """Build tool functions closed over a concrete retrieval stack."""

    def search_knowledge_base(query: str, top_k: int = 10, chapter_filter: str | None = None) -> str:
        filters = RetrievalFilters(chapter=chapter_filter)
        results = retriever.search(query, top_k=top_k, filters=filters)
        if not results:
            return "No matching knowledge-base chunks found."
        return "\n\n".join(_format_result(result) for result in results)

    def get_chunk_by_id(chunk_id: str) -> str:
        result = store.get_chunk(chunk_id)
        if result is None:
            return f"Chunk '{chunk_id}' was not found."
        return _format_result(result)

    def get_section_context(section_id: str) -> str:
        results = store.get_section_chunks(section_id)
        if not results:
            return f"Section '{section_id}' was not found."
        return "\n\n".join(_format_result(result) for result in results)

    return [search_knowledge_base, get_chunk_by_id, get_section_context]
