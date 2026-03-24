"""Chunk parsed sections for indexing."""

from dataclasses import dataclass

from .parser import ParsedSection


@dataclass(frozen=True)
class ChunkRecord:
    """A chunk derived from a parsed section."""

    chunk_id: str
    section_id: str
    chapter: str
    section_label: str
    title: str
    source_file: str
    chunk_index: int
    text: str


def chunk_sections(sections: list[ParsedSection], max_chars: int = 1600) -> list[ChunkRecord]:
    """Split sections into sized chunks while preserving metadata."""
    chunks: list[ChunkRecord] = []
    for section in sections:
        paragraphs = [part.strip() for part in section.text.split("\n\n") if part.strip()]
        current_parts: list[str] = []
        current_length = 0
        chunk_index = 0

        def flush_chunk() -> None:
            nonlocal current_parts, current_length, chunk_index
            if not current_parts:
                return
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{section.section_id}#{chunk_index}",
                    section_id=section.section_id,
                    chapter=section.chapter,
                    section_label=section.section_label,
                    title=section.title,
                    source_file=section.source_file,
                    chunk_index=chunk_index,
                    text="\n\n".join(current_parts).strip(),
                )
            )
            chunk_index += 1
            current_parts = []
            current_length = 0

        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            if current_parts and current_length + paragraph_length > max_chars:
                flush_chunk()
            current_parts.append(paragraph)
            current_length += paragraph_length

        flush_chunk()

    return chunks
