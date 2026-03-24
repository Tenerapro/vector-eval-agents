"""Parse the Florida building code source text into structured sections."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path


_MARKER_RE = re.compile(r"^\[(?P<slug>[^\]]+)\]$")
_SECTION_RE = re.compile(r"^(?P<label>(?:TABLE\s+[A-Z0-9.\-]+)|(?:[A-Z0-9][A-Z0-9.\-()]*))\s+(?P<title>.+)$")


@dataclass(frozen=True)
class ParsedSection:
    """Parsed source section."""

    section_id: str
    code_family: str
    chapter: str
    section_label: str
    title: str
    source_slug: str
    source_file: str
    text: str


def _normalize_code_family(line: str) -> str:
    upper = line.upper()
    if "BUILDING CODE, BUILDING" in upper:
        return "Building Code"
    if "BUILDING CODE, RESIDENTIAL" in upper:
        return "Residential Code"
    return line.strip()


def _looks_like_section_label(label: str) -> bool:
    if label.startswith("TABLE "):
        return True
    if not any(char.isdigit() for char in label):
        return False
    has_alpha = any(char.isalpha() for char in label)
    has_hyphen = "-" in label
    has_decimal_section = re.search(r"\d+\.\d+", label) is not None
    return has_alpha or has_hyphen or has_decimal_section


def parse_source_file(source_path: str | Path) -> list[ParsedSection]:
    """Parse the combined source text into structured sections."""
    path = Path(source_path)
    lines = path.read_text(encoding="utf-8").splitlines()

    current_code_family = "Florida Building Code"
    current_slug = "root"
    current_chapter = current_code_family
    current_title_lines: list[str] = []
    current_section_label: str | None = None
    current_section_title: str | None = None
    current_section_lines: list[str] = []
    sections: list[ParsedSection] = []
    section_counts: dict[str, int] = {}

    def flush_section() -> None:
        nonlocal current_section_label, current_section_title, current_section_lines
        if not current_section_label or not current_section_lines:
            current_section_label = None
            current_section_title = None
            current_section_lines = []
            return

        base_section_id = f"{current_slug}:{current_section_label}"
        occurrence = section_counts.get(base_section_id, 0)
        section_counts[base_section_id] = occurrence + 1
        section_id = base_section_id if occurrence == 0 else f"{base_section_id}:{occurrence}"
        sections.append(
            ParsedSection(
                section_id=section_id,
                code_family=current_code_family,
                chapter=current_chapter,
                section_label=current_section_label,
                title=current_section_title or current_section_label,
                source_slug=current_slug,
                source_file=str(path),
                text="\n".join(line for line in current_section_lines if line.strip()).strip(),
            )
        )
        current_section_label = None
        current_section_title = None
        current_section_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("2023 FLORIDA BUILDING CODE"):
            current_code_family = _normalize_code_family(stripped)
            continue
        if not stripped or stripped.startswith("=") or stripped.startswith("-"):
            continue

        marker_match = _MARKER_RE.match(stripped)
        if marker_match:
            flush_section()
            current_slug = marker_match.group("slug")
            current_title_lines = []
            current_chapter = current_code_family
            continue

        section_match = _SECTION_RE.match(stripped)
        if section_match and _looks_like_section_label(section_match.group("label")):
            flush_section()
            label = section_match.group("label").strip()
            title = section_match.group("title").strip()
            current_section_label = label
            current_section_title = title
            current_section_lines = [stripped]
            continue

        if current_section_label is None:
            current_title_lines.append(stripped)
            if current_title_lines:
                primary_title = current_title_lines[0].title()
                current_chapter = f"{current_code_family} - {primary_title}"
            continue

        current_section_lines.append(stripped)

    flush_section()
    return sections
