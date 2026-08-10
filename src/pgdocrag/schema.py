"""Shared data contract.

Every stage reads and writes JSONL through these types. The HTML and PDF
extractors are independent implementations that must both produce `Document`
objects, which is what makes the two formats comparable downstream.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

# Block kinds
PROSE = "prose"
CODE = "code"
TABLE = "table"
ADMONITION = "admonition"
SYNOPSIS = "synopsis"

# Section kinds
SECTION = "section"
PARAMETER = "parameter"  # a single dl.variablelist entry, e.g. a GUC

# Chunk kinds
CHUNK_TYPES = (PROSE, PARAMETER, CODE, TABLE, "mixed")

HTML = "html"
PDF = "pdf"


@dataclass
class Block:
    """A leaf unit of content within a section."""

    kind: str
    text: str
    lang: str | None = None


@dataclass
class Section:
    """One node of a document's section tree, flattened to its leaf content.

    `path` is the full breadcrumb including this section's own title, so a chunk
    can render its context without walking back up the tree.
    """

    title: str
    level: int
    path: list[str]
    blocks: list[Block] = field(default_factory=list)
    anchor: str | None = None
    number: str | None = None
    kind: str = SECTION
    # Set for PDF sections so a chunk can cite the page it came from.
    page: int | None = None

    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text.strip())

    def has_code(self) -> bool:
        return any(block.kind in (CODE, SYNOPSIS) for block in self.blocks)

    def has_table(self) -> bool:
        return any(block.kind == TABLE for block in self.blocks)


@dataclass
class Document:
    """One source page (HTML) or one top-level manual section (PDF)."""

    doc_id: str
    source_format: str
    pg_version: str
    title: str
    sections: list[Section] = field(default_factory=list)
    source_url: str | None = None
    source_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    ordinal: int = 0

    def section_count(self) -> int:
        return len(self.sections)


@dataclass
class Chunk:
    """The unit that gets embedded and stored."""

    chunk_id: str
    text: str
    doc_id: str
    source_format: str
    pg_version: str
    title: str
    section_path: list[str]
    chunk_type: str
    token_count: int
    char_count: int
    content_hash: str
    ordinal: int
    source_url: str | None = None
    source_page: int | None = None
    section_anchor: str | None = None
    has_code: bool = False
    has_table: bool = False

    def breadcrumb(self) -> str:
        return " > ".join(self.section_path)

    def to_metadata(self) -> dict[str, Any]:
        """Chroma metadata values must be scalars, so the path is flattened."""
        return {
            "doc_id": self.doc_id,
            "source_format": self.source_format,
            "pg_version": self.pg_version,
            "title": self.title,
            "breadcrumb": self.breadcrumb(),
            "chunk_type": self.chunk_type,
            "token_count": self.token_count,
            "content_hash": self.content_hash,
            "ordinal": self.ordinal,
            "source_url": self.source_url or "",
            "source_page": self.source_page if self.source_page is not None else -1,
            "section_anchor": self.section_anchor or "",
            "has_code": self.has_code,
            "has_table": self.has_table,
        }


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_chunk_id(source_format: str, doc_id: str, anchor: str | None, ordinal: int) -> str:
    """Stable across runs so re-ingesting upserts rather than duplicates."""
    basis = f"{source_format}|{doc_id}|{anchor or ''}|{ordinal}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


# --- JSONL serialisation ----------------------------------------------------

T = TypeVar("T", Document, Chunk)


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def document_from_dict(payload: dict[str, Any]) -> Document:
    sections = [
        Section(
            title=section["title"],
            level=section["level"],
            path=section["path"],
            blocks=[Block(**block) for block in section.get("blocks", [])],
            anchor=section.get("anchor"),
            number=section.get("number"),
            kind=section.get("kind", SECTION),
            page=section.get("page"),
        )
        for section in payload.get("sections", [])
    ]
    return Document(
        doc_id=payload["doc_id"],
        source_format=payload["source_format"],
        pg_version=payload["pg_version"],
        title=payload["title"],
        sections=sections,
        source_url=payload.get("source_url"),
        source_path=payload.get("source_path"),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        ordinal=payload.get("ordinal", 0),
    )


def chunk_from_dict(payload: dict[str, Any]) -> Chunk:
    return Chunk(**payload)


def read_documents(path: Path) -> Iterator[Document]:
    for payload in read_jsonl(path):
        yield document_from_dict(payload)


def read_chunks(path: Path) -> Iterator[Chunk]:
    for payload in read_jsonl(path):
        yield chunk_from_dict(payload)
