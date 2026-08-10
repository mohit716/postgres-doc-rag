"""Stage 3: turn documents into section-aware chunks.

Chunk boundaries follow the document's own structure rather than a fixed token
stride. The unit is a leaf section, or a single dl.variablelist entry for
configuration parameters and command clauses. Sections too small to retrieve well
are merged into their predecessor; sections too large for the embedding model are
split on natural boundaries without ever cutting through a fenced code block or a
markdown table header.

Every chunk is prefixed with its breadcrumb, so an embedded chunk carries the
context ("Server Configuration > Connection Settings > listen_addresses") that
its body text alone would not express.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..normalize import collapse_blank_lines
from ..schema import (
    CODE,
    PARAMETER,
    PROSE,
    SYNOPSIS,
    TABLE,
    Chunk,
    Document,
    Section,
    content_hash,
    make_chunk_id,
    read_documents,
    write_jsonl,
)
from .tokenizer import count_tokens

# Sentence boundary: period/question/exclamation followed by whitespace and the
# start of something that looks like a new sentence.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[`\"])")

# Special tokens the model adds around every input.
_SPECIAL_TOKEN_BUDGET = 4


@dataclass
class _Atom:
    """An indivisible-by-default piece of content with its token cost."""

    text: str
    tokens: int
    kind: str


@dataclass
class _Piece:
    """A prospective chunk: one section, or one part of an oversized section."""

    section: Section
    document: Document
    atoms: list[_Atom] = field(default_factory=list)
    part_index: int = 0
    part_count: int = 1

    @property
    def body(self) -> str:
        return collapse_blank_lines("\n\n".join(atom.text for atom in self.atoms))

    @property
    def tokens(self) -> int:
        return sum(atom.tokens for atom in self.atoms)

    @property
    def kinds(self) -> set[str]:
        return {atom.kind for atom in self.atoms}


# --- splitting primitives ---------------------------------------------------


def _pack_by_tokens(units: list[str], budget: int, joiner: str) -> list[str]:
    """Greedily group units so each group fits the budget."""
    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > budget:
            groups.append(joiner.join(current))
            current, current_tokens = [], 0
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        groups.append(joiner.join(current))
    return groups


def _split_words(text: str, budget: int) -> list[str]:
    """Last resort for a single unsplittable run longer than the budget."""
    return _pack_by_tokens(text.split(), budget, " ")


def _split_prose(text: str, budget: int) -> list[str]:
    # Overlap is prepended after packing, so its cost must be reserved up front
    # or every part but the first would overflow the budget.
    overlap_budget = (
        int(budget * config.CHUNK_OVERLAP_RATIO) if config.CHUNK_OVERLAP_RATIO > 0 else 0
    )
    pack_budget = max(budget - overlap_budget, 16)

    safe: list[str] = []
    for sentence in _SENTENCE_BREAK.split(text):
        if count_tokens(sentence) > pack_budget:
            safe.extend(_split_words(sentence, pack_budget))
        else:
            safe.append(sentence)

    parts = _pack_by_tokens(safe, pack_budget, " ")
    return _apply_overlap(parts, overlap_budget)


def _apply_overlap(parts: list[str], overlap_budget: int) -> list[str]:
    """Prepend trailing context from the previous part.

    Overlap is only introduced where a section had to be cut, so it costs nothing
    on the majority of chunks that fit whole.
    """
    if len(parts) < 2 or overlap_budget <= 0:
        return parts

    result = [parts[0]]
    for previous, current in zip(parts, parts[1:]):
        tail: list[str] = []
        tail_tokens = 0
        for sentence in reversed(_SENTENCE_BREAK.split(previous)):
            sentence_tokens = count_tokens(sentence)
            if tail_tokens + sentence_tokens > overlap_budget:
                break
            tail.insert(0, sentence)
            tail_tokens += sentence_tokens
        result.append(" ".join(tail + [current]) if tail else current)
    return result


def _split_code(text: str, budget: int) -> list[str]:
    """Split a fenced block by lines, re-fencing each part.

    Splitting inside the fence rather than on blank lines keeps every part valid
    and prevents a code sample from bleeding into surrounding prose.
    """
    lines = text.split("\n")
    fence_open = lines[0] if lines and lines[0].startswith("```") else "```"
    fence_close = "```"
    inner = lines[1:-1] if len(lines) > 2 and lines[-1].strip() == fence_close else lines

    overhead = count_tokens(f"{fence_open}\n{fence_close}")
    parts = _pack_by_tokens(inner, max(budget - overhead, 16), "\n")
    return [f"{fence_open}\n{part}\n{fence_close}" for part in parts if part.strip()]


def _split_table(text: str, budget: int) -> list[str]:
    """Split a markdown table by rows, repeating the header on each part."""
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) < 2:
        return _split_prose(text, budget)

    header, separator, *rows = lines
    if not set(separator.replace("|", "").strip()) <= {"-", " "}:
        header, separator, rows = lines[0], None, lines[1:]

    preamble = [header] + ([separator] if separator else [])
    overhead = count_tokens("\n".join(preamble))
    row_groups = _pack_by_tokens(rows, max(budget - overhead, 16), "\n")
    return ["\n".join(preamble + [group]) for group in row_groups]


def _atomise(section: Section, budget: int) -> list[_Atom]:
    """Convert a section's blocks into atoms that each fit the budget."""
    atoms: list[_Atom] = []
    for block in section.blocks:
        text = block.text.strip()
        if not text:
            continue
        tokens = count_tokens(text)
        if tokens <= budget:
            atoms.append(_Atom(text=text, tokens=tokens, kind=block.kind))
            continue

        if block.kind in (CODE, SYNOPSIS):
            parts = _split_code(text, budget)
        elif block.kind == TABLE:
            parts = _split_table(text, budget)
        else:
            parts = _split_prose(text, budget)

        for part in parts:
            atoms.append(_Atom(text=part, tokens=count_tokens(part), kind=block.kind))
    return _enforce_budget(atoms, budget)


def _enforce_budget(atoms: list[_Atom], budget: int) -> list[_Atom]:
    """Guarantee the token ceiling regardless of which splitter produced an atom.

    Token counts are not additive, so a splitter that packs units to a budget can
    still emit a slightly oversized result. This makes the ceiling a property of
    the pipeline rather than of each splitter being individually exact.
    """
    enforced: list[_Atom] = []
    for atom in atoms:
        if atom.tokens <= budget:
            enforced.append(atom)
            continue
        for part in _split_words(atom.text, budget):
            enforced.append(_Atom(text=part, tokens=count_tokens(part), kind=atom.kind))
    return enforced


# --- chunk assembly ---------------------------------------------------------


def _breadcrumb(section: Section, document: Document) -> str:
    parts = [part for part in section.path if part] or [document.title]
    return " > ".join(parts)


def _compose(breadcrumb: str, body: str) -> str:
    return f"{breadcrumb}\n\n{body}" if body else breadcrumb


def _chunk_type(piece: _Piece) -> str:
    if piece.section.kind == PARAMETER:
        return PARAMETER
    kinds = piece.kinds
    if not kinds:
        return PROSE
    if kinds <= {CODE, SYNOPSIS}:
        return CODE
    if kinds == {TABLE}:
        return TABLE
    if len(kinds) > 1:
        return "mixed"
    return PROSE


def _pieces_for_section(section: Section, document: Document) -> list[_Piece]:
    breadcrumb = _breadcrumb(section, document)
    breadcrumb_tokens = count_tokens(breadcrumb)
    body_budget = max(
        config.MIN_CHUNK_TOKENS,
        config.MODEL_MAX_TOKENS - breadcrumb_tokens - _SPECIAL_TOKEN_BUDGET,
    )
    target = min(config.TARGET_CHUNK_TOKENS, body_budget)

    atoms = _atomise(section, body_budget)
    if not atoms:
        return []

    groups: list[list[_Atom]] = []
    current: list[_Atom] = []
    current_tokens = 0
    for atom in atoms:
        if current and current_tokens + atom.tokens > target:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(atom)
        current_tokens += atom.tokens
    if current:
        groups.append(current)

    return [
        _Piece(
            section=section,
            document=document,
            atoms=group,
            part_index=index,
            part_count=len(groups),
        )
        for index, group in enumerate(groups)
    ]


def _merge_small_pieces(pieces: list[_Piece]) -> list[_Piece]:
    """Fold undersized sections into the preceding chunk.

    Parameter sections are never merged: their anchor is the reason they exist,
    and merging would cost both retrieval precision and the ability to cite an
    exact URL.
    """
    merged: list[_Piece] = []
    for piece in pieces:
        is_mergeable = (
            piece.section.kind != PARAMETER
            and piece.part_count == 1
            and piece.tokens < config.MIN_CHUNK_TOKENS
        )
        if is_mergeable and merged:
            previous = merged[-1]
            combined = previous.tokens + piece.tokens
            same_scope = previous.section.kind != PARAMETER
            if same_scope and combined <= config.TARGET_CHUNK_TOKENS:
                title = piece.section.title
                text = f"{title}\n{piece.body}" if title else piece.body
                previous.atoms.append(
                    _Atom(text=text, tokens=count_tokens(text), kind=PROSE)
                )
                continue
        merged.append(piece)
    return merged


def chunk_document(document: Document) -> list[Chunk]:
    pieces: list[_Piece] = []
    for section in document.sections:
        pieces.extend(_pieces_for_section(section, document))

    pieces = _merge_small_pieces(pieces)

    chunks: list[Chunk] = []
    for ordinal, piece in enumerate(pieces):
        section = piece.section
        breadcrumb = _breadcrumb(section, document)
        text = _compose(breadcrumb, piece.body)
        anchor = section.anchor
        url = document.source_url
        if url and anchor:
            url = f"{url}#{anchor}"

        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(
                    document.source_format, document.doc_id, anchor, ordinal
                ),
                text=text,
                doc_id=document.doc_id,
                source_format=document.source_format,
                pg_version=document.pg_version,
                title=section.title or document.title,
                section_path=[part for part in section.path if part],
                chunk_type=_chunk_type(piece),
                token_count=count_tokens(text),
                char_count=len(text),
                content_hash=content_hash(text),
                ordinal=ordinal,
                source_url=url,
                source_page=section.page if section.page is not None else document.page_start,
                section_anchor=anchor,
                has_code=bool(piece.kinds & {CODE, SYNOPSIS}),
                has_table=TABLE in piece.kinds,
            )
        )
    return chunks


def chunk_source(source_format: str, *, verbose: bool = True) -> Path:
    config.ensure_dirs()
    docs_path = config.INTERIM_DIR / f"{source_format}_docs.jsonl"
    if not docs_path.exists():
        raise RuntimeError(f"No extracted documents at {docs_path}. Run extract first.")

    output_path = config.CHUNKS_DIR / f"{source_format}_chunks.jsonl"
    chunks: list[Chunk] = []
    for document in read_documents(docs_path):
        chunks.extend(chunk_document(document))

    write_jsonl(output_path, chunks)
    if verbose:
        _report(chunks, output_path)
    return output_path


def _report(chunks: list[Chunk], output_path: Path) -> None:
    from .tokenizer import get_counter

    if not chunks:
        print("  no chunks produced")
        return

    tokens = sorted(chunk.token_count for chunk in chunks)
    by_type: dict[str, int] = {}
    for chunk in chunks:
        by_type[chunk.chunk_type] = by_type.get(chunk.chunk_type, 0) + 1

    over_limit = sum(1 for value in tokens if value > config.MODEL_MAX_TOKENS)
    anchored = sum(1 for chunk in chunks if chunk.section_anchor)

    print(f"  chunks            {len(chunks)}")
    print(f"  token counting    {get_counter().describe()}")
    print(
        f"  tokens            min {tokens[0]}  median {int(statistics.median(tokens))}  "
        f"p95 {tokens[int(len(tokens) * 0.95) - 1]}  max {tokens[-1]}"
    )
    print(f"  over model limit  {over_limit}")
    print(f"  with anchor       {anchored} ({anchored / len(chunks):.0%})")
    print("  by type           " + ", ".join(
        f"{name}={count}" for name, count in sorted(by_type.items())
    ))
    print(f"  wrote             {output_path}")
