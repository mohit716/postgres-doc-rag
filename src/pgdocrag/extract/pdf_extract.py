"""Extract normalised Documents from the PDF manual.

Written against the structural skeleton reported by `scripts/probe_pdf.py`
(see docs/structure-notes.md):

* the bookmark outline carries 4,023 entries across six levels, which supplies
  section hierarchy directly instead of inferring it from font sizes
* body text is Times-Roman, code is Courier, headings are Helvetica-Bold
* the running chapter header and the page number occupy fixed vertical bands

Scope follows the HTML corpus: only the chapters and reference pages already
collected as HTML are extracted, so the cross-format comparison comes from the
same content rather than from different subsets of the manual.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from .. import config
from ..collect.pdf_fetch import PDF_PATH
from ..normalize import (
    fence_code,
    normalize_prose,
    split_section_number,
    title_key,
)
from ..schema import (
    CODE,
    PARAMETER,
    PDF,
    PROSE,
    SECTION,
    Block,
    Document,
    Section,
    read_documents,
    write_jsonl,
)

def output_path() -> Path:
    """Resolved per call so the active corpus is honoured, not captured at import."""
    return config.INTERIM_DIR / "pdf_docs.jsonl"

# A configuration parameter as the PDF renders it: "work_mem (integer)". The
# bookmark outline stops above this level, so without recognising these lines the
# PDF side would produce whole-section chunks while the HTML side produces one
# chunk per parameter, and the two corpora would not be comparable.
_PARAM_LINE = re.compile(r"^([a-z][a-z0-9_]{2,})\s*\(([a-z][a-z ]*)\)$")

_MAX_HEADING_LOOKAHEAD = 3


@dataclass
class _Line:
    page: int
    y0: float
    y1: float
    x0: float
    text: str
    mono: bool
    heading: bool
    size: float
    # A parameter heading such as "work_mem (integer)" sets the name in Courier
    # but the surrounding parentheses in Times-Roman, so it is not fully mono.
    starts_mono: bool = False


@dataclass
class _Entry:
    """One bookmark outline entry with the page range it owns."""

    level: int
    title: str
    page: int  # 1-based, as reported by the outline
    index: int
    end_page: int = 0
    path: list[str] = field(default_factory=list)


def _is_mono(font: str) -> bool:
    lowered = font.lower()
    return any(hint in lowered for hint in config.PDF_MONO_FONT_HINTS)


def _is_heading_font(font: str) -> bool:
    """Headings are set in Helvetica-Bold; inline emphasis uses Times-Bold."""
    lowered = font.lower()
    return "helvetica" in lowered and "bold" in lowered


def _page_lines(page) -> list[_Line]:
    height = page.rect.height
    lines: list[_Line] = []

    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [span for span in line.get("spans", []) if span.get("text")]
            if not spans:
                continue
            text = "".join(span["text"] for span in spans)
            if not text.strip():
                continue

            bbox = line["bbox"]
            relative_y = bbox[1] / height
            if relative_y < config.PDF_HEADER_BAND or relative_y > config.PDF_FOOTER_BAND:
                continue

            total_chars = sum(len(span["text"]) for span in spans)
            mono_chars = sum(
                len(span["text"]) for span in spans if _is_mono(span["font"])
            )
            heading_chars = sum(
                len(span["text"]) for span in spans if _is_heading_font(span["font"])
            )
            size = max(span["size"] for span in spans)
            lines.append(
                _Line(
                    page=page.number,
                    y0=bbox[1],
                    y1=bbox[3],
                    x0=bbox[0],
                    text=text.rstrip(),
                    mono=mono_chars == total_chars,
                    heading=heading_chars == total_chars and size > config.PDF_BODY_SIZE,
                    size=size,
                    starts_mono=_is_mono(spans[0]["font"]),
                )
            )

    lines.sort(key=lambda line: (line.y0, line.x0))
    return lines


class _LineCache:
    """Lazily extract and retain lines per page; pages are visited repeatedly."""

    def __init__(self, doc) -> None:
        self._doc = doc
        self._cache: dict[int, list[_Line]] = {}

    def get(self, page_index: int) -> list[_Line]:
        if page_index not in self._cache:
            if 0 <= page_index < self._doc.page_count:
                self._cache[page_index] = _page_lines(self._doc[page_index])
            else:
                self._cache[page_index] = []
        return self._cache[page_index]


# --- block assembly ---------------------------------------------------------


def _paragraphs(lines: list[_Line]) -> list[Block]:
    """Group prose lines into paragraphs using vertical spacing."""
    if not lines:
        return []

    gaps = [
        second.y0 - first.y0
        for first, second in zip(lines, lines[1:])
        if second.page == first.page and second.y0 > first.y0
    ]
    typical_gap = median(gaps) if gaps else 12.0
    threshold = typical_gap * config.PDF_PARAGRAPH_GAP_RATIO

    blocks: list[Block] = []
    current: list[str] = []
    previous: _Line | None = None

    for line in lines:
        starts_paragraph = previous is not None and (
            line.page != previous.page or (line.y0 - previous.y0) > threshold
        )
        if starts_paragraph and current:
            blocks.append(Block(kind=PROSE, text="\n".join(current)))
            current = []
        current.append(line.text)
        previous = line

    if current:
        blocks.append(Block(kind=PROSE, text="\n".join(current)))

    # normalize_prose rejoins the wrapped lines and repairs hyphenation, which is
    # the step that makes PDF text comparable to the HTML rendering.
    return [
        Block(kind=PROSE, text=normalize_prose(block.text, from_pdf=True))
        for block in blocks
        if block.text.strip()
    ]


def _blocks_from_lines(lines: list[_Line]) -> list[Block]:
    """Split lines into code runs and prose paragraphs."""
    blocks: list[Block] = []
    buffer: list[_Line] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line.mono:
            buffer.append(line)
            index += 1
            continue

        run_end = index
        while run_end < len(lines) and lines[run_end].mono:
            run_end += 1
        run = lines[index:run_end]

        if len(run) >= config.PDF_MIN_CODE_RUN:
            blocks.extend(_paragraphs(buffer))
            buffer = []
            code = "\n".join(item.text for item in run)
            fenced = fence_code(code)
            if fenced:
                blocks.append(Block(kind=CODE, text=fenced))
        else:
            buffer.extend(run)
        index = run_end

    blocks.extend(_paragraphs(buffer))
    return blocks


# --- parameter splitting ----------------------------------------------------


def _guc_anchor(name: str) -> str:
    """Derive the anchor the HTML docs use for the same parameter.

    The PDF carries no anchors. Reconstructing the HTML form gives the two
    formats a shared join key, so the same evaluation gold set can score both
    collections.
    """
    return f"GUC-{name.upper().replace('_', '-')}"


def _split_parameters(
    lines: list[_Line], parent_path: list[str], level: int, page: int
) -> tuple[list[_Line], list[Section]]:
    """Peel configuration parameters out of a section's lines.

    Returns the lines that belong to the section itself plus one section per
    parameter found.
    """
    starts: list[tuple[int, str, str]] = []
    for index, line in enumerate(lines):
        if not line.starts_mono:
            continue
        match = _PARAM_LINE.match(line.text.strip())
        if match:
            starts.append((index, match.group(1), line.text.strip()))

    if not starts:
        return lines, []

    own_lines = lines[: starts[0][0]]
    sections: list[Section] = []
    for position, (start_index, name, title) in enumerate(starts):
        end_index = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        body_lines = lines[start_index + 1 : end_index]
        blocks = _blocks_from_lines(body_lines)
        if not blocks:
            continue
        sections.append(
            Section(
                title=title,
                level=level,
                path=parent_path + [title],
                blocks=blocks,
                anchor=_guc_anchor(name),
                kind=PARAMETER,
                page=(body_lines[0].page + 1) if body_lines else page,
            )
        )
    return own_lines, sections


# --- outline handling -------------------------------------------------------


def _outline_entries(doc) -> list[_Entry]:
    raw = doc.get_toc(simple=True)
    entries = [
        _Entry(level=level, title=title.strip(), page=page, index=index)
        for index, (level, title, page) in enumerate(raw)
        if title and title.strip()
    ]

    stack: list[_Entry] = []
    for entry in entries:
        while stack and stack[-1].level >= entry.level:
            stack.pop()
        _, bare = split_section_number(entry.title)
        entry.path = [ancestor_title for ancestor_title in (
            split_section_number(item.title)[1] for item in stack
        )] + [bare]
        stack.append(entry)

    for position, entry in enumerate(entries):
        end_page = doc.page_count
        for following in entries[position + 1 :]:
            if following.level <= entry.level:
                end_page = following.page
                break
        entry.end_page = end_page
    return entries


@dataclass
class _Boundary:
    """Where a section's heading sits on the page.

    Two coordinates are needed: content of this section starts below the heading
    (`content_y`), while the previous section ends above it (`heading_y`). Using
    one value for both leaks each heading into the section before it.
    """

    page: int
    heading_y: float
    content_y: float


def _locate_heading(entry: _Entry, cache: _LineCache) -> _Boundary:
    """Find where an outline entry's content actually starts.

    The outline gives a page but its stored coordinate uses a different origin
    than the text layer, so the heading line is located by matching its text.
    """
    page_index = entry.page - 1
    key = title_key(entry.title)

    for line in cache.get(page_index):
        line_key = title_key(line.text)
        if not line_key:
            continue
        if line_key == key or (len(line_key) >= 8 and key.startswith(line_key)):
            return _Boundary(page=page_index, heading_y=line.y0, content_y=line.y1)
    return _Boundary(page=page_index, heading_y=0.0, content_y=0.0)


def _split_headings(lines: list[_Line]) -> list[tuple[str | None, list[_Line]]]:
    """Split a run of lines on headings the bookmark outline does not include.

    The outline omits some headings entirely — "Synopsis" on every command
    reference page, for instance — so those sections would otherwise be absorbed
    into their parent and lose the breadcrumb the HTML side gives them.
    """
    groups: list[tuple[str | None, list[_Line]]] = []
    title: str | None = None
    current: list[_Line] = []

    for line in lines:
        if line.heading and line.text.strip():
            groups.append((title, current))
            title = line.text.strip()
            current = []
        else:
            current.append(line)
    groups.append((title, current))

    return [group for group in groups if group[1]]


def _collect_lines(
    cache: _LineCache,
    start: tuple[int, float],
    end: tuple[int, float],
) -> list[_Line]:
    start_page, start_y = start
    end_page, end_y = end
    collected: list[_Line] = []

    for page_index in range(start_page, min(end_page, start_page + 400) + 1):
        for line in cache.get(page_index):
            if page_index == start_page and line.y0 < start_y:
                continue
            if page_index == end_page and end_y > 0 and line.y0 >= end_y:
                continue
            collected.append(line)
    return collected


@dataclass
class _Wanted:
    """An HTML document to find the PDF counterpart of."""

    title: str
    # Chapter the HTML breadcrumb sits under, used to disambiguate titles that
    # occur more than once in the manual ("Error Handling" appears under Server
    # Configuration, PL/pgSQL and ECPG).
    context_key: str | None


def _html_wanted() -> dict[str, _Wanted]:
    html_docs = config.INTERIM_DIR / "html_docs.jsonl"
    if not html_docs.exists():
        return {}

    wanted: dict[str, _Wanted] = {}
    for document in read_documents(html_docs):
        if not document.title:
            continue
        # Navigation-only pages (a chapter's table of contents) carry no content
        # of their own. Matching them would pull the entire chapter out of the
        # PDF, when what mirrors the HTML corpus is each individual subsection.
        has_content = any(
            block.text.strip() for section in document.sections for block in section.blocks
        )
        if not has_content:
            continue

        own_key = title_key(document.title)
        context_key = None
        for section in document.sections:
            if len(section.path) > 1:
                candidate = title_key(section.path[0])
                # On reference pages the breadcrumb starts with the page's own
                # title, which says nothing about where the page sits.
                if candidate and candidate != own_key:
                    context_key = candidate
                break

        wanted[own_key] = _Wanted(title=document.title, context_key=context_key)
    return wanted


def _matches(entry: _Entry, wanted: dict[str, _Wanted]) -> bool:
    target = wanted.get(title_key(entry.title))
    if target is None:
        return False
    if target.context_key is None:
        return True
    ancestor_keys = {title_key(title) for title in entry.path[:-1]}
    return target.context_key in ancestor_keys


def _encloses(outer: _Entry, inner: _Entry) -> bool:
    return (
        outer is not inner
        and outer.page <= inner.page
        and inner.end_page <= outer.end_page
        and outer.level < inner.level
    )


def _select_roots(entries: list[_Entry], wanted: dict[str, _Wanted]) -> list[_Entry]:
    """Pick the outline entries mirroring the HTML corpus.

    Matched entries nest — a matched chapter can contain matched subsections — and
    `config.PDF_SCOPE_RULE` decides which survives. Keeping the innermost entries
    mirrors the HTML side more closely, since HTML excludes navigation-only parent
    pages for the same reason, at the cost of skipping any chapter preamble that
    sits above the first matched subsection.
    """
    matched = [entry for entry in entries if _matches(entry, wanted)]

    if config.PDF_SCOPE_RULE == "innermost":
        return [
            entry
            for entry in matched
            if not any(_encloses(entry, other) for other in matched)
        ]
    return [
        entry for entry in matched if not any(_encloses(other, entry) for other in matched)
    ]


# --- document assembly ------------------------------------------------------


def _build_document(
    root: _Entry,
    entries: list[_Entry],
    cache: _LineCache,
) -> Document:
    _, root_title = split_section_number(root.title)
    descendants = [
        entry
        for entry in entries
        if entry.index > root.index
        and entry.page < root.end_page
        and entry.level > root.level
    ]

    boundaries = [(entry, _locate_heading(entry, cache)) for entry in [root] + descendants]

    sections: list[Section] = []
    for position, (entry, start) in enumerate(boundaries):
        if position + 1 < len(boundaries):
            following = boundaries[position + 1][1]
            end = (following.page, following.heading_y)
        else:
            end = (root.end_page - 1, 0.0)

        lines = _collect_lines(cache, (start.page, start.content_y), end)
        if not lines:
            continue

        # Paths are rebased on the selected root so PDF breadcrumbs line up with
        # the HTML ones, which start at the chapter rather than the part.
        depth = entry.level - root.level
        path = [root_title] + entry.path[len(root.path) :] if depth else [root_title]
        number, bare_title = split_section_number(entry.title)

        for sub_title, sub_lines in _split_headings(lines):
            if sub_title is None:
                section_path, section_title, level = path, bare_title, entry.level
            else:
                _, sub_bare = split_section_number(sub_title)
                section_path = path + [sub_bare]
                section_title = sub_bare
                level = entry.level + 1

            page = (sub_lines[0].page + 1) if sub_lines else start.page + 1
            own_lines, parameters = _split_parameters(
                sub_lines, section_path, level + 1, page
            )
            blocks = _blocks_from_lines(own_lines)

            if blocks:
                sections.append(
                    Section(
                        title=section_title,
                        level=level,
                        path=section_path,
                        blocks=blocks,
                        number=number if sub_title is None else None,
                        kind=SECTION,
                        page=page,
                    )
                )
            sections.extend(parameters)

    return Document(
        # The page is part of the id because a title can legitimately repeat
        # across chapters, and colliding ids would overwrite each other.
        doc_id=f"pdf:{title_key(root.title) or 'entry'}:p{root.page}",
        source_format=PDF,
        pg_version=config.PG_VERSION,
        title=root_title,
        sections=sections,
        source_path=str(PDF_PATH),
        page_start=root.page,
        page_end=root.end_page,
        ordinal=root.index,
    )


def extract_all(*, verbose: bool = True) -> Path:
    import pymupdf

    config.ensure_dirs()
    if not PDF_PATH.exists():
        raise RuntimeError(f"PDF not found at {PDF_PATH}. Run: pgdocrag collect --source pdf")

    wanted = _html_wanted()
    if not wanted:
        raise RuntimeError(
            "No HTML documents found. The PDF scope mirrors the HTML corpus, so "
            "run the HTML collect and extract stages first."
        )

    doc = pymupdf.open(PDF_PATH)
    cache = _LineCache(doc)
    entries = _outline_entries(doc)
    roots = _select_roots(entries, wanted)

    if verbose:
        print(f"  outline entries   {len(entries)}")
        print(f"  html titles       {len(wanted)}")
        print(f"  matched roots     {len(roots)}")
        matched_keys = {title_key(root.title) for root in roots}
        missing = sorted(
            target.title for key, target in wanted.items() if key not in matched_keys
        )
        if missing:
            preview = ", ".join(missing[:6])
            print(f"  unmatched titles  {len(missing)} ({preview}...)")

    documents: list[Document] = []
    for root in roots:
        document = _build_document(root, entries, cache)
        documents.append(document)
        if verbose:
            pages = f"p{root.page}-{root.end_page}"
            print(
                f"  {document.title[:44]:<44} {pages:>14} "
                f"{len(document.sections):>4} sections"
            )

    destination = output_path()
    written = write_jsonl(destination, documents)
    doc.close()
    if verbose:
        print(f"\nWrote {written} documents to {destination}")
    return destination
