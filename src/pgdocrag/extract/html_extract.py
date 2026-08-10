"""Extract normalised Documents from cached PostgreSQL HTML docs.

Written against the structural skeleton reported by `scripts/probe_structure.py`
(see docs/structure-notes.md), not against page content. Two container families
appear in the manual and both are handled here:

* chapter pages     div.sect1 > div.sect2, headings in div.titlepage as h2.title
* reference pages   div.refentry > div.refsect1, headings as bare h2/h3

Both converge on dl.variablelist, whose dt elements carry stable anchors
(GUC-LISTEN-ADDRESSES, SQL-CREATETABLE-TEMPORARY). Each dt/dd pair is promoted
to its own section so a single configuration parameter or clause becomes an
independently retrievable unit.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, NavigableString, Tag

from .. import config
from ..collect.html_crawler import PageRecord, cache_path, load_manifest
from ..normalize import (
    collapse_blank_lines,
    fence_code,
    markdown_table,
    normalize_code,
    normalize_prose,
    split_section_number,
)
from ..schema import (
    ADMONITION,
    CODE,
    HTML,
    PARAMETER,
    PROSE,
    SECTION,
    SYNOPSIS,
    TABLE,
    Block,
    Document,
    Section,
    write_jsonl,
)

OUTPUT_PATH = config.INTERIM_DIR / "html_docs.jsonl"

# refnamediv is deliberately absent: its heading duplicates the refentry title,
# so it is folded into the parent rather than becoming its own section.
_SECTION_CLASS = re.compile(
    r"^(sect[1-5]|refsect[1-3]|refentry|refsynopsisdiv|chapter|preface|"
    r"appendix|part|glossdiv|bibliodiv|article)$"
)

# div.toc is a per-page table of contents; a.id_link is the "#" permalink glyph
# that would otherwise be appended to every heading.
_NOISE_SELECTORS = (
    "div.navheader",
    "div.navfooter",
    "div.toc",
    "div.footnotes",
    "a.id_link",
    "a.indexterm",
    "script",
    "style",
)

_ADMONITIONS = {
    "note": "Note",
    "warning": "Warning",
    "tip": "Tip",
    "caution": "Caution",
    "important": "Important",
}

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_CODE_TAGS = ("code", "tt", "kbd", "samp")

_TITLE_PREFIX = re.compile(r"^PostgreSQL:\s*Documentation:\s*[^:]+:\s*(.*)$", re.IGNORECASE)
_GENERIC_TITLE = re.compile(r"documentation$", re.IGNORECASE)
_MAX_ANCESTOR_DEPTH = 8


# --- helpers ----------------------------------------------------------------


def clean_nav_title(title: str) -> str:
    """Strip the site's "PostgreSQL: Documentation: 18:" title prefix."""
    match = _TITLE_PREFIX.match(title.strip())
    return (match.group(1) if match else title).strip()


def _classes(tag: Tag) -> list[str]:
    value = tag.get("class") or []
    return [value] if isinstance(value, str) else list(value)


def _is_section(tag: Tag) -> bool:
    return tag.name == "div" and any(_SECTION_CLASS.match(c) for c in _classes(tag))


def _strip_noise(root: Tag) -> None:
    for selector in _NOISE_SELECTORS:
        for tag in root.select(selector):
            tag.decompose()


def _inline_text(node, *, in_code: bool = False) -> str:
    """Flatten inline markup, marking code spans with backticks.

    Inline code carries real signal in these docs (parameter names, types, file
    paths), so it survives into the embedded text; hyperlink targets do not.
    """
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""
    if node.name == "br":
        return "\n"
    if node.name in _CODE_TAGS and not in_code:
        inner = "".join(_inline_text(child, in_code=True) for child in node.children).strip()
        if not inner or "`" in inner:
            return inner
        return f"`{inner}`"
    return "".join(_inline_text(child, in_code=in_code) for child in node.children)


def _prose(tag: Tag) -> str:
    return normalize_prose(_inline_text(tag))


def _plain(tag: Tag) -> str:
    """Text without inline code marking, for headings and terms.

    Titles flow into breadcrumbs and result listings, where backticks around
    every parameter name are noise rather than signal.
    """
    return normalize_prose(tag.get_text(" "))


def _section_title(container: Tag) -> str:
    """Resolve a container's heading across both structural families."""
    titlepage = container.find("div", class_="titlepage", recursive=False)
    if titlepage is not None:
        heading = titlepage.find(_HEADING_TAGS)
        if heading is not None:
            text = _plain(heading)
            if text:
                return text

    if "refentry" in _classes(container):
        refname = container.find("div", class_="refnamediv", recursive=False)
        if refname is not None:
            heading = refname.find(_HEADING_TAGS)
            if heading is not None:
                return _plain(heading)

    for child in container.find_all(_HEADING_TAGS, recursive=False):
        text = _plain(child)
        if text:
            return text
    return ""


# --- block rendering --------------------------------------------------------


def _render_table(table: Tag) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if cells:
            rows.append([_prose(cell) for cell in cells])
    has_header = table.find("thead") is not None or bool(table.find("th"))
    return markdown_table(rows, has_header=has_header)


def _render_list(element: Tag, *, ordered: bool, depth: int = 0) -> str:
    lines: list[str] = []
    for index, item in enumerate(element.find_all("li", recursive=False), start=1):
        marker = f"{index}." if ordered else "-"
        own_text = normalize_prose(
            " ".join(
                _inline_text(child)
                for child in item.children
                if not (isinstance(child, Tag) and child.name in ("ul", "ol"))
            )
        )
        indent = "  " * depth
        if own_text:
            lines.append(f"{indent}{marker} {own_text}")
        for nested in item.find_all(["ul", "ol"], recursive=False):
            nested_text = _render_list(nested, ordered=nested.name == "ol", depth=depth + 1)
            if nested_text:
                lines.append(nested_text)
    return "\n".join(lines)


def _render_admonition(element: Tag, label: str) -> str:
    heading = element.find(_HEADING_TAGS)
    title = _prose(heading) if heading is not None else ""
    if heading is not None:
        heading.extract()
    body = normalize_prose(_inline_text(element))
    prefix = title if title and title.lower() != label.lower() else label
    return f"{prefix}: {body}" if body else ""


def _render_element(element: Tag) -> list[Block]:
    """Convert one direct child of a section container into blocks."""
    classes = set(_classes(element))
    name = element.name

    if name == "div" and ("titlepage" in classes or "toc" in classes):
        return []

    # Handled by the caller: promoted to standalone parameter sections.
    if name == "dl" and "variablelist" in classes:
        return []
    if name == "div" and "variablelist" in classes:
        return []

    if name == "p":
        text = _prose(element)
        return [Block(kind=PROSE, text=text)] if text else []

    if name == "pre":
        kind = SYNOPSIS if "synopsis" in classes else CODE
        text = fence_code(element.get_text(), lang="sql" if kind == SYNOPSIS else None)
        return [Block(kind=kind, text=text)] if text else []

    if name == "table":
        text = _render_table(element)
        return [Block(kind=TABLE, text=text)] if text else []

    if name in ("ul", "ol"):
        text = _render_list(element, ordered=name == "ol")
        return [Block(kind=PROSE, text=text)] if text else []

    if name == "blockquote":
        text = _prose(element)
        return [Block(kind=PROSE, text=text)] if text else []

    if name == "div":
        for cls, label in _ADMONITIONS.items():
            if cls in classes:
                text = _render_admonition(element, label)
                return [Block(kind=ADMONITION, text=text)] if text else []

        if "refnamediv" in classes:
            # The heading duplicates the refentry title; keep only the summary line.
            blocks: list[Block] = []
            for paragraph in element.find_all("p", recursive=False):
                text = _prose(paragraph)
                if text:
                    blocks.append(Block(kind=PROSE, text=text))
            return blocks

        if classes & {"informaltable", "table-contents", "example", "figure",
                      "itemizedlist", "orderedlist", "procedure", "simplelist",
                      "informalexample", "screen", "literallayout"}:
            return _render_children(element)

        return _render_children(element)

    if name in ("span", "code", "em", "strong", "a", "acronym"):
        text = _prose(element)
        return [Block(kind=PROSE, text=text)] if text else []

    return []


def _render_children(element: Tag) -> list[Block]:
    blocks: list[Block] = []
    for child in element.find_all(True, recursive=False):
        if _is_section(child):
            continue
        blocks.extend(_render_element(child))
    return blocks


def _direct_variablelists(container: Tag) -> list[Tag]:
    """Find variable lists belonging to this container, not to a nested one."""
    found: list[Tag] = []

    def walk(node: Tag) -> None:
        for child in node.find_all(True, recursive=False):
            if _is_section(child):
                continue
            child_classes = set(_classes(child))
            if child.name == "dl" and "variablelist" in child_classes:
                found.append(child)
                continue
            if child.name == "dd":
                # Nested lists stay inside their parent term's text.
                continue
            walk(child)

    walk(container)
    return found


def _parameter_sections(varlist: Tag, parent_path: list[str], level: int) -> list[Section]:
    """Promote each dt/dd pair to its own section.

    These are the atomic units of the manual: one configuration parameter or one
    command clause, each with a permanent anchor to cite.
    """
    sections: list[Section] = []
    for term in varlist.find_all("dt", recursive=False):
        spans = term.find_all("span", class_="term")
        title = normalize_prose(" / ".join(_plain(span) for span in spans)) or _plain(term)
        if not title:
            continue

        definition = term.find_next_sibling("dd")
        blocks = _render_children(definition) if definition is not None else []
        if definition is not None and not blocks:
            text = _prose(definition)
            if text:
                blocks = [Block(kind=PROSE, text=text)]

        # A parameter's definition can itself contain a nested variable list.
        if definition is not None:
            for nested in definition.find_all("dl", class_="variablelist", recursive=True):
                nested_text = _render_definition_list_inline(nested)
                if nested_text:
                    blocks.append(Block(kind=PROSE, text=nested_text))

        sections.append(
            Section(
                title=title,
                level=level,
                path=parent_path + [title],
                blocks=blocks,
                anchor=term.get("id"),
                kind=PARAMETER,
            )
        )
    return sections


def _render_definition_list_inline(varlist: Tag) -> str:
    lines: list[str] = []
    for term in varlist.find_all("dt", recursive=False):
        label = _prose(term)
        definition = term.find_next_sibling("dd")
        body = _prose(definition) if definition is not None else ""
        if label and body:
            lines.append(f"- {label}: {body}")
        elif label:
            lines.append(f"- {label}")
    return "\n".join(lines)


# --- document assembly ------------------------------------------------------


def _find_top_containers(root: Tag) -> list[Tag]:
    found: list[Tag] = []

    def walk(node: Tag) -> None:
        for child in node.find_all(True, recursive=False):
            if _is_section(child):
                found.append(child)
            else:
                walk(child)

    walk(root)
    return found


def _child_containers(container: Tag) -> list[Tag]:
    found: list[Tag] = []

    def walk(node: Tag) -> None:
        for child in node.find_all(True, recursive=False):
            if _is_section(child):
                found.append(child)
            elif child.name != "dd":
                walk(child)

    walk(container)
    return found


def _walk_sections(
    container: Tag,
    parent_path: list[str],
    level: int,
    out: list[Section],
) -> None:
    raw_title = _section_title(container)
    number, title = split_section_number(raw_title)
    path = parent_path + [title] if title else list(parent_path)

    varlists = _direct_variablelists(container)
    blocks = _render_children(container)

    out.append(
        Section(
            title=title,
            level=level,
            path=path,
            blocks=blocks,
            anchor=container.get("id"),
            number=number,
            kind=SECTION,
        )
    )

    for varlist in varlists:
        out.extend(_parameter_sections(varlist, path, level + 1))

    for child in _child_containers(container):
        _walk_sections(child, path, level + 1, out)


def extract_page(html: str, record: PageRecord, ancestor_titles: list[str]) -> Document:
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one("#docContent") or soup.body
    if root is None:
        return Document(
            doc_id=record.slug,
            source_format=HTML,
            pg_version=config.PG_VERSION,
            title=clean_nav_title(record.title),
            source_url=record.url,
            source_path=str(cache_path(record.slug)),
            ordinal=record.ordinal,
        )

    _strip_noise(root)

    sections: list[Section] = []
    for container in _find_top_containers(root):
        _walk_sections(container, ancestor_titles, len(ancestor_titles) + 1, sections)

    _, page_title = split_section_number(clean_nav_title(record.title))
    if sections and sections[0].title:
        page_title = sections[0].title

    return Document(
        doc_id=record.slug,
        source_format=HTML,
        pg_version=config.PG_VERSION,
        title=page_title,
        sections=sections,
        source_url=record.url,
        source_path=str(cache_path(record.slug)),
        ordinal=record.ordinal,
    )


def _ancestor_titles(slug: str, manifest: dict[str, PageRecord]) -> list[str]:
    """Build the cross-page breadcrumb by following "up" links in the manifest."""
    chain: list[str] = []
    seen = {slug}
    current = manifest.get(slug)
    depth = 0
    while current is not None and current.parent_slug and depth < _MAX_ANCESTOR_DEPTH:
        parent_slug = current.parent_slug
        if parent_slug in seen:
            break
        seen.add(parent_slug)
        parent = manifest.get(parent_slug)
        if parent is None:
            break
        _, title = split_section_number(clean_nav_title(parent.title))
        if title and not _GENERIC_TITLE.search(title):
            chain.append(title)
        current = parent
        depth += 1
    chain.reverse()
    return chain


def extract_all(*, verbose: bool = True) -> Path:
    config.ensure_dirs()
    manifest = load_manifest()
    if not manifest:
        raise RuntimeError("No crawled pages found. Run the collect stage first.")

    documents: list[Document] = []
    for record in sorted(manifest.values(), key=lambda item: item.ordinal):
        path = cache_path(record.slug)
        if not path.exists():
            continue
        html = path.read_text(encoding="utf-8", errors="replace")
        document = extract_page(html, record, _ancestor_titles(record.slug, manifest))
        documents.append(document)
        if verbose:
            block_count = sum(len(section.blocks) for section in document.sections)
            print(
                f"  {record.slug:<44} {len(document.sections):>4} sections "
                f"{block_count:>5} blocks"
            )

    written = write_jsonl(OUTPUT_PATH, documents)
    if verbose:
        print(f"\nWrote {written} documents to {OUTPUT_PATH}")
    return OUTPUT_PATH
