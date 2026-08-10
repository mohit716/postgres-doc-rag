"""Shared text normalisation.

Both extractors route through this module. HTML and PDF renderings of the same
DocBook source differ in whitespace, quote characters, ligatures and line
breaking, so without a single normalisation pass the cross-format comparison
would measure formatting noise rather than extraction quality.
"""

from __future__ import annotations

import re
import unicodedata

# Characters that survive NFKC but still differ between the HTML and PDF
# renderings of the same source text.
_CHAR_MAP = {
    "\u00ad": "",  # soft hyphen
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\ufeff": "",
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2032": "'",
    "\u2033": '"',
}

_TRANSLATION = str.maketrans(_CHAR_MAP)

_MULTI_SPACE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCT = re.compile(r" +([,.;:!?\)\]])")
# Flattening HTML inserts a space between adjacent inline tags, so a term like
# "listen_addresses (string)" arrives as "listen_addresses ( string)". The PDF
# text layer has no such gap, and titles must agree across formats.
_SPACE_AFTER_OPEN = re.compile(r"([\(\[]) +")

# A hyphen at end of line inside a word: either a syllable break introduced by
# PDF line wrapping, or a genuine compound hyphen that happened to wrap.
_LINEBREAK_HYPHEN = re.compile(r"(\w)-\n(\w)")


def _fold_characters(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.translate(_TRANSLATION)


def dehyphenate(text: str) -> str:
    """Rejoin words split across a line break by PDF wrapping.

    The hyphen is kept only where the case changes across the break, which is the
    signature of a real compound (``pre-\\nPostgreSQL``). It is dropped when both
    sides share a case class, covering ordinary syllable breaks
    (``config-\\nuration``) and all-caps keywords the PDF wrapped mid-word
    (``CRE-\\nATE INDEX``).

    This remains a heuristic: ``non-\\nblocking`` is indistinguishable from a
    syllable break and loses its hyphen. Nothing in the extracted text resolves
    that case.
    """

    def replace(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        crosses_case = left.islower() and (right.isupper() or right.isdigit())
        return f"{left}-{right}" if crosses_case else f"{left}{right}"

    return _LINEBREAK_HYPHEN.sub(replace, text)


def normalize_prose(text: str, *, from_pdf: bool = False) -> str:
    """Collapse a prose run to a single normalised paragraph."""
    if not text:
        return ""
    text = _fold_characters(text)
    if from_pdf:
        text = dehyphenate(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", " ")
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    return text.strip()


def normalize_code(text: str) -> str:
    """Normalise a code block while preserving its line structure."""
    if not text:
        return ""
    text = _fold_characters(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return _dedent(lines)


def _dedent(lines: list[str]) -> str:
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    shift = min(indents) if indents else 0
    return "\n".join(line[shift:] if line.strip() else "" for line in lines)


def collapse_blank_lines(text: str) -> str:
    return _MULTI_NEWLINE.sub("\n\n", text).strip()


# --- Section titles ---------------------------------------------------------

# "Chapter 20. Server Configuration", "Part III. Server Administration"
_LABELLED = re.compile(
    r"^(?:Chapter|Part|Appendix)\s+([0-9]+|[IVXLC]+|[A-Z])\.?\s+(.+)$",
    re.IGNORECASE,
)
# "20.3.1. Connection Settings", and the appendix form "F.1. adminpack"
_NUMBERED = re.compile(r"^((?:[A-Z]\.)?[0-9]+(?:\.[0-9]+)*)\.\s+(.+)$")


def split_section_number(title: str) -> tuple[str | None, str]:
    """Separate a section's numbering from its text.

    HTML headings and PDF bookmarks number sections differently in places, so the
    number is stored separately and only the bare title is used for cross-format
    matching.
    """
    title = normalize_prose(title)
    match = _LABELLED.match(title)
    if match:
        return match.group(1), match.group(2).strip()
    match = _NUMBERED.match(title)
    if match:
        return match.group(1), match.group(2).strip()
    return None, title


_KEY_STRIP = re.compile(r"[^a-z0-9]+")


def title_key(title: str) -> str:
    """Aggressive key for joining HTML sections to PDF sections."""
    _, bare = split_section_number(title)
    return _KEY_STRIP.sub("", bare.lower())


def token_key_set(text: str) -> set[str]:
    """Word set used for cross-format text similarity."""
    return set(_KEY_STRIP.sub(" ", text.lower()).split())


# --- Block rendering --------------------------------------------------------


def fence_code(code: str, lang: str | None = None) -> str:
    code = normalize_code(code)
    if not code:
        return ""
    return f"```{lang or ''}\n{code}\n```"


def markdown_table(rows: list[list[str]], *, has_header: bool = True) -> str:
    """Render a table identically from either source format.

    Both extractors emit GitHub-flavoured markdown so that a table's chunk text
    is comparable across formats and stays readable to an embedding model.
    """
    cleaned = [[normalize_prose(cell).replace("|", r"\|") for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    lines = ["| " + " | ".join(cleaned[0]) + " |"]
    if has_header:
        lines.append("| " + " | ".join(["---"] * width) + " |")
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
