"""Measure whether HTML and PDF extraction agree.

This is the point of ingesting the same manual twice. Chunks from the two formats
are joined on a shared key — the anchor for configuration parameters, the
breadcrumb otherwise — and then compared three ways:

* word overlap (Jaccard), which is insensitive to reordering
* character-level similarity (difflib), which catches dropped or mangled runs
* cosine similarity of the stored embeddings, which is what retrieval actually
  sees; two chunks can differ in whitespace yet be identical to the retriever

Inline code marking is stripped before comparison: the HTML docs mark parameter
names up as code and the PDF's text layer cannot, which is a rendering
difference rather than an extraction failure.
"""

from __future__ import annotations

import difflib
import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import config
from ..normalize import title_key, token_key_set
from ..schema import Chunk, read_chunks

_FENCE = re.compile(r"^```.*$", re.MULTILINE)
_TABLE_RULE = re.compile(r"^\|[\s\-|]+\|$", re.MULTILINE)


@dataclass
class PairComparison:
    key: str
    breadcrumb: str
    jaccard: float
    ratio: float
    cosine: float | None
    html_chars: int
    pdf_chars: int

    @property
    def length_ratio(self) -> float:
        longest = max(self.html_chars, self.pdf_chars) or 1
        return min(self.html_chars, self.pdf_chars) / longest


@dataclass
class ComparisonReport:
    html_keys: int
    pdf_keys: int
    matched: int
    html_coverage: float
    pdf_coverage: float
    median_jaccard: float
    median_ratio: float
    median_cosine: float | None
    cosine_pairs: int
    buckets: dict[str, int] = field(default_factory=dict)
    unmatched: dict[str, int] = field(default_factory=dict)
    worst: list[PairComparison] = field(default_factory=list)


def _body(chunk: Chunk) -> str:
    """Chunk text minus the breadcrumb prefix the chunker adds."""
    if "\n\n" in chunk.text:
        return chunk.text.split("\n\n", 1)[1]
    return chunk.text


def _normalise_for_comparison(text: str) -> str:
    text = text.replace("`", "")
    text = _FENCE.sub("", text)
    text = _TABLE_RULE.sub("", text)
    text = text.replace("|", " ")
    return " ".join(text.split()).lower()


def _candidate_keys(chunk: Chunk) -> list[str]:
    """Join keys for one chunk, most specific first.

    An anchor is an exact identity. Failing that, breadcrumbs are compared by
    their last two elements rather than in full, because the two formats root
    them differently: the HTML crawler recovers "SQL Commands > CREATE TABLE >
    Description" from parent-page links, while the PDF rebases on the selected
    outline entry and yields "CREATE TABLE > Description".
    """
    if chunk.section_anchor:
        return [f"anchor:{chunk.section_anchor.upper()}"]

    parts = [title_key(part) for part in chunk.section_path if title_key(part)]
    if not parts:
        return []
    keys = []
    if len(parts) >= 2:
        keys.append("path2:" + "|".join(parts[-2:]))
    # A single-element key is only safe near the top of the tree. Deeper down it
    # produces false joins: HTML's "EXPLAIN > Parameters > ANALYZE" would match
    # the PDF's ANALYZE command page, which is unrelated content.
    if len(parts) <= 2:
        keys.append("path1:" + parts[-1])
    return keys


def _group(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Group chunks by their most specific key, keeping document order."""
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        keys = _candidate_keys(chunk)
        if keys:
            grouped.setdefault(keys[0], []).append(chunk)
    for group in grouped.values():
        group.sort(key=lambda chunk: chunk.ordinal)
    return grouped


def _build_index(groups: dict[str, list[Chunk]]) -> dict[str, str]:
    """Map every usable key to the group it identifies.

    A bare final title ("Notes", "Description") repeats across the manual, so
    those keys are only usable when they resolve to exactly one group.
    """
    alias_counts: dict[str, int] = {}
    aliases: dict[str, str] = {}
    for key, chunks in groups.items():
        for candidate in _candidate_keys(chunks[0]):
            alias_counts[candidate] = alias_counts.get(candidate, 0) + 1
            aliases.setdefault(candidate, key)
    return {
        candidate: key
        for candidate, key in aliases.items()
        if not candidate.startswith("path1:") or alias_counts[candidate] == 1
    }


def _load_vector_cache() -> dict[str, np.ndarray]:
    path = config.EMBED_CACHE_DIR / f"{config.EMBED_MODEL_NAME.split('/')[-1]}.npz"
    if not path.exists():
        return {}
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def compare() -> ComparisonReport:
    html_path = config.CHUNKS_DIR / "html_chunks.jsonl"
    pdf_path = config.CHUNKS_DIR / "pdf_chunks.jsonl"
    for path in (html_path, pdf_path):
        if not path.exists():
            raise RuntimeError(f"Missing {path}. Chunk both sources before comparing.")

    html_groups = _group(list(read_chunks(html_path)))
    pdf_groups = _group(list(read_chunks(pdf_path)))
    pdf_index = _build_index(pdf_groups)
    vectors = _load_vector_cache()

    comparisons: list[PairComparison] = []
    unmatched: dict[str, int] = {"anchor_missing_in_pdf": 0, "section_missing_in_pdf": 0}

    for key, html_chunks in html_groups.items():
        pdf_chunks = None
        for candidate in _candidate_keys(html_chunks[0]):
            target = pdf_index.get(candidate)
            if target is not None:
                pdf_chunks = pdf_groups[target]
                break

        if not pdf_chunks:
            reason = (
                "anchor_missing_in_pdf"
                if key.startswith("anchor:")
                else "section_missing_in_pdf"
            )
            unmatched[reason] += 1
            continue

        html_text = _normalise_for_comparison(" ".join(_body(c) for c in html_chunks))
        pdf_text = _normalise_for_comparison(" ".join(_body(c) for c in pdf_chunks))
        if not html_text or not pdf_text:
            continue

        html_words = token_key_set(html_text)
        pdf_words = token_key_set(pdf_text)
        union = html_words | pdf_words
        jaccard = len(html_words & pdf_words) / len(union) if union else 0.0
        ratio = difflib.SequenceMatcher(None, html_text, pdf_text).ratio()

        cosine = None
        # Embeddings are cached per chunk, so a pair is only comparable in vector
        # space when neither side was split into multiple chunks.
        if len(html_chunks) == 1 and len(pdf_chunks) == 1:
            html_vector = vectors.get(html_chunks[0].content_hash)
            pdf_vector = vectors.get(pdf_chunks[0].content_hash)
            if html_vector is not None and pdf_vector is not None:
                cosine = _cosine(html_vector, pdf_vector)

        comparisons.append(
            PairComparison(
                key=key,
                breadcrumb=" > ".join(html_chunks[0].section_path),
                jaccard=jaccard,
                ratio=ratio,
                cosine=cosine,
                html_chars=len(html_text),
                pdf_chars=len(pdf_text),
            )
        )

    cosines = [c.cosine for c in comparisons if c.cosine is not None]
    buckets = {">=0.95": 0, "0.85-0.95": 0, "0.70-0.85": 0, "0.50-0.70": 0, "<0.50": 0}
    for comparison in comparisons:
        value = comparison.ratio
        if value >= 0.95:
            buckets[">=0.95"] += 1
        elif value >= 0.85:
            buckets["0.85-0.95"] += 1
        elif value >= 0.70:
            buckets["0.70-0.85"] += 1
        elif value >= 0.50:
            buckets["0.50-0.70"] += 1
        else:
            buckets["<0.50"] += 1

    return ComparisonReport(
        html_keys=len(html_groups),
        pdf_keys=len(pdf_groups),
        matched=len(comparisons),
        html_coverage=len(comparisons) / (len(html_groups) or 1),
        pdf_coverage=len(comparisons) / (len(pdf_groups) or 1),
        median_jaccard=statistics.median([c.jaccard for c in comparisons]) if comparisons else 0.0,
        median_ratio=statistics.median([c.ratio for c in comparisons]) if comparisons else 0.0,
        median_cosine=statistics.median(cosines) if cosines else None,
        cosine_pairs=len(cosines),
        buckets=buckets,
        unmatched=unmatched,
        worst=sorted(comparisons, key=lambda c: c.ratio)[:8],
    )


def _print_report(report: ComparisonReport) -> None:
    print("\ncross-format extraction consistency")
    print(f"  html sections     {report.html_keys}")
    print(f"  pdf sections      {report.pdf_keys}")
    print(f"  joined pairs      {report.matched}")
    print(f"  html covered      {report.html_coverage:.1%}")
    print(f"  pdf covered       {report.pdf_coverage:.1%}")
    print(f"  median jaccard    {report.median_jaccard:.3f}")
    print(f"  median difflib    {report.median_ratio:.3f}")
    if report.median_cosine is not None:
        print(
            f"  median cosine     {report.median_cosine:.4f} "
            f"({report.cosine_pairs} single-chunk pairs)"
        )
    print("  difflib buckets   " + ", ".join(
        f"{name}={count}" for name, count in report.buckets.items()
    ))
    if report.unmatched:
        print("  unmatched html    " + ", ".join(
            f"{name}={count}" for name, count in report.unmatched.items()
        ))

    if report.worst:
        print("\n  least consistent pairs:")
        for comparison in report.worst:
            print(
                f"    ratio {comparison.ratio:.2f}  jaccard {comparison.jaccard:.2f}  "
                f"len {comparison.html_chars}/{comparison.pdf_chars}"
            )
            print(f"      {comparison.breadcrumb[:86]}")


def main() -> ComparisonReport:
    report = compare()
    _print_report(report)

    config.ensure_dirs()
    path = config.REPORTS_DIR / "format_comparison.json"
    payload = asdict(report)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {path}")
    return report
