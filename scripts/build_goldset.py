"""Sample candidate targets for a gold set, and verify a written one.

Authoring questions is manual — the point of a gold set is that a human decides
what a user would plausibly ask — but choosing *what* to write about and checking
the result afterwards are both mechanical, and this handles those two ends.

`sample` walks the corpus and proposes target chunks spread across topic areas,
so the set covers the manual rather than whichever corner comes to mind. `verify`
checks a finished set for the four ways it can be quietly wrong: an expectation
that matches no chunk at all, one so loose it matches half the corpus, a question
that leaks its answer's vocabulary, and lopsided topic coverage.

Verification deliberately never runs retrieval. A gold set filtered by what the
retriever already finds measures nothing.

Usage:
    python scripts/build_goldset.py sample --per-area 12 > worksheet.md
    python scripts/build_goldset.py verify --goldset full
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pgdocrag import config  # noqa: E402
from pgdocrag.evaluate.run_eval import load_goldset, resolve_goldset  # noqa: E402

# Topic areas, defined by breadcrumb prefix. These follow the manual's own
# structure rather than an invented taxonomy, so every chunk lands in at most one.
AREAS: dict[str, tuple[str, ...]] = {
    "sql-commands": ("Reference > SQL Commands",),
    "tools": (
        "Reference > PostgreSQL Client Applications",
        "Reference > PostgreSQL Server Applications",
    ),
    "admin": (
        "Server Administration > Server Configuration",
        "Server Administration > Server Setup and Operation",
        "Server Administration > Monitoring Database Activity",
        "Server Administration > Routine Database Maintenance Tasks",
        "Server Administration > Database Roles",
        "Server Administration > Localization",
    ),
    "authentication": ("Server Administration > Client Authentication",),
    "backup": (
        "Server Administration > Backup and Restore",
        "Server Administration > Reliability and the Write-Ahead Log",
    ),
    "replication": (
        "Server Administration > High Availability, Load Balancing, and Replication",
        "Server Administration > Logical Replication",
    ),
    "indexing": (
        "The SQL Language > Indexes",
        "Internals > Built-in Index Access Methods",
    ),
    "planning": (
        "The SQL Language > Performance Tips",
        "Internals > How the Planner Uses Statistics",
    ),
    "datatypes": (
        "The SQL Language > Data Types",
        "The SQL Language > Type Conversion",
    ),
    "functions": ("The SQL Language > Functions and Operators",),
    "internals": (
        "Internals > System Catalogs",
        "Internals > System Views",
        "Internals > Database Physical Storage",
        "Internals > Frontend/Backend Protocol",
        "The SQL Language > Concurrency Control",
    ),
}

# An expectation matching more than this many chunks is too vague to be evidence
# that the right passage was found.
SPECIFICITY_LIMIT = 30

MIN_TOKENS = 60


def area_of(breadcrumb: str) -> str | None:
    for area, prefixes in AREAS.items():
        if any(breadcrumb.startswith(prefix) for prefix in prefixes):
            return area
    return None


def load_rows(corpus: str, source: str) -> list[dict]:
    config.use_corpus(corpus)
    path = config.CHUNKS_DIR / f"{source}_chunks.jsonl"
    if not path.exists():
        raise SystemExit(f"No chunks at {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample(args: argparse.Namespace) -> int:
    rows = load_rows(args.corpus, args.source)

    by_area: dict[str, list[dict]] = {area: [] for area in AREAS}
    for row in rows:
        breadcrumb = " > ".join(row["section_path"])
        area = area_of(breadcrumb)
        if area and row["token_count"] >= MIN_TOKENS:
            by_area[area].append(row)

    print(f"# Gold set worksheet ({args.corpus}/{args.source})\n")
    for area, candidates in by_area.items():
        # Stride across documents rather than chunks. Chunks arrive in corpus
        # order, which is alphabetical within the command reference, so striding
        # over chunks would return eleven flavours of ALTER and nothing else.
        by_doc: dict[str, list[dict]] = {}
        for row in candidates:
            by_doc.setdefault(row["doc_id"], []).append(row)

        doc_ids = list(by_doc)
        step = max(1, len(doc_ids) // args.per_area)
        picked = []
        for doc_id in doc_ids[::step][: args.per_area]:
            rows_for_doc = by_doc[doc_id]
            # An anchored chunk gives a stable, unambiguous label; a breadcrumb
            # substring is the fallback when the page publishes no usable anchor.
            picked.append(
                next((row for row in rows_for_doc if row.get("section_anchor")), rows_for_doc[0])
            )

        print(f"\n## {area}  ({len(candidates)} candidate chunks)\n")
        for row in picked:
            breadcrumb = " > ".join(row["section_path"])
            anchor = row.get("section_anchor") or ""
            excerpt = " ".join(row["text"].split())[:240]
            print(f"- **{breadcrumb}**")
            print(f"  - anchor: `{anchor or '(none)'}`  type: {row['chunk_type']}  tokens: {row['token_count']}")
            print(f"  - url: {row.get('source_url') or ''}")
            print(f"  - excerpt: {excerpt}")
    return 0


def _distinctive_terms(anchors: list[str], paths: list[str]) -> list[str]:
    """Words whose presence in a question would give the answer away."""
    terms: list[str] = []
    for anchor in anchors:
        # GUC-WORK-MEM -> work_mem, the identifier a user might paste verbatim.
        body = re.sub(r"^GUC-", "", anchor, flags=re.IGNORECASE)
        terms.append(body.replace("-", "_").lower())
    for path in paths:
        terms.append(path.lower())
    return [term for term in terms if len(term) > 3]


def verify(args: argparse.Namespace) -> int:
    queries = load_goldset(resolve_goldset(args.goldset))
    rows = {source: load_rows(args.corpus, source) for source in ("html", "pdf")}

    index = {}
    for source, source_rows in rows.items():
        index[source] = [
            ((row.get("section_anchor") or "").upper(), " > ".join(row["section_path"]).lower())
            for row in source_rows
        ]

    def match_count(query, source: str) -> int:
        anchors = {anchor.upper() for anchor in query.anchors()}
        paths = [path.lower() for path in query.paths()]
        total = 0
        for anchor, breadcrumb in index[source]:
            if anchor and anchor in anchors:
                total += 1
            elif any(path in breadcrumb for path in paths):
                total += 1
        return total

    unresolved: list[str] = []
    vague: list[str] = []
    leaked: list[str] = []
    pdf_missing: list[str] = []
    topics = Counter()
    seen: Counter = Counter()

    for query in queries:
        topics[query.topic or "(none)"] += 1
        for anchor in query.anchors():
            seen[anchor.upper()] += 1

        html_hits = match_count(query, "html")
        if html_hits == 0:
            unresolved.append(f"{query.label()}  <- {query.question[:60]}")
        elif html_hits > SPECIFICITY_LIMIT:
            vague.append(f"{query.label()} matches {html_hits} chunks")

        if match_count(query, "pdf") == 0:
            pdf_missing.append(query.label())

        question = query.question.lower()
        hit_terms = [
            term for term in _distinctive_terms(query.anchors(), query.paths())
            if term in question
        ]
        if hit_terms:
            leaked.append(f"{query.label()}: {', '.join(hit_terms)}")

    print(f"gold set '{args.goldset}': {len(queries)} questions\n")
    print("topics")
    for topic, count in sorted(topics.items()):
        print(f"  {topic:<18} {count}")

    duplicates = [anchor for anchor, count in seen.items() if count > 1]

    print("\nchecks")
    print(f"  resolvable in HTML     {len(queries) - len(unresolved)}/{len(queries)}")
    print(f"  resolvable in PDF      {len(queries) - len(pdf_missing)}/{len(queries)}")
    print(f"  labels within limit    {len(queries) - len(vague)}/{len(queries)}  (<= {SPECIFICITY_LIMIT} chunks)")
    print(f"  questions leaking term {len(leaked)}")
    print(f"  duplicated anchors     {len(duplicates)}")

    for title, rows_out in (
        ("UNRESOLVED in HTML (fatal)", unresolved),
        ("too vague", vague),
        ("leaking the answer's vocabulary", leaked),
        ("absent from PDF (expected for a few)", pdf_missing),
        ("duplicated anchors", duplicates),
    ):
        if rows_out:
            print(f"\n{title} ({len(rows_out)}):")
            for row in rows_out[:12]:
                print(f"  {row}")

    return 1 if unresolved else 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sampler = sub.add_parser("sample", help="Propose target chunks by topic area.")
    sampler.add_argument("--corpus", default="full", choices=config.CORPORA)
    sampler.add_argument("--source", default="html", choices=("html", "pdf"))
    sampler.add_argument("--per-area", type=int, default=10)
    sampler.set_defaults(func=sample)

    verifier = sub.add_parser("verify", help="Check a written gold set.")
    verifier.add_argument("--corpus", default="full", choices=config.CORPORA)
    verifier.add_argument("--goldset", default="full")
    verifier.set_defaults(func=verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
