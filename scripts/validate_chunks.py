"""Check chunk-file invariants and report violations.

The important one is the token ceiling: a chunk longer than the embedding model's
window is truncated silently at embed time, so the tail of that chunk would never
be searchable. Run this after any change to the chunker.

Usage:
    python scripts/validate_chunks.py data/chunks/html_chunks.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pgdocrag import config  # noqa: E402

FENCE = "`" * 3


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    path = Path(sys.argv[1])
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        print(f"{path} is empty")
        return 1

    failures: list[str] = []

    oversized = [row for row in rows if row["token_count"] > config.MODEL_MAX_TOKENS]
    if oversized:
        failures.append(f"{len(oversized)} chunks exceed {config.MODEL_MAX_TOKENS} tokens")
        for row in oversized[:5]:
            print(
                f"  OVER  {row['token_count']:>4} tokens  {row['chunk_type']:<9} "
                f"{row['doc_id']}"
            )
            print(f"        path   {' > '.join(row['section_path'])[:100]}")
            text = row["text"]
            print(
                f"        fences {text.count(FENCE)}  chars {len(text)}  "
                f"lines {text.count(chr(10))}"
            )
            longest = max(text.split("\n\n"), key=len)
            print(f"        longest block {len(longest)} chars: {longest[:120]!r}")

    empty = [row for row in rows if not row["text"].strip()]
    if empty:
        failures.append(f"{len(empty)} chunks have empty text")

    ids = Counter(row["chunk_id"] for row in rows)
    duplicates = [chunk_id for chunk_id, count in ids.items() if count > 1]
    if duplicates:
        failures.append(f"{len(duplicates)} duplicate chunk_ids (first: {duplicates[0]})")

    unbalanced = [row for row in rows if row["text"].count(FENCE) % 2 != 0]
    if unbalanced:
        failures.append(f"{len(unbalanced)} chunks have unbalanced code fences")
        for row in unbalanced[:3]:
            print(f"  FENCE {row['doc_id']}  {' > '.join(row['section_path'])[:80]}")

    no_breadcrumb = [row for row in rows if not row["section_path"]]
    if no_breadcrumb:
        failures.append(f"{len(no_breadcrumb)} chunks have no section path")

    print(f"\nchecked {len(rows)} chunks from {path.name}")
    anchored = sum(1 for row in rows if row.get("section_anchor"))
    print(f"  anchored          {anchored} ({anchored / len(rows):.0%})")
    print(f"  distinct docs     {len({row['doc_id'] for row in rows})}")

    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nall invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
