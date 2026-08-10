"""Inspect pipeline artifacts by anchor, title or free text.

Works on both document and chunk JSONL files, which makes it the quickest way to
check whether a change to the extractor or chunker did what was intended.

Usage:
    python scripts/peek.py data/interim/html_docs.jsonl --anchor GUC-LISTEN-ADDRESSES
    python scripts/peek.py data/chunks/html_chunks.jsonl --match "listen_addresses" -n 2
    python scripts/peek.py data/interim/pdf_docs.jsonl --title "Connection Settings"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def emit(label: str, value: str) -> None:
    print(f"{label:>14}: {value}")


def show_section(document: dict, section: dict, *, chars: int) -> None:
    print("-" * 78)
    emit("doc", document.get("doc_id", ""))
    emit("title", section.get("title", ""))
    emit("anchor", section.get("anchor") or "-")
    emit("kind", section.get("kind", ""))
    emit("level", str(section.get("level", "")))
    emit("path", " > ".join(section.get("path", [])))
    emit("url", document.get("source_url") or "-")
    if document.get("page_start") is not None:
        emit("pdf page", str(document.get("page_start")))
    for index, block in enumerate(section.get("blocks", [])):
        text = block.get("text", "")
        clipped = text if chars <= 0 or len(text) <= chars else text[:chars] + " ..."
        print(f"    [{index}] {block.get('kind')}: {clipped}")


def show_chunk(chunk: dict, *, chars: int) -> None:
    print("-" * 78)
    emit("chunk_id", chunk.get("chunk_id", ""))
    emit("doc", chunk.get("doc_id", ""))
    emit("title", chunk.get("title", ""))
    emit("anchor", chunk.get("section_anchor") or "-")
    emit("type", chunk.get("chunk_type", ""))
    emit("tokens", str(chunk.get("token_count", "")))
    emit("path", " > ".join(chunk.get("section_path", [])))
    emit("url", chunk.get("source_url") or "-")
    text = chunk.get("text", "")
    clipped = text if chars <= 0 or len(text) <= chars else text[:chars] + " ..."
    print(clipped)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--anchor", help="Match section/chunk anchor (case-insensitive).")
    parser.add_argument("--title", help="Match title substring.")
    parser.add_argument("--match", help="Match anywhere in the text.")
    parser.add_argument("--doc", help="Match doc_id substring.")
    parser.add_argument("-n", "--limit", type=int, default=3)
    parser.add_argument("--chars", type=int, default=400, help="0 for full text.")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Not found: {args.path}")
        return 1

    def matches(*fields: str) -> bool:
        haystack = " ".join(field.lower() for field in fields if field)
        if args.anchor and args.anchor.lower() not in haystack:
            return False
        if args.title and args.title.lower() not in haystack:
            return False
        if args.match and args.match.lower() not in haystack:
            return False
        return True

    shown = 0
    total = 0
    with args.path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            is_chunk = "chunk_id" in record

            if args.doc and args.doc.lower() not in record.get("doc_id", "").lower():
                continue

            if is_chunk:
                total += 1
                if matches(
                    record.get("section_anchor") or "",
                    record.get("title", ""),
                    record.get("text", ""),
                ):
                    show_chunk(record, chars=args.chars)
                    shown += 1
            else:
                for section in record.get("sections", []):
                    total += 1
                    blocks = " ".join(b.get("text", "") for b in section.get("blocks", []))
                    if matches(section.get("anchor") or "", section.get("title", ""), blocks):
                        show_section(record, section, chars=args.chars)
                        shown += 1
                        if shown >= args.limit:
                            break
            if shown >= args.limit:
                break

    print("-" * 78)
    print(f"showed {shown} of {total} scanned records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
