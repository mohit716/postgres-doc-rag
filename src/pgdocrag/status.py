"""Report what each pipeline stage has produced so far."""

from __future__ import annotations

import json
from pathlib import Path

from . import config


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _size_mb(path: Path) -> float:
    return path.stat().st_size / 1_048_576


def _row(label: str, value: str) -> None:
    print(f"  {label:<22} {value}")


def report() -> None:
    print(f"PostgresDocRAG - PostgreSQL {config.PG_VERSION} documentation")
    print(f"corpus: {config.CORPUS}  ({config.CORPUS_DIR})\n")

    print("collect")
    manifest = config.HTML_MANIFEST_PATH
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        cached = len(list(config.RAW_HTML_DIR.glob("*.html")))
        _row("html pages", f"{payload.get('page_count', 0)} in manifest, {cached} cached")
    else:
        _row("html pages", "not collected")

    pdf = config.RAW_PDF_DIR / config.PDF_FILENAME
    _row("pdf manual", f"{_size_mb(pdf):.1f} MB" if pdf.exists() else "not downloaded")

    print("\nextract")
    for source in ("html", "pdf"):
        path = config.INTERIM_DIR / f"{source}_docs.jsonl"
        _row(f"{source} documents", str(_count_lines(path)) if path.exists() else "-")

    print("\nchunk")
    for source in ("html", "pdf"):
        path = config.CHUNKS_DIR / f"{source}_chunks.jsonl"
        _row(f"{source} chunks", str(_count_lines(path)) if path.exists() else "-")

    print("\nstore")
    if not config.CHROMA_DIR.exists():
        _row("chroma", "no collections")
        return

    for source in ("html", "pdf"):
        name = config.collection_name(source)
        try:
            from .store.chroma_store import ChromaStore

            _row(name, f"{ChromaStore(name).count()} vectors")
        except Exception as error:
            _row(name, f"unavailable ({type(error).__name__})")
