"""Retrieval: the end the whole pipeline exists to serve."""

from __future__ import annotations

from typing import Any

from . import config
from .embed.embedder import Embedder
from .store.base import SearchResult
from .store.chroma_store import open_store

_PREVIEW_CHARS = 420


def search(
    question: str,
    *,
    source_format: str = "html",
    top_k: int = config.DEFAULT_TOP_K,
    where: dict[str, Any] | None = None,
    embedder: Embedder | None = None,
) -> list[SearchResult]:
    store = open_store(source_format)
    if store.count() == 0:
        raise RuntimeError(
            f"Collection '{store.collection_name}' is empty. "
            f"Run: pgdocrag embed --source {source_format}"
        )

    embedder = embedder or Embedder()
    vector = embedder.embed_query(question)
    return store.query(vector, top_k, where)


def render_results(question: str, results: list[SearchResult], *, full: bool = False) -> None:
    print(f"\nQ: {question}\n")
    if not results:
        print("No results.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"[{rank}] score {result.score:.3f}  {result.breadcrumb}")
        if result.source_url:
            print(f"    {result.source_url}")
        page = result.metadata.get("source_page")
        if isinstance(page, int) and page > 0:
            print(f"    PDF page {page}")

        body = result.text
        # The breadcrumb is already shown above; don't repeat it in the body.
        if "\n\n" in body:
            body = body.split("\n\n", 1)[1]
        if not full and len(body) > _PREVIEW_CHARS:
            body = body[:_PREVIEW_CHARS].rstrip() + " ..."
        print("    " + body.replace("\n", "\n    "))
        print()
