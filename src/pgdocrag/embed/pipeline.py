"""Drive stages 4 and 5: embed chunks, then write them to the vector store."""

from __future__ import annotations

import time

from .. import config
from ..schema import read_chunks
from ..store.chroma_store import open_store
from .embedder import Embedder


def embed_and_store(source_format: str, *, reset: bool = False, verbose: bool = True) -> int:
    chunks_path = config.CHUNKS_DIR / f"{source_format}_chunks.jsonl"
    if not chunks_path.exists():
        raise RuntimeError(f"No chunks at {chunks_path}. Run the chunk stage first.")

    chunks = list(read_chunks(chunks_path))
    if not chunks:
        if verbose:
            print("  no chunks to embed")
        return 0

    embedder = Embedder()
    started = time.monotonic()
    vectors = embedder.embed_texts(
        [chunk.text for chunk in chunks],
        [chunk.content_hash for chunk in chunks],
        show_progress=verbose,
    )
    elapsed = time.monotonic() - started

    store = open_store(source_format)
    if reset:
        store.reset()
    written = store.upsert(chunks, vectors)

    if verbose:
        print(f"  model             {config.EMBED_MODEL_NAME} ({vectors.shape[1]}d)")
        print(f"  embedded          {embedder.cache_misses} new, {embedder.cache_hits} cached")
        print(f"  elapsed           {elapsed:.1f}s")
        print(f"  collection        {store.collection_name}")
        print(f"  stored            {written} chunks (collection now {store.count()})")
    return written
