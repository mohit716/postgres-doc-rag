"""Lexical retrieval over the same chunks the dense index holds.

Dense retrieval matches meaning and is weak on exact tokens: a query naming
`pg_resetwal` competes against every other utility described in similar prose.
BM25 has the opposite bias, which is the entire reason for fusing them.

The index is built from the chunks file rather than the vector store so that the
two retrievers see byte-identical text, and so this needs no embedding at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .. import config
from ..schema import Chunk, read_chunks
from ..store.base import SearchResult

_WORD = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase words, with identifiers additionally split on underscores.

    `work_mem` is emitted as `work_mem`, `work` and `mem`. Keeping the whole
    identifier lets an exact mention score strongly; keeping the parts means a
    question phrased as "working memory" still finds it. Emitting only one or the
    other loses one of those two behaviours.
    """
    tokens: list[str] = []
    for word in _WORD.findall(text.lower()):
        tokens.append(word)
        if "_" in word:
            tokens.extend(part for part in word.split("_") if part)
    return tokens


@dataclass
class BM25Index:
    chunks: list[Chunk]
    model: object

    @classmethod
    def build(cls, source_format: str) -> BM25Index:
        from rank_bm25 import BM25Okapi

        path = config.CHUNKS_DIR / f"{source_format}_chunks.jsonl"
        if not path.exists():
            raise RuntimeError(f"No chunks at {path}. Run the chunk stage first.")

        chunks = list(read_chunks(path))
        if not chunks:
            raise RuntimeError(f"No chunks in {path}")

        # The breadcrumb is already the first line of chunk.text, so section
        # titles are indexed alongside the body without special handling.
        return cls(chunks=chunks, model=BM25Okapi([tokenize(chunk.text) for chunk in chunks]))

    def query(self, question: str, top_k: int) -> list[SearchResult]:
        scores = self.model.get_scores(tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [
            SearchResult(
                chunk_id=self.chunks[index].chunk_id,
                text=self.chunks[index].text,
                score=float(scores[index]),
                metadata=self.chunks[index].to_metadata(),
            )
            for index in ranked[:top_k]
        ]
