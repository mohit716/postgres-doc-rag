"""Stage 5: persist vectors in a local Chroma collection."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .. import config
from ..schema import Chunk
from .base import SearchResult

# Chroma rejects very large writes; this stays well inside its per-request cap.
_UPSERT_BATCH = 500


class ChromaStore:
    def __init__(self, collection: str, *, distance: str = "cosine") -> None:
        import chromadb
        from chromadb.config import Settings

        self.collection_name = collection
        self.distance = distance
        self._client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._get_or_create()

    def _get_or_create(self):
        # Chroma moved space configuration from `metadata` to `configuration`;
        # accept either so the store works across 0.5.x and 1.x.
        try:
            return self._client.get_or_create_collection(
                name=self.collection_name,
                configuration={"hnsw": {"space": self.distance}},
            )
        except TypeError:
            return self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance},
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}"
            )

        written = 0
        for start in range(0, len(chunks), _UPSERT_BATCH):
            batch = chunks[start : start + _UPSERT_BATCH]
            batch_vectors = vectors[start : start + _UPSERT_BATCH]
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=[vector.tolist() for vector in batch_vectors],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.to_metadata() for chunk in batch],
            )
            written += len(batch)
        return written

    def query(
        self,
        vector: np.ndarray,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        response = self._collection.query(
            query_embeddings=[vector.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results: list[SearchResult] = []
        for index, chunk_id in enumerate(ids):
            distance = distances[index] if index < len(distances) else 1.0
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=documents[index] if index < len(documents) else "",
                    # Cosine distance in Chroma is 1 - similarity.
                    score=1.0 - float(distance),
                    metadata=dict(metadatas[index]) if index < len(metadatas) else {},
                )
            )
        return results

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._get_or_create()


def open_store(source_format: str) -> ChromaStore:
    return ChromaStore(config.collection_name(source_format))
