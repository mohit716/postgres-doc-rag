"""Vector store interface.

Chroma is the default because it is embedded and needs no service to run, but
retrieval code depends only on this interface. Swapping in Qdrant means writing
one adapter, not touching the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from ..schema import Chunk


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]

    @property
    def breadcrumb(self) -> str:
        return str(self.metadata.get("breadcrumb", ""))

    @property
    def source_url(self) -> str:
        return str(self.metadata.get("source_url", ""))

    @property
    def anchor(self) -> str:
        return str(self.metadata.get("section_anchor", ""))


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int: ...

    def query(
        self,
        vector: np.ndarray,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...
