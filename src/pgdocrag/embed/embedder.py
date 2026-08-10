"""Stage 4: turn chunk text into vectors.

Uses fastembed, which runs bge-small-en-v1.5 through ONNX Runtime on CPU. That
keeps the install to roughly 130 MB with no PyTorch dependency, while still
giving a 512-token context window — the chunker's budget is derived from it.

Vectors are cached by content hash, so re-running after an unrelated pipeline
change only re-embeds text that actually changed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config


class VectorCache:
    """Content-hash keyed vector cache backed by a single .npz file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (config.EMBED_CACHE_DIR / f"{config.EMBED_MODEL_NAME.split('/')[-1]}.npz")
        self._vectors: dict[str, np.ndarray] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with np.load(self.path) as archive:
                self._vectors = {key: archive[key] for key in archive.files}
        except Exception:
            # A corrupt cache is not worth failing the run over; re-embedding is
            # slower but always correct.
            self._vectors = {}

    def get(self, key: str) -> np.ndarray | None:
        return self._vectors.get(key)

    def put(self, key: str, vector: np.ndarray) -> None:
        self._vectors[key] = vector
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **self._vectors)
        self._dirty = False

    def __len__(self) -> int:
        return len(self._vectors)


class Embedder:
    def __init__(
        self,
        model_name: str = config.EMBED_MODEL_NAME,
        *,
        use_cache: bool = True,
    ) -> None:
        self.model_name = model_name
        self._model = None
        self.cache = VectorCache() if use_cache else None
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _embed_raw(self, texts: list[str], batch_size: int) -> list[np.ndarray]:
        return [
            np.asarray(vector, dtype=np.float32)
            for vector in self.model.embed(texts, batch_size=batch_size)
        ]

    def embed_texts(
        self,
        texts: list[str],
        keys: list[str] | None = None,
        *,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed texts, reusing cached vectors where the key already exists."""
        if not texts:
            return np.zeros((0, config.EMBED_DIM), dtype=np.float32)

        if self.cache is None or keys is None:
            vectors = self._embed_raw(texts, batch_size)
            self.cache_misses += len(texts)
            return np.vstack(vectors)

        result: list[np.ndarray | None] = [None] * len(texts)
        pending_indices: list[int] = []
        pending_texts: list[str] = []

        for index, (text, key) in enumerate(zip(texts, keys)):
            cached = self.cache.get(key)
            if cached is not None:
                result[index] = cached
                self.cache_hits += 1
            else:
                pending_indices.append(index)
                pending_texts.append(text)

        if pending_texts:
            self.cache_misses += len(pending_texts)
            iterator = self.model.embed(pending_texts, batch_size=batch_size)
            if show_progress:
                from tqdm import tqdm

                iterator = tqdm(
                    iterator, total=len(pending_texts), desc="embedding", unit="chunk"
                )
            for index, vector in zip(pending_indices, iterator):
                array = np.asarray(vector, dtype=np.float32)
                result[index] = array
                self.cache.put(keys[index], array)
            self.cache.save()

        return np.vstack([vector for vector in result if vector is not None])

    def embed_query(self, question: str) -> np.ndarray:
        """Embed a question.

        bge models are trained asymmetrically: the instruction prefix belongs on
        the query side only, and adding it to passages would degrade retrieval.
        """
        prefixed = f"{config.QUERY_PREFIX}{question}"
        vectors = self._embed_raw([prefixed], batch_size=1)
        return vectors[0]
