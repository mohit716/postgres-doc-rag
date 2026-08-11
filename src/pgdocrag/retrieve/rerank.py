"""Cross-encoder reranking of an existing candidate list.

A bi-encoder embeds question and chunk separately, so the only thing it can
compare is two summaries of meaning made without reference to each other. A
cross-encoder reads both at once and scores the pair directly, which is far more
accurate and far too slow to run over 11,000 chunks — hence reranking a shortlist
that dense retrieval has already narrowed.

The models come from fastembed's ONNX cross-encoders, keeping the stack free of
PyTorch and reusing the same execution-provider selection as embedding.
"""

from __future__ import annotations

from .. import config
from ..embed.embedder import CUDA_PROVIDER, providers_for, session_providers
from ..store.base import SearchResult

MODELS = {
    # The standard reranking baseline: small, quick, and trained on exactly this
    # task. 22M parameters, so a 50-candidate shortlist costs milliseconds.
    "minilm": "Xenova/ms-marco-MiniLM-L-6-v2",
    # Same family as the bge-small embedder and roughly ten times the size.
    # Included to test whether reranker capacity buys anything on technical prose.
    "bge": "BAAI/bge-reranker-base",
}
DEFAULT_MODEL = "minilm"


class Reranker:
    def __init__(self, model: str = DEFAULT_MODEL, *, device: str = config.EMBED_DEVICE) -> None:
        if model not in MODELS:
            raise ValueError(f"Unknown reranker {model!r}; expected one of {', '.join(MODELS)}")
        self.key = model
        self.model_name = MODELS[model]
        self.device = device
        self._encoder = None
        self.providers: list[str] = []

    @property
    def encoder(self):
        if self._encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._encoder = TextCrossEncoder(
                model_name=self.model_name,
                providers=providers_for(self.device),
                # fastembed treats `cuda` and `providers` as mutually exclusive and
                # warns when both are meaningful. The explicit provider list is the
                # more precise of the two, so the flag is switched off in its favour.
                cuda=False,
            )
            self.providers = session_providers(self._encoder)
        return self._encoder

    @property
    def on_gpu(self) -> bool:
        return any(provider == CUDA_PROVIDER for provider in self.providers)

    def score(
        self, question: str, candidates: list[SearchResult], *, batch_size: int = 32
    ) -> list[float]:
        if not candidates:
            return []
        return [
            float(score)
            for score in self.encoder.rerank(
                question, [result.text for result in candidates], batch_size=batch_size
            )
        ]

    def rerank(
        self, question: str, candidates: list[SearchResult], *, depth: int
    ) -> list[SearchResult]:
        return reorder(candidates, self.score(question, candidates[:depth]), depth)


def reorder(
    candidates: list[SearchResult], scores: list[float], depth: int
) -> list[SearchResult]:
    """Sort the first `depth` candidates by score, leaving the tail untouched.

    Keeping the tail matters for measurement: reranking a shortlist cannot
    introduce a chunk the shortlist never contained, so recall at the candidate
    depth must come out identical to the dense run. If it does not, the harness
    is wrong rather than the reranker being good.

    Taking `scores` as an argument rather than computing them lets one scoring
    pass over 50 candidates serve every depth at or below 50.
    """
    head, tail = candidates[:depth], candidates[depth:]
    if len(head) < 2:
        return list(candidates)

    order = sorted(range(len(head)), key=lambda index: scores[index], reverse=True)
    return [
        SearchResult(
            chunk_id=head[index].chunk_id,
            text=head[index].text,
            score=scores[index],
            metadata=head[index].metadata,
        )
        for index in order
    ] + tail
