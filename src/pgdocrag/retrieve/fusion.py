"""Combining a dense ranking with a lexical one."""

from __future__ import annotations

from ..store.base import SearchResult

# The constant from the original RRF paper. It damps the influence of the very
# top ranks, so one retriever being confidently wrong cannot dominate the fusion.
RRF_K = 60


def _by_id(results: list[SearchResult]) -> dict[str, SearchResult]:
    return {result.chunk_id: result for result in results}


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]],
    *,
    k: int = RRF_K,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Fuse rankings by summed reciprocal rank.

    Rank-based on purpose. A cosine similarity sits in [-1, 1] while a BM25 score
    is unbounded and scaled by corpus statistics, so any weighted sum of the raw
    values is really a comparison of two incompatible units. Ranks discard
    magnitude, which costs some information but needs no normalisation and no
    tuning per corpus.
    """
    scores: dict[str, float] = {}
    lookup: dict[str, SearchResult] = {}

    for ranking in rankings:
        lookup.update(_by_id(ranking))
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)

    order = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    fused = []
    for chunk_id in order[: top_k or len(order)]:
        result = lookup[chunk_id]
        fused.append(
            SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=scores[chunk_id],
                metadata=result.metadata,
            )
        )
    return fused


def weighted_fusion(
    dense: list[SearchResult],
    sparse: list[SearchResult],
    *,
    alpha: float = 0.5,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Fuse min-max normalised scores, weighting dense by `alpha`.

    The counterpart to RRF: it keeps score magnitude, at the cost of a
    normalisation that is only valid within one query's candidate set, and of a
    weight that has to be chosen. Included to test whether the magnitude RRF
    throws away was worth anything.
    """

    def normalise(results: list[SearchResult]) -> dict[str, float]:
        if not results:
            return {}
        values = [result.score for result in results]
        low, high = min(values), max(values)
        span = high - low
        if span <= 0:
            return {result.chunk_id: 1.0 for result in results}
        return {result.chunk_id: (result.score - low) / span for result in results}

    dense_scores = normalise(dense)
    sparse_scores = normalise(sparse)
    lookup = {**_by_id(sparse), **_by_id(dense)}

    combined = {
        chunk_id: alpha * dense_scores.get(chunk_id, 0.0)
        + (1 - alpha) * sparse_scores.get(chunk_id, 0.0)
        for chunk_id in set(dense_scores) | set(sparse_scores)
    }
    order = sorted(combined, key=lambda chunk_id: combined[chunk_id], reverse=True)
    return [
        SearchResult(
            chunk_id=chunk_id,
            text=lookup[chunk_id].text,
            score=combined[chunk_id],
            metadata=lookup[chunk_id].metadata,
        )
        for chunk_id in order[: top_k or len(order)]
    ]
