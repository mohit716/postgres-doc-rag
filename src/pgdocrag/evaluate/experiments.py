"""Compare retrieval strategies with the ingestion pipeline held constant.

Every configuration here reads the same chunks and the same vectors. The only
thing that changes is how candidates are found and how they are ordered, so any
difference in the numbers is attributable to the retrieval method and nothing
else.

Two quantities are reported per configuration and the distinction between them
is the point of the exercise:

  candidate recall  whether the answer is anywhere in the pool at all
  Recall@5          whether it was ranked highly enough to be seen

Reranking can only move the second. When Recall@5 is far below candidate recall,
the retriever is finding the answer and failing to order it, which reranking
fixes. When candidate recall itself is low, no amount of reordering helps and the
candidate generator has to change — which is what hybrid retrieval is for.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..embed.embedder import Embedder
from ..retrieve import rerank as rerank_module
from ..retrieve.bm25 import BM25Index
from ..retrieve.fusion import reciprocal_rank_fusion, weighted_fusion
from ..store.base import SearchResult
from ..store.chroma_store import open_store
from .run_eval import DEFAULT_GOLDSET, GoldQuery, load_goldset, resolve_goldset

CANDIDATE_DEPTH = 50

# Fusion draws from deeper lists than it returns. Fusing two top-50s and cutting
# back to 50 would push out dense hits sitting at rank 40-50 to make room for
# lexical ones, lowering the ceiling hybrid retrieval is supposed to raise; the
# resulting drop would say more about the truncation than about the method.
FUSION_POOL = 100

RECALL_DEPTHS = (5, 10, 20, 50)
RERANK_DEPTHS = (20, 50)
ALPHA_SWEEP = (0.3, 0.5, 0.7)

# Buckets rather than raw ranks: with 107 questions a per-rank histogram is
# mostly ones, and the shape of the distribution is the thing worth seeing.
BUCKETS = ((1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 50))


@dataclass
class Outcome:
    question: str
    expectation: str
    topic: str | None
    rank: int | None


@dataclass
class Measurement:
    name: str
    recall: dict[str, float]
    mrr: float
    ndcg_at_5: float
    candidate_recall: float
    buckets: dict[str, int]
    outcomes: list[Outcome] = field(default_factory=list)


def _first_rank(query: GoldQuery, results: list[SearchResult]) -> int | None:
    for rank, result in enumerate(results, start=1):
        if query.matches(result):
            return rank
    return None


def _ndcg(rank: int | None, cutoff: int) -> float:
    if rank is None or rank > cutoff:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _measure(
    name: str,
    queries: list[GoldQuery],
    rankings: list[list[SearchResult]],
    candidate_ranks: list[int | None],
) -> Measurement:
    ranks = [_first_rank(query, ranking) for query, ranking in zip(queries, rankings)]
    count = len(queries) or 1

    buckets = {f"{low}-{high}" if low != high else str(low): 0 for low, high in BUCKETS}
    buckets["miss"] = 0
    for rank in ranks:
        if rank is None:
            buckets["miss"] += 1
            continue
        for low, high in BUCKETS:
            if low <= rank <= high:
                buckets[f"{low}-{high}" if low != high else str(low)] += 1
                break

    return Measurement(
        name=name,
        recall={
            str(depth): sum(1 for rank in ranks if rank and rank <= depth) / count
            for depth in RECALL_DEPTHS
        },
        mrr=sum(1.0 / rank for rank in ranks if rank) / count,
        ndcg_at_5=sum(_ndcg(rank, 5) for rank in ranks) / count,
        candidate_recall=sum(1 for rank in candidate_ranks if rank) / count,
        buckets=buckets,
        outcomes=[
            Outcome(
                question=query.question,
                expectation=query.label(),
                topic=query.topic,
                rank=rank,
            )
            for query, rank in zip(queries, ranks)
        ],
    )


def run(
    source_format: str = "html",
    *,
    goldset: str = "full",
    rerankers: tuple[str, ...] = ("minilm", "bge"),
    device: str = config.EMBED_DEVICE,
    verbose: bool = True,
) -> list[Measurement]:
    queries = load_goldset(resolve_goldset(goldset))
    store = open_store(source_format)
    if store.count() == 0:
        raise RuntimeError(
            f"Collection '{store.collection_name}' is empty. "
            f"Run: pgdocrag embed --source {source_format}"
        )

    if verbose:
        print(f"corpus {config.CORPUS}/{source_format}, gold set '{goldset}', {len(queries)} questions")
        print(f"collection {store.collection_name} ({store.count()} chunks)")

    embedder = Embedder()
    started = time.monotonic()
    if verbose:
        print("\nbuilding the lexical index")
    bm25 = BM25Index.build(source_format)
    if verbose:
        print(f"  {len(bm25.chunks)} chunks indexed in {time.monotonic() - started:.1f}s")

    # One retrieval pass. Every configuration below is a different view of these
    # same candidates, so nothing is queried or embedded twice.
    dense_deep: list[list[SearchResult]] = []
    sparse_deep: list[list[SearchResult]] = []
    if verbose:
        print("\nretrieving candidates")
    started = time.monotonic()
    for query in queries:
        vector = embedder.embed_query(query.question)
        dense_deep.append(store.query(vector, FUSION_POOL))
        sparse_deep.append(bm25.query(query.question, FUSION_POOL))
    if verbose:
        print(f"  dense and lexical top-{FUSION_POOL} in {time.monotonic() - started:.1f}s")

    # Every ranking is judged over the same number of candidates, whatever depth
    # it was drawn from, so the comparison is like for like.
    dense = [hits[:CANDIDATE_DEPTH] for hits in dense_deep]
    sparse = [hits[:CANDIDATE_DEPTH] for hits in sparse_deep]
    hybrid = [
        reciprocal_rank_fusion([dense_hits, sparse_hits], top_k=CANDIDATE_DEPTH)
        for dense_hits, sparse_hits in zip(dense_deep, sparse_deep)
    ]

    dense_pool = [_first_rank(query, hits) for query, hits in zip(queries, dense)]
    hybrid_pool = [_first_rank(query, hits) for query, hits in zip(queries, hybrid)]
    sparse_pool = [_first_rank(query, hits) for query, hits in zip(queries, sparse)]

    measurements = [
        _measure("dense", queries, dense, dense_pool),
        _measure("bm25", queries, sparse, sparse_pool),
        _measure("hybrid (RRF)", queries, hybrid, hybrid_pool),
    ]

    for alpha in ALPHA_SWEEP:
        weighted = [
            weighted_fusion(dense_hits, sparse_hits, alpha=alpha, top_k=CANDIDATE_DEPTH)
            for dense_hits, sparse_hits in zip(dense_deep, sparse_deep)
        ]
        pool = [_first_rank(query, hits) for query, hits in zip(queries, weighted)]
        measurements.append(_measure(f"hybrid (weighted a={alpha})", queries, weighted, pool))

    for key in rerankers:
        reranker = rerank_module.Reranker(key, device=device)
        if verbose:
            reranker.encoder  # create the session so the device is known before timing
            where = "GPU" if reranker.on_gpu else "CPU"
            print(f"\nreranking with {reranker.model_name} on {where}")

        for base_name, base, pool in (("dense", dense, dense_pool), ("hybrid", hybrid, hybrid_pool)):
            started = time.monotonic()
            # Scored once at the deepest depth; shallower depths reuse the scores.
            scores = [reranker.score(query.question, hits) for query, hits in zip(queries, base)]
            elapsed = time.monotonic() - started
            if verbose:
                pairs = sum(len(row) for row in scores)
                print(f"  {base_name:<7} {pairs} pairs in {elapsed:.1f}s ({pairs / elapsed:.0f}/s)")

            for depth in RERANK_DEPTHS:
                ordered = [
                    rerank_module.reorder(hits, score_row, depth)
                    for hits, score_row in zip(base, scores)
                ]
                measurements.append(
                    _measure(f"{base_name} + {key}@{depth}", queries, ordered, pool)
                )

    if verbose:
        _print_table(measurements)
    _save(measurements, source_format, goldset)
    return measurements


def _print_table(measurements: list[Measurement]) -> None:
    print(f"\n{'configuration':<28} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'R@50':>7} "
          f"{'MRR':>7} {'nDCG@5':>7} {'pool':>7}")
    for row in measurements:
        print(
            f"{row.name:<28} "
            + "".join(f"{row.recall[str(depth)]:>6.1%} " for depth in RECALL_DEPTHS)
            + f"{row.mrr:>7.3f} {row.ndcg_at_5:>7.3f} {row.candidate_recall:>6.1%}"
        )

    print("\nrank of the first relevant result")
    header = list(measurements[0].buckets)
    print(f"  {'configuration':<28} " + "".join(f"{name:>7}" for name in header))
    for row in measurements:
        print(f"  {row.name:<28} " + "".join(f"{row.buckets[name]:>7}" for name in header))


def _save(measurements: list[Measurement], source_format: str, goldset: str) -> Path:
    config.ensure_dirs()
    path = config.REPORTS_DIR / f"experiments_{source_format}_{goldset}.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "corpus": config.CORPUS,
                "source_format": source_format,
                "goldset": goldset,
                "embed_model": config.EMBED_MODEL_NAME,
                "candidate_depth": CANDIDATE_DEPTH,
                "measurements": [asdict(row) for row in measurements],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
