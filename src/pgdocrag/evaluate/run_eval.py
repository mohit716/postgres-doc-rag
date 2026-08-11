"""Score retrieval quality against the gold set.

Reports Recall@k, MRR and nDCG@k. Running the same questions against the HTML
and PDF collections is what turns "the pipeline handles both formats" into a
measurement.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .. import config
from ..embed.embedder import Embedder
from ..query import search
from ..store.base import SearchResult

GOLDSET_PATH = Path(__file__).parent / "goldset.yaml"

# `config` is the original 52 questions, aimed almost entirely at configuration
# parameters. `full` is the corpus-representative set written for the full
# manual. They measure different things and neither replaces the other.
DEFAULT_GOLDSET = "config"
GOLDSETS = {
    "config": GOLDSET_PATH,
    "full": Path(__file__).parent / "goldset_full.yaml",
}


def resolve_goldset(name: str) -> Path:
    if name not in GOLDSETS:
        raise ValueError(f"Unknown gold set {name!r}; expected one of {', '.join(GOLDSETS)}")
    return GOLDSETS[name]


@dataclass
class GoldQuery:
    """A question and the chunk (or chunks) that would answer it.

    A result is relevant when it satisfies **any** listed expectation. The
    plural fields exist because a question can have more than one genuinely
    correct answer, and scoring an equally good alternative as a miss
    understates retrieval. The original 52-question set never sets more than one
    expectation per question, so any-of and all-of agree there and its published
    numbers are unaffected.
    """

    question: str
    expect_anchor: str | None = None
    expect_path: str | None = None
    expect_anchors: list[str] = field(default_factory=list)
    expect_paths: list[str] = field(default_factory=list)
    topic: str | None = None

    def anchors(self) -> list[str]:
        return ([self.expect_anchor] if self.expect_anchor else []) + list(self.expect_anchors)

    def paths(self) -> list[str]:
        return ([self.expect_path] if self.expect_path else []) + list(self.expect_paths)

    def matches(self, result: SearchResult) -> bool:
        for anchor in self.anchors():
            if result.anchor.upper() == anchor.upper():
                return True
        for path in self.paths():
            if path.lower() in result.breadcrumb.lower():
                return True
        return False

    def label(self) -> str:
        anchors = self.anchors()
        paths = self.paths()
        parts = anchors + [f"path:{path}" for path in paths]
        head = parts[0] if parts else "?"
        return f"{head} (+{len(parts) - 1})" if len(parts) > 1 else head


@dataclass
class QueryOutcome:
    question: str
    expectation: str
    rank: int | None  # 1-based rank of the first relevant result
    top_score: float
    top_breadcrumb: str
    topic: str | None = None

    @property
    def hit(self) -> bool:
        return self.rank is not None


@dataclass
class Report:
    source_format: str
    top_k: int
    query_count: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    hits: int
    goldset: str = DEFAULT_GOLDSET
    outcomes: list[QueryOutcome] = field(default_factory=list)


def load_goldset(path: Path = GOLDSET_PATH) -> list[GoldQuery]:
    if not path.exists():
        raise FileNotFoundError(f"No gold set at {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    queries = [GoldQuery(**entry) for entry in payload]
    unlabelled = [query.question for query in queries if not (query.anchors() or query.paths())]
    if unlabelled:
        raise ValueError(f"Gold set entries without an expectation: {unlabelled[:3]}")
    return queries


def _first_relevant_rank(query: GoldQuery, results: list[SearchResult]) -> int | None:
    for rank, result in enumerate(results, start=1):
        if query.matches(result):
            return rank
    return None


def _ndcg(rank: int | None) -> float:
    """Binary-gain nDCG for a single relevant document.

    With one relevant item the ideal DCG is 1, so nDCG reduces to the discount
    at the rank where that item was found.
    """
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def run(
    source_format: str,
    top_k: int,
    *,
    verbose: bool = True,
    goldset: str = DEFAULT_GOLDSET,
) -> Report:
    queries = load_goldset(resolve_goldset(goldset))
    embedder = Embedder()

    outcomes: list[QueryOutcome] = []
    for query in queries:
        results = search(
            query.question,
            source_format=source_format,
            top_k=top_k,
            embedder=embedder,
        )
        rank = _first_relevant_rank(query, results)
        outcomes.append(
            QueryOutcome(
                question=query.question,
                expectation=query.label(),
                rank=rank,
                top_score=results[0].score if results else 0.0,
                top_breadcrumb=results[0].breadcrumb if results else "",
                topic=query.topic,
            )
        )

    hits = sum(1 for outcome in outcomes if outcome.hit)
    count = len(outcomes) or 1
    report = Report(
        source_format=source_format,
        top_k=top_k,
        query_count=len(outcomes),
        recall_at_k=hits / count,
        mrr=sum(1.0 / outcome.rank for outcome in outcomes if outcome.rank) / count,
        ndcg_at_k=sum(_ndcg(outcome.rank) for outcome in outcomes) / count,
        hits=hits,
        goldset=goldset,
        outcomes=outcomes,
    )

    if verbose:
        _print_report(report)
    _save_report(report)
    return report


def _print_report(report: Report) -> None:
    print(f"\n{report.source_format.upper()} collection, top_k={report.top_k}")
    print(f"  gold set          {report.goldset}")
    print(f"  queries           {report.query_count}")
    print(f"  Recall@{report.top_k}          {report.recall_at_k:.1%} ({report.hits}/{report.query_count})")
    print(f"  MRR               {report.mrr:.3f}")
    print(f"  nDCG@{report.top_k}            {report.ndcg_at_k:.3f}")

    rank_counts: dict[str, int] = {}
    for outcome in report.outcomes:
        key = str(outcome.rank) if outcome.rank else "miss"
        rank_counts[key] = rank_counts.get(key, 0) + 1
    ordered = sorted(
        rank_counts.items(), key=lambda item: (item[0] == "miss", item[0])
    )
    print("  rank of first hit " + ", ".join(f"{key}:{value}" for key, value in ordered))

    topics = sorted({outcome.topic for outcome in report.outcomes if outcome.topic})
    if topics:
        print("\n  by topic")
        for topic in topics:
            rows = [outcome for outcome in report.outcomes if outcome.topic == topic]
            topic_hits = sum(1 for outcome in rows if outcome.hit)
            print(f"    {topic:<18} {topic_hits / len(rows):>6.0%}  ({topic_hits}/{len(rows)})")

    misses = [outcome for outcome in report.outcomes if not outcome.hit]
    if misses:
        print(f"\n  misses ({len(misses)}):")
        for outcome in misses:
            print(f"    {outcome.expectation:<44} {outcome.question[:52]}")
            print(f"      got: {outcome.top_breadcrumb[:88]}")


def _save_report(report: Report) -> Path:
    config.ensure_dirs()
    # The default gold set keeps its original filename, so the reports behind the
    # published metrics are updated in place rather than orphaned beside new ones.
    label = "" if report.goldset == DEFAULT_GOLDSET else f"_{report.goldset}"
    path = config.REPORTS_DIR / f"eval_{report.source_format}{label}_k{report.top_k}.json"
    payload = asdict(report)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["embed_model"] = config.EMBED_MODEL_NAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main(
    *, source_format: str = "html", top_k: int = 5, goldset: str = DEFAULT_GOLDSET
) -> None:
    formats = ["html", "pdf"] if source_format in ("both", "all") else [source_format]
    reports = [run(fmt, top_k, goldset=goldset) for fmt in formats]

    if len(reports) > 1:
        print("\nside by side")
        print(f"  {'metric':<12} " + "".join(f"{r.source_format:>10}" for r in reports))
        for label, getter in (
            (f"Recall@{top_k}", lambda r: f"{r.recall_at_k:.1%}"),
            ("MRR", lambda r: f"{r.mrr:.3f}"),
            (f"nDCG@{top_k}", lambda r: f"{r.ndcg_at_k:.3f}"),
        ):
            print(f"  {label:<12} " + "".join(f"{getter(r):>10}" for r in reports))
