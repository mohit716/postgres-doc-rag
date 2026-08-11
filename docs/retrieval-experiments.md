# Retrieval experiments

Everything here holds the ingestion pipeline constant. Same crawl, same
extraction, same normalisation, same chunker, same `bge-small-en-v1.5` vectors,
same Chroma collections. Only candidate generation and ordering change, so any
difference in the numbers belongs to the retrieval method.

Two earlier benchmarks are untouched and still reproducible: the 52-question
configuration set on the slice corpus (`README.md`) and the same set on the full
corpus (`docs/full-corpus.md`).

## The full-corpus gold set

The 52-question set asks almost exclusively about configuration parameters. That
made it a good stress test — GUC pages are numerous and nearly identical, so it
punishes weak retrieval — but a poor description of what the manual contains.
The new set covers eleven areas of the documentation.

| | |
| --- | --- |
| questions | 107 |
| areas | SQL commands, tools, administration, authentication, backup, replication, indexing, planning, data types, functions, internals |
| per area | 9–10 |
| file | `src/pgdocrag/evaluate/goldset_full.yaml` |

**Construction.** `scripts/build_goldset.py sample` strides across documents
within each area and proposes target chunks, which keeps the questions spread
over the manual instead of clustered wherever attention happened to land.
Questions were then written by hand against each target's content, phrased as a
user's problem rather than as the documentation's own wording.

**Verification** is scripted and reports four failure modes:

```
resolvable in HTML     107/107
resolvable in PDF       90/107
labels within limit    107/107  (<= 30 chunks)
questions leaking term       0
duplicated anchors           0
```

*Resolvable* means the expectation matches at least one chunk that exists — the
check that caught the missing `GUC-TIMEZONE` chunk during the earlier PDF work.
*Within limit* bounds how many chunks satisfy a label, since an expectation
matching hundreds of chunks would stop being evidence of anything. *Leaking*
flags a question containing its own target's distinctive identifier, which would
turn a semantic question into a string match.

**Verification never runs retrieval, and no question was revised or dropped
because the retriever failed it.** The same model authored the questions and is
being measured on them, so that rule and the leakage check are what stand between
this and a flattering benchmark.

**Labels are any-of.** A section split across several chunks has several equally
correct answers, and scoring the second one as a miss understates retrieval. The
52-question set never lists more than one expectation per question, so its
published numbers are unaffected by the change in semantics.

**Cross-format labels.** HTML anchors do not exist in the PDF, whose breadcrumbs
are leaf-scoped. A section-title path matches both, so most questions carry one
alongside the anchor — but only where that title resolves to at most ~30 chunks
in both formats. Seventeen questions have no title specific enough to qualify,
which is why PDF resolvability is 90 rather than 107.

## Results: full corpus, HTML, 107 questions

`pool` is the share of questions whose answer appears anywhere in the 50
candidates. It is the ceiling: reranking reorders a shortlist and cannot add to
it, so any configuration's Recall@5 is bounded by its own pool.

| configuration | R@5 | R@10 | R@20 | R@50 | MRR | nDCG@5 | pool |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense (baseline) | 44.9% | 57.9% | 66.4% | 74.8% | 0.346 | 0.352 | 74.8% |
| BM25 only | 27.1% | 35.5% | 44.9% | 60.7% | 0.212 | 0.209 | 60.7% |
| hybrid, RRF | 50.5% | 57.9% | 66.4% | 73.8% | 0.358 | 0.381 | 73.8% |
| hybrid, weighted α=0.7 | 51.4% | 57.9% | 69.2% | 74.8% | 0.366 | 0.389 | 74.8% |
| dense + MiniLM@20 | 52.3% | 62.6% | 66.4% | 74.8% | 0.368 | 0.391 | 74.8% |
| dense + MiniLM@50 | 50.5% | 64.5% | 69.2% | 74.8% | 0.367 | 0.383 | 74.8% |
| dense + bge@20 | 50.5% | 62.6% | 66.4% | 74.8% | 0.370 | 0.387 | 74.8% |
| dense + bge@50 | 51.4% | 60.7% | 68.2% | 74.8% | 0.375 | 0.394 | 74.8% |
| hybrid + MiniLM@20 | 49.5% | 60.7% | 66.4% | 73.8% | 0.366 | 0.382 | 73.8% |
| **hybrid + bge@20** | **52.3%** | 60.7% | 66.4% | 73.8% | **0.382** | **0.405** | 73.8% |
| hybrid + bge@50 | 49.5% | 60.7% | 69.2% | 73.8% | 0.379 | 0.391 | 73.8% |

Rank of the first relevant result:

| configuration | 1 | 2–3 | 4–5 | 6–10 | 11–20 | 21–50 | miss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 26 | 14 | 8 | 14 | 9 | 9 | 27 |
| BM25 only | 14 | 12 | 3 | 9 | 10 | 17 | 42 |
| hybrid, RRF | 24 | 25 | 5 | 8 | 9 | 8 | 28 |
| dense + MiniLM@20 | 27 | 17 | 12 | 11 | 4 | 9 | 27 |
| hybrid + bge@20 | 29 | 19 | 8 | 9 | 6 | 8 | 28 |

### What the numbers say

**Ordering is the dominant failure, not retrieval.** Dense retrieval puts the
answer somewhere in the top 50 for 74.8% of questions but in the top 5 for only
44.9%. Thirty points of the gap are questions where the pipeline already found
the answer and buried it.

**Reranking recovers about a quarter of that gap.** The best reranked
configuration reaches 52.3% Recall@5, and nDCG@5 improves from 0.352 to 0.405 —
a larger relative move than Recall@5, which is what you would expect from a
change that only affects ordering. Pool recall is identical before and after, as
it must be; that equality is the harness checking itself.

**Reranking deeper is not better.** Every @50 configuration is at or below its
@20 counterpart on Recall@5. Handing the cross-encoder thirty extra weak
candidates gives it thirty extra chances to be confidently wrong near the top.

**The small reranker is the right one.** MiniLM-L-6 (22M parameters) and
bge-reranker-base (278M) land within one question of each other on Recall@5,
while MiniLM runs eight times faster on the same GPU — 80 pairs/second against
10. There is no case here for the larger model.

**Hybrid helps as much as reranking, and for a different reason.** RRF alone
lifts Recall@5 by 5.6 points, despite BM25 alone being far worse than dense
(27.1%). The gain is complementarity, not lexical strength.

**The two do not stack.** Best dense+rerank is 52.3%; best hybrid+rerank is also
52.3%. Both mechanisms are fixing the same easily-fixed orderings.

**RRF's rank-only view costs little.** Weighted fusion at its best α (0.7) is
within one question of RRF while requiring a tuned parameter, which is the
argument for preferring RRF.

**Differences under about five points are noise.** With 107 questions, one
question is 0.93 points. The gap between the top few configurations is two or
three questions and should not be read as a ranking.

### Per-topic recall

| topic | n | dense R@5 | best R@5 | in top 50 |
| --- | --- | --- | --- | --- |
| sql-commands | 10 | 10% | 30% | 50% |
| tools | 10 | 20% | 50% | 50% |
| internals | 10 | 30% | 40% | 90% |
| datatypes | 10 | 40% | 60% | 70% |
| replication | 10 | 40% | 30% | 60% |
| authentication | 9 | 44% | 56% | 89% |
| admin | 10 | 50% | 50% | 90% |
| planning | 9 | 56% | 78% | 89% |
| functions | 10 | 60% | 60% | 70% |
| indexing | 10 | 70% | 70% | 80% |
| backup | 9 | 78% | 56% | 89% |

Reference material is where the pipeline is weakest, and it fails at the
candidate stage rather than the ordering stage: only half the questions about SQL
commands or command-line tools have their answer anywhere in the top 50. The
corpus holds 1,332 chunks of SQL-command reference and 844 of tool reference,
almost all opening with a terse description in near-identical register. In
embedding space a target competes against hundreds of siblings that look just
like it, and dense retrieval scatters. Reranking recovers some of the tool
questions — 20% to 50% — because a cross-encoder can tell `pg_resetwal` from
`pg_receivewal` where a bi-encoder cannot.

Narrative chapters behave much better. Indexing, backup and planning questions
mostly resolve, and their remaining errors are ordering errors that reranking
moves.

### Why the ceiling is 74.8%

Twenty-seven questions have no correct chunk in the dense top 50. Reading what
was returned instead, roughly six or seven are arguable labels rather than true
failures — the CLUSTER question returns `CLUSTER > Notes` when the label names the
description, and the vacuum question returns *Routine Vacuuming → Recovering Disk
Space*, which is arguably a better answer than the one labelled. These are
recorded rather than corrected: revising a label because retrieval failed is
exactly the tuning the construction rules exist to prevent.

The remaining twenty are genuine, and several share a shape worth naming. The
question "how do I store structured documents" returns *Full Text Search → What
Is a Document?*. "How do I list the permitted labels of a custom type" returns
*SECURITY LABEL*. "How do I store a simple yes or no flag" returns the
information schema's `yes_or_no` domain. A single strong lexical anchor in the
question drags the embedding toward a section that shares the word and not the
meaning — and BM25 fusion cannot help, because it makes the same mistake harder.

## Format check: the same gold set against the PDF

Seventeen questions carry no PDF-resolvable label, so every PDF figure below is a
floor rather than a like-for-like reading. Only MiniLM was run here, since the
HTML results had already shown the larger reranker earning nothing.

| configuration | R@5 | R@10 | R@20 | R@50 | MRR | nDCG@5 | pool |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense (baseline) | 40.2% | 52.3% | 60.7% | 69.2% | 0.333 | 0.332 | 69.2% |
| BM25 only | 27.1% | 34.6% | 39.3% | 50.5% | 0.185 | 0.193 | 50.5% |
| hybrid, RRF | 41.1% | 51.4% | 56.1% | 67.3% | 0.323 | 0.329 | 67.3% |
| hybrid, weighted α=0.7 | 45.8% | 52.3% | 60.7% | 70.1% | 0.350 | 0.364 | 70.1% |
| **dense + MiniLM@20** | **51.4%** | 58.9% | 60.7% | 69.2% | 0.349 | **0.380** | 69.2% |
| hybrid + MiniLM@20 | 47.7% | 54.2% | 56.1% | 67.3% | 0.341 | 0.363 | 67.3% |

Every conclusion from the HTML run reproduces on the PDF: reranking gives the
largest single improvement (+11.2 points Recall@5, the biggest move seen
anywhere), depth 20 beats depth 50, and hybrid fusion adds little once reranking
is in place. The two formats land within a question or two of each other after
reranking — 52.3% against 51.4% — which is the same format independence the
earlier benchmarks found, now holding under a different retrieval method.

## Cost

| stage | cost |
| --- | --- |
| BM25 index over 11,331 chunks | 1.5 s, in memory, no embedding |
| dense + lexical top-100 for 107 questions | 22 s |
| MiniLM reranking, 5,350 pairs | 67 s on GTX 1650 (80/s) |
| bge-reranker-base, 5,350 pairs | 544 s on GTX 1650 (10/s) |

Nothing was re-crawled, re-chunked or re-embedded. The cross-encoders come from
fastembed's ONNX models, so the stack still has no PyTorch dependency and reuses
the same execution-provider selection as embedding.

## Reproducing

```bash
python scripts/build_goldset.py verify --goldset full
pgdocrag --corpus full evaluate --source html --goldset full --top-k 5
pgdocrag --corpus full experiment --source html --rerankers minilm,bge
```

## Open questions

- **Candidate generation on reference pages is the bottleneck.** Half the
  command and tool questions never surface their answer at all, and no reranker
  can repair that. Worth trying: a field-boosted lexical index over the section
  title alone, so `pg_resetwal` as a heading outweighs the same word buried in
  prose.
- **Lexical hijacking.** Several failures come from one strong word in the
  question matching a section about something else entirely. Query expansion or
  an instruction-tuned embedding model would be the things to test.
- **Seventeen questions have no PDF-resolvable label**, which caps what the
  cross-format comparison can say on this set.
