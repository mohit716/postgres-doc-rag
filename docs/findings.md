# Findings

Reproduce with `pgdocrag evaluate --source both -k 5` for the headline scores,
`pgdocrag evaluate --source both -k 50` for the depth figures, and
`pgdocrag compare`. Raw reports land in `data/reports/`.

## Setup

| | HTML | PDF |
| --- | --- | --- |
| source | 66 pages of the web docs | ~250 pages of the 3,130-page A4 manual |
| documents | 66 | 45 |
| chunks | 1,840 | 1,322 |
| chunks carrying an anchor | 69% | 35% |
| median chunk size | 105 tokens | 190 tokens |

Embedding model `BAAI/bge-small-en-v1.5` (384-dimensional, 512-token window) via
ONNX Runtime on CPU. Cosine distance, Chroma HNSW, one collection per format.

The PDF scope is derived from the HTML corpus rather than chosen independently,
so the two collections cover the same content and the comparison is not
confounded by ingesting different subsets.

## Retrieval quality

52 questions, each phrased by intent rather than by keyword, each labelled with
the anchor or breadcrumb that should be retrieved.

| | HTML | PDF |
| --- | --- | --- |
| Recall@5 | 78.8% (41/52) | 80.8% (42/52) |
| MRR | 0.653 | 0.619 |
| nDCG@5 | 0.687 | 0.667 |

Rank of the first relevant result:

| rank | 1 | 2 | 3 | 4 | 5 | miss |
| --- | --- | --- | --- | --- | --- | --- |
| HTML | 30 | 4 | 3 | 3 | 1 | 11 |
| PDF | 25 | 10 | 5 | 2 | 0 | 10 |

**The headline result: retrieval quality is essentially format-independent.** The
PDF collection finds a relevant chunk slightly more often, the HTML collection
ranks it first more often (30 vs 25), and the two land within a few points on
every metric. Since these are two genuinely different parsers — a DOM walk versus
font and coordinate analysis over a page layout — agreement at this level is
evidence that the normalisation layer is doing its job.

HTML's better MRR against PDF's better recall has a plausible cause: HTML chunks
are smaller (median 105 vs 190 tokens) because parameter-level anchors let the
chunker cut more finely. Smaller chunks are more precisely targeted when they
match, while larger chunks cover more ground and are more likely to contain the
answer somewhere.

## Recall by retrieval depth

Re-running the same gold set at `-k 50` separates two failure modes that
Recall@5 conflates: the answer being absent from the collection, and the answer
being present but ranked below the cutoff.

| | HTML | PDF |
| --- | --- | --- |
| Recall@5 | 78.8% (41/52) | 80.8% (42/52) |
| Recall@10 | 86.5% (45/52) | 88.5% (46/52) |
| Recall@20 | 96.2% (50/52) | 92.3% (48/52) |
| Recall@50 | 100.0% (52/52) | 98.1% (51/52) |

**Every expected chunk that exists in a collection is retrieved within the top
50.** HTML reaches 100%. PDF's one remaining failure is not a ranking problem:
scanning the metadata of all 1,322 stored PDF chunks finds no chunk carrying the
`GUC-TIMEZONE` anchor at all, so 98.1% is that collection's ceiling and it
reaches it exactly. The same scan confirms all 52 expectations are present in
the HTML collection.

Recall@5 computed from the depth run reproduces the shallow run exactly (41/52
and 42/52), so the ranking is stable and the cutoffs are directly comparable.

Ranks of every question that misses at k=5 in either format:

| expectation | HTML | PDF |
| --- | --- | --- |
| `archive_mode` | 6 | 4 |
| `lock_timeout` | 6 | 3 |
| `fsync` | 6 | 6 |
| `shared_buffers` | 7 | 8 |
| `autovacuum` | 11 | 8 |
| `search_path` | 11 | 3 |
| `hot_standby` | 12 | 14 |
| Trust Authentication | 13 | 6 |
| `max_wal_size` | 18 | 26 |
| `maintenance_work_mem` | 35 | 30 |
| `default_statistics_target` | 36 | 33 |
| VACUUM full rewrite | 2 | 11 |
| `TimeZone` | 1 | absent |

This changes what the highest-value fix is. Since the correct chunk is retrieved
whenever it exists and is merely ordered badly, a cross-encoder reranking the
top 20 — where
96.2% of HTML answers already sit — addresses the failure directly, and only has
to reorder candidates dense search already surfaces. Hybrid lexical scoring is
still worth adding, but it is no longer the whole story.

Two questions resist a top-20 rerank: `maintenance_work_mem` at rank 35 and
`default_statistics_target` at rank 36. Both describe an *effect* ("speed up
VACUUM", "more detailed column statistics") that the manual discusses at length
in command notes, so dozens of genuinely topical chunks crowd out the parameter
definition.

## Where retrieval fails

Eight questions miss in **both** formats: `shared_buffers`,
`maintenance_work_mem`, `max_wal_size`, `autovacuum`,
`default_statistics_target`, `fsync`, `hot_standby`, and Trust Authentication.
Failing identically in two independently written parsers points at retrieval
rather than extraction, and the depth table above confirms it — every one is
found further down the same ranking. Three more miss only in HTML
(`archive_mode`, `lock_timeout`, `search_path`) and two only in PDF: the VACUUM
full-rewrite question, and `TimeZone`, which is an extraction gap rather than a
ranking failure and is covered below.

The failure mode is consistent: a semantically adjacent sibling outranks the
correct parameter.

| question | expected | top result |
| --- | --- | --- |
| "turn off flushing writes to disk" | `fsync` | `bgwriter_flush_after` |
| "how large the WAL grows between checkpoints" | `max_wal_size` | `checkpoint_flush_after` |
| "memory dedicated to caching data pages" | `shared_buffers` | `vacuum_buffer_usage_limit` |
| "read only queries on a standby" | `hot_standby` | `max_standby_archive_delay` |

These are all cases where the query paraphrases a concept that the correct
document states in terms the embedding treats as near-identical to its
neighbours. Dense retrieval has no way to prefer the parameter *named* for the
concept. Two fixes apply, in this order: reranking the top 20 with a
cross-encoder, which the depth figures show is enough to recover most of these,
and hybrid retrieval — BM25 or SPLADE fused with the dense score — which adds
the lexical channel dense embeddings lack.

Worth noting what did *not* fail. 30 of 52 questions were answered at rank 1,
including several that require a genuine semantic leap with no lexical overlap to
lean on:

- "how do I tell the planner that random disk reads are cheap on SSDs?" →
  `random_page_cost`
- "how much memory the operating system caches" → `effective_cache_size`
- "how do I disconnect sessions that sit idle inside an open transaction?" →
  `idle_in_transaction_session_timeout`, correctly ranked above the nearly
  identically worded `idle_session_timeout`

That last case is the encouraging one: the two parameters differ by one clause in
their descriptions, and the retriever ordered them correctly.

## Cross-format extraction consistency

429 section pairs were joined across formats — by anchor where one exists, and by
the last two breadcrumb elements otherwise.

| metric | value |
| --- | --- |
| median word overlap (Jaccard) | 1.000 |
| median character similarity (difflib) | 1.000 |
| median embedding cosine | 0.9981 (393 single-chunk pairs) |
| pairs at difflib >= 0.95 | 414 of 429 (96.5%) |
| pairs at difflib < 0.50 | 5 |

**Where the two formats describe the same section, they agree almost exactly.** A
median cosine of 0.9981 means the retriever sees the HTML and PDF renderings of a
section as very nearly the same point in embedding space, which is the strongest
form of the claim: the normalisation is good enough that downstream behaviour does
not depend on input format.

### The three metrics disagree, informatively

The worst pair by character similarity is `COPY > Synopsis` at difflib 0.05 — with
word overlap of 0.98 and near-identical length (714 vs 710 characters). Same
words, almost no shared character runs: the PDF wraps the command synopsis at page
width while the HTML wraps it at its own margins, so line breaks fall in
completely different places. `data_sync_retry` (0.35 / 0.99) and `ssl_ciphers`
(0.58 / 1.00) show the same signature.

This is why three metrics are reported instead of one. Character-level similarity
is the strictest and flags reflowed text as a difference; word overlap ignores
arrangement; embedding cosine measures the only thing that actually affects
retrieval. Judged by difflib alone, these would look like extraction failures.
They are not.

### Coverage is limited by granularity, not disagreement

Only 30.0% of HTML sections and 55.4% of PDF sections found a counterpart. The
unmatched breakdown explains why:

- **641 HTML anchors have no PDF equivalent.** Nearly all are command clause
  anchors (`SQL-CREATETABLE-TEMPORARY`, `SQL-CREATETABLE-UNLOGGED`). The HTML docs
  publish an anchor for every `dt` in a variable list, while the PDF bookmark
  outline stops above clause level and clause names carry no distinguishing font
  signature. The PDF extractor recovers *configuration parameters* by their
  `name (type)` shape, but command clauses have no comparable pattern.
- **360 HTML sections have no PDF section** at the same breadcrumb, mostly small
  reference subsections that the PDF merges into their parent.

So the gap is a structural granularity difference between the two publications,
not a case of the same content being extracted differently. It is also why the
HTML collection carries 1,840 chunks against the PDF's 1,322 from equivalent
source material.

### Real extraction gaps the comparison exposed

Five pairs fall below 0.50 character similarity for reasons that *are* extraction
defects, all on the PDF side and all the same shape — the PDF version is several
times longer than the HTML one:

| section | HTML chars | PDF chars |
| --- | --- | --- |
| `external_pid_file` | 178 | 1,450 |
| `vacuum_cost_limit` | 118 | 562 |
| `autovacuum_analyze_scale_factor` | 346 | 1,437 |

**The last parameter in a PDF section absorbs the trailing content of that
section.** Parameter boundaries in the PDF are inferred from the next parameter
line, so the final parameter has no terminator other than the end of its outline
section — and where that section boundary is imprecise, the surplus text lands in
the last parameter. The HTML side has explicit `dt`/`dd` markup and so has exact
boundaries.

This is a bounded, well-understood defect affecting the tail parameter of each
PDF section. It is also a good argument for the general principle that structured
markup should be preferred as the primary source when a document is available in
more than one format, with the PDF as a fallback for content the HTML lacks.

### Parameters the PDF anchor pattern misses

Chasing the one `TimeZone` failure in the depth run exposed a second bounded
defect. The PDF carries no anchors, so they are reconstructed from parameter
lines matching `name (type)` — a pattern that requires a lowercase name and one
parameter per line. Eight GUC anchors present in the HTML collection are absent
from the PDF collection, six of them for that reason:

| anchor | how the PDF renders it | why the pattern rejects it |
| --- | --- | --- |
| `GUC-TIMEZONE` | `TimeZone (string)` | mixed-case name |
| `GUC-DATESTYLE` | `DateStyle (string)` | mixed-case name |
| `GUC-INTERVALSTYLE` | `IntervalStyle (enum)` | mixed-case name |
| `GUC-RECOVERY-TARGET-LSN` | `recovery_target_lsn (pg_lsn)` | underscore in the type |
| `GUC-DEBUG-PRINT-PARSE` | `debug_print_parse (boolean) / ...` | several parameters share one line |
| `GUC-LOG-STATEMENT-STATS` | `log_statement_stats (boolean) / ...` | several parameters share one line |

The scale is small — the PDF carries 417 GUC anchors against HTML's 403, so this
is a handful of specific rejections rather than a coverage deficit. Only
`GUC-TIMEZONE` is exercised by the gold set, which is why the other five cost
nothing in the score and went unnoticed until recall was measured at depth.
Widening the name pattern to accept capitals would recover three of the six.

The general lesson is that a reconstructed join key is only as good as the
pattern that produces it, and a gold set exercises only the fraction of keys it
happens to name. Measuring whether an expected answer is *present* is a
different check from measuring whether it *ranks*, and it caught what the
retrieval metrics could not.

## Conclusions

1. Two independent parsers over the same content converge to a median embedding
   cosine of 0.9981, and retrieval scores land within a few points of each other.
   Format-independent ingestion is achievable, but it depends on an explicit
   shared normalisation stage — unicode folding, de-hyphenation, whitespace and
   quote handling — not on the parsers happening to agree.
2. Structure-only parser development was sufficient. Every parser decision in
   this pipeline traces to a probe output rather than to reading documentation
   prose, and the resulting extraction agrees with an independent implementation
   to within a rounding error.
3. Chunk granularity is set by what the source format exposes. HTML anchors made
   1,363 parameter-level chunks possible; the PDF outline supports only 468. This
   is the most consequential downstream difference between the two formats.
4. The binding constraint on answer quality is result *ordering*, not retrieval.
   Every expected chunk that exists in a collection comes back within the top 50
   — Recall@50 is 100% for HTML and 98.1% for PDF, the latter being that
   collection's coverage ceiling rather than a ranking failure. The ~20%
   shortfall at k=5 is therefore entirely a ranking problem, which makes
   reranking the top 20 a more direct fix than replacing the retriever.
5. Measuring recall at one cutoff hides the distinction between "not found" and
   "found too late". Separating them turned a vague "retrieval is weak"
   conclusion into a specific ranking problem plus one concrete extraction bug.
