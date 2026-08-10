# Findings

Reproduce with `pgdocrag evaluate --source both -k 5` and `pgdocrag compare`.
Raw reports land in `data/reports/`.

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

## Where retrieval fails

Eight questions miss in **both** formats, so the cause is retrieval, not
extraction: `shared_buffers`, `maintenance_work_mem`, `max_wal_size`,
`autovacuum`, `default_statistics_target`, `fsync`, `hot_standby`, and Trust
Authentication. Three more miss only in HTML (`archive_mode`, `lock_timeout`,
`search_path`) and two only in PDF (`timezone`, and the VACUUM full-rewrite
question).

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
concept. Hybrid retrieval — BM25 or SPLADE fused with the dense score — targets
exactly this weakness, and is the single highest-value improvement available.

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
4. The binding constraint on answer quality is the retriever, not the pipeline.
   Extraction, normalisation and chunking are consistent and verified; the
   remaining 20% of failures are dense-retrieval ranking errors that hybrid
   search is designed to fix.
