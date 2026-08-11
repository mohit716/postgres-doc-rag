# PostgresDocRAG

A RAG ingestion pipeline that turns the PostgreSQL manual into a searchable
vector store — **collect → extract → chunk → embed → store** — and then measures
whether it worked.

The same content is ingested twice, from the official HTML docs and from the
official PDF manual, on purpose. Real knowledge bases are never one clean format,
so normalising two very different renderings of identical content into one chunk
schema is both the interesting engineering problem and a built-in correctness
test: if extraction is sound, retrieval quality should barely care which format
it came from.

Ask a question, get the passage that answers it, with a citation:

```console
$ pgdocrag ask "How do I disconnect sessions that sit idle inside an open transaction?" -k 2

[1] score 0.817  Server Configuration > Client Connection Defaults > Statement Behavior > idle_in_transaction_session_timeout (integer)
    https://www.postgresql.org/docs/18/runtime-config-client.html#GUC-IDLE-IN-TRANSACTION-SESSION-TIMEOUT
    Terminate any session that has been idle (that is, waiting for a client
    query) within an open transaction for longer than the specified amount of
    time. If this value is specified without units, it is taken as milliseconds...

[2] score 0.786  Server Configuration > Client Connection Defaults > Statement Behavior > idle_session_timeout (integer)
    https://www.postgresql.org/docs/18/runtime-config-client.html#GUC-IDLE-SESSION-TIMEOUT
    Terminate any session that has been idle ... but not within an open
    transaction, for longer than the specified amount of time...
```

Note the second result: the retriever separates
`idle_in_transaction_session_timeout` from the nearly identically worded
`idle_session_timeout`, and ranks them the right way round.

## Results

PostgreSQL 18 documentation, a 66-page slice of the HTML docs and the matching
250-ish pages of the 3,130-page PDF manual, scored on 52 hand-written questions
phrased by intent rather than by keyword ("how do I stop dead rows from being
cleaned up automatically?" → `autovacuum`).

| | HTML | PDF |
| --- | --- | --- |
| chunks | 1,840 | 1,322 |
| Recall@5 | 78.8% | 80.8% |
| Recall@50 | 100% | 98.1% |
| MRR | 0.653 | 0.619 |
| nDCG@5 | 0.687 | 0.667 |

Retrieval works about equally well from either format, which is the result the
dual-format design was built to test.

Recall@50 says something more specific about where the remaining errors live:
every expected chunk that exists in a collection is returned within the top 50,
so the shortfall at k=5 is purely a question of ordering. The one exception is
not a ranking failure — the PDF collection contains no chunk for `TimeZone` at
all, making 98.1% its ceiling. Details in [docs/findings.md](docs/findings.md).

Those numbers describe a curated slice. Ingesting the **entire** manual — 1,146
HTML pages, 20,431 chunks — costs about 15 points of Recall@5: 63.5% for HTML and
65.4% for PDF. The two formats stay within two points of each other even at six
times the size, which is the more interesting half of the result. That
experiment, including the extraction bug that only appeared at scale, is in
[docs/full-corpus.md](docs/full-corpus.md).

## How it works

Each stage reads and writes JSONL on disk instead of calling the next stage in
memory. Stages are independently runnable, resumable and inspectable, and a
change to the chunker does not mean re-crawling 66 pages or re-embedding 1,840
chunks.

```
                  HTML docs                     PDF manual
                      |                              |
   collect      data/raw/html/*.html      data/raw/pdf/*.pdf  + manifests
                      |                              |
   extract      html_extract.py                pdf_extract.py
                      \                              /
                       -->  data/interim/*_docs.jsonl  <--   one Document schema
                                      |
   chunk                    data/chunks/*_chunks.jsonl        section-aware
                                      |
   embed            bge-small-en-v1.5 via ONNX, cached by content hash
                                      |
   store          Chroma: pgdocs_18_html | pgdocs_18_pdf      behind VectorStore
                                      |
                          ask / evaluate / compare
```

**Collect.** Every docs page carries `accesskey` navigation links, so following
`accesskey="n"` walks the manual in reading order — more reliable than parsing
the sitemap, and it supplies a page ordinal for free. Following `accesskey="u"`
gives the parent page, which is how a breadcrumb comes to span pages: a page
knows it is "Connection Settings", but only its ancestors know that lives under
"Server Configuration". Pages are cached with content hashes so re-runs are
idempotent, rate-limited to one request per second with an identifying
User-Agent. `robots.txt` disallows only `/docs/devel/`.

**Extract.** Two independent parsers that must satisfy the same `Document`
schema. The HTML manual contains two DocBook renderings — chapter pages
(`div.sect1`, headings in `div.titlepage`) and reference pages (`div.refentry`,
bare `h2` headings) — and handling only the first silently drops the entire
command reference. The PDF side reads hierarchy from the 4,023-entry bookmark
outline, strips running headers and page numbers by vertical band, detects code
by Courier font runs, and rebuilds paragraphs from line spacing. Both route
through one normalisation pass, without which the comparison would measure
formatting noise instead of extraction quality.

**Chunk.** Boundaries follow document structure, not a fixed token stride. The
unit is a leaf section, or a single `dl.variablelist` entry — one configuration
parameter, one command clause — which is why 69% of HTML chunks carry a citable
anchor. Undersized sections merge into their predecessor; oversized ones split on
sentence, line or table-row boundaries without ever cutting a fenced code block
or orphaning a table header. Every chunk is prefixed with its breadcrumb, so the
embedded text carries context its body alone would not.

**Embed and store.** `bge-small-en-v1.5` through ONNX Runtime: 384 dimensions, a
512-token window, no PyTorch dependency. Running on ONNX also makes CPU and GPU
a choice of execution provider rather than a change of model, which is why the
two produce interchangeable vectors. Vectors are cached by content hash, so
re-running after an unrelated change re-embeds only what actually changed. HTML
and PDF live in separate Chroma collections because they are the same content —
one collection would return each answer twice.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; use source .venv/bin/activate elsewhere
pip install -e .

pgdocrag collect  --source all --scope slice   # ~1 min HTML + 16 MB PDF
pgdocrag extract  --source all
pgdocrag chunk    --source all
pgdocrag embed    --source all                 # downloads the model on first run

pgdocrag ask "How do I build an index without blocking writes?"
pgdocrag evaluate --source both -k 5
pgdocrag compare
pgdocrag info
```

Embedding is the slow part: roughly 2-3 chunks/second on CPU, so about 15 minutes
for the slice.

To ingest the whole manual instead of the slice, run the same stages under a
second corpus. Raw pages and the vector cache are shared, while documents,
chunks, collections and reports are namespaced, so the benchmark above cannot be
overwritten:

```bash
pgdocrag --corpus full collect --source all    # 1,146 pages, ~18 min at 1 req/s
pgdocrag --corpus full extract --source all
pgdocrag --corpus full chunk   --source all
```

That comes to ~20,400 chunks, and on CPU they embed at 0.85 chunks/second rather
than 2-3 because full-corpus chunks are longer — around 6.7 hours.

An NVIDIA GPU collapses that to about 19 minutes. Embedding runs through ONNX
Runtime, so moving it to the GPU is an execution-provider change rather than a
model change: same `bge-small-en-v1.5` graph, same tokenizer, vectors that agree
with the CPU's to a cosine of 0.999996. On a 4 GB GTX 1650 it measures 16.7
against 0.85 chunks/second, a 19.6x speedup, peaking at 1.8 GB of VRAM.

```bash
pgdocrag device                                # what would embedding actually use?
pgdocrag --corpus full embed --source all --device cuda
python scripts/benchmark_embed.py --corpus full --source html
```

`--device cuda` refuses to fall back to CPU, which is what a long run wants;
`auto`, the default, prefers the GPU when there is one and stays on CPU
otherwise. Setup, verification and the measurements are in
[docs/full-corpus.md](docs/full-corpus.md), alongside the ingestion statistics
and what the larger corpus revealed.

## Structure-only extraction

The parsers were written against structural summaries, never against document
text. Two probe scripts produce those summaries:

```bash
python scripts/probe_structure.py https://www.postgresql.org/docs/18/sql-createtable.html
python scripts/probe_pdf.py data/raw/pdf/postgresql-18-A4.pdf --page 1903
```

They report tag and class frequencies, heading outlines, container trees, anchor
inventories, font distributions and the vertical position of repeated page
furniture — enough to generate a parser, without the prose. This mirrors working
on a confidential corpus, where the documents cannot be pasted into an external
model but a structural skeleton can be shared freely.

The findings and the parser decisions they drove are written up in
[docs/structure-notes.md](docs/structure-notes.md). Two examples of why this
matters in practice:

- On reference pages the parameter lines are **not** fully monospace: `work_mem`
  is Courier but the surrounding parentheses are Times-Roman, so the obvious
  "all spans are monospace" test matches nothing at all.
- Some `refsect1` anchors are content-addressed (`#id-1.9.3.85.6`) rather than
  semantic, so they work as URLs today but should not be assumed stable across
  releases.

## Design decisions

**Chunk size is derived from the model, not chosen.** `bge-small-en-v1.5`
truncates at 512 word-pieces, and it truncates *silently* — no error, no warning,
the tail simply never becomes searchable. So the budget is computed per chunk as
512 minus the breadcrumb minus special tokens, measured with the model's own
tokenizer, and enforced structurally rather than trusted from each splitter.
`scripts/validate_chunks.py` asserts the ceiling holds. (`all-MiniLM-L6-v2`, the
common default, truncates at 256 — half of a 512-token chunk would vanish.)

**Parameter chunks are never merged.** A parameter's anchor is the reason it
exists: it gives retrieval precision and an exact citation URL. Merging a short
one into its neighbour to hit a size target would trade both away.

**One vector store interface.** Chroma is the default because it is embedded and
needs no service, but retrieval depends only on the `VectorStore` protocol, so
adding Qdrant is an adapter rather than a rewrite.

**PDF scope mirrors the HTML corpus.** Rather than ingesting all 3,130 pages, the
PDF extractor matches the documents already collected as HTML, disambiguating
repeated titles ("Error Handling" appears under Server Configuration, PL/pgSQL
and ECPG) via the HTML breadcrumb. The two corpora therefore cover the same
content by construction.

## Repo layout

```
src/pgdocrag/
  config.py  schema.py  normalize.py  cli.py  query.py  status.py
  collect/   fetcher.py  html_crawler.py  pdf_fetch.py
  extract/   html_extract.py  pdf_extract.py
  chunk/     sectioner.py  tokenizer.py
  embed/     embedder.py  pipeline.py
  store/     base.py  chroma_store.py
  evaluate/  goldset.yaml  run_eval.py  compare_formats.py
scripts/     probe_structure.py  probe_pdf.py  peek.py  validate_chunks.py
             benchmark_embed.py
tests/       test_normalize.py  test_chunker.py
docs/        structure-notes.md  findings.md  full-corpus.md
data/        gitignored; fully reproducible from the CLI
```

## Limitations and next steps

- **No reranking, which is where the biggest win is.** Failures are a sibling
  parameter outranking the right one (`bgwriter_flush_after` for a question about
  `fsync`), not a missing answer — with the one PDF exception below. 96.2% of
  HTML answers already sit inside the top 20, so a cross-encoder reranking that
  window is the highest-value addition, with hybrid BM25 scoring second.
- **PDF anchors are reconstructed from a pattern, and it has gaps.** Parameter
  names are matched as `name (type)`, which rejects mixed-case names like
  `TimeZone` and lines listing several parameters at once — six anchors present
  in HTML are missing from the PDF collection as a result.
- **No answer generation.** The goal is returning the correct passage; retrieval
  quality is what the pipeline is judged on. Generation would sit on top
  unchanged.
- **Tables are not recovered from the PDF**, and `colspan`/`rowspan` are flattened
  even from HTML, because markdown cannot express them.
- **The gold set is single-label.** Each question expects one anchor or breadcrumb,
  so a genuinely relevant alternative passage scores as a miss.
