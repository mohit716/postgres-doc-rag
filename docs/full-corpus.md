# Full-corpus experiment

A second ingestion run over the **entire** PostgreSQL 18 manual, kept separate
from the benchmark that [findings.md](findings.md) reports. The published metrics
were measured on a curated slice; nothing here replaces them.

**Status: complete.** 1,146 HTML pages and the whole PDF manual, ingested,
embedded on the GPU and evaluated against the same 52-question gold set. The
headline is that **Recall@5 falls from 78.8% to 63.5% at six times the corpus
size, while the two formats stay within two points of each other** — see
[Retrieval at full scale](#retrieval-at-full-scale).

## Isolation

Two corpora coexist without either being able to overwrite the other. The split
follows what is content-addressed and what is derived:

| | shared | namespaced per corpus |
| --- | --- | --- |
| raw HTML pages | yes | |
| PDF manual | yes | |
| vector cache | yes | |
| HTML manifest | | `data/full/html_manifest.json` |
| documents, chunks | | `data/full/interim/`, `data/full/chunks/` |
| Chroma collections | | `data/full/vectorstore/chroma/` |
| evaluation reports | | `data/full/reports/` |

Sharing the raw downloads and the vector cache is what makes the second
experiment affordable: the full crawl re-fetched only the 1,079 pages the slice
did not already have, and re-embedding never repeats text already vectorised.
Both caches are keyed by content, so reuse cannot silently serve stale data.

What bounds a corpus is therefore not the page cache but the **manifest** — the
extractor walks the manifest, not the directory. This is the one subtlety worth
remembering: after the full crawl, `data/raw/html/` holds 1,146 pages while the
slice still extracts exactly 67, because its manifest still lists 67.

The slice keeps the flat layout its artifacts were written under, so introducing
the second corpus neither moved nor invalidated them. Verified by re-running
`extract` and `chunk` on the slice after the refactor: all four artifact files
re-hashed byte-identically, and both Chroma collections still hold 1,840 and
1,322 vectors.

Select a corpus with a global option, before the subcommand:

```bash
pgdocrag --corpus full collect     # scope defaults to match the corpus
pgdocrag --corpus full info
pgdocrag info                      # unqualified: the slice, unchanged
```

## Ingestion results

| | slice HTML | full HTML | slice PDF | full PDF |
| --- | --- | --- | --- | --- |
| pages | 67 | 1,146 | ~250 | 3,066 |
| documents | 67 | 1,146 | 45 | 26 |
| chunks | 1,840 | 11,331 | 1,322 | 9,100 |
| carrying an anchor | 69% | 73% | 35% | 7% |
| median chunk | 105 tok | 140 tok | 189 tok | 230 tok |
| over the 512-token model limit | 0 | 0 | 0 | 0 |

The HTML side scales cleanly. Chunk count grows slightly sub-linearly with pages
(27 chunks/page on the config-heavy slice against 10 across the whole manual,
which is mostly prose), anchor coverage *improves* to 73% because the command
reference is dense with `refsect` anchors, and the token ceiling still holds
everywhere — the structural enforcement described in the README is not a
small-corpus artifact.

## The PDF scope selector, and why scale broke it

The first full PDF run produced **fewer** documents than the slice: 26 against
45. Diagnosing that turned out to be the most instructive part of the
experiment.

PDF scope is derived from the HTML corpus by design, so both formats cover the
same content. The extractor matches HTML document titles against the manual's
4,023-entry bookmark outline, and matched entries nest — a matched chapter can
contain matched subsections — so a rule is needed to decide which wins. The
original rule kept the **outermost** match, on the reasoning that selecting a
chapter should not also select its subsections as separate documents.

That rule depends on chapter-level pages being excluded, which `_html_wanted()`
does for navigation-only pages. At slice scale it holds. At full scale it does
not: the manual's own top-level parts have HTML pages carrying enough content to
qualify, so they match, and the outermost rule collapses 1,134 matches into the
26 parts themselves — "Reference" as a single document spanning pages 1633-2394
with 1,585 sections.

The fix is to keep the **innermost** match instead, which mirrors the HTML side
more faithfully: one page, one document, exactly as HTML excludes navigation-only
parents. The cost is that any chapter preamble sitting above the first matched
subsection is skipped, which is the same content HTML drops anyway.

Inverting the rule outright was not an option, because it is **not** a no-op on
the slice: "Chapter 19. Server Configuration" and "Chapter 20. Client
Authentication" are matched containers there, and innermost selection would have
taken the slice from 45 documents to 66 and invalidated the published metrics.
So the rule is per corpus, set in `use_corpus()` alongside the other choices that
pin the slice to its original behaviour:

```python
PDF_SCOPE_RULE = "outermost" if name == DEFAULT_CORPUS else "innermost"
```

The slice stays on `outermost` because its metrics were measured there, not
because it is the better rule. Re-measuring the slice under `innermost` would be
a separate experiment.

| | before | after |
| --- | --- | --- |
| documents | 26 | 1,022 |
| HTML titles with no PDF counterpart | 1,065 | 106 |
| chunks | 7,896 | 9,100 |
| gold-set expectations present | — | 51 of 52 |

Verified byte-identical on the slice afterwards: 45 documents, same SHA-256.

Anchor share stayed at 7%, which initially looked like the fix had not worked.
It had — the share is dilution, not a gap. Parameter anchors exist only in the
configuration chapters, and the full manual buries them under prose from the
tutorial, internals and command reference. The absolute count went *up*, from
468 in the slice to 674. What matters for evaluation is whether the gold set's
targets are present, and 51 of 52 are; the exception is `GUC-TIMEZONE`, absent
for the unrelated mixed-case reason documented in the README.

## Retrieval at full scale

The same 52 questions, the same model, the same gold set, scored against a
corpus roughly six times larger.

| | slice HTML | full HTML | slice PDF | full PDF |
| --- | --- | --- | --- | --- |
| chunks | 1,840 | 11,331 | 1,322 | 9,100 |
| Recall@5 | 78.8% | **63.5%** | 80.8% | **65.4%** |
| Recall@50 | 100% | 92.3% | 98.1% | 92.3% |
| MRR | 0.653 | 0.429 | 0.619 | 0.444 |
| nDCG@5 | 0.687 | 0.480 | 0.667 | 0.495 |

**Scale costs about 15 points of Recall@5.** Six times the candidate pool moves
roughly one question in seven out of the top 5. Nothing about extraction or
chunking changed, so this is dense retrieval alone: more near-duplicate
neighbours to rank against. Anyone quoting a retrieval score from a curated
subset should expect to lose this much on the real corpus.

**Format independence survives.** HTML 63.5% against PDF 65.4%, MRR 0.429
against 0.444 — the same few-point spread as the slice, with PDF again slightly
ahead on recall. Two very different parsers still agree at six times the size,
which is the result the dual-format design exists to test.

**What does not survive is "the answer is always in the top 50".** On the slice,
every expectation present in a collection was returned within 50 results; the
shortfall at k=5 was purely ordering. At full scale, four questions per format
are absent from the top 50 entirely:

| | missing at k=50 |
| --- | --- |
| both formats | `maintenance_work_mem`, `max_wal_size`, `default_statistics_target` |
| HTML only | `hot_standby` (PDF finds it at rank 48) |
| PDF only | `TimeZone` (absent from the collection; the known mixed-case gap) |

Three of the four are the *same* in both formats, and the coverage check
confirms those chunks exist in both. Two independently written parsers failing
identically points at the question wording and the embedding, not extraction —
these are the questions phrased furthest from their parameter's vocabulary
("speed up VACUUM and CREATE INDEX by giving them more memory"), competing now
against the whole manual's discussion of vacuuming and indexing.

Reranking looks even more valuable here than on the slice: 15 HTML answers and
14 PDF answers land between ranks 6 and 50, several as deep as 38, 45 and 49.
A cross-encoder over the top 50 has a lot to recover.

## Cost

Measured on this machine, an ASUS TUF FX505DT: laptop CPU, and a GTX 1650 with
4 GB of VRAM.

| stage | CPU | GPU |
| --- | --- | --- |
| collect (HTML) | 18.4 min | network-bound at 1 req/s, not a compute stage |
| extract (HTML) | 80 s | |
| extract (PDF) | 26 s | whole 3,130-page manual |
| chunk (both) | 57 s | |
| embed HTML, 11,331 chunks | ~3.7 h projected | 10.2 min actual, 18.4 chunks/s |
| embed PDF, 9,100 chunks | ~3.0 h projected | 7.7 min actual, 18.0 chunks/s |
| **embedding total** | **~6.7 h** | **18 min** |

Embedding dominates on CPU and effectively disappears on GPU. The benchmark
predicted 16.7 chunks/s; the real run sustained 18.4, about **21x** the CPU's
0.85. See [GPU embedding](#gpu-embedding) for how that is set up and verified.

The shared vector cache also earned its keep: 767 of the PDF corpus's 9,100
chunks were already vectorised by the slice and cost nothing to reuse.

The CPU rate is worth stating carefully because it is easy to over-estimate. A
benchmark over the slice's first 256 chunks gave 3.0 chunks/s; the real
full-corpus rate is 0.85. The difference is token count, not overhead —
full-corpus chunks are longer (median 140 against 105, with 583 table chunks and
2,123 mixed) and embedding cost scales with word-pieces. Extrapolating from a
head sample overstates throughput by roughly 3.5x, so
`scripts/benchmark_embed.py` samples at an even stride through the corpus
instead. Sampling that way, the same CPU measures 0.62 chunks/s.

One caveat that matters more on CPU than GPU: the vector cache is written once
per format, at the end, so interrupting mid-format loses that format's progress.
HTML completing does bank its vectors before PDF begins.

## GPU embedding

Embedding is the only stage that benefits from a GPU, and on this hardware it
turns the experiment from an overnight job into a coffee break. The model does
not change: the same `bge-small-en-v1.5` ONNX graph and the same tokenizer, run
through ONNX Runtime's CUDA execution provider instead of its CPU one. Only the
kernels differ, which is what keeps results comparable.

The GPU stack lives in a **separate virtual environment**. `onnxruntime-gpu`
replaces `onnxruntime` rather than coexisting with it, and `.venv` is the
environment the published metrics were measured in, so it is left alone:

```bash
python -m venv .venv-gpu
.venv-gpu\Scripts\pip install -e .
.venv-gpu\Scripts\pip uninstall -y onnxruntime
.venv-gpu\Scripts\pip install onnxruntime-gpu==1.23.2 ^
    nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 ^
    nvidia-cufft-cu12 nvidia-curand-cu12
```

No CUDA toolkit is required — only the NVIDIA driver. The CUDA and cuDNN
libraries come from the `nvidia-*-cu12` wheels, and ONNX Runtime 1.21+ loads
them from `site-packages` via `preload_dlls()`. Expect ~2.5 GB of downloads,
mostly cuDNN; pip's default 15-second read timeout is not generous enough for a
737 MB wheel, so `--timeout 300` is worth passing.

Confirm the device before starting anything long:

```console
$ pgdocrag device
CUDA available    True  (CUDAExecutionProvider is registered)
onnxruntime       1.23.2
providers built   TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
session providers CUDAExecutionProvider, CPUExecutionProvider
embedding would run on GPU via CUDAExecutionProvider
```

Two traps make that command worth having, and both were hit here:

- **A registered provider is not a working one.** `get_available_providers()`
  listed `CUDAExecutionProvider` while no CUDA library was installed at all — the
  list reflects how the wheel was compiled, not what can initialise. Only a live
  session settles it, so `Embedder` reads the providers back off the session and
  `--device cuda` refuses to start if CUDA is not among them.
- **On a GPU build, the default is already CUDA.** Passing no provider list does
  not mean CPU; ORT's default leads with CUDA, so `--device cpu` has to name the
  CPU provider explicitly. Without that, a "CPU" benchmark silently measures the
  GPU and reports a 1.0x speedup.

Measured with `python scripts/benchmark_embed.py --corpus full --source html
--sample 256 --baseline-rate 0.85`:

| | chunks/s | full corpus | VRAM |
| --- | --- | --- | --- |
| GPU (CUDA) | 16.69 | ~19 min | 1.1 GB, peak 1.8 GB of 4 GB |
| CPU, same sample | 0.62 | ~8.7 h | |
| CPU, observed in the real run | 0.85 | ~6.3 h | |

Batch size 64 peaks at 1.8 GB of the 4 GB card, so there is headroom; reduce it
if a larger model is ever swapped in.

Switching device does not move the vectors. Across the 256-chunk sample, CUDA
and CPU agree to a minimum cosine of 0.999996 and a maximum absolute component
difference of 8.6e-4 — ordinary floating-point kernel differences. The end-to-end
check is stronger: re-running the slice evaluation with queries embedded on the
GPU, against collections built on the CPU, reproduces every published figure
exactly (Recall@5 78.8%/80.8%, MRR 0.653/0.619, nDCG 0.687/0.667). Cached
vectors therefore stay valid across a device switch, which is why the cache key
is content only and carries no device.

## Reproducing

Every stage is on disk, so the results above re-derive from:

```bash
.venv-gpu\Scripts\pgdocrag --corpus full evaluate --source both -k 5
.venv-gpu\Scripts\pgdocrag --corpus full evaluate --source both -k 50
.venv-gpu\Scripts\pgdocrag --corpus full info
```

Rebuilding the vectors from the chunks takes 18 minutes:

```bash
.venv-gpu\Scripts\pgdocrag --corpus full embed --source all --device cuda --reset
```

`--device cuda` rather than the default `auto` is deliberate for a long run: it
fails in seconds if the GPU is unavailable instead of quietly spending hours on
the CPU.

On a fresh machine — a rented instance, say — `data/` is gitignored, so the
corpus is rebuilt from the CLI rather than copied. The crawl is the only step
that touches the network:

```bash
pip install -e .
pgdocrag --corpus full collect --source all    # ~18 min, 1 req/s
pgdocrag --corpus full extract --source all
pgdocrag --corpus full chunk   --source all
pgdocrag --corpus full embed   --source all --device cuda
```

Embedding is the only stage worth renting hardware for, and a GPU is a far
better answer than more cores: a four-year-old 4 GB laptop GPU already beats
this CPU by ~20x. If the instance has an NVIDIA card, install the GPU stack
above and pass `--device cuda`; if it does not, `auto` falls back to CPU and the
run costs hours instead of minutes.

## Still open

- **The gold set no longer fits the corpus.** It was written against the slice,
  so it exercises configuration parameters almost exclusively, while the full
  corpus is mostly command reference, tutorial and internals. The 52 questions
  now probe a small and unrepresentative corner of what was ingested, and every
  score above should be read with that in mind.
- **Reranking is untested and now clearly the highest-value addition.** Roughly
  a third of correct answers sit between ranks 6 and 50.
- **Cross-format `compare` has not been re-run** on the finer PDF documents.
- **The slice under `innermost` selection** is unmeasured: it would take that
  corpus from 45 PDF documents to 66 and is a separate experiment, deliberately
  not folded into the published numbers.
