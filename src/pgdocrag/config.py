"""Central configuration: paths, source URLs, model and chunking parameters."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

# Raw downloads and the embedding cache are shared by every corpus. Both are
# content-addressed, so a larger run reuses whatever a smaller one already paid
# for instead of re-fetching pages or re-embedding identical text.
RAW_HTML_DIR = DATA_DIR / "raw" / "html"
RAW_PDF_DIR = DATA_DIR / "raw" / "pdf"
EMBED_CACHE_DIR = DATA_DIR / "embed_cache"


# --- Corpora ----------------------------------------------------------------

# Two corpora coexist. `slice` is the benchmark the published metrics describe:
# a curated sample chosen to exercise both HTML structure families. `full`
# ingests the entire manual. Everything derived from the source — documents,
# chunks, vectors, reports — is namespaced per corpus, so running one experiment
# can never overwrite the other's results.
DEFAULT_CORPUS = "slice"
CORPORA = ("slice", "full")

CORPUS = DEFAULT_CORPUS
PDF_SCOPE_RULE = "outermost"
CORPUS_DIR = DATA_DIR
INTERIM_DIR = DATA_DIR / "interim"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
REPORTS_DIR = DATA_DIR / "reports"
CHROMA_DIR = VECTORSTORE_DIR / "chroma"
HTML_MANIFEST_PATH = RAW_HTML_DIR / "manifest.json"
ALL_DIRS: list[Path] = []


def use_corpus(name: str) -> None:
    """Point every derived-artifact path at the named corpus.

    Runs once at import time and again from the CLI's `--corpus` option. The CLI
    imports stage modules lazily, so this always lands before any stage reads a
    path.
    """
    global CORPUS, CORPUS_DIR, INTERIM_DIR, CHUNKS_DIR, VECTORSTORE_DIR
    global REPORTS_DIR, CHROMA_DIR, HTML_MANIFEST_PATH, ALL_DIRS, PDF_SCOPE_RULE

    if name not in CORPORA:
        raise ValueError(
            f"Unknown corpus {name!r}; expected one of {', '.join(CORPORA)}"
        )

    CORPUS = name
    # Matched PDF outline entries nest, and which one wins changes the result.
    # "innermost" is the truer mirror of the HTML corpus — one page, one document
    # — and the only rule that survives full scale, where the manual's own parts
    # match and "outermost" collapses 1,134 matches into 26 documents spanning
    # hundreds of pages each. The default corpus stays on "outermost" because its
    # published metrics were measured there, not because it is the better rule.
    PDF_SCOPE_RULE = "outermost" if name == DEFAULT_CORPUS else "innermost"
    # The default corpus keeps the flat layout its artifacts were written under,
    # so introducing a second corpus neither moves nor invalidates them.
    CORPUS_DIR = DATA_DIR if name == DEFAULT_CORPUS else DATA_DIR / name
    INTERIM_DIR = CORPUS_DIR / "interim"
    CHUNKS_DIR = CORPUS_DIR / "chunks"
    VECTORSTORE_DIR = CORPUS_DIR / "vectorstore"
    REPORTS_DIR = CORPUS_DIR / "reports"
    CHROMA_DIR = VECTORSTORE_DIR / "chroma"
    # The manifest, not the page cache, is what bounds a corpus: the cache is
    # shared, and the extractor walks the manifest. The default corpus keeps its
    # original location for the same reason as the directories above.
    HTML_MANIFEST_PATH = (
        RAW_HTML_DIR / "manifest.json"
        if name == DEFAULT_CORPUS
        else CORPUS_DIR / "html_manifest.json"
    )
    ALL_DIRS = [
        RAW_HTML_DIR,
        RAW_PDF_DIR,
        EMBED_CACHE_DIR,
        INTERIM_DIR,
        CHUNKS_DIR,
        VECTORSTORE_DIR,
        REPORTS_DIR,
    ]


def ensure_dirs() -> None:
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


use_corpus(os.environ.get("PGDOCRAG_CORPUS", DEFAULT_CORPUS))


# --- Source data ------------------------------------------------------------

PG_VERSION = "18"
DOCS_BASE_URL = f"https://www.postgresql.org/docs/{PG_VERSION}/"
PDF_URL = (
    f"https://www.postgresql.org/files/documentation/pdf/"
    f"{PG_VERSION}/postgresql-{PG_VERSION}-A4.pdf"
)
PDF_FILENAME = f"postgresql-{PG_VERSION}-A4.pdf"

USER_AGENT = (
    "PostgresDocRAG/0.1 (portfolio RAG ingestion project; "
    "polite crawler, 1 req/sec)"
)
CRAWL_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 30

# robots.txt disallows /docs/devel/ only; /docs/<version>/ is crawlable.
DISALLOWED_URL_FRAGMENTS = ("/docs/devel/", "/search/", "/account/", "/admin/")

# Seed pages for the development slice, each walked forward for N pages via the
# "next" navigation link. The slice covers both HTML structure families: the
# configuration chapters (div.sect*, dl.variablelist, GUC-* anchors) and the
# command reference (div.refentry, div.refsect*, pre.synopsis).
#
# Reference pages are listed individually because walking alphabetically from
# sql-commands.html spends the whole budget on ALTER statements and never
# reaches the commands users actually ask about.
SLICE_SEEDS: list[tuple[str, int]] = [
    ("runtime-config.html", 20),
    ("client-authentication.html", 6),
    ("sql-createtable.html", 1),
    ("sql-altertable.html", 1),
    ("sql-createindex.html", 1),
    ("sql-select.html", 1),
    ("sql-insert.html", 1),
    ("sql-update.html", 1),
    ("sql-delete.html", 1),
    ("sql-copy.html", 1),
    ("sql-vacuum.html", 1),
    ("sql-analyze.html", 1),
    ("sql-explain.html", 1),
    ("sql-grant.html", 1),
    ("sql-createrole.html", 1),
    ("sql-createdatabase.html", 1),
    ("app-psql.html", 1),
    ("app-pgdump.html", 1),
    ("app-pgbasebackup.html", 1),
]

FULL_CRAWL_SEED = "index.html"


# --- PDF layout -------------------------------------------------------------

# Measured with scripts/probe_pdf.py on the A4 manual: pages are 841.9pt tall,
# the running chapter header sits at y/height 0.040 and the page number at 0.940,
# while body text starts no higher than 0.083.
PDF_HEADER_BAND = 0.065
PDF_FOOTER_BAND = 0.925

# Body text is Times-Roman at 10pt, code is Courier at 10pt, and headings are
# Helvetica-Bold at 14.4pt and above.
PDF_MONO_FONT_HINTS = ("courier", "mono")
PDF_BODY_SIZE = 11.5

# A vertical gap this many times the median line spacing starts a new paragraph.
PDF_PARAGRAPH_GAP_RATIO = 1.55

# A single monospace line is usually an inline-heavy prose line or a term, not a
# code block, so a run of at least this many is required to emit code.
PDF_MIN_CODE_RUN = 2


# --- Embedding model --------------------------------------------------------

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# bge-small-en-v1.5 truncates input at 512 word-pieces. Anything above this is
# silently dropped by the model rather than raising, so the chunk budget below
# is derived from it instead of being chosen independently.
MODEL_MAX_TOKENS = 512

# Every chunk is prefixed with a section breadcrumb, which consumes part of the
# model's window. Reserving it here keeps body text from being truncated away.
BREADCRUMB_TOKEN_RESERVE = 48

MAX_CHUNK_TOKENS = MODEL_MAX_TOKENS - BREADCRUMB_TOKEN_RESERVE
TARGET_CHUNK_TOKENS = 320

# Sections below this size are merged into a neighbour rather than stored alone,
# since one-sentence chunks retrieve poorly.
MIN_CHUNK_TOKENS = 48

# Overlap applies only when a single oversized section must be split.
CHUNK_OVERLAP_RATIO = 0.12

# bge models were trained with an asymmetric prefix on the query side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# --- Embedding device -------------------------------------------------------

# Which ONNX Runtime execution provider embeds chunks. The model, its weights
# and the tokenizer are identical either way — only the kernels differ — so
# vectors stay comparable across devices and the cache needs no device key.
#
# "auto" takes CUDA when ONNX Runtime can genuinely offer it and stays on CPU
# otherwise. "cuda" refuses to fall back, which is what a multi-hour job wants:
# a misconfigured GPU should fail in seconds, not silently cost hours of CPU.
EMBED_DEVICES = ("auto", "cuda", "cpu")
DEFAULT_EMBED_DEVICE = "auto"
EMBED_DEVICE = os.environ.get("PGDOCRAG_EMBED_DEVICE", DEFAULT_EMBED_DEVICE)


# --- Vector store -----------------------------------------------------------

COLLECTION_PREFIX = "pgdocs"


# Collection names carry no corpus label: each corpus has its own Chroma
# directory, so the names cannot collide and the default corpus keeps the
# collections its published metrics were measured against.
def collection_name(source_format: str) -> str:
    """HTML and PDF live in separate collections: they are the same content in
    two formats, so a single collection would return duplicate hits."""
    return f"{COLLECTION_PREFIX}_{PG_VERSION}_{source_format}"


DEFAULT_TOP_K = 5
