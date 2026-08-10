"""Central configuration: paths, source URLs, model and chunking parameters."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_HTML_DIR = DATA_DIR / "raw" / "html"
RAW_PDF_DIR = DATA_DIR / "raw" / "pdf"
INTERIM_DIR = DATA_DIR / "interim"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
EMBED_CACHE_DIR = DATA_DIR / "embed_cache"
REPORTS_DIR = DATA_DIR / "reports"

ALL_DIRS = [
    RAW_HTML_DIR,
    RAW_PDF_DIR,
    INTERIM_DIR,
    CHUNKS_DIR,
    VECTORSTORE_DIR,
    EMBED_CACHE_DIR,
    REPORTS_DIR,
]


def ensure_dirs() -> None:
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


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


# --- Vector store -----------------------------------------------------------

CHROMA_DIR = VECTORSTORE_DIR / "chroma"
COLLECTION_PREFIX = "pgdocs"


def collection_name(source_format: str) -> str:
    """HTML and PDF live in separate collections: they are the same content in
    two formats, so a single collection would return duplicate hits."""
    return f"{COLLECTION_PREFIX}_{PG_VERSION}_{source_format}"


DEFAULT_TOP_K = 5
