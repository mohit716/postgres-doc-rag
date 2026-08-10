"""Command line entry point for the ingestion pipeline.

Stage modules are imported inside each command: fastembed and chromadb pull in
onnxruntime and take seconds to import, which would otherwise be paid even by
`pgdocrag --help`.
"""

from __future__ import annotations

import sys
from enum import Enum

import typer

from . import config

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="RAG ingestion pipeline for PostgreSQL documentation (HTML + PDF).",
)


class Source(str, Enum):
    html = "html"
    pdf = "pdf"
    all = "all"


class Scope(str, Enum):
    slice = "slice"
    full = "full"


def _configure_stdout() -> None:
    """The docs contain em dashes; the Windows console codepage cannot encode them."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _wants(source: Source, target: str) -> bool:
    return source in (Source.all, Source(target))


@app.callback()
def main() -> None:
    _configure_stdout()
    config.ensure_dirs()


@app.command()
def collect(
    source: Source = typer.Option(Source.all, help="Which source format to collect."),
    scope: Scope = typer.Option(Scope.slice, help="Development slice or the full manual."),
    refresh: bool = typer.Option(False, help="Re-download even if cached."),
) -> None:
    """Stage 1: fetch raw documentation to data/raw/."""
    if _wants(source, "html"):
        from .collect import html_crawler

        typer.secho(f"Crawling HTML docs ({scope.value})", fg=typer.colors.CYAN)
        pages = (
            html_crawler.crawl_slice(refresh=refresh)
            if scope is Scope.slice
            else html_crawler.crawl_full(refresh=refresh)
        )
        typer.secho(f"Collected {len(pages)} HTML pages", fg=typer.colors.GREEN)

    if _wants(source, "pdf"):
        from .collect import pdf_fetch

        typer.secho("Downloading PDF manual", fg=typer.colors.CYAN)
        path = pdf_fetch.download(refresh=refresh)
        typer.secho(f"PDF ready at {path}", fg=typer.colors.GREEN)


@app.command()
def extract(
    source: Source = typer.Option(Source.all, help="Which source format to extract."),
) -> None:
    """Stage 2: parse raw sources into normalised documents."""
    if _wants(source, "html"):
        from .extract import html_extract

        typer.secho("Extracting HTML documents", fg=typer.colors.CYAN)
        html_extract.extract_all()

    if _wants(source, "pdf"):
        from .extract import pdf_extract

        typer.secho("Extracting PDF documents", fg=typer.colors.CYAN)
        pdf_extract.extract_all()


@app.command()
def chunk(
    source: Source = typer.Option(Source.all, help="Which source format to chunk."),
) -> None:
    """Stage 3: split documents into section-aware chunks."""
    from .chunk import sectioner

    for target in ("html", "pdf"):
        if _wants(source, target):
            typer.secho(f"Chunking {target} documents", fg=typer.colors.CYAN)
            sectioner.chunk_source(target)


@app.command()
def embed(
    source: Source = typer.Option(Source.all, help="Which source format to embed."),
    reset: bool = typer.Option(False, help="Drop the collection before writing."),
) -> None:
    """Stages 4 and 5: embed chunks and store them in Chroma."""
    from .embed import pipeline

    for target in ("html", "pdf"):
        if _wants(source, target):
            typer.secho(f"Embedding and storing {target} chunks", fg=typer.colors.CYAN)
            pipeline.embed_and_store(target, reset=reset)


@app.command()
def ask(
    question: str = typer.Argument(..., help="A natural language question."),
    source: str = typer.Option("html", help="Collection to search: html or pdf."),
    top_k: int = typer.Option(config.DEFAULT_TOP_K, "--top-k", "-k"),
    full: bool = typer.Option(False, help="Print full chunk text instead of a preview."),
) -> None:
    """Retrieve the most relevant documentation chunks for a question."""
    from .query import search, render_results

    results = search(question, source_format=source, top_k=top_k)
    render_results(question, results, full=full)


@app.command()
def evaluate(
    source: str = typer.Option("html", help="Collection to evaluate."),
    top_k: int = typer.Option(5, "--top-k", "-k"),
) -> None:
    """Score retrieval quality against the anchored gold set."""
    from .evaluate import run_eval

    run_eval.main(source_format=source, top_k=top_k)


@app.command()
def compare() -> None:
    """Compare HTML and PDF extraction of the same sections."""
    from .evaluate import compare_formats

    compare_formats.main()


@app.command()
def info() -> None:
    """Show the current state of each pipeline stage."""
    from .status import report

    report()


if __name__ == "__main__":
    app()
