"""Command line entry point for the ingestion pipeline.

Stage modules are imported inside each command: fastembed and chromadb pull in
onnxruntime and take seconds to import, which would otherwise be paid even by
`pgdocrag --help`.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Optional

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


class Corpus(str, Enum):
    slice = "slice"
    full = "full"


class Device(str, Enum):
    auto = "auto"
    cuda = "cuda"
    cpu = "cpu"


class GoldSet(str, Enum):
    config = "config"
    full = "full"


def _configure_stdout() -> None:
    """The docs contain em dashes; the Windows console codepage cannot encode them."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _wants(source: Source, target: str) -> bool:
    return source in (Source.all, Source(target))


@app.callback()
def main(
    corpus: Corpus = typer.Option(
        Corpus(config.DEFAULT_CORPUS),
        help="Namespace for derived artifacts: the published benchmark slice, or the full manual.",
    ),
) -> None:
    _configure_stdout()
    config.use_corpus(corpus.value)
    config.ensure_dirs()


@app.command()
def collect(
    source: Source = typer.Option(Source.all, help="Which source format to collect."),
    scope: Optional[Scope] = typer.Option(
        None, help="Crawl breadth. Defaults to matching the selected corpus."
    ),
    refresh: bool = typer.Option(False, help="Re-download even if cached."),
) -> None:
    """Stage 1: fetch raw documentation to data/raw/."""
    scope = scope or Scope(config.CORPUS)

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
    device: Device = typer.Option(
        Device(config.DEFAULT_EMBED_DEVICE),
        help="Execution provider: auto prefers the GPU, cuda refuses to fall back to CPU.",
    ),
) -> None:
    """Stages 4 and 5: embed chunks and store them in Chroma."""
    from .embed import pipeline

    for target in ("html", "pdf"):
        if _wants(source, target):
            typer.secho(f"Embedding and storing {target} chunks", fg=typer.colors.CYAN)
            try:
                pipeline.embed_and_store(target, reset=reset, device=device.value)
            except RuntimeError as error:
                # Chiefly an unusable GPU or missing chunks: both are the user's
                # to fix, so a traceback would only bury the message.
                typer.secho(str(error), fg=typer.colors.RED)
                raise typer.Exit(code=1) from error


@app.command()
def device() -> None:
    """Report which execution provider embedding would use, without embedding."""
    from .embed.embedder import CUDA_PROVIDER, Embedder, cuda_available

    available, reason = cuda_available()
    typer.secho(f"CUDA available    {available}  ({reason})", fg=typer.colors.CYAN)

    import onnxruntime as ort

    typer.echo(f"onnxruntime       {ort.__version__}")
    typer.echo(f"providers built   {', '.join(ort.get_available_providers())}")

    # Loading the real model is the point: a provider that registers can still
    # fail to initialise, and only a live session settles the question.
    probe = Embedder(use_cache=False, device=config.DEFAULT_EMBED_DEVICE)
    probe.model
    typer.echo(f"session providers {', '.join(probe.providers) or 'none reported'}")
    colour = typer.colors.GREEN if probe.on_gpu else typer.colors.YELLOW
    typer.secho(
        f"embedding would run on {'GPU via ' + CUDA_PROVIDER if probe.on_gpu else 'CPU'}",
        fg=colour,
    )


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
    goldset: GoldSet = typer.Option(
        GoldSet.config,
        help="config: the 52 configuration questions. full: the corpus-wide set.",
    ),
) -> None:
    """Score retrieval quality against a gold set."""
    from .evaluate import run_eval

    run_eval.main(source_format=source, top_k=top_k, goldset=goldset.value)


@app.command()
def experiment(
    source: Source = typer.Option(Source.html, help="Collection to run against."),
    goldset: GoldSet = typer.Option(GoldSet.full, help="Which gold set to score."),
    rerankers: str = typer.Option(
        "minilm,bge", help="Comma-separated cross-encoders, or 'none' to skip reranking."
    ),
    device: Device = typer.Option(
        Device(config.DEFAULT_EMBED_DEVICE), help="Execution provider for the reranker."
    ),
) -> None:
    """Compare dense, lexical, hybrid and reranked retrieval on one gold set."""
    from .evaluate import experiments

    chosen = tuple(
        name.strip()
        for name in rerankers.split(",")
        if name.strip() and name.strip().lower() != "none"
    )
    for target in ("html", "pdf"):
        if _wants(source, target):
            try:
                experiments.run(target, goldset=goldset.value, rerankers=chosen, device=device.value)
            except (RuntimeError, ValueError) as error:
                typer.secho(str(error), fg=typer.colors.RED)
                raise typer.Exit(code=1) from error


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
