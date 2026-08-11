"""Measure embedding throughput per device and project a full run from it.

Run this before committing to a long embedding job. It reports chunks/second for
each available execution provider, projects the whole corpus from that rate, and
checks that changing device does not change the vectors — which is what keeps
results comparable across a device switch.

The sample is taken at an even stride through the chunk file, never as the first
N lines. Chunk length varies through a corpus and embedding cost scales with
word-pieces, so a head sample overstates throughput badly: on the full HTML
corpus the first 256 chunks suggest 3.0 chunks/s where the true rate is 0.85.

Usage:
    python scripts/benchmark_embed.py
    python scripts/benchmark_embed.py --corpus full --source html --sample 512
    python scripts/benchmark_embed.py --devices cuda --baseline-rate 0.85
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pgdocrag import config  # noqa: E402
from pgdocrag.embed.embedder import Embedder, cuda_available  # noqa: E402


def chunk_texts(corpus: str, source: str, sample: int) -> tuple[list[str], int]:
    """An evenly spaced sample of chunk texts, plus the corpus total."""
    config.use_corpus(corpus)
    path = config.CHUNKS_DIR / f"{source}_chunks.jsonl"
    if not path.exists():
        raise SystemExit(f"No chunks at {path}. Run the chunk stage for corpus {corpus!r} first.")

    with path.open(encoding="utf-8") as handle:
        texts = [json.loads(line)["text"] for line in handle if line.strip()]
    if not texts:
        raise SystemExit(f"{path} is empty")

    if sample >= len(texts):
        return texts, len(texts)
    step = len(texts) / sample
    return [texts[int(index * step)] for index in range(sample)], len(texts)


def gpu_memory_mib() -> int | None:
    """Currently allocated VRAM, or None when nvidia-smi is unavailable."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except Exception:
        return None
    return int(completed.stdout.strip().splitlines()[0])


def measure(device: str, texts: list[str], batch_size: int) -> dict:
    """Time one device over the sample, excluding model load and warm-up.

    The first batch on CUDA pays for context creation and kernel selection, and
    counting it would understate a long run by a wide margin.
    """
    idle_vram = gpu_memory_mib()

    embedder = Embedder(use_cache=False, device=device)
    embedder.model
    embedder.embed_texts(texts[: min(16, len(texts))], batch_size=batch_size, show_progress=False)

    started = time.monotonic()
    vectors = embedder.embed_texts(texts, batch_size=batch_size, show_progress=False)
    elapsed = time.monotonic() - started

    busy_vram = gpu_memory_mib()
    return {
        "device": device,
        "providers": embedder.providers,
        "on_gpu": embedder.on_gpu,
        "elapsed": elapsed,
        "rate": len(texts) / elapsed,
        "vectors": vectors,
        "vram": (busy_vram - idle_vram) if (busy_vram and idle_vram) else None,
    }


def projection(rate: float, counts: dict[str, int]) -> str:
    parts = [f"{name} {count / rate / 60:.0f} min" for name, count in counts.items()]
    total = sum(counts.values()) / rate / 3600
    return f"{', '.join(parts)}  (total {total:.1f} h)"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="full", choices=config.CORPORA)
    parser.add_argument("--source", default="html", choices=("html", "pdf"))
    parser.add_argument("--sample", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        choices=("cuda", "cpu"),
        help="Defaults to every device available, so GPU and CPU are compared directly.",
    )
    parser.add_argument(
        "--baseline-rate",
        type=float,
        default=None,
        help="A previously observed chunks/s to compare against, e.g. from a real CPU run.",
    )
    args = parser.parse_args()

    texts, corpus_total = chunk_texts(args.corpus, args.source, args.sample)
    available, reason = cuda_available()

    print(f"corpus            {args.corpus} / {args.source}")
    print(f"sample            {len(texts)} of {corpus_total} chunks, evenly strided")
    print(f"batch size        {args.batch_size}")
    print(f"model             {config.EMBED_MODEL_NAME}")
    print(f"CUDA available    {available}  ({reason})")

    devices = args.devices or (["cuda", "cpu"] if available else ["cpu"])
    results = []
    for device in devices:
        print(f"\nmeasuring {device} ...")
        try:
            result = measure(device, texts, args.batch_size)
        except Exception as error:
            print(f"  failed: {error}")
            continue
        results.append(result)
        label = "GPU" if result["on_gpu"] else "CPU"
        print(f"  ran on          {label}  [{', '.join(result['providers']) or 'unreported'}]")
        print(f"  elapsed         {result['elapsed']:.1f}s")
        print(f"  throughput      {result['rate']:.2f} chunks/s")
        if result["vram"] is not None:
            print(f"  VRAM delta      {result['vram']} MiB")

    if not results:
        print("\nno device produced a measurement")
        return 1

    counts: dict[str, int] = {}
    for source in ("html", "pdf"):
        path = config.CHUNKS_DIR / f"{source}_chunks.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                counts[source] = sum(1 for line in handle if line.strip())

    print(f"\nprojected runtime for corpus {args.corpus!r}")
    for result in results:
        label = "GPU" if result["on_gpu"] else "CPU"
        print(f"  {label:<4} {result['rate']:>6.2f} chunks/s   {projection(result['rate'], counts)}")
    if args.baseline_rate:
        print(
            f"  prev {args.baseline_rate:>6.2f} chunks/s   "
            f"{projection(args.baseline_rate, counts)}   (supplied baseline)"
        )

    fastest = max(results, key=lambda item: item["rate"])
    slowest = min(results, key=lambda item: item["rate"])
    if fastest is not slowest:
        print(f"\nspeedup           {fastest['rate'] / slowest['rate']:.1f}x")
    if args.baseline_rate:
        print(f"vs baseline       {fastest['rate'] / args.baseline_rate:.1f}x")

    # A device switch must not move the vectors, or the existing evaluation stops
    # being comparable to anything measured before it.
    if len(results) > 1:
        first, second = results[0], results[1]
        a, b = first["vectors"], second["vectors"]
        cosine = np.sum(a * b, axis=1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        )
        print(
            f"\nvector agreement  {first['device']} vs {second['device']}: "
            f"min cosine {cosine.min():.6f}, max abs delta {np.abs(a - b).max():.2e}"
        )
        if cosine.min() < 0.9999:
            print("  WARNING: vectors differ enough to affect ranking; re-embed rather than mix")
        else:
            print("  identical to float noise, so cached vectors stay valid across devices")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
