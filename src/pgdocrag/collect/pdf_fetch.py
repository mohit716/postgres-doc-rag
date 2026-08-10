"""Download the official PostgreSQL PDF manual."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .. import config
from . import fetcher

PDF_PATH = config.RAW_PDF_DIR / config.PDF_FILENAME
PDF_MANIFEST_PATH = config.RAW_PDF_DIR / "manifest.json"

_CHUNK_BYTES = 1 << 16


def download(*, refresh: bool = False, verbose: bool = True) -> Path:
    config.ensure_dirs()

    if PDF_PATH.exists() and not refresh:
        if verbose:
            size_mb = PDF_PATH.stat().st_size / 1_048_576
            print(f"  cached  {PDF_PATH.name} ({size_mb:.1f} MB)")
        return PDF_PATH

    session = fetcher.make_session()
    response = fetcher.get(session, config.PDF_URL, stream=True)
    total = int(response.headers.get("Content-Length", 0))

    digest = hashlib.sha256()
    with PDF_PATH.open("wb") as handle:
        progress = tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=PDF_PATH.name,
            disable=not verbose,
        )
        with progress:
            for block in response.iter_content(chunk_size=_CHUNK_BYTES):
                if not block:
                    continue
                handle.write(block)
                digest.update(block)
                progress.update(len(block))

    PDF_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "pg_version": config.PG_VERSION,
                "url": config.PDF_URL,
                "filename": PDF_PATH.name,
                "sha256": digest.hexdigest(),
                "bytes": PDF_PATH.stat().st_size,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PDF_PATH
