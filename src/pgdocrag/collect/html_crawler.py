"""Crawl the PostgreSQL HTML documentation.

Every docs page carries navigation links annotated with `accesskey` attributes
(p=prev, u=up, h=home, n=next). Following `accesskey="n"` walks the manual in
reading order, which is more reliable than parsing the sitemap and yields a
natural ordinal for each page for free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .. import config
from . import fetcher

def manifest_path() -> Path:
    """Resolved per call so switching corpus is picked up, not captured at import."""
    return config.HTML_MANIFEST_PATH


@dataclass
class PageRecord:
    slug: str
    url: str
    title: str
    sha256: str
    fetched_at: str
    ordinal: int
    bytes: int
    # The "up" link. A page only knows its own section, so the parent chain is
    # what lets the extractor build a breadcrumb that spans pages
    # ("Server Configuration > Connections and Authentication > ...").
    parent_slug: str | None = None


def load_manifest() -> dict[str, PageRecord]:
    path = manifest_path()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {slug: PageRecord(**record) for slug, record in payload.get("pages", {}).items()}


def save_manifest(records: dict[str, PageRecord]) -> None:
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pg_version": config.PG_VERSION,
        "base_url": config.DOCS_BASE_URL,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(records),
        "pages": {slug: asdict(record) for slug, record in sorted(records.items())},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def slug_for(url: str) -> str:
    name = Path(urlparse(url).path).name
    return name or "index.html"


def cache_path(slug: str) -> Path:
    return config.RAW_HTML_DIR / slug


def _page_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    heading = soup.select_one("#docContent h1, #docContent h2")
    return heading.get_text(strip=True) if heading else ""


def _nav_url(soup: BeautifulSoup, current_url: str, accesskey: str) -> str | None:
    link = soup.select_one(f'a[accesskey="{accesskey}"]')
    if not link or not link.get("href"):
        return None
    candidate = urljoin(current_url, str(link["href"]))
    # Stay inside this version's docs tree.
    if not candidate.startswith(config.DOCS_BASE_URL):
        return None
    return candidate


def _next_url(soup: BeautifulSoup, current_url: str) -> str | None:
    return _nav_url(soup, current_url, "n")


def _parent_slug(soup: BeautifulSoup, current_url: str) -> str | None:
    up = _nav_url(soup, current_url, "u")
    return slug_for(up) if up else None


def crawl(
    seed: str,
    limit: int | None,
    *,
    refresh: bool = False,
    visited: set[str] | None = None,
    start_ordinal: int = 0,
    verbose: bool = True,
) -> list[PageRecord]:
    """Walk forward from `seed`, caching each page to disk.

    Cached pages are reused unless `refresh` is set, so re-running the crawl is
    cheap and does not hammer the server.
    """
    config.ensure_dirs()
    manifest = load_manifest()
    visited = visited if visited is not None else set()
    session = fetcher.make_session()

    url: str | None = urljoin(config.DOCS_BASE_URL, seed)
    collected: list[PageRecord] = []
    ordinal = start_ordinal

    while url and (limit is None or len(collected) < limit):
        slug = slug_for(url)
        if slug in visited:
            break
        visited.add(slug)

        path = cache_path(slug)
        use_cache = path.exists() and not refresh
        if use_cache:
            html = path.read_text(encoding="utf-8", errors="replace")
        else:
            response = fetcher.get(session, url)
            response.encoding = response.encoding or "utf-8"
            html = response.text
            path.write_text(html, encoding="utf-8")

        soup = BeautifulSoup(html, "lxml")
        record = PageRecord(
            slug=slug,
            url=url,
            title=_page_title(soup),
            sha256=fetcher.sha256_bytes(html.encode("utf-8")),
            fetched_at=datetime.now(timezone.utc).isoformat(),
            ordinal=ordinal,
            bytes=len(html),
            parent_slug=_parent_slug(soup, url),
        )
        manifest[slug] = record
        collected.append(record)
        ordinal += 1

        if verbose:
            marker = "cached" if use_cache else "fetched"
            print(f"  [{len(collected):>4}] {marker:>7}  {slug}")

        url = _next_url(soup, url)

    save_manifest(manifest)
    return collected


def crawl_slice(*, refresh: bool = False, verbose: bool = True) -> list[PageRecord]:
    """Crawl the development slice: configuration chapters plus SQL reference.

    The two seeds deliberately cover both HTML structure families, so the
    extractor is exercised against `sect*`/`variablelist` pages and against
    `refentry`/`refsect*` pages.
    """
    visited: set[str] = set()
    collected: list[PageRecord] = []
    for seed, limit in config.SLICE_SEEDS:
        if verbose:
            print(f"Seed: {seed} (limit {limit})")
        collected.extend(
            crawl(
                seed,
                limit,
                refresh=refresh,
                visited=visited,
                start_ordinal=len(collected),
                verbose=verbose,
            )
        )
    return collected


def crawl_full(*, refresh: bool = False, verbose: bool = True) -> list[PageRecord]:
    return crawl(config.FULL_CRAWL_SEED, None, refresh=refresh, verbose=verbose)
