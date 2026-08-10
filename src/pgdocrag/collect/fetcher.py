"""Polite HTTP access shared by the HTML crawler and the PDF downloader."""

from __future__ import annotations

import hashlib
import time

import requests

from .. import config

_last_request_at = 0.0


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": config.USER_AGENT})
    return session


def is_allowed(url: str) -> bool:
    return not any(fragment in url for fragment in config.DISALLOWED_URL_FRAGMENTS)


def _throttle() -> None:
    """Enforce the crawl delay across all callers, not per-loop."""
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < config.CRAWL_DELAY_SECONDS:
        time.sleep(config.CRAWL_DELAY_SECONDS - elapsed)
    _last_request_at = time.monotonic()


def get(session: requests.Session, url: str, *, stream: bool = False,
        attempts: int = 3) -> requests.Response:
    if not is_allowed(url):
        raise ValueError(f"URL is disallowed by robots.txt policy: {url}")

    last_error: Exception | None = None
    for attempt in range(attempts):
        _throttle()
        try:
            response = session.get(
                url, timeout=config.REQUEST_TIMEOUT_SECONDS, stream=stream
            )
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch {url} after {attempts} attempts") from last_error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
