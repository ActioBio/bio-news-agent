"""RSS feed collector with URL normalization and retry logic."""

import calendar
import html
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import feedparser
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from config import (
        RSS_MAX_FEED_BYTES,
        RSS_MAX_WORKERS,
        RSS_RETRIES,
        RSS_TIMEOUT,
        RSS_USER_AGENT,
    )
except ModuleNotFoundError:  # pragma: no cover - module execution fallback
    from .config import (
        RSS_MAX_FEED_BYTES,
        RSS_MAX_WORKERS,
        RSS_RETRIES,
        RSS_TIMEOUT,
        RSS_USER_AGENT,
    )

logger = logging.getLogger(__name__)

_DAY = timedelta(days=1)
_FEEDS_FILE = Path(__file__).resolve().parent.parent / "feeds.json"
_TRACKING_PARAMS = frozenset(
    [
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "ref",
        "source",
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
    ]
)


def _load_feeds() -> dict[str, dict[str, str]]:
    try:
        raw_feeds = json.loads(_FEEDS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error("feeds.json not found at %s", _FEEDS_FILE)
        return {}
    except json.JSONDecodeError as exc:
        logger.error("feeds.json is invalid JSON: %s", exc)
        return {}

    if not isinstance(raw_feeds, dict):
        logger.error("feeds.json must be a JSON object mapping URL to metadata")
        return {}

    validated: dict[str, dict[str, str]] = {}
    for raw_url, raw_meta in raw_feeds.items():
        if not isinstance(raw_url, str) or not raw_url.strip():
            logger.warning("Skipping feed with invalid URL key: %r", raw_url)
            continue
        if not isinstance(raw_meta, dict):
            logger.warning("Skipping feed %s because metadata is not an object", raw_url)
            continue

        source = str(raw_meta.get("source", "")).strip()
        category = str(raw_meta.get("category", "All")).strip() or "All"
        if not source:
            source = urlparse(raw_url).netloc or "Unknown Source"
            logger.warning("Feed %s missing source; using %s", raw_url, source)

        source_type = str(raw_meta.get("type", "news")).strip().lower() or "news"

        validated[raw_url] = {
            "source": source,
            "category": category,
            "type": source_type,
        }

    logger.info("Loaded %d valid feeds from feeds.json", len(validated))
    return validated


def normalize_url(url: str) -> str:
    """Normalize URL to improve duplicate detection."""
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = parsed.path.rstrip("/")

    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=True)
        filtered = {k: v for k, v in params.items() if k.lower() not in _TRACKING_PARAMS}
        query = urlencode(filtered, doseq=True) if filtered else ""
    else:
        query = ""

    return urlunparse((scheme, netloc, path, "", query, ""))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date(entry: dict[str, Any]) -> datetime | None:
    for attr in ("published_parsed", "updated_parsed"):
        tup = entry.get(attr)
        if tup:
            return datetime.fromtimestamp(calendar.timegm(tup), tz=timezone.utc)
    return None


def _clean_summary(value: Any) -> str:
    if not value:
        return ""

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_title(value: Any) -> str:
    if not value:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _log_retry(retry_state) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning("Fetch attempt %s failed, retrying: %s", retry_state.attempt_number, exc)


@retry(
    stop=stop_after_attempt(RSS_RETRIES + 1),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((URLError, TimeoutError)),
    before_sleep=_log_retry,
    reraise=True,
)
def _fetch_with_retry(url: str) -> feedparser.FeedParserDict:
    """Fetch RSS feed with retry logic and request timeout."""
    request = Request(url, headers={"User-Agent": RSS_USER_AGENT})
    with urlopen(request, timeout=RSS_TIMEOUT) as response:
        payload = response.read(RSS_MAX_FEED_BYTES + 1)
        if len(payload) > RSS_MAX_FEED_BYTES:
            raise ValueError(
                f"Feed response exceeded max size ({RSS_MAX_FEED_BYTES} bytes): {url}"
            )
        response_headers = {
            header.lower(): value for header, value in response.headers.items()
        }
        response_headers.setdefault("content-type", "application/rss+xml")
    return feedparser.parse(payload, response_headers=response_headers)


def _fetch_feed_entries(
    url: str,
    meta: dict[str, str],
) -> tuple[str, str, str, list[dict[str, Any]]]:
    category = meta.get("category", "All")
    source = meta.get("source", urlparse(url).netloc or "Unknown Source")
    source_type = meta.get("type", "news")
    try:
        logger.info("Fetching %s...", source)
        parsed = _fetch_with_retry(url)
    except Exception as exc:
        logger.warning("Feed error for %s: %s", url, exc)
        return source, category, source_type, []

    if parsed.bozo:
        logger.warning(
            "Parse warning for %s: %s",
            source,
            getattr(parsed, "bozo_exception", "unknown warning"),
        )

    entries = list(parsed.entries) if hasattr(parsed, "entries") else []
    logger.info("Found %d entries from %s", len(entries), source)
    return source, category, source_type, entries


def collect_items() -> list[dict[str, Any]]:
    """Return list[dict] fresh within 24 h."""
    cutoff = _now() - _DAY
    logger.info("Collecting items newer than %s", cutoff)
    items: list[dict[str, Any]] = []
    feeds = _load_feeds()
    feed_results: list[tuple[str, str, str, list[dict[str, Any]]]] = []

    max_workers = max(1, min(RSS_MAX_WORKERS, len(feeds)))
    if max_workers == 1:
        for url, meta in feeds.items():
            feed_results.append(_fetch_feed_entries(url, meta))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_fetch_feed_entries, url, meta)
                for url, meta in feeds.items()
            ]
            for future in as_completed(futures):
                feed_results.append(future.result())

    for source, category, source_type, entries in feed_results:
        for entry in entries:
            published = _parse_date(entry)
            if not published or published < cutoff:
                continue

            title = _clean_title(entry.get("title", ""))
            link = str(entry.get("link", "")).strip()
            if not (title and link):
                continue

            normalized_link = normalize_url(link)
            summary = _clean_summary(entry.get("summary") or entry.get("description") or "")
            items.append(
                {
                    "id": normalized_link,
                    "title": title,
                    "original_title": title,
                    "link": link,
                    "source": source,
                    "published": published,
                    "category": category,
                    "summary": summary,
                    "source_type": source_type,
                }
            )

    logger.info("Collected %d total items from all feeds", len(items))
    return items
