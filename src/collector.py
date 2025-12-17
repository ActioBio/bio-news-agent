"""RSS feed collector with URL normalization and retry logic."""

import hashlib
import json
import logging
import socket
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import feedparser
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import RSS_RETRIES, RSS_TIMEOUT, RSS_USER_AGENT

logger = logging.getLogger(__name__)

_DAY = timedelta(days=1)

# Location of feeds configuration file (project root)
_FEEDS_FILE = Path(__file__).resolve().parent.parent / "feeds.json"

try:
    FEEDS = json.loads(_FEEDS_FILE.read_text())
    logger.info(f"Loaded {len(FEEDS)} feeds from feeds.json")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(f"Error loading feeds.json: {e}")
    FEEDS = {}

# Tracking parameters to strip from URLs
_TRACKING_PARAMS = frozenset([
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "source", "fbclid", "gclid", "mc_cid", "mc_eid",
])


def normalize_url(url: str) -> str:
    """Normalize URL to improve duplicate detection.

    Only lowercases scheme and domain (case-insensitive per RFC).
    Preserves path/query case since many servers are case-sensitive.
    """
    parsed = urlparse(url.strip())

    # Lowercase scheme and domain only (these are case-insensitive)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Remove www. prefix
    if netloc.startswith("www."):
        netloc = netloc[4:]

    # Remove trailing slashes from path (preserve case)
    path = parsed.path.rstrip("/")

    # Remove tracking parameters (check param names case-insensitively)
    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=True)
        filtered = {k: v for k, v in params.items() if k.lower() not in _TRACKING_PARAMS}
        query = urlencode(filtered, doseq=True) if filtered else ""
    else:
        query = ""

    # Rebuild URL without fragment
    normalized = urlunparse((scheme, netloc, path, "", query, ""))
    return normalized


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date(entry: dict) -> datetime | None:
    for attr in ("published_parsed", "updated_parsed"):
        tup = entry.get(attr)
        if tup:
            return datetime.fromtimestamp(time.mktime(tup), tz=timezone.utc)
    return None


@retry(
    stop=stop_after_attempt(RSS_RETRIES + 1),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((URLError, socket.timeout, TimeoutError)),
)
def _fetch_with_retry(url: str) -> feedparser.FeedParserDict:
    """Fetch RSS feed with retry logic."""
    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(RSS_TIMEOUT)
        parsed = feedparser.parse(url, request_headers={"User-Agent": RSS_USER_AGENT})
        return parsed
    finally:
        socket.setdefaulttimeout(old_timeout)


def collect_items() -> list[dict]:
    """Return list[dict] fresh within 24 h."""
    cutoff = _now() - _DAY
    logger.info(f"Collecting items newer than {cutoff}")
    items = []

    for url, meta in FEEDS.items():
        category = meta["category"]
        src = meta["source"]
        try:
            logger.info(f"Fetching {src}...")
            parsed = _fetch_with_retry(url)

            if parsed.bozo:
                logger.warning(f"Parse error for {src}: {parsed.bozo_exception}")
                continue

            entries_count = len(parsed.entries) if hasattr(parsed, "entries") else 0
            logger.info(f"Found {entries_count} entries from {src}")

        except Exception as exc:
            logger.warning(f"Feed error for {url}: {exc}")
            continue

        for e in parsed.entries:
            ts = _parse_date(e)
            if not ts:
                continue

            if ts < cutoff:
                continue

            title, link = e.get("title", "").strip(), e.get("link", "").strip()
            if not (title and link):
                continue

            # Normalize URL for better dedup
            normalized_link = normalize_url(link)
            uid = hashlib.sha1(normalized_link.encode()).hexdigest()

            items.append(
                {
                    "id": uid,
                    "title": title,
                    "link": link,
                    "source": src,
                    "published": ts,
                    "category": category,
                }
            )

    logger.info(f"Collected {len(items)} total items from all feeds")
    return items
