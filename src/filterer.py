"""Helpers for filtering collected news items."""

import re
from typing import Any

_NOISE_TITLE_PATTERNS = (
    re.compile(r"\bwebinar\b", re.IGNORECASE),
    re.compile(r"\bsponsored\b", re.IGNORECASE),
    re.compile(r"\badvertorial\b", re.IGNORECASE),
    re.compile(r"\bpartner content\b", re.IGNORECASE),
    re.compile(r"^opinion:", re.IGNORECASE),
    re.compile(r"\bpharmalittle\b", re.IGNORECASE),
    re.compile(r"\bup and down the ladder\b", re.IGNORECASE),
    re.compile(r"\bcomings and goings\b", re.IGNORECASE),
    re.compile(r"\bwhat we're reading\b", re.IGNORECASE),
)
_NOISE_LINK_PATTERNS = (
    re.compile(r"/spons/", re.IGNORECASE),
    re.compile(r"/webinars?/", re.IGNORECASE),
)
_ROUNDUP_BUNDLE_SOURCES = frozenset(
    {"Endpoints News", "STAT Pharma", "STAT Biotech", "BioPharma Dive"}
)
_SOURCE_TITLE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "MHRA": (
        re.compile(r"\bpilot pathway\b", re.IGNORECASE),
    ),
    "NIH": (
        re.compile(r"\bawards?\b", re.IGNORECASE),
        re.compile(r"\bnamed\b", re.IGNORECASE),
        re.compile(r"\binvests?\b", re.IGNORECASE),
        re.compile(r"\bchief of staff\b", re.IGNORECASE),
    ),
}
_SOURCE_LINK_REQUIRED_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "MHRA": (
        re.compile(r"/government/news/", re.IGNORECASE),
    ),
}


def deduplicate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return items keeping only the newest entry for each ``id``."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in sorted(items, key=lambda x: x["published"], reverse=True):
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        out.append(it)
    return out


def exclude_noise(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop obviously low-signal editorial or sponsored titles."""
    filtered: list[dict[str, Any]] = []
    skipped = 0

    for item in items:
        source = str(item.get("source", "")).strip()
        title = str(item.get("title", "")).strip()
        link = str(item.get("link", "")).strip()
        if any(pattern.search(title) for pattern in _NOISE_TITLE_PATTERNS):
            skipped += 1
            continue
        if any(pattern.search(link) for pattern in _NOISE_LINK_PATTERNS):
            skipped += 1
            continue
        if any(pattern.search(title) for pattern in _SOURCE_TITLE_PATTERNS.get(source, ())):
            skipped += 1
            continue
        if source in _ROUNDUP_BUNDLE_SOURCES and ";" in title:
            skipped += 1
            continue
        required_link_patterns = _SOURCE_LINK_REQUIRED_PATTERNS.get(source, ())
        if required_link_patterns and not any(
            pattern.search(link) for pattern in required_link_patterns
        ):
            skipped += 1
            continue
        filtered.append(item)

    return filtered, skipped
