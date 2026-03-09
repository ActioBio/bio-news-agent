"""Helpers for filtering collected news items."""

import re
from typing import Any

_NOISE_TITLE_PATTERNS = (
    re.compile(r"\bwebinar\b", re.IGNORECASE),
    re.compile(r"\bsponsored\b", re.IGNORECASE),
    re.compile(r"\badvertorial\b", re.IGNORECASE),
    re.compile(r"\bpartner content\b", re.IGNORECASE),
    re.compile(r"^opinion:", re.IGNORECASE),
)
_NOISE_LINK_PATTERNS = (
    re.compile(r"/spons/", re.IGNORECASE),
    re.compile(r"/webinars?/", re.IGNORECASE),
)


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
        title = str(item.get("title", "")).strip()
        link = str(item.get("link", "")).strip()
        if any(pattern.search(title) for pattern in _NOISE_TITLE_PATTERNS):
            skipped += 1
            continue
        if any(pattern.search(link) for pattern in _NOISE_LINK_PATTERNS):
            skipped += 1
            continue
        filtered.append(item)

    return filtered, skipped
