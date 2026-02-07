"""Helpers for filtering collected news items."""

from typing import Any


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
