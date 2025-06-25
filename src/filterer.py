"""Helpers for filtering collected news items."""


def deduplicate(items):
    """Return items keeping only the newest entry for each ``id``."""
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x["published"], reverse=True):
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        out.append(it)
    return out