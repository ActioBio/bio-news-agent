"""Render news items to markdown."""

from collections import defaultdict
from typing import Any

try:
    from config import CATEGORIES
except ModuleNotFoundError:  # pragma: no cover - module execution fallback
    from .config import CATEGORIES

_CATEGORY_ORDER = list(CATEGORIES)


def to_markdown(items: list[dict[str, Any]]) -> str:
    if not items:
        return "_No fresh biotech/pharma headlines in the last 24 h._"

    sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        cat = it.get("category", "Other")
        # Map unknown categories to "Other"
        if cat not in CATEGORIES:
            cat = "Company News"
        sections[cat].append(it)

    lines = ["## Daily Biotech / Pharma Headlines\n"]

    for cat in _CATEGORY_ORDER:
        if cat not in sections:
            continue
        lines.append(f"### {cat}")

        # Sort items: first by recency (newest first), then by source
        sorted_items = sorted(
            sections[cat],
            key=lambda x: (-x["published"].timestamp(), x["source"]),
        )

        for i in sorted_items:
            title = i["title"].strip()
            lines.append(f"- [{title}]({i['link']}) — {i['source']}")
        lines.append("")  # blank line

    return "\n".join(lines)
