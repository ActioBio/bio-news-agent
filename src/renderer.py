"""Render news items to markdown."""

from collections import defaultdict
from typing import Any

try:
    from config import CATEGORIES
except ModuleNotFoundError:  # pragma: no cover - module execution fallback
    from .config import CATEGORIES

_CATEGORY_ORDER = list(CATEGORIES)
_SOURCE_ROLE_PRIORITY = {
    "primary": 3,
    "independent_reporting": 2,
    "commentary": 1,
    "community": 0,
}


def _source_role_priority(value: Any) -> int:
    source_role = str(value).strip().lower()
    return _SOURCE_ROLE_PRIORITY.get(source_role, _SOURCE_ROLE_PRIORITY["independent_reporting"])


def _render_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        -(1 if item.get("tier") == "high" else 0),
        -float(_source_role_priority(item.get("source_role"))),
        -float(len(item.get("coverage_sources", []))),
        -item["published"].timestamp(),
        str(item.get("source", "")),
    )


def _render_item_line(item: dict[str, Any], *, bold: bool = False) -> str:
    title = item["title"].strip()
    link_text = f"**[{title}]({item['link']})**" if bold else f"[{title}]({item['link']})"
    coverage_sources: list[str] = item.get("coverage_sources", [])
    coverage_suffix = f" ({len(coverage_sources) + 1} sources)" if coverage_sources else ""
    return f"- {link_text} — {item['source']}{coverage_suffix}"


def to_markdown(
    items: list[dict[str, Any]],
    *,
    executive_summary: str = "",
    top_stories: list[str] | None = None,
) -> str:
    if not items:
        return "_No fresh biotech/pharma headlines in the last 24 h._"

    if top_stories is None:
        top_stories = []

    lines = ["## Daily Biotech / Pharma Headlines\n"]
    top_story_ids: set[str] = set()

    if top_stories:
        prompt_id_lookup = {item.get("_prompt_id", ""): item for item in items}
        resolved = [prompt_id_lookup[sid] for sid in top_stories if sid in prompt_id_lookup]
        if resolved:
            top_story_ids = {str(item.get("_prompt_id", "")) for item in resolved}
            lines.append("### Top Stories")
            for item in resolved:
                lines.append(_render_item_line(item, bold=True))
            lines.append("")

    sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        cat = it.get("category", "Other")
        if cat not in CATEGORIES:
            cat = "Company News"
        sections[cat].append(it)

    for cat in _CATEGORY_ORDER:
        if cat not in sections:
            continue
        sorted_items = sorted(
            sections[cat],
            key=_render_sort_key,
        )
        visible_items = [item for item in sorted_items if str(item.get("_prompt_id", "")) not in top_story_ids]
        if not visible_items:
            continue

        lines.append(f"### {cat}")

        for item in visible_items:
            lines.append(_render_item_line(item))
        lines.append("")

    return "\n".join(lines)
