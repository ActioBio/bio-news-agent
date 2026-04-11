"""Render news items to markdown."""

from collections import defaultdict

try:
    from config import CATEGORIES
    from item_types import ResolvedItem
    from ranking import CATEGORY_ORDER as _CATEGORY_ORDER, render_sort_key as _render_sort_key
except ModuleNotFoundError:  # pragma: no cover - module execution fallback
    from .config import CATEGORIES
    from .item_types import ResolvedItem
    from .ranking import CATEGORY_ORDER as _CATEGORY_ORDER, render_sort_key as _render_sort_key

def _render_item_line(item: ResolvedItem, *, bold: bool = False) -> str:
    title = item["title"].strip()
    link_text = f"**[{title}]({item['link']})**" if bold else f"[{title}]({item['link']})"
    coverage_sources: list[str] = item.get("coverage_sources", [])
    coverage_suffix = f" ({len(coverage_sources) + 1} sources)" if coverage_sources else ""
    return f"- {link_text} — {item['source']}{coverage_suffix}"


def to_markdown(
    items: list[ResolvedItem],
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

    sections: dict[str, list[ResolvedItem]] = defaultdict(list)
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
