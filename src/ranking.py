"""Ranking and ordering helpers for bio-news-agent."""

from typing import Any

try:
    from config import CATEGORIES
    from item_types import CollectedItem, FeedMode, ResolvedItem, SourceRole
except ModuleNotFoundError:  # pragma: no cover - module execution fallback
    from .config import CATEGORIES
    from .item_types import CollectedItem, FeedMode, ResolvedItem, SourceRole

CATEGORY_ORDER = list(CATEGORIES)
_FEED_MODE_PRIORITY = {
    "core": 1,
    "discovery_only": 0,
}
_SOURCE_ROLE_PRIORITY = {
    "primary": 3,
    "independent_reporting": 2,
    "commentary": 1,
    "community": 0,
}


def normalize_source_role(value: Any) -> SourceRole:
    source_role = str(value).strip().lower()
    if source_role in _SOURCE_ROLE_PRIORITY:
        return source_role
    return "independent_reporting"


def normalize_feed_mode(value: Any) -> FeedMode:
    feed_mode = str(value).strip().lower()
    if feed_mode in _FEED_MODE_PRIORITY:
        return feed_mode
    return "core"


def feed_mode_priority(value: Any) -> int:
    return _FEED_MODE_PRIORITY[normalize_feed_mode(value)]


def source_role_priority(value: Any) -> int:
    return _SOURCE_ROLE_PRIORITY[normalize_source_role(value)]


def group_item_sort_key(item: CollectedItem) -> tuple[float, float, float, str]:
    return (
        -float(feed_mode_priority(item.get("feed_mode"))),
        -float(source_role_priority(item.get("source_role"))),
        -item["published"].timestamp(),
        str(item.get("source", "")),
    )


def story_rank_key(item: ResolvedItem) -> tuple[int, float, float, float, float, str]:
    return (
        -(1 if item.get("tier") == "high" else 0),
        -float(feed_mode_priority(item.get("feed_mode"))),
        -float(source_role_priority(item.get("source_role"))),
        -float(len(item.get("coverage_sources", []))),
        -item["published"].timestamp(),
        str(item.get("source", "")),
    )


def render_sort_key(item: ResolvedItem) -> tuple[int, float, float, float, str]:
    return (
        -(1 if item.get("tier") == "high" else 0),
        -float(source_role_priority(item.get("source_role"))),
        -float(len(item.get("coverage_sources", []))),
        -item["published"].timestamp(),
        str(item.get("source", "")),
    )


def sort_items_by_source_role(items: list[CollectedItem]) -> list[CollectedItem]:
    return sorted(items, key=group_item_sort_key)


def select_top_story_ids(items: list[ResolvedItem], requested_ids: list[str]) -> list[str]:
    prompt_id_lookup = {
        str(item.get("_prompt_id", "")).strip(): item
        for item in items
        if str(item.get("_prompt_id", "")).strip()
    }

    selected_ids: list[str] = []
    for story_id in requested_ids:
        if story_id in prompt_id_lookup and story_id not in selected_ids:
            selected_ids.append(story_id)
    if selected_ids:
        return selected_ids[:3]

    ranked_items = sorted(prompt_id_lookup.values(), key=story_rank_key)
    return [str(item["_prompt_id"]) for item in ranked_items[:3]]
