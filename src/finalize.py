"""Finalization helpers for bio-news-agent."""

from collections import defaultdict
from collections.abc import Mapping
from logging import Logger
from typing import Any

try:
    from config import COMPANY_NEWS_LIMIT, PAPER_LIMIT
    from item_types import ResolvedItem
    from ranking import normalize_feed_mode, select_top_story_ids, story_rank_key
except ModuleNotFoundError:  # pragma: no cover - module execution fallback
    from .config import COMPANY_NEWS_LIMIT, PAPER_LIMIT
    from .item_types import ResolvedItem
    from .ranking import normalize_feed_mode, select_top_story_ids, story_rank_key


def _sort_items_by_recency(items: list[ResolvedItem]) -> list[ResolvedItem]:
    return sorted(items, key=lambda item: item["published"], reverse=True)


def is_paper_item(item: Mapping[str, Any]) -> bool:
    source_type = str(item.get("source_type", "")).lower()
    return source_type == "paper" or "papers" in str(item.get("source", "")).lower()


def _limit_papers(items: list[ResolvedItem], *, limit: int) -> tuple[list[ResolvedItem], int]:
    paper_count = 0
    filtered_items: list[ResolvedItem] = []
    skipped_papers = 0

    for item in items:
        if is_paper_item(item):
            if paper_count >= limit:
                skipped_papers += 1
                continue
            paper_count += 1
        filtered_items.append(item)

    return filtered_items, skipped_papers


def _apply_category_cap(
    items: list[ResolvedItem],
    *,
    category: str,
    limit: int,
) -> tuple[list[ResolvedItem], int]:
    if limit <= 0:
        return items, 0

    category_items = [item for item in items if item.get("category") == category]
    if len(category_items) <= limit:
        return items, 0

    kept_item_ids = {
        id(item)
        for item in sorted(category_items, key=story_rank_key)[:limit]
    }
    filtered_items: list[ResolvedItem] = []
    skipped_items = 0

    for item in items:
        if item.get("category") != category or id(item) in kept_item_ids:
            filtered_items.append(item)
            continue
        skipped_items += 1

    return filtered_items, skipped_items


def _filter_discovery_only_items(items: list[ResolvedItem]) -> tuple[list[ResolvedItem], int]:
    filtered_items = [
        item for item in items if normalize_feed_mode(item.get("feed_mode")) != "discovery_only"
    ]
    return filtered_items, len(items) - len(filtered_items)


def finalize_items(
    state: dict[str, Any],
    items: list[ResolvedItem],
    skipped_items: int,
    *,
    log_label: str,
    logger: Logger,
    paper_limit: int = PAPER_LIMIT,
    company_news_limit: int = COMPANY_NEWS_LIMIT,
) -> dict[str, Any]:
    sorted_items = _sort_items_by_recency(items)
    final_items, skipped_discovery = _filter_discovery_only_items(sorted_items)
    final_items, skipped_papers = _limit_papers(final_items, limit=paper_limit)
    final_items, skipped_company_news = _apply_category_cap(
        final_items,
        category="Company News",
        limit=company_news_limit,
    )

    if skipped_discovery:
        logger.info("Skipped %d discovery-only items before rendering", skipped_discovery)
    if skipped_papers:
        logger.info("Skipped %d additional papers (kept top %d)", skipped_papers, paper_limit)
    if skipped_company_news:
        logger.info(
            "Skipped %d lower-priority Company News items (kept top %d)",
            skipped_company_news,
            company_news_limit,
        )

    logger.info(
        "%s: %d items (skipped %d items)",
        log_label,
        len(final_items),
        skipped_items + skipped_discovery + skipped_papers + skipped_company_news,
    )

    categories: dict[str, int] = defaultdict(int)
    for item in final_items:
        categories[item.get("category", "Unknown")] += 1
    logger.info("Category distribution: %s", dict(categories))

    requested_top_stories = state.get("top_stories", [])
    if not isinstance(requested_top_stories, list):
        requested_top_stories = []
    state["top_stories"] = select_top_story_ids(final_items, requested_top_stories)
    state["items"] = final_items
    return state
