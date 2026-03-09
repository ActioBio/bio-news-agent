"""
LangGraph pipeline for bio-news-agent.

Flow:
    collect -> filter -> categorize -> render
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import StateGraph
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from collector import collect_items
    from config import (
        CATEGORIES,
        COMPANY_NAMES,
        DIGEST_OUTPUT_FILE,
        MAX_ITEMS_PER_SOURCE,
        OPENAI_API_KEY as CONFIG_OPENAI_API_KEY,
        OPENAI_MODEL,
        OPENAI_RETRIES,
        OPENAI_TIMEOUT_SECONDS,
        PAPER_LIMIT,
    )
    from filterer import deduplicate
    from renderer import to_markdown
except ModuleNotFoundError:  # pragma: no cover - module execution fallback
    from .collector import collect_items
    from .config import (
        CATEGORIES,
        COMPANY_NAMES,
        DIGEST_OUTPUT_FILE,
        MAX_ITEMS_PER_SOURCE,
        OPENAI_API_KEY as CONFIG_OPENAI_API_KEY,
        OPENAI_MODEL,
        OPENAI_RETRIES,
        OPENAI_TIMEOUT_SECONDS,
        PAPER_LIMIT,
    )
    from .filterer import deduplicate
    from .renderer import to_markdown

logger = logging.getLogger(__name__)


def _resolve_output_file(path_value: str) -> Path:
    output = Path(path_value)
    if output.is_absolute():
        return output
    return (Path(__file__).resolve().parent.parent / output).resolve()


_NEWS_FILE = _resolve_output_file(DIGEST_OUTPUT_FILE)
_MAX_PROMPT_SUMMARY_CHARS = 280
_SHORT_TITLE_MAX_WORDS = 8
_DUPLICATE_STOP_WORDS = {
    "about",
    "after",
    "announces",
    "been",
    "breaking",
    "data",
    "disease",
    "drug",
    "from",
    "have",
    "into",
    "latest",
    "named",
    "patients",
    "phase",
    "results",
    "says",
    "study",
    "that",
    "their",
    "therapy",
    "this",
    "treatment",
    "trial",
    "what",
    "with",
    "year",
}


class DigestState(TypedDict, total=False):
    items: list[dict[str, Any]]
    markdown: str


class ItemMatchData(TypedDict):
    company: str | None
    context_tokens: set[str]
    title_tokens: set[str]


def _log_openai_retry(retry_state) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "OpenAI request attempt %s failed, retrying: %s",
        retry_state.attempt_number,
        exc,
    )


def _get_openai_api_key() -> str:
    return (os.getenv("OPENAI_API_KEY") or CONFIG_OPENAI_API_KEY).strip()


@lru_cache(maxsize=1)
def _get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS)


@retry(
    stop=stop_after_attempt(OPENAI_RETRIES + 1),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(
        (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
    ),
    before_sleep=_log_openai_retry,
    reraise=True,
)
def _chat_completion_text(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


def _title_for_matching(item: dict[str, Any]) -> str:
    original_title = str(item.get("original_title", "")).strip()
    if original_title:
        return original_title
    return str(item.get("title", "")).strip()


def _significant_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\b[a-z0-9]+\b", text.lower())
        if len(token) > 3 and token not in _DUPLICATE_STOP_WORDS
    }


def _summary_tokens(item: dict[str, Any]) -> set[str]:
    summary = str(item.get("summary", "")).strip()
    if not summary:
        return set()
    return _significant_tokens(summary)


def _mentioned_company(title: str) -> str | None:
    title_lower = title.lower()
    for company in COMPANY_NAMES:
        company_name = str(company)
        if company_name in title_lower:
            return company_name
    return None


def _build_item_match_data(item: dict[str, Any]) -> ItemMatchData:
    title = _title_for_matching(item)
    title_tokens = _significant_tokens(title)
    return {
        "company": _mentioned_company(title),
        "context_tokens": title_tokens | _summary_tokens(item),
        "title_tokens": title_tokens,
    }


def _is_high_confidence_duplicate_data(
    existing: ItemMatchData,
    candidate: ItemMatchData,
) -> bool:
    existing_tokens = existing["title_tokens"]
    candidate_tokens = candidate["title_tokens"]
    if len(existing_tokens) < 4 or len(candidate_tokens) < 4:
        return False

    overlap = existing_tokens & candidate_tokens
    if len(overlap) < 4:
        return False

    overlap_ratio = len(overlap) / min(len(existing_tokens), len(candidate_tokens))
    existing_company = existing["company"]
    candidate_company = candidate["company"]

    if existing_company or candidate_company:
        if existing_company != candidate_company:
            return False
        return overlap_ratio >= 0.8

    return overlap_ratio >= 0.9 and len(overlap) >= 5


def _is_high_confidence_duplicate(
    existing: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    return _is_high_confidence_duplicate_data(
        _build_item_match_data(existing),
        _build_item_match_data(candidate),
    )


def _is_candidate_duplicate_data(
    existing: ItemMatchData,
    candidate: ItemMatchData,
) -> bool:
    if _is_high_confidence_duplicate_data(existing, candidate):
        return True

    title_overlap = existing["title_tokens"] & candidate["title_tokens"]
    if len(title_overlap) >= 3:
        return True

    existing_company = existing["company"]
    candidate_company = candidate["company"]
    if (
        existing_company
        and candidate_company
        and existing_company == candidate_company
        and len(title_overlap) >= 2
    ):
        return True

    context_overlap = existing["context_tokens"] & candidate["context_tokens"]
    return len(title_overlap) >= 2 and len(context_overlap) >= 5


def _build_candidate_groups(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not items:
        return []

    match_data = [_build_item_match_data(item) for item in items]
    adjacency = {index: set() for index in range(len(items))}
    for left_index in range(len(items)):
        for right_index in range(left_index + 1, len(items)):
            if _is_candidate_duplicate_data(match_data[left_index], match_data[right_index]):
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    groups: list[list[dict[str, Any]]] = []
    visited: set[int] = set()
    for index in range(len(items)):
        if index in visited:
            continue

        stack = [index]
        component: list[int] = []
        visited.add(index)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)

        groups.append(
            sorted(
                (items[item_index] for item_index in component),
                key=lambda item: item["published"],
                reverse=True,
            )
        )

    return sorted(groups, key=lambda group: group[0]["published"], reverse=True)


def _clean_prompt_text(text: str, limit: int) -> str:
    normalized = " ".join(text.split()).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _extract_json_object(text: str) -> dict[str, Any]:
    payload = text.strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload)
        payload = re.sub(r"\s*```$", "", payload)

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(payload[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object from LLM")
    return parsed


def _clean_short_title(value: Any, fallback: str) -> str:
    title = " ".join(str(value).split()).strip()
    if not title:
        return fallback

    words = title.split()
    if len(words) > _SHORT_TITLE_MAX_WORDS:
        title = " ".join(words[:_SHORT_TITLE_MAX_WORDS])
    return title


def _keyword_categorize(title: str) -> str:
    title_lower = title.lower()

    if any(word in title_lower for word in ["fda", "approve", "reject", "ema", "regulatory"]):
        return "Regulatory & FDA"
    if any(word in title_lower for word in ["trial", "phase", "study", "efficacy", "therapy", "research"]):
        return "Clinical & Research"
    if any(word in title_lower for word in ["partner", "deal", "raise", "funding", "$", "acquisition", "ipo", "merger"]):
        return "Deals & Finance"
    if any(word in title_lower for word in ["layoff", "cuts", "ceo", "executive", "hire", "appoint"]):
        return "Company News"
    if any(word in title_lower for word in ["trump", "congress", "medicare", "medicaid", "policy", "legislation"]):
        return "Policy & Politics"
    if any(word in title_lower for word in ["market", "spending", "forecast", "trend", "billion", "outlook"]):
        return "Market Insights"

    if any(company in title_lower for company in COMPANY_NAMES):
        return "Company News"

    return "Company News"


def _validate_category(value: Any, item: dict[str, Any]) -> str:
    category = str(value).strip()
    if category in CATEGORIES:
        return category
    return _keyword_categorize(_title_for_matching(item))


def _sort_items_by_recency(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: item["published"], reverse=True)


def _apply_source_cap(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if MAX_ITEMS_PER_SOURCE <= 0:
        return items, 0

    source_counts: dict[str, int] = defaultdict(int)
    filtered_items: list[dict[str, Any]] = []
    skipped_items = 0

    for item in items:
        source = str(item.get("source", ""))
        if source_counts[source] >= MAX_ITEMS_PER_SOURCE:
            skipped_items += 1
            continue
        source_counts[source] += 1
        filtered_items.append(item)

    return filtered_items, skipped_items


def _is_paper_item(item: dict[str, Any]) -> bool:
    source_type = str(item.get("source_type", "")).lower()
    if source_type == "paper":
        return True
    return "papers" in str(item.get("source", "")).lower()


def _limit_papers(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    paper_count = 0
    filtered_items: list[dict[str, Any]] = []
    skipped_papers = 0

    for item in items:
        if _is_paper_item(item):
            if paper_count >= PAPER_LIMIT:
                skipped_papers += 1
                continue
            paper_count += 1
        filtered_items.append(item)

    return filtered_items, skipped_papers


def _fallback_resolve_groups(groups: list[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], int]:
    kept_items: list[dict[str, Any]] = []
    skipped_items = 0

    for group in groups:
        resolved_group: list[dict[str, Any]] = []
        resolved_match_data: list[ItemMatchData] = []
        for item in group:
            item_match_data = _build_item_match_data(item)
            if any(
                _is_candidate_duplicate_data(existing_data, item_match_data)
                for existing_data in resolved_match_data
            ):
                skipped_items += 1
                continue

            item["title"] = _title_for_matching(item)
            item["category"] = _keyword_categorize(_title_for_matching(item))
            resolved_group.append(item)
            resolved_match_data.append(item_match_data)

        kept_items.extend(resolved_group)

    return kept_items, skipped_items


def _finalize_items(
    state: DigestState,
    items: list[dict[str, Any]],
    skipped_items: int,
    *,
    log_label: str,
) -> DigestState:
    sorted_items = _sort_items_by_recency(items)
    final_items, skipped_papers = _limit_papers(sorted_items)

    if skipped_papers:
        logger.info("Skipped %d additional papers (kept top %d)", skipped_papers, PAPER_LIMIT)

    logger.info("%s: %d items (skipped %d items)", log_label, len(final_items), skipped_items)

    categories: dict[str, int] = defaultdict(int)
    for item in final_items:
        categories[item.get("category", "Unknown")] += 1
    logger.info("Category distribution: %s", dict(categories))

    state["items"] = final_items
    return state


def node_collect(state: DigestState) -> DigestState:
    items = collect_items()
    logger.info("Collected %d items", len(items))
    state["items"] = items
    return state


def node_filter(state: DigestState) -> DigestState:
    items = deduplicate(state.get("items", []))
    logger.info("After URL deduplication: %d items", len(items))

    filtered_items, skipped_source_cap = _apply_source_cap(items)
    if skipped_source_cap:
        logger.info(
            "Skipped %d items due to source cap (%d/source)",
            skipped_source_cap,
            MAX_ITEMS_PER_SOURCE,
        )

    logger.info("After filtering: %d items", len(filtered_items))
    state["items"] = filtered_items
    return state


def node_categorize(state: DigestState) -> DigestState:
    items = state.get("items", [])
    if not items:
        logger.warning("No items to categorize")
        return state

    groups = _build_candidate_groups(items)
    ambiguous_groups = [group for group in groups if len(group) > 1]
    logger.info(
        "Built %d candidate groups (%d ambiguous)",
        len(groups),
        len(ambiguous_groups),
    )

    api_key = _get_openai_api_key()
    if not api_key:
        logger.warning("No OPENAI_API_KEY found, using local duplicate resolution")
        resolved_items, skipped_items = _fallback_resolve_groups(groups)
        return _finalize_items(
            state,
            resolved_items,
            skipped_items,
            log_label="After categorization",
        )

    try:
        client = _get_openai_client(api_key)

        prompt_groups: list[dict[str, Any]] = []
        for group_index, group in enumerate(groups, start=1):
            group_id = f"g{group_index}"
            prompt_items: list[dict[str, Any]] = []
            for item_index, item in enumerate(group, start=1):
                prompt_items.append(
                    {
                        "item_id": f"{group_id}i{item_index}",
                        "source": item["source"],
                        "published": item["published"].isoformat(),
                        "title": _title_for_matching(item),
                        "summary": _clean_prompt_text(
                            str(item.get("summary", "")),
                            _MAX_PROMPT_SUMMARY_CHARS,
                        ),
                    }
                )
            prompt_groups.append({"group_id": group_id, "items": prompt_items})

        prompt_payload = json.dumps({"groups": prompt_groups}, ensure_ascii=True)
        categories_str = "\n".join(f"- {category}" for category in CATEGORIES)

        prompt = f"""Deduplicate and categorize these biotech and pharma news items.

Each group contains possible near-duplicates. Only compare items within the same group.

Rules:
- If multiple items cover the same event, keep the most informative one.
- Prefer primary or official sources when they cover the same event as secondary coverage.
- Preserve distinct stories from the same company.
- Use the original title and summary to decide duplicates.
- Mark clearly off-topic items for a biotech/pharma industry digest as off-topic.
- For each kept item, assign exactly one category from this list:
{categories_str}
- For each kept item, provide a short_title of {_SHORT_TITLE_MAX_WORDS} words or fewer.

Return JSON only with this schema:
{{
  "groups": [
    {{
      "group_id": "g1",
      "off_topic_ids": ["g1i3"],
      "clusters": [
        {{
          "keep_id": "g1i1",
          "duplicate_ids": ["g1i2"],
          "category": "Clinical & Research",
          "short_title": "Example short title"
        }}
      ]
    }}
  ]
}}

Input JSON:
{prompt_payload}
"""

        response_text = _chat_completion_text(client, prompt)
        response_payload = _extract_json_object(response_text)
        response_groups = response_payload.get("groups", [])
        if not isinstance(response_groups, list):
            raise ValueError("LLM response missing groups list")

        response_group_lookup: dict[str, dict[str, Any]] = {}
        for response_group in response_groups:
            if not isinstance(response_group, dict):
                continue
            group_id = str(response_group.get("group_id", "")).strip()
            clusters = response_group.get("clusters", [])
            if not group_id or not isinstance(clusters, list):
                continue
            off_topic_ids = response_group.get("off_topic_ids", [])
            if not isinstance(off_topic_ids, list):
                off_topic_ids = []
            response_group_lookup[group_id] = {
                "clusters": [cluster for cluster in clusters if isinstance(cluster, dict)],
                "off_topic_ids": [str(item_id).strip() for item_id in off_topic_ids],
            }

        kept_items: list[dict[str, Any]] = []
        skipped_items = 0

        for group_index, group in enumerate(groups, start=1):
            group_id = f"g{group_index}"
            group_prompt_ids = {
                f"{group_id}i{item_index}": item
                for item_index, item in enumerate(group, start=1)
            }
            group_response = response_group_lookup.get(group_id, {})
            group_clusters = group_response.get("clusters", [])
            off_topic_ids = {
                item_id
                for item_id in group_response.get("off_topic_ids", [])
                if item_id in group_prompt_ids
            }
            used_ids: set[str] = set(off_topic_ids)
            skipped_items += len(off_topic_ids)

            for cluster in group_clusters:
                keep_id = str(cluster.get("keep_id", "")).strip()
                if keep_id not in group_prompt_ids or keep_id in used_ids:
                    continue

                duplicate_ids: list[str] = []
                for raw_duplicate_id in cluster.get("duplicate_ids", []):
                    duplicate_id = str(raw_duplicate_id).strip()
                    if (
                        duplicate_id
                        and duplicate_id != keep_id
                        and duplicate_id in group_prompt_ids
                        and duplicate_id not in used_ids
                    ):
                        duplicate_ids.append(duplicate_id)

                keep_item = group_prompt_ids[keep_id]
                keep_item["category"] = _validate_category(cluster.get("category"), keep_item)
                keep_item["title"] = _clean_short_title(
                    cluster.get("short_title"),
                    _title_for_matching(keep_item),
                )

                kept_items.append(keep_item)
                used_ids.add(keep_id)
                used_ids.update(duplicate_ids)
                skipped_items += len(duplicate_ids)

            for prompt_id, item in group_prompt_ids.items():
                if prompt_id in used_ids:
                    continue
                item["title"] = _title_for_matching(item)
                item["category"] = _keyword_categorize(_title_for_matching(item))
                kept_items.append(item)

        return _finalize_items(
            state,
            kept_items,
            skipped_items,
            log_label="After categorization",
        )
    except Exception as exc:
        logger.error(
            "Categorization error: %s. Falling back to local duplicate resolution.",
            exc,
        )
        resolved_items, skipped_items = _fallback_resolve_groups(groups)
        return _finalize_items(
            state,
            resolved_items,
            skipped_items,
            log_label="After local categorization",
        )


def node_render(state: DigestState) -> DigestState:
    items = state.get("items", [])
    logger.info("Rendering %d items", len(items))
    markdown = to_markdown(items)
    _NEWS_FILE.write_text(markdown, encoding="utf-8")
    state["markdown"] = markdown
    return state


def build_graph():
    graph = StateGraph(DigestState)

    graph.add_node("collect", node_collect)
    graph.add_node("filter", node_filter)
    graph.add_node("categorize", node_categorize)
    graph.add_node("render", node_render)

    graph.set_entry_point("collect")
    graph.add_edge("collect", "filter")
    graph.add_edge("filter", "categorize")
    graph.add_edge("categorize", "render")

    return graph.compile()
