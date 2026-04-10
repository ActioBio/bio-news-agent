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
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from dotenv import dotenv_values
from langgraph.graph import StateGraph
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

try:
    from collector import collect_items
    from config import (
        CATEGORIES,
        COMPANY_NAMES,
        DIGEST_OUTPUT_FILE,
        MAX_ITEMS_PER_SOURCE,
        _ENV_FILE as CONFIG_ENV_FILE,
        OPENAI_API_KEY as CONFIG_OPENAI_API_KEY,
        OPENAI_MODEL,
        OPENAI_RETRIES,
        OPENAI_TIMEOUT_SECONDS,
        PAPER_LIMIT,
    )
    from filterer import deduplicate, exclude_noise
    from renderer import to_markdown
except ModuleNotFoundError:  # pragma: no cover - module execution fallback
    from .collector import collect_items
    from .config import (
        CATEGORIES,
        COMPANY_NAMES,
        DIGEST_OUTPUT_FILE,
        MAX_ITEMS_PER_SOURCE,
        _ENV_FILE as CONFIG_ENV_FILE,
        OPENAI_API_KEY as CONFIG_OPENAI_API_KEY,
        OPENAI_MODEL,
        OPENAI_RETRIES,
        OPENAI_TIMEOUT_SECONDS,
        PAPER_LIMIT,
    )
    from .filterer import deduplicate, exclude_noise
    from .renderer import to_markdown

logger = logging.getLogger(__name__)
_OPENAI_API_KEY_PLACEHOLDERS = {
    "your_api_key_here",
    "your_openai_api_key_here",
    "sk-...",
    "sk-proj-...",
}
_SUMMARY_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
_SUMMARY_ABBREVIATION_PATTERN = re.compile(
    r"(?:\b(?:mr|mrs|ms|dr|prof|st|vs|etc|inc|co|corp|ltd|llc|plc|jr|sr|mt|no|fig|dept|est|e\.g|i\.e)\.|(?:[A-Z]\.){2,})$",
    re.IGNORECASE,
)


def _resolve_output_file(path_value: str) -> Path:
    output = Path(path_value)
    if output.is_absolute():
        return output
    return (Path(__file__).resolve().parent.parent / output).resolve()


_NEWS_FILE = _resolve_output_file(DIGEST_OUTPUT_FILE)
_CANDIDATE_SNAPSHOT_VERSION = 3
_MAX_PROMPT_SUMMARY_CHARS = 280
_SHORT_TITLE_MAX_WORDS = 8
_SOURCE_ROLE_PRIORITY = {
    "primary": 3,
    "independent_reporting": 2,
    "commentary": 1,
    "community": 0,
}
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
    executive_summary: str
    top_stories: list[str]


def _normalize_source_role(value: Any) -> str:
    source_role = str(value).strip().lower()
    if source_role in _SOURCE_ROLE_PRIORITY:
        return source_role
    return "independent_reporting"


def _source_role_priority(value: Any) -> int:
    return _SOURCE_ROLE_PRIORITY[_normalize_source_role(value)]


def _group_item_sort_key(item: dict[str, Any]) -> tuple[float, float, str]:
    return (
        -float(_source_role_priority(item.get("source_role"))),
        -item["published"].timestamp(),
        str(item.get("source", "")),
    )


def _story_rank_key(item: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        -(1 if item.get("tier") == "high" else 0),
        -float(_source_role_priority(item.get("source_role"))),
        -float(len(item.get("coverage_sources", []))),
        -item["published"].timestamp(),
        str(item.get("source", "")),
    )


def _sort_items_by_source_role(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=_group_item_sort_key)


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


def _is_placeholder_openai_api_key(api_key: str) -> bool:
    normalized = api_key.strip().strip("\"'").lower()
    if not normalized:
        return False
    if normalized in _OPENAI_API_KEY_PLACEHOLDERS:
        return True
    if "your_api" in normalized or "your_openai" in normalized:
        return True
    if "changeme" in normalized:
        return True
    if "replace" in normalized and "key" in normalized:
        return True
    return normalized.endswith("...") and normalized.startswith(("sk-", "sk-proj-"))


def _normalize_openai_api_key(api_key: str | None) -> str:
    if api_key is None:
        return ""
    api_key = api_key.strip()
    if not api_key or _is_placeholder_openai_api_key(api_key):
        return ""
    return api_key


@lru_cache(maxsize=1)
def _get_dotenv_openai_api_key() -> str:
    dotenv_values_map = dotenv_values(CONFIG_ENV_FILE)
    api_key = dotenv_values_map.get("OPENAI_API_KEY")
    if not isinstance(api_key, str):
        return ""
    return _normalize_openai_api_key(api_key)


def _get_openai_api_key() -> str:
    env_api_key = _normalize_openai_api_key(os.getenv("OPENAI_API_KEY"))
    if env_api_key:
        return env_api_key
    configured_api_key = _normalize_openai_api_key(CONFIG_OPENAI_API_KEY)
    if configured_api_key:
        return configured_api_key
    return _get_dotenv_openai_api_key()


@lru_cache(maxsize=1)
def _get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=0,
    )


def _should_retry_openai_error(exc: BaseException) -> bool:
    return isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError))


@retry(
    stop=stop_after_attempt(OPENAI_RETRIES + 1),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception(_should_retry_openai_error),
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
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(items))}
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
                key=_group_item_sort_key,
            )
        )

    return sorted(
        groups,
        key=lambda group: max(item["published"].timestamp() for item in group),
        reverse=True,
    )


def _clean_prompt_text(text: str, limit: int) -> str:
    normalized = " ".join(text.split()).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _serialize_candidate_item(item_id: str, item: dict[str, Any]) -> dict[str, Any]:
    original_title = _title_for_matching(item)
    return {
        "item_id": item_id,
        "id": str(item.get("id", item_id)),
        "title": original_title,
        "original_title": original_title,
        "link": str(item.get("link", "")),
        "source": str(item.get("source", "")),
        "published": item["published"].isoformat(),
        "summary": _clean_prompt_text(
            str(item.get("summary", "")),
            _MAX_PROMPT_SUMMARY_CHARS,
        ),
        "category": str(item.get("category", "")),
        "source_type": str(item.get("source_type", "")),
        "source_role": _normalize_source_role(item.get("source_role")),
    }


def _serialize_dedupe_candidate_item(item_id: str, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "title": _title_for_matching(item),
        "source": str(item.get("source", "")),
        "source_role": _normalize_source_role(item.get("source_role")),
        "summary": _clean_prompt_text(
            str(item.get("summary", "")),
            _MAX_PROMPT_SUMMARY_CHARS,
        ),
    }


def _build_group_payloads(
    indexed_groups: list[tuple[int, list[dict[str, Any]]]],
    *,
    item_serializer,
) -> list[dict[str, Any]]:
    payload_groups: list[dict[str, Any]] = []

    for group_index, group in indexed_groups:
        group_id = f"g{group_index}"
        payload_groups.append(
            {
                "group_id": group_id,
                "items": [
                    item_serializer(f"{group_id}i{item_index}", item)
                    for item_index, item in enumerate(group, start=1)
                ],
            }
        )

    return payload_groups


def _build_candidate_snapshot_groups(
    groups: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return _build_group_payloads(
        list(enumerate(groups, start=1)),
        item_serializer=_serialize_candidate_item,
    )


def _build_dedupe_prompt_groups(
    indexed_groups: list[tuple[int, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    return _build_group_payloads(
        indexed_groups,
        item_serializer=_serialize_dedupe_candidate_item,
    )


def build_candidate_snapshot(items: list[dict[str, Any]]) -> dict[str, Any]:
    groups = _build_candidate_groups(items)
    return {
        "schema_version": _CANDIDATE_SNAPSHOT_VERSION,
        "kind": "bio-news-agent.candidates",
        "categories": list(CATEGORIES),
        "groups": _build_candidate_snapshot_groups(groups),
    }


def _deserialize_candidate_item(item_payload: dict[str, Any]) -> dict[str, Any]:
    published_value = item_payload.get("published")
    if not isinstance(published_value, str):
        raise ValueError("Candidate item is missing published timestamp")

    title = str(item_payload.get("title", "")).strip()
    original_title = str(item_payload.get("original_title", title)).strip() or title

    return {
        "id": str(item_payload.get("id", item_payload.get("item_id", ""))),
        "title": original_title,
        "original_title": original_title,
        "link": str(item_payload.get("link", "")),
        "source": str(item_payload.get("source", "")),
        "published": datetime.fromisoformat(published_value),
        "summary": str(item_payload.get("summary", "")),
        "category": str(item_payload.get("category", "")),
        "source_type": str(item_payload.get("source_type", "")),
        "source_role": _normalize_source_role(item_payload.get("source_role")),
    }


def _candidate_groups_from_snapshot(snapshot_payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    raw_groups = snapshot_payload.get("groups", [])
    if not isinstance(raw_groups, list):
        raise ValueError("Candidate snapshot missing groups list")

    groups: list[list[dict[str, Any]]] = []
    for group_payload in raw_groups:
        if not isinstance(group_payload, dict):
            raise ValueError("Candidate group must be an object")
        raw_items = group_payload.get("items", [])
        if not isinstance(raw_items, list):
            raise ValueError("Candidate group missing items list")
        groups.append(
            [
                _deserialize_candidate_item(item_payload)
                for item_payload in raw_items
                if isinstance(item_payload, dict)
            ]
        )
    return groups


def _serialize_enrichment_item(item: dict[str, Any]) -> dict[str, Any]:
    prompt_id = str(item.get("_prompt_id", "")).strip()
    return {
        "item_id": prompt_id,
        "title": _title_for_matching(item),
        "source": str(item.get("source", "")),
        "source_role": _normalize_source_role(item.get("source_role")),
        "published": item["published"].isoformat(),
        "summary": _clean_prompt_text(
            str(item.get("summary", "")),
            _MAX_PROMPT_SUMMARY_CHARS,
        ),
        "source_type": str(item.get("source_type", "")),
        "coverage_count": len(item.get("coverage_sources", [])) + 1,
    }


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


def _clean_summary_line(value: Any) -> str:
    line = " ".join(str(value).split()).strip()
    if not line:
        return ""
    for boundary in _SUMMARY_BOUNDARY_PATTERN.finditer(line):
        prefix = line[: boundary.start()].rstrip()
        if not prefix:
            continue
        if _SUMMARY_ABBREVIATION_PATTERN.search(prefix):
            continue
        if re.search(r"\b\d+\.\d+$", prefix):
            continue
        suffix = line[boundary.start() :].lstrip().lstrip("\"'([{")
        if not suffix or not suffix[0].isalnum():
            continue
        line = prefix
        break
    return line


def _fallback_summary_line(item: dict[str, Any]) -> str:
    return _clean_summary_line(item.get("summary", ""))


def _best_available_summary_line(items: list[dict[str, Any]]) -> str:
    for item in items:
        summary_line = _fallback_summary_line(item)
        if summary_line:
            return summary_line
    return ""


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

    for group_index, group in enumerate(groups, start=1):
        resolved_group: list[dict[str, Any]] = []
        resolved_match_data: list[ItemMatchData] = []
        for item_index, item in enumerate(group, start=1):
            item_match_data = _build_item_match_data(item)
            duplicate_index = next(
                (
                    idx
                    for idx, existing_data in enumerate(resolved_match_data)
                    if _is_candidate_duplicate_data(existing_data, item_match_data)
                ),
                None,
            )
            if duplicate_index is not None:
                skipped_items += 1
                existing_item = resolved_group[duplicate_index]
                if not existing_item.get("summary_line"):
                    existing_item["summary_line"] = _fallback_summary_line(item)
                dup_source = str(item.get("source", "")).strip()
                coverage_sources = existing_item.setdefault("coverage_sources", [])
                if dup_source and dup_source not in coverage_sources:
                    coverage_sources.append(dup_source)
                continue

            item["_prompt_id"] = f"g{group_index}i{item_index}"
            item["title"] = _title_for_matching(item)
            item["category"] = _keyword_categorize(_title_for_matching(item))
            item["summary_line"] = _fallback_summary_line(item)
            item["tier"] = "normal"
            item["coverage_sources"] = []
            resolved_group.append(item)
            resolved_match_data.append(item_match_data)

        kept_items.extend(resolved_group)

    return kept_items, skipped_items


def _select_top_story_ids(items: list[dict[str, Any]], requested_ids: list[str]) -> list[str]:
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

    ranked_items = sorted(
        prompt_id_lookup.values(),
        key=_story_rank_key,
    )
    return [str(item["_prompt_id"]) for item in ranked_items[:3]]


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

    requested_top_stories = state.get("top_stories", [])
    if not isinstance(requested_top_stories, list):
        requested_top_stories = []
    state["top_stories"] = _select_top_story_ids(final_items, requested_top_stories)
    state["items"] = final_items
    return state


def _seed_resolved_item(
    item: dict[str, Any],
    prompt_id: str,
    *,
    duplicate_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    duplicate_items = duplicate_items or []
    item["title"] = _title_for_matching(item)
    item["category"] = _keyword_categorize(_title_for_matching(item))
    item["_prompt_id"] = prompt_id
    item["summary_line"] = _best_available_summary_line([item, *duplicate_items])
    item["tier"] = "normal"
    item["coverage_sources"] = [
        str(duplicate_item.get("source", "")).strip()
        for duplicate_item in duplicate_items
        if str(duplicate_item.get("source", "")).strip()
    ]
    return item


def _prepare_distinct_items(
    indexed_groups: list[tuple[int, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    prepared_items: list[dict[str, Any]] = []
    for group_index, group in indexed_groups:
        for item_index, item in enumerate(group, start=1):
            prepared_items.append(_seed_resolved_item(item, f"g{group_index}i{item_index}"))
    return prepared_items


def _apply_dedupe_response(
    indexed_groups: list[tuple[int, list[dict[str, Any]]]],
    response_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    response_groups = response_payload.get("groups", [])
    if not isinstance(response_groups, list):
        raise ValueError("LLM dedupe response missing groups list")

    response_group_lookup: dict[str, list[dict[str, Any]]] = {}
    for response_group in response_groups:
        if not isinstance(response_group, dict):
            continue
        group_id = str(response_group.get("group_id", "")).strip()
        clusters = response_group.get("clusters", [])
        if not group_id or not isinstance(clusters, list):
            continue
        response_group_lookup[group_id] = [
            cluster for cluster in clusters if isinstance(cluster, dict)
        ]

    kept_items: list[dict[str, Any]] = []
    skipped_items = 0

    for group_index, group in indexed_groups:
        group_id = f"g{group_index}"
        if group_id not in response_group_lookup:
            raise ValueError(f"LLM dedupe response missing group {group_id}")

        group_prompt_ids = {
            f"{group_id}i{item_index}": item
            for item_index, item in enumerate(group, start=1)
        }
        used_ids: set[str] = set()

        for cluster in response_group_lookup[group_id]:
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

            duplicate_items = _sort_items_by_source_role(
                [group_prompt_ids[dup_id] for dup_id in duplicate_ids]
            )
            kept_items.append(
                _seed_resolved_item(
                    group_prompt_ids[keep_id],
                    keep_id,
                    duplicate_items=duplicate_items,
                )
            )
            used_ids.add(keep_id)
            used_ids.update(duplicate_ids)
            skipped_items += len(duplicate_ids)

        for prompt_id, item in group_prompt_ids.items():
            if prompt_id in used_ids:
                continue
            kept_items.append(_seed_resolved_item(item, prompt_id))

    return kept_items, skipped_items


def _apply_enrichment_response(
    state: DigestState,
    items: list[dict[str, Any]],
    response_payload: dict[str, Any],
    *,
    skipped_items: int,
    log_label: str,
) -> DigestState:
    response_items = response_payload.get("items", [])
    if not isinstance(response_items, list):
        raise ValueError("LLM enrichment response missing items list")

    executive_summary = str(response_payload.get("executive_summary", "")).strip()
    raw_top_stories = response_payload.get("top_stories", [])
    if not isinstance(raw_top_stories, list):
        raw_top_stories = []
    top_story_ids = [str(s).strip() for s in raw_top_stories if isinstance(s, str)]
    raw_off_topic_ids = response_payload.get("off_topic_ids", [])
    if not isinstance(raw_off_topic_ids, list):
        raw_off_topic_ids = []
    off_topic_ids = {str(item_id).strip() for item_id in raw_off_topic_ids if isinstance(item_id, str)}

    item_lookup = {
        str(item.get("_prompt_id", "")).strip(): item
        for item in items
        if str(item.get("_prompt_id", "")).strip()
    }
    enriched_items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for response_item in response_items:
        if not isinstance(response_item, dict):
            continue
        item_id = str(response_item.get("item_id", "")).strip()
        if item_id not in item_lookup or item_id in seen_ids or item_id in off_topic_ids:
            continue

        item = item_lookup[item_id]
        item["category"] = _validate_category(response_item.get("category"), item)
        item["title"] = _clean_short_title(
            response_item.get("short_title"),
            _title_for_matching(item),
        )
        existing_summary_line = _clean_summary_line(item.get("summary_line", ""))
        item["summary_line"] = (
            _clean_summary_line(response_item.get("summary_line", ""))
            or existing_summary_line
            or _fallback_summary_line(item)
        )
        raw_tier = str(response_item.get("tier", "normal")).strip().lower()
        item["tier"] = raw_tier if raw_tier in ("high", "normal") else "normal"
        enriched_items.append(item)
        seen_ids.add(item_id)

    for item_id, item in item_lookup.items():
        if item_id in seen_ids or item_id in off_topic_ids:
            continue
        enriched_items.append(item)

    state["executive_summary"] = executive_summary
    state["top_stories"] = top_story_ids
    return _finalize_items(
        state,
        enriched_items,
        skipped_items + len(off_topic_ids),
        log_label=log_label,
    )


def _apply_structured_response(
    state: DigestState,
    groups: list[list[dict[str, Any]]],
    response_payload: dict[str, Any],
    *,
    log_label: str,
) -> DigestState:
    response_groups = response_payload.get("groups", [])
    if not isinstance(response_groups, list):
        raise ValueError("LLM response missing groups list")

    executive_summary = str(response_payload.get("executive_summary", "")).strip()
    raw_top_stories = response_payload.get("top_stories", [])
    if not isinstance(raw_top_stories, list):
        raw_top_stories = []
    top_story_ids = [str(s).strip() for s in raw_top_stories if isinstance(s, str)]

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
            duplicate_items = _sort_items_by_source_role(
                [group_prompt_ids[dup_id] for dup_id in duplicate_ids]
            )
            keep_item["category"] = _validate_category(cluster.get("category"), keep_item)
            keep_item["title"] = _clean_short_title(
                cluster.get("short_title"),
                _title_for_matching(keep_item),
            )
            keep_item["_prompt_id"] = keep_id
            keep_item["summary_line"] = (
                _clean_summary_line(cluster.get("summary_line", ""))
                or _best_available_summary_line([keep_item, *duplicate_items])
            )
            raw_tier = str(cluster.get("tier", "normal")).strip().lower()
            keep_item["tier"] = raw_tier if raw_tier in ("high", "normal") else "normal"

            coverage_sources: list[str] = []
            for duplicate_item in duplicate_items:
                dup_source = str(duplicate_item.get("source", "")).strip()
                if dup_source and dup_source not in coverage_sources:
                    coverage_sources.append(dup_source)
            keep_item["coverage_sources"] = coverage_sources

            kept_items.append(keep_item)
            used_ids.add(keep_id)
            used_ids.update(duplicate_ids)
            skipped_items += len(duplicate_ids)

        for prompt_id, item in group_prompt_ids.items():
            if prompt_id in used_ids:
                continue
            item["title"] = _title_for_matching(item)
            item["category"] = _keyword_categorize(_title_for_matching(item))
            item["_prompt_id"] = prompt_id
            item["summary_line"] = _fallback_summary_line(item)
            item["tier"] = "normal"
            item["coverage_sources"] = []
            kept_items.append(item)

    state["executive_summary"] = executive_summary
    state["top_stories"] = top_story_ids
    return _finalize_items(
        state,
        kept_items,
        skipped_items,
        log_label=log_label,
    )


def node_collect(state: DigestState) -> DigestState:
    items = collect_items()
    logger.info("Collected %d items", len(items))
    state["items"] = items
    return state


def export_candidate_snapshot(output_path: Path) -> dict[str, Any]:
    state = node_collect({})
    state = node_filter(state)
    snapshot_payload = build_candidate_snapshot(state.get("items", []))
    output_path.write_text(
        json.dumps(snapshot_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return snapshot_payload


def apply_decisions_file(
    decisions_path: Path,
    candidates_path: Path,
) -> DigestState:
    snapshot_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not isinstance(snapshot_payload, dict):
        raise ValueError("Candidate snapshot must be a JSON object")

    groups = _candidate_groups_from_snapshot(snapshot_payload)
    decisions_payload = _extract_json_object(
        decisions_path.read_text(encoding="utf-8")
    )

    state = _apply_structured_response(
        {},
        groups,
        decisions_payload,
        log_label="After categorization",
    )
    return node_render(state)


def node_filter(state: DigestState) -> DigestState:
    items = deduplicate(state.get("items", []))
    logger.info("After URL deduplication: %d items", len(items))

    items, skipped_noise = exclude_noise(items)
    if skipped_noise:
        logger.info("Skipped %d noise titles", skipped_noise)

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
    indexed_groups = list(enumerate(groups, start=1))
    ambiguous_groups = [group for group in groups if len(group) > 1]
    ambiguous_indexed_groups = [
        (group_index, group)
        for group_index, group in indexed_groups
        if len(group) > 1
    ]
    distinct_indexed_groups = [
        (group_index, group)
        for group_index, group in indexed_groups
        if len(group) == 1
    ]
    logger.info(
        "Built %d candidate groups (%d ambiguous)",
        len(groups),
        len(ambiguous_groups),
    )

    api_key = _get_openai_api_key()
    if not api_key:
        logger.warning(
            "No valid OPENAI_API_KEY found, using local duplicate resolution"
        )
        resolved_items, skipped_items = _fallback_resolve_groups(groups)
        state["executive_summary"] = ""
        state["top_stories"] = []
        return _finalize_items(
            state,
            resolved_items,
            skipped_items,
            log_label="After categorization",
        )

    try:
        client = _get_openai_client(api_key)
        skipped_items = 0
        resolved_items = _prepare_distinct_items(distinct_indexed_groups)

        if ambiguous_indexed_groups:
            dedupe_payload = json.dumps(
                {"groups": _build_dedupe_prompt_groups(ambiguous_indexed_groups)},
                ensure_ascii=True,
            )
            dedupe_prompt = f"""Deduplicate these biotech and pharma news groups.

Each group contains possible near-duplicates. Only compare items within the same group.

Rules:
- If multiple items cover the same event, keep the most informative one.
- Input items include source_role. When stories are otherwise equivalent, prefer keep_id using this order: primary, independent_reporting, commentary, community.
- Preserve distinct stories from the same company.
- Use the title and summary to decide duplicates.
- If items in a group are distinct, return separate clusters for them.

Return JSON only with this schema:
{{
  "groups": [
    {{
      "group_id": "g1",
      "clusters": [
        {{
          "keep_id": "g1i1",
          "duplicate_ids": ["g1i2"]
        }}
      ]
    }}
  ]
}}

Input JSON:
{dedupe_payload}
"""

            dedupe_response = _extract_json_object(_chat_completion_text(client, dedupe_prompt))
            deduped_items, skipped_items = _apply_dedupe_response(
                ambiguous_indexed_groups,
                dedupe_response,
            )
            resolved_items.extend(deduped_items)

        categories_str = "\n".join(f"- {category}" for category in CATEGORIES)
        enrichment_payload = json.dumps(
            {"items": [_serialize_enrichment_item(item) for item in resolved_items]},
            ensure_ascii=True,
        )
        enrichment_prompt = f"""Enrich these deduplicated biotech and pharma news items for the daily digest.

Each item is already deduplicated.

Rules:
- Mark clearly off-topic items for a biotech/pharma industry digest as off-topic.
- For each remaining item, assign exactly one category from this list:
{categories_str}
- For each item, provide a short_title of {_SHORT_TITLE_MAX_WORDS} words or fewer.
- For each item, provide a summary_line: one plain-English sentence explaining why a general reader should care about this story.
- For each item, assign a tier: "high" if the story has broad impact, strong novelty, solid evidence, or high practical relevance; otherwise "normal".
- Use source_role and coverage_count as signals for authority and importance.
- Write an executive_summary: 2-3 sentences capturing the day's most important biotech/pharma themes.
- Pick up to 3 most significant item_ids as top_stories, drawn from high-tier items when possible.

Return JSON only with this schema:
{{
  "executive_summary": "2-3 sentence overview of today's biotech news.",
  "top_stories": ["g1i1", "g3i2", "g5i1"],
  "off_topic_ids": ["g2i1"],
  "items": [
    {{
      "item_id": "g1i1",
      "category": "Clinical & Research",
      "short_title": "Example short title",
      "summary_line": "Why this matters in one sentence.",
      "tier": "high"
    }}
  ]
}}

Input JSON:
{enrichment_payload}
"""

        enrichment_response = _extract_json_object(
            _chat_completion_text(client, enrichment_prompt)
        )
        return _apply_enrichment_response(
            state,
            resolved_items,
            enrichment_response,
            skipped_items=skipped_items,
            log_label="After categorization",
        )
    except Exception as exc:
        logger.error(
            "Categorization error: %s. Falling back to local duplicate resolution.",
            exc,
        )
        resolved_items, skipped_items = _fallback_resolve_groups(groups)
        state["executive_summary"] = ""
        state["top_stories"] = []
        return _finalize_items(
            state,
            resolved_items,
            skipped_items,
            log_label="After local categorization",
        )


def node_render(state: DigestState) -> DigestState:
    items = state.get("items", [])
    logger.info("Rendering %d items", len(items))
    markdown = to_markdown(
        items,
        executive_summary=state.get("executive_summary", ""),
        top_stories=state.get("top_stories", []),
    )
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
