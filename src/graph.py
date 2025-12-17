"""
LangGraph pipeline for bio-news-agent

Flow:
    collect ─▶ filter ─▶ shortify ─▶ categorize ─▶ render
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from openai import OpenAI
from langgraph.graph import StateGraph

from collector import collect_items
from config import (
    CATEGORIES,
    COMPANY_NAMES,
    MAX_ITEMS_PER_SOURCE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PAPER_LIMIT,
)
from filterer import deduplicate
from renderer import to_markdown

logger = logging.getLogger(__name__)

BATCH_SIZE = 20


# ──────────────────────────────────────────────────────────────
# 1. Shared state definition
class DigestState(TypedDict, total=False):
    items: List[Dict[str, Any]]
    markdown: str


# ──────────────────────────────────────────────────────────────
# 2. Nodes
def node_collect(state: DigestState) -> DigestState:
    items = collect_items()
    logger.info(f"Collected {len(items)} items")
    state["items"] = items
    return state


def node_filter(state: DigestState) -> DigestState:
    items = deduplicate(state.get("items", []))

    # Limit papers and apply source cap
    paper_count = 0
    source_counts: Dict[str, int] = defaultdict(int)
    filtered_items = []
    skipped_papers = 0
    skipped_source_cap = 0

    for item in sorted(items, key=lambda x: x["published"], reverse=True):
        # Apply per-source cap for diversity
        source = item.get("source", "")
        if MAX_ITEMS_PER_SOURCE > 0 and source_counts[source] >= MAX_ITEMS_PER_SOURCE:
            logger.debug(f"Skipping (source cap): {item['title'][:50]}...")
            skipped_source_cap += 1
            continue
        source_counts[source] += 1

        # Limit papers
        if "Papers" in source and paper_count >= PAPER_LIMIT:
            logger.debug(f"Skipping paper: {item['title'][:50]}...")
            skipped_papers += 1
            continue
        if "Papers" in source:
            paper_count += 1

        filtered_items.append(item)

    logger.info(f"After deduplication: {len(items)} items")
    if skipped_source_cap > 0:
        logger.info(f"Skipped {skipped_source_cap} items due to source cap ({MAX_ITEMS_PER_SOURCE}/source)")
    if skipped_papers > 0:
        logger.info(f"Skipped {skipped_papers} additional papers (kept top {PAPER_LIMIT})")
    logger.info(f"After filtering: {len(filtered_items)} items")
    state["items"] = filtered_items
    return state


def node_shortify(state: DigestState) -> DigestState:
    """Shorten titles to ≤10 words using batched LLM calls."""
    if not OPENAI_API_KEY:
        logger.warning("No OPENAI_API_KEY found, skipping shortify")
        return state

    items = state.get("items", [])
    if not items:
        logger.warning("No items to shortify")
        return state

    client = OpenAI(api_key=OPENAI_API_KEY)
    shortified_count = 0

    # Process in batches of BATCH_SIZE
    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start : batch_start + BATCH_SIZE]

        # Build batch prompt
        titles_list = "\n".join(
            f"{i+1}. {item['title']}" for i, item in enumerate(batch)
        )
        prompt = f"""Shorten each headline to 8 words MAX. Keep the core news. Remove fluff.

Rules:
- STRICT 8-word limit
- Use active voice
- Remove source attributions, company descriptors
- Keep drug names, company names, key numbers

{titles_list}

Return ONLY shortened headlines, numbered (1. headline, 2. headline, etc.)"""

        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.choices[0].message.content
            if not content:
                continue
            response_text = content.strip()

            # Parse response
            lines = [
                re.sub(r"^\d+\.\s*", "", line.strip())
                for line in response_text.split("\n")
                if line.strip() and re.match(r"^\d+\.", line.strip())
            ]

            # Apply shortened titles
            for i, new_title in enumerate(lines):
                if i < len(batch) and new_title:
                    batch[i]["title"] = new_title
                    shortified_count += 1

        except Exception as exc:
            logger.warning(f"LLM error for batch starting at {batch_start}: {exc}")

    logger.info(f"Shortified {shortified_count}/{len(items)} items")
    return state


def _extract_keywords(title: str) -> list[str]:
    """Extract significant keywords from title for fingerprinting.

    Returns a sorted list (not set) for deterministic fingerprinting.
    """
    title_lower = title.lower()
    # Remove common words
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "its", "it", "this", "that", "these", "those", "new", "says", "said",
    }
    words = sorted(set(
        w for w in re.findall(r"\b\w+\b", title_lower)
        if len(w) > 3 and w not in stopwords
    ))
    return words


def _extract_entities(title: str) -> set[str]:
    """Extract company/biotech names from title for entity-based dedup.

    Looks for capitalized multi-word names and known patterns.
    """
    entities: set[str] = set()
    title_lower = title.lower()

    # Check for known company names from config
    for company in COMPANY_NAMES:
        if company in title_lower:
            entities.add(company)

    # Extract capitalized words that look like company names (ending in common suffixes)
    # Pattern: Capitalized word(s) followed by Therapeutics, Pharma, Bio, etc.
    company_suffixes = [
        "therapeutics", "pharma", "pharmaceuticals", "biotech", "biosciences",
        "sciences", "genomics", "oncology", "medicine", "health", "bio",
    ]
    words = title.split()
    for i, word in enumerate(words):
        word_lower = word.lower().rstrip(",'s")
        if word_lower in company_suffixes and i > 0:
            # Include the word before the suffix as part of company name
            prev_word = words[i - 1].rstrip(",'s").lower()
            if prev_word and prev_word[0].isupper() or len(prev_word) > 2:
                entities.add(f"{prev_word}_{word_lower}")

    # Also extract standalone capitalized proper nouns that might be companies
    # Look for CamelCase or all-caps abbreviations
    for word in words:
        clean = word.strip("',.-()[]")
        if len(clean) >= 3:
            # All caps (like "FDA", "NIH", "J&J")
            if clean.isupper() and len(clean) <= 5:
                entities.add(clean.lower())
            # Capitalized and looks like a name (not common words)
            elif clean[0].isupper() and clean.lower() not in {
                "the", "and", "for", "with", "from", "into", "after", "phase",
                "new", "first", "trial", "drug", "data", "study", "cancer",
                "treatment", "therapy", "patients", "disease", "health",
            }:
                entities.add(clean.lower())

    return entities


def _run_keyword_dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run keyword-based duplicate detection on items."""
    seen_keywords: Dict[str, int] = {}
    seen_entities: Dict[str, int] = {}

    for i, item in enumerate(items):
        if item.get("skip"):
            continue

        keywords = _extract_keywords(item["title"])
        # Create fingerprint from top keywords (already sorted by _extract_keywords)
        fingerprint = "_".join(keywords[:5])

        if fingerprint and fingerprint in seen_keywords:
            item["skip"] = True
            item["skip_reason"] = f"keyword duplicate of item {seen_keywords[fingerprint]}"
            logger.debug(f"Keyword duplicate found: {item['title']}")
        elif fingerprint:
            seen_keywords[fingerprint] = i

    # Second pass: entity-based dedup for same company/topic from different sources
    for i, item in enumerate(items):
        if item.get("skip"):
            continue

        entities = _extract_entities(item["title"])
        curr_keywords = set(_extract_keywords(item["title"]))

        # Create entity key from significant entities (companies, orgs)
        # Only flag as duplicate if multiple specific entities match
        for entity in entities:
            # Skip generic entities
            if entity in {"fda", "nih", "cdc", "ema"}:
                continue
            entity_key = entity
            if entity_key in seen_entities:
                prev_idx = seen_entities[entity_key]
                # Check if titles share multiple keywords (not just the entity)
                prev_keywords = set(_extract_keywords(items[prev_idx]["title"]))
                shared = prev_keywords & curr_keywords
                # If they share the entity AND 2+ other keywords, likely duplicate
                if len(shared) >= 2:
                    item["skip"] = True
                    item["skip_reason"] = f"entity duplicate ({entity}) of item {prev_idx}"
                    logger.debug(f"Entity duplicate found: {item['title']}")
                    break
            else:
                seen_entities[entity_key] = i

    original_count = len(items)
    items = [item for item in items if not item.get("skip")]
    if original_count != len(items):
        logger.info(f"Keyword dedup removed {original_count - len(items)} duplicates")
    return items


def node_categorize(state: DigestState) -> DigestState:
    """Use LLM to categorize items and identify duplicates."""
    items = state.get("items", [])
    if not items:
        logger.warning("No items to categorize")
        return state

    if not OPENAI_API_KEY:
        logger.warning("No OPENAI_API_KEY found, using keyword categorization")
        for item in items:
            item["category"] = _keyword_categorize(item["title"])
        # Still run keyword-based duplicate detection
        items = _run_keyword_dedup(items)
        state["items"] = items
        return state

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Create item list
        items_text = "\n".join(
            f"{i+1}. {item['title']} — {item['source']}"
            for i, item in enumerate(items)
        )

        categories_str = ", ".join(CATEGORIES)
        prompt = f"""Analyze these headlines for a BIOTECH/PHARMA industry digest.

TASK 1 - RELEVANCE: Mark OFF-TOPIC items that don't belong in a biotech/pharma digest:
- General consumer health (school lunches, diet trends, fitness tips)
- Mental health lifestyle content (social media and mental health)
- Non-industry wellness/nutrition advice
- General news that happens to mention health tangentially

TASK 2 - DUPLICATES: Mark duplicate stories about the SAME EVENT:
- "FDA approves drug" = "Drug gets FDA nod" = "Regulatory approval for drug"
- "Company A acquires B" = "B bought by A" = "A-B merger complete"
Keep the most informative version.

TASK 3 - CATEGORIZE remaining items:
- Regulatory & FDA: FDA/EMA approvals, rejections, submissions, regulatory decisions
- Clinical & Research: Clinical trials, study results, drug mechanisms, research findings
- Deals & Finance: M&A, funding rounds, partnerships, licensing deals, financial news
- Company News: Leadership changes, layoffs, company strategies, lawsuits, product launches, devices
- Policy & Politics: Government policy, legislation, NIH/CDC decisions, healthcare reform
- Market Insights: Industry trends, market analysis, forecasts, rankings

Items:
{items_text}

For each item, respond with ONLY ONE of:
- Category name (if relevant and keeping)
- SKIP (if duplicate or off-topic)

One per line."""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )

        content = resp.choices[0].message.content
        if not content:
            logger.warning("Empty LLM response for categorization")
            for item in items:
                item["category"] = _keyword_categorize(item["title"])
            items = _run_keyword_dedup(items)
            state["items"] = items
            return state
        response_text = content.strip()
        logger.info("LLM categorization response received")

        lines = [line.strip() for line in response_text.split("\n") if line.strip()]

        # First pass: LLM categorization
        for i, line in enumerate(lines):
            if i >= len(items):
                break

            if "SKIP" in line.upper():
                items[i]["skip"] = True
                items[i]["skip_reason"] = "duplicate"
                continue

            # Check for category match
            category_found = False
            for valid_cat in CATEGORIES:
                if valid_cat.lower() in line.lower():
                    items[i]["category"] = valid_cat
                    category_found = True
                    break

            if not category_found:
                items[i]["category"] = _keyword_categorize(items[i]["title"])

        # Second pass: Keyword-based duplicate detection as safety net
        original_count = len(items)
        items = _run_keyword_dedup(items)
        state["items"] = items

        logger.info(
            f"After categorization: {len(items)} items "
            f"(skipped {original_count - len(items)} duplicates)"
        )

        # Log category distribution
        cats: Dict[str, int] = defaultdict(int)
        for item in items:
            cats[item.get("category", "Unknown")] += 1
        logger.info(f"Category distribution: {dict(cats)}")

    except Exception as exc:
        logger.warning(f"Categorization error: {exc}")
        for item in items:
            item["category"] = _keyword_categorize(item["title"])
        items = _run_keyword_dedup(items)
        state["items"] = items

    return state


def _keyword_categorize(title: str) -> str:
    """Fallback keyword-based categorization."""
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

    # Check for company names -> Company News
    if any(company in title_lower for company in COMPANY_NAMES):
        return "Company News"

    return "Company News"


def node_render(state: DigestState) -> DigestState:
    items = state.get("items", [])
    logger.info(f"Rendering {len(items)} items")
    markdown = to_markdown(items)
    Path("news.md").write_text(markdown, encoding="utf-8")
    state["markdown"] = markdown
    return state


# ──────────────────────────────────────────────────────────────
# 3. Build LangGraph
def build_graph():
    g = StateGraph(DigestState)

    g.add_node("collect", node_collect)
    g.add_node("filter", node_filter)
    g.add_node("shortify", node_shortify)
    g.add_node("categorize", node_categorize)
    g.add_node("render", node_render)

    g.set_entry_point("collect")
    g.add_edge("collect", "filter")
    g.add_edge("filter", "shortify")
    g.add_edge("shortify", "categorize")
    g.add_edge("categorize", "render")

    return g.compile()
