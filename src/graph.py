"""
LangGraph pipeline for bio-news-agent

Flow:
    collect ─▶ filter ─▶ categorize ─▶ shortify ─▶ render
"""
from __future__ import annotations

import os
from openai import OpenAI
from pathlib import Path
from typing import TypedDict, List, Dict
from collections import defaultdict

from dotenv import load_dotenv
from langgraph.graph import StateGraph

from collector import collect_items
from filterer import deduplicate
from renderer import to_markdown

# load .env once
load_dotenv()

# ──────────────────────────────────────────────────────────────
# 1. Shared state definition
class DigestState(TypedDict, total=False):
    items:    List[Dict]   # list of headline dicts
    markdown: str          # final rendered MD


# ──────────────────────────────────────────────────────────────
# 2. Nodes
def node_collect(state: DigestState) -> DigestState:
    items = collect_items()
    print(f"📊 Collected {len(items)} items")
    state["items"] = items
    return state


def node_filter(state: DigestState) -> DigestState:
    items = deduplicate(state.get("items", []))
    
    # Limit papers to avoid overwhelming the digest
    paper_count = 0
    filtered_items = []
    skipped_papers = 0
    
    for item in sorted(items, key=lambda x: x["published"], reverse=True):
        if "Papers" in item.get("source", "") and paper_count >= 7:
            print(f"   Skipping paper: {item['title'][:50]}...")
            skipped_papers += 1
            continue  # Skip additional papers
        if "Papers" in item.get("source", ""):
            paper_count += 1
        filtered_items.append(item)
    
    print(f"📊 After deduplication: {len(items)} items")
    if skipped_papers > 0:
        print(f"📊 Skipped {skipped_papers} additional papers (kept top 7)")
    print(f"📊 After limiting papers: {len(filtered_items)} items")
    state["items"] = filtered_items
    return state


def node_categorize(state: DigestState) -> DigestState:
    """
    Use LLM to categorize items and identify duplicates.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If no API key, skip categorization but set varied categories
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found, using default categorization")
        # Distribute items across categories for testing
        categories = [
            "Regulatory & FDA",
            "Clinical & Research", 
            "Deals & Finance",
            "Company News",
            "Policy & Politics",
            "Market Insights"
        ]
        for i, item in enumerate(state.get("items", [])):
            item["category"] = categories[i % len(categories)]
        return state
        
    if not state.get("items"):
        print("⚠️  No items to categorize")
        return state
    
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Create item list for categorization
        items_text = "\n".join([
            f"{i+1}. {item['title']} — {item['source']}"
            for i, item in enumerate(state["items"])
        ])
        
        prompt = f"""Analyze these biotech/pharma headlines:

STEP 1: FIND DUPLICATES
These are the SAME story if they describe the same event:
- "Company X wins case" = "Judge rules for Company X" = "Court favors Company X"
- "FDA approves Drug Y" = "Drug Y gets FDA approval" = "Company receives approval for Drug Y"
- "Company raises $Z" = "Company valued at $W after funding" = "Company secures funding"

STEP 2: For duplicates, mark all but the best/most detailed version as "SKIP"

STEP 3: Categorize remaining items into EXACTLY one of these categories:
   - Regulatory & FDA
   - Clinical & Research
   - Deals & Finance
   - Company News
   - Policy & Politics
   - Market Insights

Mark opinion pieces or off-topic items as "SKIP".

Items:
{items_text}

OUTPUT: One line per item. Write ONLY the category name OR write SKIP.
NO NUMBERS. NO PUNCTUATION. Just the category or SKIP.

Example correct output:
Regulatory & FDA
SKIP
Clinical & Research
Company News"""

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        
        # Parse categorization response
        response_text = resp.choices[0].message.content.strip()
        print(f"📊 LLM response preview: {response_text[:200]}...")
        
        lines = response_text.split('\n')
        for i, line in enumerate(lines):
            if i >= len(state["items"]):
                break
                
            # Clean the line
            line = line.strip()
            
            # Remove line number if present
            if ". " in line:
                parts = line.split(". ", 1)
                if len(parts) > 1:
                    category = parts[1].strip()
                else:
                    category = line
            else:
                category = line
            
            if "SKIP" in category.upper():
                state["items"][i]["skip"] = True
                state["items"][i]["skip_reason"] = "duplicate or off-topic"
            else:
                # Only use valid categories
                valid_categories = [
                    "Regulatory & FDA",
                    "Clinical & Research", 
                    "Deals & Finance",
                    "Company News",
                    "Policy & Politics",
                    "Market Insights"
                ]
                
                # Find exact matching category
                if category in valid_categories:
                    state["items"][i]["category"] = category
                else:
                    # Try partial match
                    matched = False
                    for valid_cat in valid_categories:
                        if any(word in category.lower() for word in valid_cat.lower().split()):
                            state["items"][i]["category"] = valid_cat
                            matched = True
                            break
                    
                    if not matched:
                        # Default category if no match
                        state["items"][i]["category"] = "Company News"
                        print(f"⚠️  No match for category '{category}', using Company News")
                    
        # Remove skipped items
        original_count = len(state["items"])
        skipped_items = [item for item in state["items"] if item.get("skip")]
        state["items"] = [item for item in state["items"] if not item.get("skip")]
        
        # Log skipped items
        if skipped_items:
            print(f"📊 Skipped items:")
            for item in skipped_items[:5]:  # Show first 5
                print(f"   - {item['title'][:60]}... ({item.get('skip_reason', 'duplicate')})")
            if len(skipped_items) > 5:
                print(f"   ... and {len(skipped_items) - 5} more")
        
        print(f"📊 After categorization: {len(state['items'])} items (skipped {original_count - len(state['items'])})")
        
        # Debug: show categories assigned
        cats = defaultdict(int)
        for item in state["items"]:
            cats[item.get("category", "Unknown")] += 1
        print(f"📊 Category distribution: {dict(cats)}")
        
    except Exception as exc:
        print(f"⚠️  Categorization error: {exc}")
        import traceback
        traceback.print_exc()
        
        # Set default varied categories on error
        categories = ["Company News", "Clinical & Research", "Regulatory & FDA"]
        for i, item in enumerate(state.get("items", [])):
            item["category"] = categories[i % len(categories)]
    
    return state


def node_shortify(state: DigestState) -> DigestState:
    """
    Rewrite each item's title in ≤10 words using an LLM.
    Skips silently if no OPENAI_API_KEY is set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found, skipping shortify")
        return state

    if not state.get("items"):
        print("⚠️  No items to shortify")
        return state

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    shortified_count = 0
    for item in state["items"]:
        prompt = (
            "Rewrite this headline in ≤10 words, keep the core idea:\n"
            f"{item['title']}"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
                temperature=0.3,
            )
            item["title"] = resp.choices[0].message.content.strip()
            shortified_count += 1
        except Exception as exc:
            # fail gracefully – just log & continue
            print(f"⚠️  LLM error for item {shortified_count + 1}: {exc}")
            
    print(f"📊 Shortified {shortified_count}/{len(state.get('items', []))} items")
    return state


def node_render(state: DigestState) -> DigestState:
    items = state.get("items", [])
    print(f"📊 Rendering {len(items)} items")
    markdown = to_markdown(items)
    Path("news.md").write_text(markdown, encoding="utf-8")
    state["markdown"] = markdown
    return state


# ──────────────────────────────────────────────────────────────
# 3. Build LangGraph
def build_graph():
    g = StateGraph(DigestState)

    g.add_node("collect", node_collect)
    g.add_node("filter",  node_filter)
    g.add_node("categorize", node_categorize)
    g.add_node("shortify",   node_shortify)
    g.add_node("render",  node_render)

    g.set_entry_point("collect")
    g.add_edge("collect", "filter")
    g.add_edge("filter",  "categorize")
    g.add_edge("categorize", "shortify")
    g.add_edge("shortify",   "render")

    return g.compile()