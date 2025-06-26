"""
LangGraph pipeline for bio-news-agent

Flow:
    collect ─▶ filter ─▶ shortify ─▶ categorize ─▶ render
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


def node_shortify(state: DigestState) -> DigestState:
    """
    Shorten titles to ≤10 words for cleaner digest and better duplicate detection
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found, skipping shortify")
        return state

    if not state.get("items"):
        print("⚠️  No items to shortify")
        return state

    client = OpenAI(api_key=api_key)

    shortified_count = 0
    for item in state["items"]:
        prompt = (
            "Rewrite this headline in ≤10 words, keep the core idea:\n"
            f"{item['title']}"
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            item["title"] = resp.choices[0].message.content.strip()
            shortified_count += 1
        except Exception as exc:
            print(f"⚠️  LLM error for item {shortified_count + 1}: {exc}")
            
    print(f"📊 Shortified {shortified_count}/{len(state.get('items', []))} items")
    return state


def node_categorize(state: DigestState) -> DigestState:
    """
    Use LLM to categorize items and identify duplicates.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found, using default categorization")
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
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Create item list with shortened titles
        items_text = "\n".join([
            f"{i+1}. {item['title']} — {item['source']}"
            for i, item in enumerate(state["items"])
        ])
        
        prompt = f"""Analyze these biotech/pharma headlines for duplicates and categorization.

CRITICAL DUPLICATE DETECTION:
Look for stories about the SAME EVENT even if worded differently:
- "ACIP reviews vaccines" = "CDC panel reviews vaccines" = "Vaccine committee meeting" 
- "Trump nominee hearing" = "CDC nominee Senate" = "Nominee grilled" = "Pick faces questions"
- "Cancer drugs fail" = "Generic drugs quality" = "Chemo drugs fail tests"
- "X partners Y" = "Y deal with X" = "X and Y announce"
- "Kymera Gilead deal" = "Kymera partners Gilead" = "Gilead Kymera partnership"

Mark ALL BUT ONE as SKIP for each duplicate group. Keep the most informative version.

CATEGORIES:
- Regulatory & FDA: FDA/CDC decisions, drug approvals, vaccine policies
- Clinical & Research: Trials, research findings, scientific papers
- Deals & Finance: M&A, partnerships, funding, IPOs
- Company News: Executive changes, layoffs, company updates
- Policy & Politics: Government policy, political appointments, hearings
- Market Insights: Market analysis, industry trends, forecasts

Items:
{items_text}

For each item, respond with ONLY:
- The category name (if keeping)
- SKIP (if duplicate)

One per line. Be VERY aggressive marking duplicates."""

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        
        # Parse categorization response
        response_text = resp.choices[0].message.content.strip()
        print(f"📊 LLM categorization response received")
        
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        # Valid categories
        valid_categories = [
            "Regulatory & FDA",
            "Clinical & Research", 
            "Deals & Finance",
            "Company News",
            "Policy & Politics",
            "Market Insights"
        ]
        
        # First pass: LLM categorization
        for i, line in enumerate(lines):
            if i >= len(state["items"]):
                break
            
            # Check for SKIP
            if "SKIP" in line.upper():
                state["items"][i]["skip"] = True
                state["items"][i]["skip_reason"] = "duplicate"
                continue
            
            # Check for exact category match
            category_found = False
            for valid_cat in valid_categories:
                if valid_cat.lower() in line.lower():
                    state["items"][i]["category"] = valid_cat
                    category_found = True
                    break
            
            # If no match found, use default based on keywords
            if not category_found:
                title = state["items"][i]["title"].lower()
                
                # More specific keyword matching
                if any(word in title for word in ["fda", "approve", "reject", "cdc advise", "regulatory"]):
                    state["items"][i]["category"] = "Regulatory & FDA"
                elif any(word in title for word in ["trial", "phase", "study", "research", "efficacy", "therapy"]):
                    state["items"][i]["category"] = "Clinical & Research"
                elif any(word in title for word in ["partner", "deal", "raise", "funding", "$", "acquisition"]):
                    state["items"][i]["category"] = "Deals & Finance"
                elif any(word in title for word in ["layoff", "cuts", "ceo", "executive", "hire"]):
                    state["items"][i]["category"] = "Company News"
                elif any(word in title for word in ["trump", "congress", "medicare", "medicaid", "policy", "politics"]):
                    state["items"][i]["category"] = "Policy & Politics"
                elif any(word in title for word in ["market", "spending", "forecast", "trend", "billion", "long game"]):
                    state["items"][i]["category"] = "Market Insights"
                else:
                    state["items"][i]["category"] = "Company News"
        
        # Second pass: Additional duplicate detection based on key terms
        seen_topics = {}
        for i, item in enumerate(state["items"]):
            if item.get("skip"):
                continue
                
            title_lower = item["title"].lower()
            
            # Define topic signatures
            topic_key = None
            if all(word in title_lower for word in ["cdc", "vaccine"]) or all(word in title_lower for word in ["acip", "vaccine"]):
                topic_key = "cdc_vaccine_panel"
            elif "kymera" in title_lower and "gilead" in title_lower:
                topic_key = "kymera_gilead_deal"
            elif "trump" in title_lower and ("nominee" in title_lower or "cdc" in title_lower):
                topic_key = "trump_cdc_nominee"
            elif ("cancer" in title_lower or "chemo" in title_lower) and "drug" in title_lower:
                topic_key = "cancer_drug_quality"
                
            if topic_key:
                if topic_key in seen_topics:
                    # Mark as duplicate
                    item["skip"] = True
                    item["skip_reason"] = f"duplicate of item {seen_topics[topic_key]}"
                    print(f"📊 Additional duplicate found: {item['title']}")
                else:
                    seen_topics[topic_key] = i
                    
        # Remove skipped items
        original_count = len(state["items"])
        state["items"] = [item for item in state["items"] if not item.get("skip")]
        
        print(f"📊 After categorization: {len(state['items'])} items (skipped {original_count - len(state['items'])} duplicates)")
        
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