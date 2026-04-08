# bio-news-agent

A lightweight AI agent that grabs fresh biotech/pharma headlines and posts a daily digest to GitHub Issues.

🔔 **Watch this repository** to receive the daily biotech news digest email delivered straight to your inbox.

## Architecture

```mermaid
flowchart LR
    subgraph Trigger[Triggers]
        GH[GitHub Actions<br/>schedule or manual dispatch]
        LOCAL[Local CLI run<br/>uv run python src/main.py]
    end

    subgraph App[Application]
        G[LangGraph]
        C[node_collect]
        F[node_filter]
        K[node_categorize]
        R[node_render]
        G --> C --> F --> K --> R
    end

    subgraph In[Inputs]
        FEEDS[feeds.json]
        RSS[RSS feed endpoints]
        CONF[.env + config.py]
        OAI[OpenAI API]
    end

    subgraph Out[Outputs]
        MD[news.md]
        ISSUE[GitHub Issue<br/>daily biotech digest]
    end

    GH --> G
    LOCAL --> G
    FEEDS --> C
    RSS --> C
    CONF --> C
    CONF --> K
    OAI --> K
    R --> MD --> ISSUE

    classDef io fill:#eef7ff,stroke:#1f6feb,stroke-width:1px,color:#0b1f3a;
    classDef proc fill:#f7f7f7,stroke:#555,stroke-width:1px,color:#111;
    class FEEDS,RSS,CONF,OAI,MD,ISSUE io;
    class GH,LOCAL,G,C,F,K,R proc;
```

```mermaid
flowchart LR
    C[Collect] --> F[Filter]
    F --> G[Build candidate groups]
    G --> K1{API key for LLM clustering?}
    K1 -- Yes --> L[Structured LLM dedupe + categorize]
    K1 -- No --> R[Local duplicate resolution + fallback categorization]
    L --> W[Render + Write news.md]
    R --> W
```

## Prerequisites

- Python 3.12+ with pip

## Quick Start

### 1. Install UV

```bash
pip install uv
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
# Placeholder values such as sk-... or your_api_key_here are treated as missing
```

### 3. Run
```bash
cd bio-news-agent
uv run python src/main.py
```

## Feed configuration

The collector reads RSS feed URLs from [`feeds.json`](feeds.json) in the project root. The
file should contain a JSON object where each key is a feed URL and each value
specifies the `category` and human-readable `source` name.

An optional `type` field can be used for source-specific handling such as paper limits:

```json
{
  "https://example.com/feed.xml": {
    "source": "Example Feed",
    "category": "All",
    "type": "news"
  }
}
```

Pipeline notes:

- Exact duplicates are removed by normalized URL before any LLM call.
- The collector preserves `original_title` and RSS `summary` for duplicate resolution.
- Obvious noise titles such as webinars, sponsored posts, and opinion items are dropped before grouping.
- The source cap is still applied before LLM dedupe for diversity and lower cost.
- Placeholder OpenAI API keys from either the shell environment or `.env` are ignored for local runs.
- The LLM receives candidate groups and returns structured duplicate clusters instead of line-based `SKIP` output.
- Short display titles are generated only for kept items after duplicates are resolved.
