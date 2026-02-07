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
        S[node_shortify]
        K[node_categorize]
        R[node_render]
        G --> C --> F --> S --> K --> R
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
    CONF --> S
    CONF --> K
    OAI --> S
    OAI --> K
    R --> MD --> ISSUE

    classDef io fill:#eef7ff,stroke:#1f6feb,stroke-width:1px,color:#0b1f3a;
    classDef proc fill:#f7f7f7,stroke:#555,stroke-width:1px,color:#111;
    class FEEDS,RSS,CONF,OAI,MD,ISSUE io;
    class GH,LOCAL,G,C,F,S,K,R proc;
```

```mermaid
flowchart LR
    C[Collect] --> F[Filter]
    F --> K1{API key for shortify?}
    K1 -- Yes --> S[Shortify]
    K1 -- No --> N1[Skip shortify]
    S --> K2{API key for categorize?}
    N1 --> K2
    K2 -- Yes --> L[LLM Categorize]
    K2 -- No --> R[Rule Categorize]
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
```

### 3. Run
```bash
cd bio-news-agent
uv run python src/main.py
```

## Feed configuration

The collector reads RSS feed URLs from [`feeds.json`](feeds.json) in the project root. The
file should contain a JSON object where each key is a feed URL and each value
specifies the `category` and human‑readable `source` name.
