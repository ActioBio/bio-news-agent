# Architecture

## Pipeline

```mermaid
flowchart LR
    subgraph Trigger[Triggers]
        GH[GitHub Actions<br/>schedule or manual dispatch]
        AGENT[Codex / Claude Code<br/>automation]
        LOCAL[Local CLI run<br/>UV_CACHE_DIR=.uv-cache uv run python src/main.py]
    end

    subgraph Guard[Issue Guard]
        CHECK[Check today's GitHub issue]
    end

    subgraph App[Application]
        C[Collect]
        F[Filter]
        G[Group candidates]
        K[Categorize]
        R[Render]
        C --> F --> G --> K --> R
    end

    subgraph In[Inputs]
        FEEDS[feeds.json]
        RSS[RSS feed endpoints]
        CONF[.env + config.py]
        OAI[OpenAI API<br/>GitHub fallback]
        MODEL[Codex / Claude model<br/>agent mode]
    end

    subgraph Out[Outputs]
        JSON[digest-candidates.json<br/>digest-decisions.json]
        MD[news.md]
        ISSUE[GitHub Issue<br/>label: ai-digest]
    end

    GH --> CHECK --> C
    AGENT --> CHECK
    LOCAL --> C
    FEEDS --> C
    RSS --> C
    CONF --> C
    CONF --> K
    OAI --> K
    MODEL --> K
    G --> JSON
    JSON --> K
    R --> MD
    MD --> ISSUE

    classDef io fill:#eef7ff,stroke:#1f6feb,stroke-width:1px,color:#0b1f3a;
    classDef proc fill:#f7f7f7,stroke:#555,stroke-width:1px,color:#111;
    class FEEDS,RSS,CONF,OAI,MODEL,JSON,MD,ISSUE io;
    class GH,AGENT,LOCAL,CHECK,C,F,G,K,R proc;
```

```mermaid
flowchart LR
    GH[GitHub Actions] --> T{Today's issue<br/>already open?}
    AGENT[Codex / Claude] --> T
    T -- Yes --> S[Stop]
    T -- No --> C[Collect + filter + build candidate groups]
    C --> P{Path}
    P -- GitHub Actions --> K1{OPENAI_API_KEY available?}
    K1 -- Yes --> L[OpenAI dedupe + categorize]
    K1 -- No --> R[Local duplicate resolution + fallback categorization]
    P -- Codex / Claude --> X[Write digest-candidates.json]
    X --> Y[Agent writes digest-decisions.json]
    Y --> Z[Apply decisions]
    L --> W[Render + write news.md]
    R --> W
    Z --> W
```

## Pipeline Notes

- Exact duplicates are removed by normalized URL before any LLM call.
- The collector preserves `original_title` and RSS `summary` for duplicate resolution.
- Source-specific low-signal items such as webinars, sponsored posts, opinion pieces, people-move roundups, and bundled roundup headlines are dropped before grouping.
- Mixed regulator and institutional feeds can be gated by source-specific title or link rules before grouping.
- The source cap is applied before LLM dedupe for diversity and lower cost.
- Candidate export writes `digest-run-status.json` with feed health, group counts, and sample `feed_errors` for automation use.
- `--check-issue` writes `digest-issue-status.json`, preferring authenticated `gh` locally and `DIGEST_GITHUB_TOKEN`, `GITHUB_TOKEN`, or `GH_TOKEN` in GitHub Actions.
- On GitHub failure, the issue-status artifact includes `ok: false`, a `reason`, an `error_kind`, and a `retryable` flag.
- `--candidates-only` exits nonzero only when feed health is bad enough to make the snapshot unreliable. Empty days are reported as `reason: "no_fresh_items"` without failing, even when optional feeds have warnings.
- Lower-priority `Company News` items are capped after ranking to keep fallback digests compact.
- `discovery_only` feeds can still merge into a core story and contribute coverage context, but standalone discovery-only items are dropped before final render.
- Placeholder OpenAI API keys from either the shell environment or `.env` are ignored for local runs.
- LLM request timeouts retry before falling back to local duplicate resolution.
- The LLM receives candidate groups and returns structured duplicate clusters.
- `--dispatch-publish` triggers `.github/workflows/publish-digest.yml` with a compressed digest payload; that workflow runs the repo-local `--publish-issue` path on GitHub Actions.
- Direct `--publish-issue` remains a manual fallback.
- Short display titles are generated only for kept items after duplicates are resolved.
