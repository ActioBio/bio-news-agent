# Architecture

## Pipeline

```mermaid
flowchart LR
    subgraph Trigger[Triggers]
        GH[GitHub Actions<br/>manual dispatch]
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
        OAI[OpenAI API<br/>manual graph path]
        MODEL[Codex / Claude model<br/>agent mode]
    end

    subgraph Out[Outputs]
        JSON[digest-candidates.json<br/>digest-decisions.json]
        MD[news.md]
        ISSUE[GitHub Issue<br/>label: ai-digest]
    end

    GH --> C
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
    MD --> DATE{Publication date valid?}
    DATE -- Yes --> ISSUE
    DATE -- No --> STOP[Stop without publishing]

    classDef io fill:#eef7ff,stroke:#1f6feb,stroke-width:1px,color:#0b1f3a;
    classDef proc fill:#f7f7f7,stroke:#555,stroke-width:1px,color:#111;
    class FEEDS,RSS,CONF,OAI,MODEL,JSON,MD,ISSUE io;
    class GH,AGENT,LOCAL,CHECK,C,F,G,K,R,DATE,STOP proc;
```

```mermaid
flowchart LR
    GH[GitHub Actions<br/>manual digest] --> C[Collect + filter + build candidate groups]
    AGENT[Codex / Claude] --> T
    T -- Yes --> S[Stop]
    T -- No --> C
    C --> P{Path}
    P -- GitHub manual --> K1{OPENAI_API_KEY available?}
    K1 -- Yes --> L[OpenAI dedupe + categorize]
    K1 -- No --> R[Local duplicate resolution + fallback categorization]
    P -- Codex / Claude --> X[Write digest-candidates.json]
    X --> Y[Agent writes exhaustive decisions v2]
    Y --> V{Snapshot binding and dispositions valid?}
    V -- Yes --> Z[Apply decisions]
    V -- No --> F[Stop: fail closed, nothing published]
    L -- Responses valid --> W[Render + write news.md]
    L -- API error or invalid response --> R
    R --> W
    Z --> W
```

## Pipeline Notes

- Exact duplicates are removed by normalized URL before any LLM call.
- The collector preserves `original_title` and RSS `summary` for duplicate resolution.
- Regular comparison grouping also uses article-URL basename hints: at least two distinct alphabetic tokens of six or more characters must match the other item's original headline or summary, excluding configured company names and generic news/date words. Hostnames, parent directories, queries, and fragments do not supply hints. These hints only create editorial comparison opportunities; groups can contain distinct events and every merge still requires an explicit decision. Local fallback grouping does not use URL hints.
- Local fallback merges only nonempty original headlines equal after case folding and whitespace normalization within an existing candidate group. A usable `title` is used only when the original is absent or unusable. Candidate grouping remains fuzzy, including the broader fallback grouping; numbers, punctuation, word order, negation, and differing originals prevent local merging. More near-duplicates may remain, and identical generic headlines do not prove semantic identity; see the [local fallback contract](development.md#api-response-contract).
- Source-specific low-signal items such as webinars, sponsored posts, opinion pieces, people-move roundups, and bundled roundup headlines are dropped before grouping.
- Mixed regulator and institutional feeds can be gated by source-specific title or link rules before grouping.
- The source cap is applied before LLM dedupe for diversity and lower cost.
- Candidate export writes `digest-run-status.json` with feed health, group counts, and sample `feed_errors` for automation use.
- Candidate schema v5 binds decisions schema v2 through the exact exported `snapshot_id`. Decisions must include every candidate group and disposition every item exactly once. Validation completes before keep promotion; only afterward does rendering remove standalone `discovery_only` keeps.
- `--check-issue` writes `digest-issue-status.json`, preferring authenticated `gh` locally and `DIGEST_GITHUB_TOKEN`, `GITHUB_TOKEN`, or `GH_TOKEN` in GitHub Actions.
- On GitHub failure, the issue-status artifact includes `ok: false`, a `reason`, an `error_kind`, and a `retryable` flag.
- `--candidates-only` exits nonzero only when feed health is bad enough to make the snapshot unreliable. Empty days are reported as `reason: "no_fresh_items"` without failing, even when optional feeds have warnings.
- Lower-priority `Company News` items are capped after ranking to keep fallback digests compact.
- `discovery_only` feeds can still merge into a core story and contribute coverage context, but standalone discovery-only items are dropped before final render.
- Placeholder OpenAI API keys from either the shell environment or `.env` are ignored for local runs.
- LLM request timeouts retry before falling back to local duplicate resolution. Malformed responses and invalid dispositions also enter that existing fallback. Each complete API response is validated before applying its contents or promoting a keep; see the [API response contract and local fallback limits](development.md#api-response-contract).
- The LLM receives candidate groups and returns structured duplicate clusters.
- `--dispatch-publish` triggers `.github/workflows/publish-digest.yml` with a compressed digest payload; that workflow runs the repo-local `--publish-issue` path on GitHub Actions.
- `.github/workflows/digest.yml` is manual-only. Manual dispatch skips its schedule-only preflight, runs the API graph, and rechecks the issue idempotently only when publishing.
- Direct `--publish-issue` remains a manual fallback.
- Short display titles are generated only for kept items after duplicates are resolved.
- Publication uses one frozen Eastern date for dispatch inputs, title generation and issue selection. Actions requires `DIGEST_DATE`; missing, malformed or noncurrent dates fail closed, with a second date check after issue lookup before starting a write.
- `digest.yml` and `publish-digest.yml` share one repository-wide Actions concurrency group, retain pending runs with `queue: max`, and do not cancel the active run. Direct manual publication bypasses that lock. See [publication safety and rollout limits](development.md#publication-safety).
