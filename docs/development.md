# Development

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)

## Quick Start

```bash
UV_CACHE_DIR=.uv-cache uv sync --locked
cp .env.example .env
UV_CACHE_DIR=.uv-cache uv run python src/main.py
```

Add `OPENAI_API_KEY` to `.env` if you want the default graph path to use OpenAI directly. Placeholder values such as `sk-...` or `your_api_key_here` are treated as missing.

The default API model is `gpt-5.6-luna`. Set `OPENAI_MODEL` to override it.

## Agent-Driven Mode

This path keeps feed collection and filtering in Python, then lets Codex or Claude Code write editorial decisions without needing `OPENAI_API_KEY`.

**The canonical operational runbook is [AGENTS.md](../AGENTS.md).**

```bash
UV_CACHE_DIR=.uv-cache uv run python src/main.py --check-issue --issue-status-file digest-issue-status.json
UV_CACHE_DIR=.uv-cache uv run python src/main.py --candidates-only
# agent reads digest-candidates.json and writes digest-decisions.json
UV_CACHE_DIR=.uv-cache uv run python src/main.py --apply-decisions digest-decisions.json
UV_CACHE_DIR=.uv-cache uv run python src/main.py --dispatch-publish
```

`--check-issue` writes `digest-issue-status.json` by default. `--candidates-only` writes `digest-candidates.json` and `digest-run-status.json` by default. Use `--candidates-file <path>`, `--status-file <path>`, and `--issue-status-file <path>` to override these artifacts.

If `digest-run-status.json` reports `reason: "no_fresh_items"`, stop without writing decisions or dispatching publish. Partial feed failures remain visible in `feed_errors`, but only an all-feed failure makes candidate export fail.

Local runs prefer authenticated `gh` for `--dispatch-publish`; GitHub Actions and CI-style environments prefer `DIGEST_GITHUB_TOKEN`, `GITHUB_TOKEN`, or `GH_TOKEN` with workflow-dispatch access. Direct `--publish-issue` is still available as a manual fallback.

## Decision Schema

Agent decisions should use this JSON shape:

```json
{
  "executive_summary": "2-3 sentence overview of today's biotech/pharma news.",
  "top_stories": ["g1i1"],
  "groups": [
    {
      "group_id": "g1",
      "off_topic_ids": ["g1i3"],
      "clusters": [
        {
          "keep_id": "g1i1",
          "duplicate_ids": ["g1i2"],
          "category": "Clinical & Research",
          "short_title": "Pfizer posts oncology trial results",
          "summary_line": "Why this matters in one sentence.",
          "tier": "high"
        }
      ]
    }
  ]
}
```

`keep_id` always refers to one item id from a candidate group. Use `off_topic_ids` for items that should not appear in the digest. `summary_line` and `executive_summary` are kept as decision metadata and are not rendered in the issue body. The published issue title appends the leading top story, e.g. `Biotech / Pharma Headlines - Jun 12: MHRA approves oral GLP-1 for weight loss`, while same-day deduplication matches on the `ai-digest` label and creation date rather than the title.

By default, `Company News` is capped to the top 3 ranked items to keep the daily digest quick to scan.

## Feed Configuration

The collector reads RSS feed URLs from [`feeds.json`](../feeds.json). The file contains a JSON object where each key is a feed URL and each value specifies the `category` and human-readable `source` name.

RSS fetches use `RSS_USER_AGENT` first. If a feed returns HTTP 403, the collector retries
that request with `RSS_FALLBACK_USER_AGENT` because some feed CDNs reject non-browser
user agents.

Optional fields:

- `type`: source-specific handling such as paper limits
- `source_role`: source authority for duplicate tie-breaks and ranking. Supported values: `primary`, `independent_reporting`, `commentary`, `community`.
- `feed_mode`: whether a feed is part of the main digest or supporting discovery only. Supported values: `core`, `discovery_only`.

```json
{
  "https://example.com/feed.xml": {
    "source": "Example Feed",
    "category": "All",
    "type": "news",
    "source_role": "independent_reporting",
    "feed_mode": "core"
  }
}
```

## CI

Push and pull request CI runs `pytest` and `mypy`. Scheduled agent runs generate locally and dispatch the final publish through GitHub Actions so the final issue author is `app/github-actions`. Publisher issue matching and generated title dates use `America/New_York`, so a delayed run does not shift the digest to the wrong calendar day.
