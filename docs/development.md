# Development

## Prerequisites

- Python 3.12+
- Node.js for the executable workflow contract tests (available on GitHub-hosted runners).
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

### Publication safety

`--dispatch-publish` freezes one `America/New_York` date and sends it as the required `digest_date` workflow input alongside the title and compressed body. The receiver passes it unchanged as `DIGEST_DATE`. Actions publishing rejects missing, malformed, noncanonical, past or future dates before contacting GitHub, and checks the date again after issue lookup before starting a write. An observed Eastern date change stops publication; a UTC date change alone does not invalidate the payload. The guard cannot undo a request already accepted by GitHub.

The API-build workflow freezes `DIGEST_DATE` before preflight and build. Title generation and issue selection use that same date. Both publishing workflows share the repository-wide `digest-publish-${{ github.repository }}` concurrency group with `cancel-in-progress: false` and `queue: max`. This serializes Actions publishers, retains up to 100 pending runs, and leaves the existing final issue lookup inside the lock. Same-day retries update the existing issue rather than create another. Queue order is not a promise of dispatch order. See [GitHub concurrency documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).

Direct local `--publish-issue` without `DIGEST_DATE` remains a manual fallback using the date when the command starts; an explicitly supplied date is still validated. It does not participate in the Actions lock and must not run concurrently with Actions publishers. Standalone `--check-issue` still checks today's issue. The date binds dispatch/build to publication, not the age of an arbitrary local `news.md`; regenerate stale local output instead of redispatching it as new news.

For rollout, merge the producer, receiver and workflow lock together in each repo, then update the local checkout before the next agent run. Older callers missing `digest_date` fail closed against the new receiver. Already-running workflows using older code are not retroactively protected; let those finish or handle them explicitly before relying on the new gate. No schedule or fallback-policy change is part of this batch.


## Decision Schema

Agent decisions should use this JSON shape:

```json
{
  "schema_version": 2,
  "kind": "bio-news-agent.decisions",
  "snapshot_id": "sha256:<copy exactly from digest-candidates.json>",
  "executive_summary": "2-3 sentence overview of today's biotech/pharma news.",
  "top_stories": ["g1i1"],
  "groups": []
}
```

The empty `groups` skeleton above is valid only for a candidate snapshot with no groups; it is invalid for every nonempty snapshot. Every candidate group must appear exactly once, and every candidate item must be dispositioned exactly once as a keep, duplicate, or off-topic item. For example, given `g1i1` as a kept singleton, `g2i1` and `g2i2` as duplicate coverage of one story, and `g3i1` as off-topic, the exhaustive decisions are:

```json
{
  "schema_version": 2,
  "kind": "bio-news-agent.decisions",
  "snapshot_id": "sha256:<copy exactly from digest-candidates.json>",
  "executive_summary": "2-3 sentence overview of today's biotech/pharma news.",
  "top_stories": ["g1i1"],
  "groups": [
    {
      "group_id": "g1",
      "off_topic_ids": [],
      "clusters": [
        {
          "keep_id": "g1i1",
          "duplicate_ids": [],
          "category": "Clinical & Research",
          "short_title": "Pfizer posts oncology trial results",
          "summary_line": "Why this matters in one sentence.",
          "tier": "high"
        }
      ]
    },
    {
      "group_id": "g2",
      "off_topic_ids": [],
      "clusters": [
        {
          "keep_id": "g2i1",
          "duplicate_ids": ["g2i2"],
          "category": "Regulatory & FDA",
          "short_title": "FDA updates gene therapy guidance",
          "summary_line": "Why this matters in one sentence.",
          "tier": "normal"
        }
      ]
    },
    {
      "group_id": "g3",
      "off_topic_ids": ["g3i1"],
      "clusters": []
    }
  ]
}
```

Every cluster must contain a list-valued `duplicate_ids`; use `[]` for a kept singleton. A standalone `discovery_only` item is valid decision input and must still be represented as an explicit singleton keep, but it is removed later during rendering. Decisions are fully validated, including snapshot binding and exhaustive dispositions, before any keep is promoted. Failed validation invalidates and removes any prior generated `news.md`, then stops before rendering or dispatch.

Use a canonical category from the candidate snapshot and a `tier` of `high` or `normal`. Optional `short_title` values that are missing, null, non-string, or blank retain the original source title. Valid strings have whitespace normalized and are limited to 8 words. This applies to agent decisions and API enrichment.

The resolved `coverage_sources` list contains additional distinct sources, excluding the kept source. Source labels have leading, trailing, and repeated whitespace normalized and are compared case-insensitively; empty labels are ignored and the first normalized display spelling is preserved. Coverage counts used in rendering, ranking, and API enrichment represent distinct source/newsroom labels, not the number of URLs. Publisher aliases are not merged.

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
