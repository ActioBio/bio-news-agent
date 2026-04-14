# Bio News Agent

## Purpose
- Build the daily biotech/pharma digest locally with an agent.
- Publish the final issue through GitHub Actions so the issue author is `app/github-actions`.

## Preferred Agent Flow
1. `UV_CACHE_DIR=.uv-cache uv sync --locked`
2. `uv run python src/main.py --check-issue --issue-status-file digest-issue-status.json`
3. Stop if `digest-issue-status.json` says `ok: true` and `exists: true`.
4. Stop if it says `ok: false` and `retryable: false`.
5. If it says `ok: false` and `retryable: true`, continue.
6. `RSS_MAX_WORKERS=2 RSS_TIMEOUT=15 uv run python src/main.py --candidates-only --status-file digest-run-status.json`
7. If `digest-run-status.json` says `feed_fetch_failed` or `empty_snapshot_with_feed_errors`, retry once with `RSS_MAX_WORKERS=1 RSS_TIMEOUT=20`.
8. If the retry still fails, stop and report the failure reason.
9. If status is healthy, write `digest-decisions.json` using the repo decision schema.
10. `uv run python src/main.py --apply-decisions digest-decisions.json`
11. `uv run python src/main.py --dispatch-publish`
12. Wait briefly, then re-run `--check-issue` to confirm the issue exists.

## Defaults
- Prefer the agent path above for daily runs.
- Do not use `uv run python src/main.py` for the daily automation path unless explicitly asked. That is the repo's default graph path and may use the repo LLM API path.
- `--publish-issue` is a manual fallback only.
- `--dispatch-publish` is the normal final publish path.
- `--dispatch-publish` needs workflow-dispatch-capable GitHub auth through `GITHUB_TOKEN` / `GH_TOKEN` or authenticated local `gh`.

## Status Artifacts
- `digest-run-status.json`: feed health and candidate export status.
- `digest-issue-status.json`: issue existence or GitHub preflight status.
- `digest-candidates.json`: grouped candidate snapshot.
- `digest-decisions.json`: agent editorial decisions.
- `news.md`: rendered digest body.

## Decision Shape
- For the full input contract, inspect the current `digest-candidates.json` and the README decision-schema section. `keep_id` always refers to one item id from a candidate group.
- `clusters`: duplicate groups with `keep_id` and `duplicate_ids`
- `top_stories`
- `executive_summary`
- `off_topic_ids`

## Generated Files
- Treat `digest-candidates.json`, `digest-decisions.json`, `digest-run-status.json`, `digest-issue-status.json`, and `news.md` as generated local artifacts.
- Do not commit them unless explicitly asked.

## Troubleshooting
- If candidate export fails, inspect `digest-run-status.json` first.
- Check `digest-run-status.json.feed_errors` first for sample feed failures before looking elsewhere.
