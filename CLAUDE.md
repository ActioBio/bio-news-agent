# Claude Notes

Use [AGENTS.md](./AGENTS.md) as the canonical repo guide.

## Short Version
1. Run `--check-issue` first.
2. Run `--candidates-only` and inspect `digest-run-status.json`.
3. Write `digest-decisions.json`.
4. Run `--apply-decisions`.
5. Run `--dispatch-publish`.
6. Re-run `--check-issue` to confirm the issue exists.

## Important Defaults
- Prefer the local agent path over the repo's default `uv run python src/main.py`.
- Keep `--publish-issue` as manual fallback only.
- Do not commit generated digest artifacts unless explicitly asked.
