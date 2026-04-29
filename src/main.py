"""Command-line entry point for generating the daily digest."""

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    from config import DIGEST_STATUS_FILE, LOG_LEVEL
    from graph import apply_decisions_file, build_graph, export_candidate_snapshot
    from publisher import check_issue_status, dispatch_publish_workflow, publish_issue
except ModuleNotFoundError:  # pragma: no cover - module execution fallback
    from .config import DIGEST_STATUS_FILE, LOG_LEVEL
    from .graph import apply_decisions_file, build_graph, export_candidate_snapshot
    from .publisher import check_issue_status, dispatch_publish_workflow, publish_issue


_DEFAULT_CANDIDATES_FILE = Path("digest-candidates.json")
_DEFAULT_STATUS_FILE = Path(DIGEST_STATUS_FILE)
_DEFAULT_ISSUE_STATUS_FILE = Path("digest-issue-status.json")
_TRANSIENT_ISSUE_ERROR_MARKERS = (
    "error connecting to api.github.com",
    "githubstatus.com",
    "timed out",
    "temporary failure",
    "temporary unavailable",
    "connection reset",
    "connection refused",
    "name or service not known",
    "nodename nor servname provided",
    "could not resolve host",
    "api rate limit exceeded",
    "secondary rate limit",
    "rate limit",
    "502",
    "503",
    "504",
)
_AUTH_ISSUE_ERROR_MARKERS = (
    "bad credentials",
    "requires authentication",
    "resource not accessible by integration",
    "401",
)
_CONFIG_ISSUE_ERROR_MARKERS = (
    "missing digest_github_token, github_token, or gh_token",
    "missing github_token or gh_token",
    "gh cli is unavailable",
    "invalid repository name",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--candidates-only",
        action="store_true",
        help="Write grouped digest candidates to JSON and stop before LLM categorization.",
    )
    mode_group.add_argument(
        "--apply-decisions",
        type=Path,
        metavar="FILE",
        help="Apply agent-produced decisions JSON and render the digest.",
    )
    mode_group.add_argument(
        "--publish-issue",
        action="store_true",
        help="Create or update today's GitHub issue from the rendered digest.",
    )
    mode_group.add_argument(
        "--dispatch-publish",
        action="store_true",
        help="Trigger the publish-only GitHub Actions workflow for today's rendered digest.",
    )
    mode_group.add_argument(
        "--check-issue",
        action="store_true",
        help="Check whether today's digest issue already exists.",
    )
    parser.add_argument(
        "--candidates-file",
        type=Path,
        default=_DEFAULT_CANDIDATES_FILE,
        help="Path used for exported candidates or imported candidate snapshots.",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=_DEFAULT_STATUS_FILE,
        help="Path used for candidate export run status JSON.",
    )
    parser.add_argument(
        "--issue-status-file",
        type=Path,
        default=_DEFAULT_ISSUE_STATUS_FILE,
        help="Path used for issue preflight status JSON.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _classify_issue_error(reason: str) -> tuple[str, bool]:
    lowered = reason.lower()
    if any(marker in lowered for marker in _CONFIG_ISSUE_ERROR_MARKERS):
        return "config", False
    if any(marker in lowered for marker in _TRANSIENT_ISSUE_ERROR_MARKERS):
        return "transient", True
    if any(marker in lowered for marker in _AUTH_ISSUE_ERROR_MARKERS):
        return "auth", False
    return "unknown", False


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()
    logger.info("Starting Bio News Agent")
    if args.candidates_only:
        snapshot_payload, run_status = export_candidate_snapshot(args.candidates_file)
        args.status_file.write_text(json.dumps(run_status, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "Wrote %d candidate groups to %s",
            len(snapshot_payload.get("groups", [])),
            args.candidates_file,
        )
        logger.info("Wrote candidate export status to %s", args.status_file)
        if not run_status["ok"]:
            logger.error(
                "Candidate export failed health checks: %s (groups=%d, feeds_failed=%d)",
                run_status["reason"],
                run_status["groups"],
                run_status["feeds_failed"],
            )
            if run_status.get("feed_errors"):
                sample_error = run_status["feed_errors"][0]
                logger.error(
                    "Feed failure sample: %s: %s",
                    sample_error.get("source", "unknown"),
                    sample_error.get("error", "unknown error"),
                )
            raise SystemExit(1)
    elif args.apply_decisions:
        apply_decisions_file(args.apply_decisions, args.candidates_file)
        logger.info("news.md generated successfully from agent decisions")
    elif args.publish_issue:
        try:
            result = publish_issue()
        except RuntimeError as exc:
            logger.error("Issue publish failed: %s", exc)
            raise SystemExit(1) from exc
        logger.info(
            "%s digest issue #%d (%s)",
            result["action"].capitalize(),
            result["issue_number"],
            result["title"],
        )
    elif args.dispatch_publish:
        try:
            result = dispatch_publish_workflow()
        except RuntimeError as exc:
            logger.error("Publish workflow dispatch failed: %s", exc)
            raise SystemExit(1) from exc
        logger.info(
            "Dispatched publish workflow %s on %s for %s",
            result["workflow"],
            result["ref"],
            result["title"],
        )
    elif args.check_issue:
        try:
            issue_status = check_issue_status()
        except RuntimeError as exc:
            error_kind, retryable = _classify_issue_error(str(exc))
            issue_status = {
                "ok": False,
                "exists": False,
                "issue_number": None,
                "title": "",
                "reason": str(exc),
                "error_kind": error_kind,
                "retryable": retryable,
            }
            args.issue_status_file.write_text(
                json.dumps(issue_status, indent=2) + "\n",
                encoding="utf-8",
            )
            logger.error("Issue preflight failed: %s", exc)
            raise SystemExit(1) from exc
        issue_status = {
            "ok": True,
            "reason": "ok",
            "error_kind": "none",
            "retryable": False,
            **issue_status,
        }
        args.issue_status_file.write_text(
            json.dumps(issue_status, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Wrote issue preflight status to %s (exists=%s)",
            args.issue_status_file,
            issue_status["exists"],
        )
    else:
        graph = build_graph()
        graph.invoke({})
        logger.info("news.md generated successfully")
