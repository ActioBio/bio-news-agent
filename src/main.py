"""Command-line entry point for generating the daily digest."""

import argparse
import logging
import sys
from pathlib import Path

try:
    from config import LOG_LEVEL
    from graph import apply_decisions_file, build_graph, export_candidate_snapshot
except ModuleNotFoundError:  # pragma: no cover - module execution fallback
    from .config import LOG_LEVEL
    from .graph import apply_decisions_file, build_graph, export_candidate_snapshot


_DEFAULT_CANDIDATES_FILE = Path("digest-candidates.json")


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
    parser.add_argument(
        "--candidates-file",
        type=Path,
        default=_DEFAULT_CANDIDATES_FILE,
        help="Path used for exported candidates or imported candidate snapshots.",
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


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()
    logger.info("Starting Bio News Agent")
    if args.candidates_only:
        snapshot_payload = export_candidate_snapshot(args.candidates_file)
        logger.info(
            "Wrote %d candidate groups to %s",
            len(snapshot_payload.get("groups", [])),
            args.candidates_file,
        )
    elif args.apply_decisions:
        apply_decisions_file(args.apply_decisions, args.candidates_file)
        logger.info("news.md generated successfully from agent decisions")
    else:
        graph = build_graph()
        graph.invoke({})
        logger.info("news.md generated successfully")
