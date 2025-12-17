"""Command-line entry point for generating the daily digest."""

import logging
import sys

from graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


if __name__ == "__main__":
    graph = build_graph()
    graph.invoke({})
    logging.info("news.md generated.")
