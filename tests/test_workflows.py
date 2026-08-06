"""Regression tests for GitHub Actions workflow triggers."""

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_digest_workflow_is_not_scheduled() -> None:
    workflow = (_REPO_ROOT / ".github/workflows/digest.yml").read_text(
        encoding="utf-8"
    )
    trigger_block = workflow.split("on:\n", maxsplit=1)[1].split(
        "\njobs:\n", maxsplit=1
    )[0]

    assert "  schedule:" not in trigger_block
    assert "  workflow_dispatch:" in trigger_block
