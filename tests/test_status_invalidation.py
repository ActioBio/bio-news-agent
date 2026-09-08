"""Handler-entry invalidation of generated status and candidate artifacts."""

import argparse
import builtins
import json
import logging
from datetime import datetime, timezone

import graph
import main
import pytest


@pytest.fixture
def artifact_args(tmp_path):
    args = argparse.Namespace(
        issue_status_file=tmp_path / "custom-issue-status.json",
        status_file=tmp_path / "custom-run-status.json",
        candidates_file=tmp_path / "custom-candidates.json",
    )
    for output in vars(args).values():
        output.write_text('{"stale": true}', encoding="utf-8")
    return args


@pytest.mark.parametrize("error", [ValueError("checker failed"), KeyboardInterrupt()])
def test_check_issue_removes_stale_status_before_unexpected_failure(
    artifact_args, monkeypatch, error
):
    def fail_check():
        raise error

    monkeypatch.setattr(main, "check_issue_status", fail_check)

    with pytest.raises(type(error)):
        main._run_check_issue(artifact_args, logging.getLogger(__name__))

    assert not artifact_args.issue_status_file.exists()
    assert artifact_args.status_file.read_text(encoding="utf-8") == '{"stale": true}'
    assert artifact_args.candidates_file.read_text(encoding="utf-8") == '{"stale": true}'


@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize("stale", [False, True])
def test_check_issue_writes_fresh_success(artifact_args, monkeypatch, exists, stale):
    if not stale:
        artifact_args.issue_status_file.unlink()

    def check():
        assert not artifact_args.issue_status_file.exists()
        return {"exists": exists, "issue_number": 42 if exists else None, "title": "Today"}

    monkeypatch.setattr(main, "check_issue_status", check)

    main._run_check_issue(artifact_args, logging.getLogger(__name__))

    assert json.loads(artifact_args.issue_status_file.read_text(encoding="utf-8")) == {
        "ok": True,
        "reason": "ok",
        "error_kind": "none",
        "retryable": False,
        "exists": exists,
        "issue_number": 42 if exists else None,
        "title": "Today",
    }


@pytest.mark.parametrize(("reason", "kind", "retryable"), [
    ("GitHub returned 503", "transient", True),
    ("gh cli is unavailable", "config", False),
    ("401 bad credentials", "auth", False),
])
def test_check_issue_writes_fresh_expected_error(
    artifact_args, monkeypatch, reason, kind, retryable
):
    def fail_check():
        assert not artifact_args.issue_status_file.exists()
        raise RuntimeError(reason)

    monkeypatch.setattr(main, "check_issue_status", fail_check)

    with pytest.raises(SystemExit) as error:
        main._run_check_issue(artifact_args, logging.getLogger(__name__))

    assert error.value.code == 1
    assert json.loads(artifact_args.issue_status_file.read_text(encoding="utf-8")) == {
        "ok": False,
        "reason": reason,
        "error_kind": kind,
        "retryable": retryable,
        "exists": False,
        "issue_number": None,
        "title": "",
    }


def test_check_issue_unlink_failure_stops_checker(artifact_args, monkeypatch):
    artifact_args.issue_status_file.unlink()
    artifact_args.issue_status_file.mkdir()

    def unexpected_check():
        pytest.fail("Issue checker ran despite failed artifact invalidation")

    monkeypatch.setattr(main, "check_issue_status", unexpected_check)

    with pytest.raises(OSError):
        main._run_check_issue(artifact_args, logging.getLogger(__name__))

    assert artifact_args.issue_status_file.is_dir()


def test_candidates_removes_stale_artifacts_before_lazy_import_failure(
    artifact_args, monkeypatch
):
    original_import = builtins.__import__

    def fail_graph_import(name, *args, **kwargs):
        if name == "graph":
            raise ImportError("graph import failed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_graph_import)

    with pytest.raises(ImportError, match="graph import failed"):
        main._run_candidates_only(artifact_args, logging.getLogger(__name__))

    assert not artifact_args.status_file.exists()
    assert not artifact_args.candidates_file.exists()
    assert artifact_args.issue_status_file.read_text(encoding="utf-8") == '{"stale": true}'


@pytest.mark.parametrize("error", [RuntimeError("collection failed"), KeyboardInterrupt()])
def test_candidates_removes_stale_artifacts_before_collection_failure(
    artifact_args, monkeypatch, error
):
    def fail_collection():
        raise error

    monkeypatch.setattr(graph, "collect_items_with_stats", fail_collection)

    with pytest.raises(type(error)):
        main._run_candidates_only(artifact_args, logging.getLogger(__name__))

    assert not artifact_args.status_file.exists()
    assert not artifact_args.candidates_file.exists()


@pytest.mark.parametrize("blocked_output", ["status_file", "candidates_file"])
def test_candidates_unlink_failure_stops_before_graph_import(
    artifact_args, monkeypatch, blocked_output
):
    output = getattr(artifact_args, blocked_output)
    output.unlink()
    output.mkdir()
    original_import = builtins.__import__

    def unexpected_graph_import(name, *args, **kwargs):
        if name == "graph":
            pytest.fail("Graph import ran despite failed artifact invalidation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unexpected_graph_import)

    with pytest.raises(OSError):
        main._run_candidates_only(artifact_args, logging.getLogger(__name__))

    assert output.is_dir()


@pytest.mark.parametrize(("item_count", "succeeded", "failed", "ok", "reason"), [
    (1, 2, 0, True, "ok"),
    (0, 1, 1, True, "no_fresh_items"),
    (0, 0, 2, False, "feed_fetch_failed"),
    (0, 0, 0, False, "no_feeds_configured"),
])
@pytest.mark.parametrize("stale", [False, True])
def test_candidates_writes_fresh_snapshot_and_health_status(
    artifact_args, monkeypatch, item_count, succeeded, failed, ok, reason, stale
):
    if not stale:
        artifact_args.status_file.unlink()
        artifact_args.candidates_file.unlink()
    items = [{
        "id": "fresh",
        "title": "Pfizer reports phase 3 trial results",
        "original_title": "Pfizer reports phase 3 trial results",
        "link": "https://example.com/fresh",
        "source": "Example",
        "published": datetime(2026, 9, 8, 12, tzinfo=timezone.utc),
        "category": "Clinical & Research",
        "summary": "Trial results",
        "source_type": "news",
        "source_role": "independent_reporting",
        "feed_mode": "core",
    }] if item_count else []
    feed_errors = [{"source": "Example", "error": "fetch failed"}] if failed else []

    def collect():
        assert not artifact_args.status_file.exists()
        assert not artifact_args.candidates_file.exists()
        return items, {
            "feeds_total": succeeded + failed,
            "feeds_succeeded": succeeded,
            "feeds_failed": failed,
            "items_collected": item_count,
            "feed_errors": feed_errors,
        }

    monkeypatch.setattr(graph, "collect_items_with_stats", collect)

    if ok:
        main._run_candidates_only(artifact_args, logging.getLogger(__name__))
    else:
        with pytest.raises(SystemExit) as error:
            main._run_candidates_only(artifact_args, logging.getLogger(__name__))
        assert error.value.code == 1

    assert json.loads(artifact_args.status_file.read_text(encoding="utf-8")) == {
        "ok": ok,
        "reason": reason,
        "groups": item_count,
        "items_collected": item_count,
        "items_filtered": item_count,
        "feeds_total": succeeded + failed,
        "feeds_succeeded": succeeded,
        "feeds_failed": failed,
        "feed_errors": feed_errors,
    }
    snapshot = json.loads(artifact_args.candidates_file.read_text(encoding="utf-8"))
    assert snapshot["kind"] == "bio-news-agent.candidates"
    assert len(snapshot["groups"]) == item_count
    if item_count:
        assert snapshot["groups"][0]["items"][0]["link"] == "https://example.com/fresh"
