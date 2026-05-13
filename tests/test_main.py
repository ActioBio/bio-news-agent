"""Tests for main module helpers."""

import importlib
import sys

import main


def test_main_import_does_not_import_graph():
    previous_main = sys.modules.pop("main", None)
    previous_graph = sys.modules.pop("graph", None)

    try:
        importlib.import_module("main")

        assert "graph" not in sys.modules
    finally:
        if previous_main is not None:
            sys.modules["main"] = previous_main
        if previous_graph is not None:
            sys.modules["graph"] = previous_graph


def test_classify_issue_error_marks_transient_network_failure():
    kind, retryable = main._classify_issue_error(
        "GitHub gh api GET /repos/ActioBio/bio-news-agent/issues failed: error connecting to api.github.com"
    )

    assert kind == "transient"
    assert retryable is True


def test_classify_issue_error_marks_missing_auth_as_config():
    kind, retryable = main._classify_issue_error(
        "Missing DIGEST_GITHUB_TOKEN, GITHUB_TOKEN, or GH_TOKEN for issue publishing"
    )

    assert kind == "config"
    assert retryable is False


def test_classify_issue_error_marks_bad_credentials_as_auth():
    kind, retryable = main._classify_issue_error(
        "GitHub API GET /repos/ActioBio/bio-news-agent/issues failed: 401 bad credentials"
    )

    assert kind == "auth"
    assert retryable is False


def test_classify_issue_error_marks_rate_limit_as_transient():
    kind, retryable = main._classify_issue_error(
        "GitHub gh api GET /repos/ActioBio/bio-news-agent/issues failed: 403 API rate limit exceeded"
    )

    assert kind == "transient"
    assert retryable is True
