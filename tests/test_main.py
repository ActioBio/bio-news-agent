"""Tests for main module helpers."""

import main


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
