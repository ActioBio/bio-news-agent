"""Tests for GitHub issue publishing."""

import subprocess
from pathlib import Path

import publisher


def test_publish_issue_updates_existing_issue(tmp_path, monkeypatch):
    news_file = tmp_path / "news.md"
    news_file.write_text("hello world", encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    monkeypatch.setattr(
        publisher,
        "_today_issue_title",
        lambda: "Biotech / Pharma Headlines – 2026-04-13",
    )

    calls: list[tuple[str, str, object | None]] = []

    def fake_request(method: str, path: str, payload=None):
        calls.append((method, path, payload))
        if method == "GET":
            return [{"number": 12, "title": "Biotech / Pharma Headlines – 2026-04-13"}]
        if method == "PATCH":
            return {"number": 12}
        raise AssertionError(f"unexpected call: {method} {path}")

    monkeypatch.setattr(publisher, "_github_api_request", fake_request)

    result = publisher.publish_issue(news_file)

    assert result == {
        "action": "updated",
        "issue_number": 12,
        "title": "Biotech / Pharma Headlines – 2026-04-13",
    }
    assert calls[1][0] == "PATCH"


def test_check_issue_status_finds_existing_issue(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    monkeypatch.setattr(
        publisher,
        "_today_issue_title",
        lambda: "Biotech / Pharma Headlines – 2026-04-13",
    )

    def fake_request(method: str, path: str, payload=None):
        assert method == "GET"
        return [{"number": 12, "title": "Biotech / Pharma Headlines – 2026-04-13"}]

    monkeypatch.setattr(publisher, "_github_api_request", fake_request)

    result = publisher.check_issue_status()

    assert result == {
        "exists": True,
        "issue_number": 12,
        "title": "Biotech / Pharma Headlines – 2026-04-13",
    }


def test_check_issue_status_reports_missing_issue(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    monkeypatch.setattr(
        publisher,
        "_today_issue_title",
        lambda: "Biotech / Pharma Headlines – 2026-04-13",
    )

    def fake_request(method: str, path: str, payload=None):
        assert method == "GET"
        return []

    monkeypatch.setattr(publisher, "_github_api_request", fake_request)

    result = publisher.check_issue_status()

    assert result == {
        "exists": False,
        "issue_number": None,
        "title": "Biotech / Pharma Headlines – 2026-04-13",
    }


def test_publish_issue_creates_new_issue(tmp_path, monkeypatch):
    news_file = tmp_path / "news.md"
    news_file.write_text("hello world", encoding="utf-8")
    monkeypatch.setenv("GH_TOKEN", "test-token")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    monkeypatch.setattr(
        publisher,
        "_today_issue_title",
        lambda: "Biotech / Pharma Headlines – 2026-04-13",
    )

    calls: list[tuple[str, str, object | None]] = []

    def fake_request(method: str, path: str, payload=None):
        calls.append((method, path, payload))
        if method == "GET":
            return []
        if method == "POST":
            return {"number": 34}
        raise AssertionError(f"unexpected call: {method} {path}")

    monkeypatch.setattr(publisher, "_github_api_request", fake_request)

    result = publisher.publish_issue(news_file)

    assert result == {
        "action": "created",
        "issue_number": 34,
        "title": "Biotech / Pharma Headlines – 2026-04-13",
    }
    assert calls[1][0] == "POST"


def test_publish_issue_requires_token(tmp_path, monkeypatch):
    news_file = tmp_path / "news.md"
    news_file.write_text("hello world", encoding="utf-8")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            1,
            stdout="",
            stderr="gh auth token missing",
        ),
    )

    try:
        publisher.publish_issue(news_file)
    except RuntimeError as exc:
        assert "gh auth token missing" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_check_issue_status_falls_back_to_gh_cli(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    monkeypatch.setattr(
        publisher,
        "_today_issue_title",
        lambda: "Biotech / Pharma Headlines – 2026-04-13",
    )

    def fake_run(command, input=None, text=None, capture_output=None, check=None, env=None):
        assert command[:4] == [
            "gh",
            "api",
            "/repos/ActioBio/bio-news-agent/issues?state=open&labels=ai-digest&per_page=100",
            "--method",
        ]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='[{"number": 12, "title": "Biotech / Pharma Headlines – 2026-04-13"}]',
            stderr="",
        )

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)

    result = publisher.check_issue_status()

    assert result == {
        "exists": True,
        "issue_number": 12,
        "title": "Biotech / Pharma Headlines – 2026-04-13",
    }


def test_publish_issue_uses_gh_cli_when_no_token(tmp_path, monkeypatch):
    news_file = tmp_path / "news.md"
    news_file.write_text("hello world", encoding="utf-8")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    monkeypatch.setattr(
        publisher,
        "_today_issue_title",
        lambda: "Biotech / Pharma Headlines – 2026-04-13",
    )

    calls: list[tuple[list[str], str | None]] = []

    def fake_run(command, input=None, text=None, capture_output=None, check=None, env=None):
        calls.append((command, input))
        if command[2] == "/repos/ActioBio/bio-news-agent/issues?state=open&labels=ai-digest&per_page=100":
            return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")
        if command[2] == "/repos/ActioBio/bio-news-agent/issues":
            assert input == '{"title": "Biotech / Pharma Headlines \\u2013 2026-04-13", "body": "hello world", "labels": ["ai-digest"]}'
            return subprocess.CompletedProcess(command, 0, stdout='{"number": 34}', stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)

    result = publisher.publish_issue(news_file)

    assert result == {
        "action": "created",
        "issue_number": 34,
        "title": "Biotech / Pharma Headlines – 2026-04-13",
    }
    assert len(calls) == 2


def test_github_api_request_via_gh_strips_token_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-your-token-here")
    monkeypatch.setenv("GH_TOKEN", "also-bad")

    captured_env = {}

    def fake_run(command, input=None, text=None, capture_output=None, check=None, env=None):
        nonlocal captured_env
        captured_env = env or {}
        return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)

    publisher._github_api_request_via_gh("GET", "/repos/ActioBio/bio-news-agent/issues")

    assert "GITHUB_TOKEN" not in captured_env
    assert "GH_TOKEN" not in captured_env


def test_get_github_token_ignores_placeholder(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-your-token-here")
    monkeypatch.delenv("GH_TOKEN", raising=False)

    assert publisher._get_github_token() == ""


def test_get_github_token_prefers_real_gh_token_over_placeholder_github_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-your-token-here")
    monkeypatch.setenv("GH_TOKEN", "real-gh-token")

    assert publisher._get_github_token() == "real-gh-token"


def test_check_issue_status_falls_back_to_gh_when_token_auth_fails(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "stale-token")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    monkeypatch.setattr(
        publisher,
        "_today_issue_title",
        lambda: "Biotech / Pharma Headlines – 2026-04-13",
    )
    monkeypatch.setattr(
        publisher,
        "_github_api_request_via_token",
        lambda method, path, *, token, payload=None: (_ for _ in ()).throw(
            RuntimeError(f"GitHub API {method} {path} failed: 401 bad credentials")
        ),
    )
    monkeypatch.setattr(
        publisher,
        "_github_api_request_via_gh",
        lambda method, path, *, payload=None: [
            {"number": 12, "title": "Biotech / Pharma Headlines – 2026-04-13"}
        ],
    )

    result = publisher.check_issue_status()

    assert result == {
        "exists": True,
        "issue_number": 12,
        "title": "Biotech / Pharma Headlines – 2026-04-13",
    }
