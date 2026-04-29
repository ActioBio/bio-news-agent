"""Tests for GitHub issue publishing."""

import base64
import gzip
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
    monkeypatch.setattr(
        publisher,
        "_list_open_issues",
        lambda owner, repo: [
            {
                "number": 12,
                "title": "Biotech / Pharma Headlines – 2026-04-13",
                "labels": [{"name": "ai-digest"}],
            }
        ],
    )

    def fake_request(method: str, path: str, payload=None):
        calls.append((method, path, payload))
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
    assert calls == [
        (
            "PATCH",
            "/repos/ActioBio/bio-news-agent/issues/12",
            {"body": "hello world"},
        )
    ]


def test_check_issue_status_finds_existing_issue(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    monkeypatch.setattr(
        publisher,
        "_today_issue_title",
        lambda: "Biotech / Pharma Headlines – 2026-04-13",
    )

    monkeypatch.setattr(
        publisher,
        "_list_open_issues",
        lambda owner, repo: [
            {
                "number": 12,
                "title": "Biotech / Pharma Headlines – 2026-04-13",
                "labels": [{"name": "ai-digest"}],
            }
        ],
    )

    result = publisher.check_issue_status()

    assert result == {
        "exists": True,
        "issue_number": 12,
        "title": "Biotech / Pharma Headlines – 2026-04-13",
    }


def test_check_issue_status_ignores_unlabeled_matching_title(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    monkeypatch.setattr(
        publisher,
        "_today_issue_title",
        lambda: "Biotech / Pharma Headlines – 2026-04-13",
    )
    monkeypatch.setattr(
        publisher,
        "_list_open_issues",
        lambda owner, repo: [
            {"number": 12, "title": "Biotech / Pharma Headlines – 2026-04-13", "labels": []}
        ],
    )

    result = publisher.check_issue_status()

    assert result == {
        "exists": False,
        "issue_number": None,
        "title": "Biotech / Pharma Headlines – 2026-04-13",
    }


def test_check_issue_status_uses_lowest_issue_number_for_duplicates(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    monkeypatch.setattr(
        publisher,
        "_today_issue_title",
        lambda: "Biotech / Pharma Headlines – 2026-04-13",
    )
    monkeypatch.setattr(
        publisher,
        "_list_open_issues",
        lambda owner, repo: [
            {
                "number": 13,
                "title": "Biotech / Pharma Headlines – 2026-04-13",
                "labels": [{"name": "ai-digest"}],
            },
            {
                "number": 12,
                "title": "Biotech / Pharma Headlines – 2026-04-13",
                "labels": [{"name": "ai-digest"}],
            },
        ],
    )

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

    monkeypatch.setattr(publisher, "_list_open_issues", lambda owner, repo: [])

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
    monkeypatch.setattr(publisher, "_list_open_issues", lambda owner, repo: [])

    def fake_request(method: str, path: str, payload=None):
        calls.append((method, path, payload))
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
    assert calls == [
        (
            "POST",
            "/repos/ActioBio/bio-news-agent/issues",
            {
                "title": "Biotech / Pharma Headlines – 2026-04-13",
                "body": "hello world",
                "labels": ["ai-digest"],
            },
        )
    ]


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
        assert command == [
            "gh",
            "issue",
            "list",
            "--repo",
            "ActioBio/bio-news-agent",
            "--state",
            "open",
            "--limit",
            "1000",
            "--json",
            "number,title,labels",
        ]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                '[{"number": 12, "title": "Biotech / Pharma Headlines – 2026-04-13", '
                '"labels": [{"name": "ai-digest"}]}]'
            ),
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
        if command[:3] == ["gh", "issue", "list"]:
            return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")
        if command[2] == "/repos/ActioBio/bio-news-agent/issues":
            assert input == (
                '{"title": "Biotech / Pharma Headlines \\u2013 2026-04-13", '
                '"body": "hello world", "labels": ["ai-digest"]}'
            )
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
    monkeypatch.setenv("DIGEST_GITHUB_TOKEN", "real-token")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-your-token-here")
    monkeypatch.setenv("GH_TOKEN", "also-bad")

    captured_env = {}

    def fake_run(command, input=None, text=None, capture_output=None, check=None, env=None):
        nonlocal captured_env
        captured_env = env or {}
        return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)

    publisher._github_api_request_via_gh("GET", "/repos/ActioBio/bio-news-agent/issues")

    assert "DIGEST_GITHUB_TOKEN" not in captured_env
    assert "GITHUB_TOKEN" not in captured_env
    assert "GH_TOKEN" not in captured_env


def test_github_api_request_prefers_gh_cli_locally(monkeypatch):
    monkeypatch.setenv("DIGEST_GITHUB_TOKEN", "digest-token")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(
        publisher,
        "_github_api_request_via_token",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected token path")),
    )
    monkeypatch.setattr(
        publisher,
        "_github_api_request_via_gh",
        lambda method, path, *, payload=None: {"ok": True},
    )

    assert publisher._github_api_request("GET", "/repos/ActioBio/bio-news-agent") == {"ok": True}


def test_github_api_request_prefers_token_in_github_actions(monkeypatch):
    monkeypatch.setenv("DIGEST_GITHUB_TOKEN", "digest-token")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(
        publisher,
        "_github_api_request_via_token",
        lambda method, path, *, token, payload=None: {"token": token},
    )
    monkeypatch.setattr(
        publisher,
        "_github_api_request_via_gh",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected gh path")),
    )

    assert publisher._github_api_request("GET", "/repos/ActioBio/bio-news-agent") == {
        "token": "digest-token"
    }


def test_list_open_issues_via_graphql_token_paginates(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_graphql_request(query, variables, *, token):
        assert query == publisher._OPEN_ISSUES_QUERY
        assert token == "test-token"
        calls.append(variables)
        if variables["cursor"] is None:
            return {
                "repository": {
                    "issues": {
                        "nodes": [
                            {
                                "number": 13,
                                "title": "Biotech / Pharma Headlines – 2026-04-13",
                                "labels": {"nodes": [{"name": "ai-digest"}]},
                            }
                        ],
                        "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
                    }
                }
            }
        return {
            "repository": {
                "issues": {
                    "nodes": [
                        {
                            "number": 12,
                            "title": "Biotech / Pharma Headlines – 2026-04-12",
                            "labels": {"nodes": [{"name": "ai-digest"}]},
                        }
                    ],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        }

    monkeypatch.setattr(publisher, "_github_graphql_request_via_token", fake_graphql_request)

    issues = publisher._list_open_issues_via_graphql_token(
        "ActioBio",
        "bio-news-agent",
        token="test-token",
    )

    assert [call["cursor"] for call in calls] == [None, "cursor-1"]
    assert issues == [
        {
            "number": 13,
            "title": "Biotech / Pharma Headlines – 2026-04-13",
            "labels": [{"name": "ai-digest"}],
        },
        {
            "number": 12,
            "title": "Biotech / Pharma Headlines – 2026-04-12",
            "labels": [{"name": "ai-digest"}],
        },
    ]


def test_get_github_token_ignores_placeholder(monkeypatch):
    monkeypatch.setenv("DIGEST_GITHUB_TOKEN", "github_pat_your_token_here")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-your-token-here")
    monkeypatch.delenv("GH_TOKEN", raising=False)

    assert publisher._get_github_token() == ""


def test_get_github_token_prefers_digest_token_over_generic_tokens(monkeypatch):
    monkeypatch.setenv("DIGEST_GITHUB_TOKEN", "digest-token")
    monkeypatch.setenv("GITHUB_TOKEN", "generic-token")
    monkeypatch.setenv("GH_TOKEN", "gh-token")

    assert publisher._get_github_token() == "digest-token"


def test_get_github_token_prefers_real_gh_token_over_placeholder_github_token(monkeypatch):
    monkeypatch.delenv("DIGEST_GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-your-token-here")
    monkeypatch.setenv("GH_TOKEN", "real-gh-token")

    assert publisher._get_github_token() == "real-gh-token"


def test_list_open_issues_prefers_gh_cli_locally(monkeypatch):
    monkeypatch.setenv("DIGEST_GITHUB_TOKEN", "digest-token")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(
        publisher,
        "_list_open_issues_via_graphql_token",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected token path")),
    )
    monkeypatch.setattr(
        publisher,
        "_list_open_issues_via_gh",
        lambda owner, repo: [
            {"number": 12, "title": "Biotech / Pharma Headlines – 2026-04-13"}
        ],
    )

    assert publisher._list_open_issues("ActioBio", "bio-news-agent") == [
        {"number": 12, "title": "Biotech / Pharma Headlines – 2026-04-13"}
    ]


def test_list_open_issues_prefers_token_in_github_actions(monkeypatch):
    monkeypatch.setenv("DIGEST_GITHUB_TOKEN", "digest-token")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(
        publisher,
        "_list_open_issues_via_graphql_token",
        lambda owner, repo, *, token: [{"number": 12, "token": token}],
    )
    monkeypatch.setattr(
        publisher,
        "_list_open_issues_via_gh",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected gh path")),
    )

    assert publisher._list_open_issues("ActioBio", "bio-news-agent") == [
        {"number": 12, "token": "digest-token"}
    ]


def test_check_issue_status_falls_back_to_gh_when_token_auth_fails(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "stale-token")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    monkeypatch.setattr(
        publisher,
        "_today_issue_title",
        lambda: "Biotech / Pharma Headlines – 2026-04-13",
    )
    monkeypatch.setattr(
        publisher,
        "_list_open_issues_via_graphql_token",
        lambda owner, repo, *, token: (_ for _ in ()).throw(
            RuntimeError("GitHub GraphQL failed: 401 bad credentials")
        ),
    )
    monkeypatch.setattr(
        publisher,
        "_list_open_issues_via_gh",
        lambda owner, repo: [
            {
                "number": 12,
                "title": "Biotech / Pharma Headlines – 2026-04-13",
                "labels": ["ai-digest"],
            }
        ],
    )

    result = publisher.check_issue_status()

    assert result == {
        "exists": True,
        "issue_number": 12,
        "title": "Biotech / Pharma Headlines – 2026-04-13",
    }


def test_encode_dispatch_body_round_trips():
    body = "Hello\n\nWorld"

    encoded = publisher._encode_dispatch_body(body)

    assert gzip.decompress(base64.b64decode(encoded)).decode("utf-8") == body


def test_dispatch_publish_workflow_posts_dispatch_payload(tmp_path, monkeypatch):
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
        return None

    monkeypatch.setattr(publisher, "_github_api_request", fake_request)

    result = publisher.dispatch_publish_workflow(news_file)

    assert result == {
        "workflow": "publish-digest.yml",
        "ref": "main",
        "title": "Biotech / Pharma Headlines – 2026-04-13",
    }
    assert calls == [
        (
            "POST",
            "/repos/ActioBio/bio-news-agent/actions/workflows/publish-digest.yml/dispatches",
            {
                "ref": "main",
                "inputs": {
                    "issue_title": "Biotech / Pharma Headlines – 2026-04-13",
                    "issue_body_gz_b64": publisher._encode_dispatch_body("hello world"),
                },
            },
        )
    ]
