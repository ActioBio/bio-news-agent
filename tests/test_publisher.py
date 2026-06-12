"""Tests for GitHub issue publishing."""

import base64
import gzip
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import publisher

_FROZEN_NOW = datetime(2026, 4, 13, 16, 0, tzinfo=timezone.utc)
_CREATED_TODAY = "2026-04-13T15:00:00Z"
_CREATED_YESTERDAY = "2026-04-12T15:00:00Z"
_BASE_TITLE = "Biotech / Pharma Headlines - Apr 13"
_TOP_STORY_BODY = (
    "## Daily Biotech / Pharma Headlines\n\n"
    "### Top Stories\n"
    "- **[FDA approves first therapy for rare disease](https://example.com/fda)** — FDA (2 sources)\n"
    "- **[Second story](https://example.com/second)** — STAT Biotech\n\n"
    "### Regulatory & FDA\n"
    "- [Plain bullet](https://example.com/plain) — Source\n"
)


def _freeze_now(monkeypatch):
    monkeypatch.setattr(publisher, "_utcnow", lambda: _FROZEN_NOW)


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


def test_publish_issue_updates_existing_issue(tmp_path, monkeypatch):
    news_file = tmp_path / "news.md"
    news_file.write_text(_TOP_STORY_BODY, encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    _freeze_now(monkeypatch)

    calls: list[tuple[str, str, object | None]] = []
    monkeypatch.setattr(
        publisher,
        "_list_open_issues",
        lambda owner, repo: [
            {
                "number": 12,
                "title": "Biotech / Pharma Headlines – 2026-04-13",
                "createdAt": _CREATED_TODAY,
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
        "title": f"{_BASE_TITLE}: FDA approves first therapy for rare disease",
    }
    assert calls == [
        (
            "PATCH",
            "/repos/ActioBio/bio-news-agent/issues/12",
            {
                "title": f"{_BASE_TITLE}: FDA approves first therapy for rare disease",
                "body": _TOP_STORY_BODY,
            },
        )
    ]


def test_check_issue_status_finds_existing_issue(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    _freeze_now(monkeypatch)

    monkeypatch.setattr(
        publisher,
        "_list_open_issues",
        lambda owner, repo: [
            {
                "number": 12,
                "title": f"{_BASE_TITLE}: Some headline",
                "createdAt": _CREATED_TODAY,
                "labels": [{"name": "ai-digest"}],
            }
        ],
    )

    result = publisher.check_issue_status()

    assert result == {
        "exists": True,
        "issue_number": 12,
        "title": f"{_BASE_TITLE}: Some headline",
    }


def test_check_issue_status_ignores_unlabeled_issue(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    _freeze_now(monkeypatch)
    monkeypatch.setattr(
        publisher,
        "_list_open_issues",
        lambda owner, repo: [
            {"number": 12, "title": _BASE_TITLE, "createdAt": _CREATED_TODAY, "labels": []}
        ],
    )

    result = publisher.check_issue_status()

    assert result == {
        "exists": False,
        "issue_number": None,
        "title": _BASE_TITLE,
    }


def test_check_issue_status_ignores_issue_created_on_other_day(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    _freeze_now(monkeypatch)
    monkeypatch.setattr(
        publisher,
        "_list_open_issues",
        lambda owner, repo: [
            {
                "number": 11,
                "title": "Biotech / Pharma Headlines - Apr 12: Old story",
                "createdAt": _CREATED_YESTERDAY,
                "labels": [{"name": "ai-digest"}],
            }
        ],
    )

    result = publisher.check_issue_status()

    assert result == {
        "exists": False,
        "issue_number": None,
        "title": _BASE_TITLE,
    }


def test_check_issue_status_uses_lowest_issue_number_for_duplicates(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    _freeze_now(monkeypatch)
    monkeypatch.setattr(
        publisher,
        "_list_open_issues",
        lambda owner, repo: [
            {
                "number": 13,
                "title": f"{_BASE_TITLE}: Later duplicate",
                "createdAt": _CREATED_TODAY,
                "labels": [{"name": "ai-digest"}],
            },
            {
                "number": 12,
                "title": f"{_BASE_TITLE}: First issue",
                "createdAt": _CREATED_TODAY,
                "labels": [{"name": "ai-digest"}],
            },
        ],
    )

    result = publisher.check_issue_status()

    assert result == {
        "exists": True,
        "issue_number": 12,
        "title": f"{_BASE_TITLE}: First issue",
    }


def test_check_issue_status_reports_missing_issue(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    _freeze_now(monkeypatch)

    monkeypatch.setattr(publisher, "_list_open_issues", lambda owner, repo: [])

    result = publisher.check_issue_status()

    assert result == {
        "exists": False,
        "issue_number": None,
        "title": _BASE_TITLE,
    }


def test_publish_issue_creates_new_issue(tmp_path, monkeypatch):
    news_file = tmp_path / "news.md"
    news_file.write_text("hello world", encoding="utf-8")
    monkeypatch.setenv("GH_TOKEN", "test-token")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    _freeze_now(monkeypatch)

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
        "title": _BASE_TITLE,
    }
    assert calls == [
        (
            "POST",
            "/repos/ActioBio/bio-news-agent/issues",
            {
                "title": _BASE_TITLE,
                "body": "hello world",
                "labels": ["ai-digest"],
            },
        )
    ]


def test_publish_issue_creates_new_issue_with_top_story_title(tmp_path, monkeypatch):
    news_file = tmp_path / "news.md"
    news_file.write_text(_TOP_STORY_BODY, encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    _freeze_now(monkeypatch)

    calls: list[tuple[str, str, object | None]] = []
    monkeypatch.setattr(publisher, "_list_open_issues", lambda owner, repo: [])

    def fake_request(method: str, path: str, payload=None):
        calls.append((method, path, payload))
        if method == "POST":
            return {"number": 35}
        raise AssertionError(f"unexpected call: {method} {path}")

    monkeypatch.setattr(publisher, "_github_api_request", fake_request)

    result = publisher.publish_issue(news_file)

    assert result == {
        "action": "created",
        "issue_number": 35,
        "title": f"{_BASE_TITLE}: FDA approves first therapy for rare disease",
    }
    assert calls[0][2]["title"] == f"{_BASE_TITLE}: FDA approves first therapy for rare disease"


def test_leading_top_story_parses_first_bold_bullet():
    assert publisher._leading_top_story(_TOP_STORY_BODY) == (
        "FDA approves first therapy for rare disease"
    )


def test_leading_top_story_empty_without_bold_bullet():
    assert publisher._leading_top_story("- [Plain bullet](https://example.com) — Source") == ""


def test_leading_top_story_truncates_long_headline():
    headline = "A" * 150
    body = f"- **[{headline}](https://example.com)** — Source"

    story = publisher._leading_top_story(body)

    assert len(story) == 100
    assert story.endswith("…")


def test_issue_title_for_body_prefers_override(monkeypatch):
    monkeypatch.setenv("DIGEST_ISSUE_TITLE_OVERRIDE", "Override Title")

    assert publisher._issue_title_for_body(_TOP_STORY_BODY) == "Override Title"


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
    _freeze_now(monkeypatch)

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
            "number,title,labels,createdAt",
        ]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                '[{"number": 12, "title": "Biotech / Pharma Headlines - Apr 13: Some headline", '
                f'"createdAt": "{_CREATED_TODAY}", "labels": [{{"name": "ai-digest"}}]}}]'
            ),
            stderr="",
        )

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)

    result = publisher.check_issue_status()

    assert result == {
        "exists": True,
        "issue_number": 12,
        "title": "Biotech / Pharma Headlines - Apr 13: Some headline",
    }


def test_publish_issue_uses_gh_cli_when_no_token(tmp_path, monkeypatch):
    news_file = tmp_path / "news.md"
    news_file.write_text("hello world", encoding="utf-8")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    _freeze_now(monkeypatch)

    calls: list[tuple[list[str], str | None]] = []

    def fake_run(command, input=None, text=None, capture_output=None, check=None, env=None):
        calls.append((command, input))
        if command[:3] == ["gh", "issue", "list"]:
            return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")
        if command[2] == "/repos/ActioBio/bio-news-agent/issues":
            assert input == (
                '{"title": "Biotech / Pharma Headlines - Apr 13", '
                '"body": "hello world", "labels": ["ai-digest"]}'
            )
            return subprocess.CompletedProcess(command, 0, stdout='{"number": 34}', stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)

    result = publisher.publish_issue(news_file)

    assert result == {
        "action": "created",
        "issue_number": 34,
        "title": _BASE_TITLE,
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
                                "title": "Biotech / Pharma Headlines - Apr 13: Story",
                                "createdAt": _CREATED_TODAY,
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
                            "title": "Biotech / Pharma Headlines - Apr 12: Story",
                            "createdAt": _CREATED_YESTERDAY,
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
            "title": "Biotech / Pharma Headlines - Apr 13: Story",
            "createdAt": _CREATED_TODAY,
            "labels": [{"name": "ai-digest"}],
        },
        {
            "number": 12,
            "title": "Biotech / Pharma Headlines - Apr 12: Story",
            "createdAt": _CREATED_YESTERDAY,
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
    _freeze_now(monkeypatch)
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
                "title": _BASE_TITLE,
                "createdAt": _CREATED_TODAY,
                "labels": ["ai-digest"],
            }
        ],
    )

    result = publisher.check_issue_status()

    assert result == {
        "exists": True,
        "issue_number": 12,
        "title": _BASE_TITLE,
    }


def test_encode_dispatch_body_round_trips():
    body = "Hello\n\nWorld"

    encoded = publisher._encode_dispatch_body(body)

    assert gzip.decompress(base64.b64decode(encoded)).decode("utf-8") == body


def test_dispatch_publish_workflow_posts_dispatch_payload(tmp_path, monkeypatch):
    news_file = tmp_path / "news.md"
    news_file.write_text(_TOP_STORY_BODY, encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    _freeze_now(monkeypatch)

    calls: list[tuple[str, str, object | None]] = []

    def fake_request(method: str, path: str, payload=None):
        calls.append((method, path, payload))
        return None

    monkeypatch.setattr(publisher, "_github_api_request", fake_request)

    result = publisher.dispatch_publish_workflow(news_file)

    assert result == {
        "workflow": "publish-digest.yml",
        "ref": "main",
        "title": f"{_BASE_TITLE}: FDA approves first therapy for rare disease",
    }
    assert calls == [
        (
            "POST",
            "/repos/ActioBio/bio-news-agent/actions/workflows/publish-digest.yml/dispatches",
            {
                "ref": "main",
                "inputs": {
                    "issue_title": f"{_BASE_TITLE}: FDA approves first therapy for rare disease",
                    "issue_body_gz_b64": publisher._encode_dispatch_body(_TOP_STORY_BODY),
                },
            },
        )
    ]


def test_dispatch_publish_workflow_token_path_accepts_empty_response(tmp_path, monkeypatch):
    news_file = tmp_path / "news.md"
    news_file.write_text("hello world", encoding="utf-8")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.delenv("DIGEST_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "ActioBio/bio-news-agent")
    _freeze_now(monkeypatch)

    calls = []

    def fake_urlopen(request, timeout=30):
        calls.append((request.get_method(), request.full_url, request.data))
        return _FakeResponse(b"")

    monkeypatch.setattr(publisher, "urlopen", fake_urlopen)

    result = publisher.dispatch_publish_workflow(news_file)

    assert result == {
        "workflow": "publish-digest.yml",
        "ref": "main",
        "title": _BASE_TITLE,
    }
    assert calls[0][0] == "POST"
    assert calls[0][1] == (
        "https://api.github.com/repos/ActioBio/bio-news-agent/actions/workflows/"
        "publish-digest.yml/dispatches"
    )


def test_github_api_request_via_token_wraps_invalid_json(monkeypatch):
    monkeypatch.setattr(
        publisher,
        "urlopen",
        lambda request, timeout=30: _FakeResponse(b"not json"),
    )

    try:
        publisher._github_api_request_via_token(
            "GET",
            "/repos/ActioBio/bio-news-agent/issues",
            token="test-token",
        )
    except RuntimeError as exc:
        assert "returned invalid JSON" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_github_api_request_via_token_wraps_timeout(monkeypatch):
    def raise_timeout(request, timeout=30):
        raise TimeoutError("timed out")

    monkeypatch.setattr(publisher, "urlopen", raise_timeout)

    try:
        publisher._github_api_request_via_token(
            "GET",
            "/repos/ActioBio/bio-news-agent/issues",
            token="test-token",
        )
    except RuntimeError as exc:
        assert (
            "GitHub API GET /repos/ActioBio/bio-news-agent/issues failed: timed out"
            in str(exc)
        )
    else:
        raise AssertionError("expected RuntimeError")


def test_github_graphql_request_via_token_wraps_timeout(monkeypatch):
    def raise_timeout(request, timeout=30):
        raise TimeoutError("timed out")

    monkeypatch.setattr(publisher, "urlopen", raise_timeout)

    try:
        publisher._github_graphql_request_via_token(
            "query { viewer { login } }",
            {},
            token="test-token",
        )
    except RuntimeError as exc:
        assert "GitHub GraphQL failed: timed out" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
