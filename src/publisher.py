"""GitHub issue publishing for the daily digest."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from config import (
        DIGEST_ISSUE_LABEL,
        DIGEST_ISSUE_REPO,
        DIGEST_ISSUE_TITLE_PREFIX,
        DIGEST_OUTPUT_FILE,
    )
except ModuleNotFoundError:  # pragma: no cover - module execution fallback
    from .config import (
        DIGEST_ISSUE_LABEL,
        DIGEST_ISSUE_REPO,
        DIGEST_ISSUE_TITLE_PREFIX,
        DIGEST_OUTPUT_FILE,
    )

_GITHUB_API_BASE = "https://api.github.com"
_MAX_ISSUE_BODY_CHARS = 65_000
_PLACEHOLDER_GITHUB_TOKENS = frozenset(
    {
        "ghp-your-token-here",
        "github-token-here",
        "gh-token-here",
        "your-token-here",
    }
)


class PublishResult(TypedDict):
    action: Literal["created", "updated"]
    issue_number: int
    title: str


class IssueStatusResult(TypedDict):
    exists: bool
    issue_number: int | None
    title: str


def _resolve_output_file(path_value: str) -> Path:
    output = Path(path_value)
    if output.is_absolute():
        return output
    return (Path(__file__).resolve().parent.parent / output).resolve()


_NEWS_FILE = _resolve_output_file(DIGEST_OUTPUT_FILE)


def _get_github_token() -> str:
    github_token = _normalize_github_token(os.getenv("GITHUB_TOKEN", ""))
    if github_token:
        return github_token
    return _normalize_github_token(os.getenv("GH_TOKEN", ""))


def _normalize_github_token(token: str) -> str:
    normalized = token.strip()
    if not normalized:
        return ""
    lowered = normalized.lower()
    if lowered in _PLACEHOLDER_GITHUB_TOKENS or "your-token-here" in lowered:
        return ""
    return normalized


def _get_repo_owner_name() -> tuple[str, str]:
    repo_full_name = os.getenv("GITHUB_REPOSITORY", "").strip() or DIGEST_ISSUE_REPO
    if "/" not in repo_full_name:
        raise RuntimeError(f"Invalid repository name for issue publishing: {repo_full_name!r}")
    owner, repo = repo_full_name.split("/", 1)
    return owner, repo


def _today_issue_title() -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    return f"{DIGEST_ISSUE_TITLE_PREFIX} – {today}"


def _read_digest_body(news_file: Path) -> str:
    body = news_file.read_text(encoding="utf-8")
    if len(body) > _MAX_ISSUE_BODY_CHARS:
        return body[:_MAX_ISSUE_BODY_CHARS] + "\n\n*(truncated)*"
    return body


def _github_api_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    token = _get_github_token()
    if token:
        try:
            return _github_api_request_via_token(method, path, token=token, payload=payload)
        except RuntimeError as exc:
            if not _is_github_auth_failure(str(exc)):
                raise
    return _github_api_request_via_gh(method, path, payload=payload)


def _is_github_auth_failure(message: str) -> bool:
    lowered = message.lower()
    return (
        " 401 " in message
        or " 403 " in message
        or "bad credentials" in lowered
        or "requires authentication" in lowered
        or "resource not accessible by integration" in lowered
    )


def _github_api_request_via_token(
    method: str,
    path: str,
    *,
    token: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = Request(
        f"{_GITHUB_API_BASE}{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "bio-news-agent/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:  # pragma: no cover - exercised via behavior, not exact body
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {method} {path} failed: {exc.code} {detail}") from exc
    except URLError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"GitHub API {method} {path} failed: {exc}") from exc


def _github_api_request_via_gh(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
) -> Any:
    command = [
        "gh",
        "api",
        path,
        "--method",
        method,
        "-H",
        "Accept: application/vnd.github+json",
        "-H",
        "X-GitHub-Api-Version: 2022-11-28",
    ]
    stdin = None
    if payload is not None:
        command.extend(["--input", "-"])
        stdin = json.dumps(payload)

    try:
        env = os.environ.copy()
        env.pop("GITHUB_TOKEN", None)
        env.pop("GH_TOKEN", None)
        result = subprocess.run(
            command,
            input=stdin,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Missing GITHUB_TOKEN or GH_TOKEN for issue publishing, and gh CLI is unavailable"
        ) from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown gh error"
        raise RuntimeError(f"GitHub gh api {method} {path} failed: {detail}")

    stdout = result.stdout.strip()
    if not stdout:
        return None
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GitHub gh api {method} {path} returned invalid JSON") from exc


def check_issue_status() -> IssueStatusResult:
    owner, repo = _get_repo_owner_name()
    issue_title = _today_issue_title()
    query = urlencode({"state": "open", "labels": DIGEST_ISSUE_LABEL, "per_page": 100})
    issues = _github_api_request(
        "GET",
        f"/repos/{owner}/{repo}/issues?{query}",
    )
    if not isinstance(issues, list):
        raise RuntimeError("Unexpected GitHub issues response")

    existing_issue = next(
        (issue for issue in issues if isinstance(issue, dict) and issue.get("title") == issue_title),
        None,
    )
    if existing_issue is None:
        return {"exists": False, "issue_number": None, "title": issue_title}
    return {
        "exists": True,
        "issue_number": int(existing_issue["number"]),
        "title": issue_title,
    }


def publish_issue(news_file: Path | None = None) -> PublishResult:
    digest_body = _read_digest_body(news_file or _NEWS_FILE)
    issue_status = check_issue_status()
    owner, repo = _get_repo_owner_name()

    if issue_status["exists"]:
        issue_number_value = issue_status["issue_number"]
        if issue_number_value is None:
            raise RuntimeError("Issue preflight returned exists=true without an issue number")
        issue_number = int(issue_number_value)
        _github_api_request(
            "PATCH",
            f"/repos/{owner}/{repo}/issues/{issue_number}",
            payload={"body": digest_body},
        )
        return {"action": "updated", "issue_number": issue_number, "title": issue_status["title"]}

    created_issue = _github_api_request(
        "POST",
        f"/repos/{owner}/{repo}/issues",
        payload={
            "title": issue_status["title"],
            "body": digest_body,
            "labels": [DIGEST_ISSUE_LABEL],
        },
    )
    if not isinstance(created_issue, dict) or "number" not in created_issue:
        raise RuntimeError("Unexpected GitHub issue create response")
    return {
        "action": "created",
        "issue_number": int(created_issue["number"]),
        "title": issue_status["title"],
    }
