"""GitHub issue publishing for the daily digest."""

from __future__ import annotations

import base64
import gzip
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from config import (
        DIGEST_ACTIONS_REF,
        DIGEST_ISSUE_LABEL,
        DIGEST_ISSUE_REPO,
        DIGEST_ISSUE_TITLE_PREFIX,
        DIGEST_OUTPUT_FILE,
        DIGEST_PUBLISH_WORKFLOW,
        resolve_repo_output_file,
    )
except ModuleNotFoundError:  # pragma: no cover - module execution fallback
    from .config import (
        DIGEST_ACTIONS_REF,
        DIGEST_ISSUE_LABEL,
        DIGEST_ISSUE_REPO,
        DIGEST_ISSUE_TITLE_PREFIX,
        DIGEST_OUTPUT_FILE,
        DIGEST_PUBLISH_WORKFLOW,
        resolve_repo_output_file,
    )

_GITHUB_API_BASE = "https://api.github.com"
_MAX_ISSUE_BODY_CHARS = 65_000
_OPEN_ISSUES_QUERY = """
query($owner: String!, $repo: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    issues(first: 100, after: $cursor, states: OPEN, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        number
        title
        labels(first: 20) {
          nodes {
            name
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
"""
_PLACEHOLDER_GITHUB_TOKENS = frozenset(
    {
        "ghp-your-token-here",
        "ghp_your_token_here",
        "gho_your_token_here",
        "github_pat_your_token_here",
        "github-token-here",
        "gh-token-here",
        "your-token-here",
        "your_token_here",
    }
)
_GITHUB_TOKEN_ENV_NAMES = ("DIGEST_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")
_MISSING_GITHUB_AUTH_MESSAGE = (
    "Missing DIGEST_GITHUB_TOKEN, GITHUB_TOKEN, or GH_TOKEN for issue publishing, "
    "and gh CLI is unavailable"
)
_GitHubResult = TypeVar("_GitHubResult")


class PublishResult(TypedDict):
    action: Literal["created", "updated"]
    issue_number: int
    title: str


class IssueStatusResult(TypedDict):
    exists: bool
    issue_number: int | None
    title: str


class DispatchResult(TypedDict):
    workflow: str
    ref: str
    title: str

_NEWS_FILE = resolve_repo_output_file(DIGEST_OUTPUT_FILE)


def _get_github_token() -> str:
    for env_name in _GITHUB_TOKEN_ENV_NAMES:
        token = _normalize_github_token(os.getenv(env_name, ""))
        if token:
            return token
    return ""


def _normalize_github_token(token: str) -> str:
    normalized = token.strip()
    if not normalized:
        return ""
    lowered = normalized.lower()
    if (
        lowered in _PLACEHOLDER_GITHUB_TOKENS
        or "your-token-here" in lowered
        or "your_token_here" in lowered
    ):
        return ""
    return normalized


def _prefer_gh_cli() -> bool:
    return os.getenv("GITHUB_ACTIONS", "").strip().lower() != "true"


def _is_gh_cli_unavailable(message: str) -> bool:
    return "gh cli is unavailable" in message.lower()


def _get_repo_owner_name() -> tuple[str, str]:
    repo_full_name = os.getenv("GITHUB_REPOSITORY", "").strip() or DIGEST_ISSUE_REPO
    if "/" not in repo_full_name:
        raise RuntimeError(f"Invalid repository name for issue publishing: {repo_full_name!r}")
    owner, repo = repo_full_name.split("/", 1)
    return owner, repo


def _today_issue_title() -> str:
    override = os.getenv("DIGEST_ISSUE_TITLE_OVERRIDE", "").strip()
    if override:
        return override
    today = datetime.now(timezone.utc).date().isoformat()
    return f"{DIGEST_ISSUE_TITLE_PREFIX} – {today}"


def _read_digest_body(news_file: Path) -> str:
    body = news_file.read_text(encoding="utf-8")
    if len(body) > _MAX_ISSUE_BODY_CHARS:
        return body[:_MAX_ISSUE_BODY_CHARS] + "\n\n*(truncated)*"
    return body


def _encode_dispatch_body(body: str) -> str:
    compressed = gzip.compress(body.encode("utf-8"))
    return base64.b64encode(compressed).decode("ascii")


def _github_api_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    return _request_with_github_auth(
        via_token=lambda token: _github_api_request_via_token(
            method,
            path,
            token=token,
            payload=payload,
        ),
        via_gh=lambda: _github_api_request_via_gh(method, path, payload=payload),
    )


def _is_github_auth_failure(message: str) -> bool:
    lowered = message.lower()
    return (
        " 401 " in message
        or " 403 " in message
        or "bad credentials" in lowered
        or "requires authentication" in lowered
        or "resource not accessible by integration" in lowered
    )


def _request_with_github_auth(
    *,
    via_token: Callable[[str], _GitHubResult],
    via_gh: Callable[[], _GitHubResult],
) -> _GitHubResult:
    token = _get_github_token()
    if _prefer_gh_cli():
        try:
            return via_gh()
        except RuntimeError as exc:
            if not token or not _is_gh_cli_unavailable(str(exc)):
                raise
            return via_token(token)

    if token:
        try:
            return via_token(token)
        except RuntimeError as exc:
            if not _is_github_auth_failure(str(exc)):
                raise
    return via_gh()


def _run_gh_json(
    command: list[str],
    *,
    stdin: str | None = None,
    error_prefix: str,
    empty_stdout: Any = None,
) -> Any:
    try:
        env = os.environ.copy()
        for env_name in _GITHUB_TOKEN_ENV_NAMES:
            env.pop(env_name, None)
        result = subprocess.run(
            command,
            input=stdin,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(_MISSING_GITHUB_AUTH_MESSAGE) from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown gh error"
        raise RuntimeError(f"{error_prefix} failed: {detail}")

    stdout = result.stdout.strip()
    if not stdout:
        return empty_stdout
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{error_prefix} returned invalid JSON") from exc


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
            body = response.read()
            if not body:
                return None
            try:
                return json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"GitHub API {method} {path} returned invalid JSON"
                ) from exc
    except HTTPError as exc:  # pragma: no cover - exercised via behavior, not exact body
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {method} {path} failed: {exc.code} {detail}") from exc
    except (URLError, TimeoutError) as exc:  # pragma: no cover - network dependent
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

    return _run_gh_json(
        command,
        stdin=stdin,
        error_prefix=f"GitHub gh api {method} {path}",
    )


def _github_graphql_request_via_token(
    query: str,
    variables: dict[str, Any],
    *,
    token: str,
) -> dict[str, Any]:
    request = Request(
        f"{_GITHUB_API_BASE}/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "bio-news-agent/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:  # pragma: no cover - exercised via behavior, not exact body
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub GraphQL failed: {exc.code} {detail}") from exc
    except (URLError, TimeoutError) as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"GitHub GraphQL failed: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected GitHub GraphQL response")
    if payload.get("errors"):
        raise RuntimeError(f"GitHub GraphQL failed: {payload['errors']}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected GitHub GraphQL response")
    return data


def _list_open_issues(owner: str, repo: str) -> list[dict[str, Any]]:
    return _request_with_github_auth(
        via_token=lambda token: _list_open_issues_via_graphql_token(
            owner,
            repo,
            token=token,
        ),
        via_gh=lambda: _list_open_issues_via_gh(owner, repo),
    )


def _list_open_issues_via_graphql_token(
    owner: str,
    repo: str,
    *,
    token: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    cursor: str | None = None

    while True:
        data = _github_graphql_request_via_token(
            _OPEN_ISSUES_QUERY,
            {"owner": owner, "repo": repo, "cursor": cursor},
            token=token,
        )
        issue_page = data["repository"]["issues"]
        for issue in issue_page["nodes"]:
            labels = issue.get("labels", {}).get("nodes", [])
            issues.append(
                {
                    "number": issue["number"],
                    "title": issue["title"],
                    "labels": labels,
                }
            )

        page_info = issue_page["pageInfo"]
        if not page_info["hasNextPage"]:
            return issues
        cursor = page_info["endCursor"]


def _list_open_issues_via_gh(owner: str, repo: str) -> list[dict[str, Any]]:
    command = [
        "gh",
        "issue",
        "list",
        "--repo",
        f"{owner}/{repo}",
        "--state",
        "open",
        "--limit",
        "1000",
        "--json",
        "number,title,labels",
    ]
    payload = _run_gh_json(
        command,
        error_prefix="GitHub issue list",
        empty_stdout=[],
    )
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected GitHub issue list response")
    return [issue for issue in payload if isinstance(issue, dict)]


def check_issue_status() -> IssueStatusResult:
    owner, repo = _get_repo_owner_name()
    issue_title = _today_issue_title()
    issues = _list_open_issues(owner, repo)

    matching_issues = [
        issue
        for issue in issues
        if isinstance(issue, dict)
        and issue.get("title") == issue_title
        and _issue_has_label(issue, DIGEST_ISSUE_LABEL)
    ]
    if not matching_issues:
        return {"exists": False, "issue_number": None, "title": issue_title}

    existing_issue = min(matching_issues, key=lambda issue: int(issue["number"]))
    return {
        "exists": True,
        "issue_number": int(existing_issue["number"]),
        "title": issue_title,
    }


def _issue_has_label(issue: dict[str, Any], label_name: str) -> bool:
    labels = issue.get("labels", [])
    if not isinstance(labels, list):
        return False
    for label in labels:
        if isinstance(label, dict) and label.get("name") == label_name:
            return True
        if isinstance(label, str) and label == label_name:
            return True
    return False


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


def dispatch_publish_workflow(news_file: Path | None = None) -> DispatchResult:
    issue_title = _today_issue_title()
    digest_body = _read_digest_body(news_file or _NEWS_FILE)
    owner, repo = _get_repo_owner_name()
    _github_api_request(
        "POST",
        f"/repos/{owner}/{repo}/actions/workflows/{DIGEST_PUBLISH_WORKFLOW}/dispatches",
        payload={
            "ref": DIGEST_ACTIONS_REF,
            "inputs": {
                "issue_title": issue_title,
                "issue_body_gz_b64": _encode_dispatch_body(digest_body),
            },
        },
    )
    return {
        "workflow": DIGEST_PUBLISH_WORKFLOW,
        "ref": DIGEST_ACTIONS_REF,
        "title": issue_title,
    }
