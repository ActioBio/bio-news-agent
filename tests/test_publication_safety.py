"""Publication-date and serialized-publication contract regressions."""

import base64
import gzip
import logging
from datetime import datetime, timezone

import pytest

import main
import publisher


_BODY = "### Top Stories\n- **[Original headline](https://example.com/story)** — Wire\n"
_DAY = "2026-09-07"
_TITLE = "Biotech / Pharma Headlines - Sep 7: Original headline"
_REPO = "ActioBio/bio-news-agent"


@pytest.fixture
def publication(tmp_path, monkeypatch):
    news_file = tmp_path / "news.md"
    news_file.write_text(_BODY, encoding="utf-8")
    monkeypatch.setattr(publisher, "_NEWS_FILE", news_file)
    monkeypatch.setenv("GITHUB_REPOSITORY", _REPO)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("DIGEST_DATE", _DAY)
    monkeypatch.delenv("DIGEST_ISSUE_TITLE_OVERRIDE", raising=False)
    clock = [datetime(2026, 9, 7, 15, tzinfo=timezone.utc)]
    monkeypatch.setattr(publisher, "_utcnow", lambda: clock[0])
    issues = []
    reads = []
    writes = []

    def list_issues(owner, repo):
        reads.append((owner, repo))
        return [dict(issue) for issue in issues]

    def request(method, path, payload=None):
        writes.append((method, path, payload))
        if method == "POST" and path == f"/repos/{_REPO}/issues":
            issue = {
                "number": 41,
                "title": payload["title"],
                "body": payload["body"],
                "createdAt": clock[0].isoformat(),
                "labels": [{"name": "ai-digest"}],
            }
            issues.append(issue)
            return {"number": 41}
        if method == "PATCH" and path == f"/repos/{_REPO}/issues/41":
            issues[0].update(payload)
            return {"number": 41}
        if method == "POST" and path == f"/repos/{_REPO}/actions/workflows/publish-digest.yml/dispatches":
            return None
        raise AssertionError(f"Unexpected GitHub request: {method} {path}")

    monkeypatch.setattr(publisher, "_list_open_issues", list_issues)
    monkeypatch.setattr(publisher, "_github_api_request", request)
    return news_file, clock, issues, reads, writes


@pytest.mark.parametrize(("instant", "day", "title_date"), [
    (datetime(2026, 9, 8, 3, 59, tzinfo=timezone.utc), "2026-09-07", "Sep 7"),
    (datetime(2026, 9, 8, 4, 0, tzinfo=timezone.utc), "2026-09-08", "Sep 8"),
    (datetime(2026, 1, 2, 4, 59, tzinfo=timezone.utc), "2026-01-01", "Jan 1"),
    (datetime(2026, 1, 2, 5, 0, tzinfo=timezone.utc), "2026-01-02", "Jan 2"),
    (datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc), "2026-11-01", "Nov 1"),
    (datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc), "2026-11-01", "Nov 1"),
])
def test_dispatch_binds_payload_and_title_to_one_eastern_date(publication, monkeypatch, instant, day, title_date):
    news_file, clock, _, _, writes = publication
    monkeypatch.delenv("DIGEST_DATE")
    clock[0] = instant

    result = publisher.dispatch_publish_workflow(news_file)

    assert result["title"] == f"Biotech / Pharma Headlines - {title_date}: Original headline"
    assert len(writes) == 1
    payload = writes[0][2]
    assert payload["inputs"]["digest_date"] == day
    assert payload["inputs"]["issue_title"] == result["title"]
    assert gzip.decompress(base64.b64decode(payload["inputs"]["issue_body_gz_b64"])).decode() == _BODY


@pytest.mark.parametrize("value", [
    None, "", " ", "20260907", "2026-9-7", "2026-02-30",
    "2026-09-07T00:00:00Z", " 2026-09-07 ", "2026-09-06", "2026-09-08",
])
def test_actions_rejects_missing_malformed_or_noncurrent_date_before_github(publication, monkeypatch, value):
    _, _, _, reads, writes = publication
    if value is None:
        monkeypatch.delenv("DIGEST_DATE")
    else:
        monkeypatch.setenv("DIGEST_DATE", value)

    with pytest.raises(SystemExit) as error:
        main._run_publish_issue(logging.getLogger(__name__))

    assert error.value.code == 1
    assert reads == []
    assert writes == []


@pytest.mark.parametrize("next_day_exists", [False, True])
def test_delayed_dispatch_cannot_create_or_overwrite_next_days_issue(publication, monkeypatch, next_day_exists):
    _, clock, issues, _, writes = publication
    clock[0] = datetime(2026, 9, 8, 4, 1, tzinfo=timezone.utc)
    monkeypatch.setenv("DIGEST_ISSUE_TITLE_OVERRIDE", _TITLE)
    if next_day_exists:
        issues.append({
            "number": 41,
            "title": "Biotech / Pharma Headlines - Sep 8: Fresh headline",
            "body": "fresh digest",
            "createdAt": "2026-09-08T04:00:00Z",
            "labels": [{"name": "ai-digest"}],
        })
    before = [dict(issue) for issue in issues]

    with pytest.raises(SystemExit) as error:
        main._run_publish_issue(logging.getLogger(__name__))

    assert error.value.code == 1
    assert writes == []
    assert issues == before


@pytest.mark.parametrize("existing", [False, True])
def test_same_eastern_day_still_creates_or_updates(publication, existing):
    news_file, clock, issues, _, writes = publication
    clock[0] = datetime(2026, 9, 8, 3, 59, tzinfo=timezone.utc)
    if existing:
        issues.append({
            "number": 41, "title": "Earlier title", "body": "earlier body",
            "createdAt": "2026-09-07T15:00:00Z",
            "labels": [{"name": "ai-digest"}],
        })

    result = publisher.publish_issue(news_file)

    assert result == {
        "action": "updated" if existing else "created",
        "issue_number": 41,
        "title": _TITLE,
    }
    assert len(writes) == 1
    assert writes[0][0] == ("PATCH" if existing else "POST")
    assert issues[0]["title"] == _TITLE
    assert issues[0]["body"] == _BODY


@pytest.mark.parametrize("existing", [False, True])
def test_midnight_during_issue_lookup_stops_before_write(publication, monkeypatch, existing):
    news_file, clock, issues, _, writes = publication
    clock[0] = datetime(2026, 9, 8, 3, 59, 59, tzinfo=timezone.utc)
    if existing:
        issues.append({
            "number": 41, "title": _TITLE, "body": "earlier body",
            "createdAt": "2026-09-07T15:00:00Z",
            "labels": [{"name": "ai-digest"}],
        })
    before = [dict(issue) for issue in issues]

    def delayed_lookup(owner, repo):
        clock[0] = datetime(2026, 9, 8, 4, 0, tzinfo=timezone.utc)
        return [dict(issue) for issue in issues]

    monkeypatch.setattr(publisher, "_list_open_issues", delayed_lookup)

    with pytest.raises(RuntimeError, match="date"):
        publisher.publish_issue(news_file)

    assert writes == []
    assert issues == before


@pytest.mark.parametrize("mode", ["publish", "dispatch"])
def test_midnight_during_body_read_does_not_redate_old_run(publication, monkeypatch, mode):
    news_file, clock, _, _, writes = publication
    monkeypatch.delenv("DIGEST_DATE")
    monkeypatch.delenv("GITHUB_ACTIONS")
    clock[0] = datetime(2026, 9, 8, 3, 59, 59, tzinfo=timezone.utc)

    def delayed_read(path):
        clock[0] = datetime(2026, 9, 8, 4, 0, tzinfo=timezone.utc)
        return path.read_text(encoding="utf-8")

    monkeypatch.setattr(publisher, "_read_digest_body", delayed_read)
    operation = publisher.publish_issue if mode == "publish" else publisher.dispatch_publish_workflow

    with pytest.raises(RuntimeError, match="date"):
        operation(news_file)

    assert writes == []


def test_serial_publishers_recheck_issue_state_instead_of_creating_duplicates(publication):
    news_file, _, issues, reads, writes = publication

    first = publisher.publish_issue(news_file)
    second = publisher.publish_issue(news_file)

    assert first["action"] == "created"
    assert second["action"] == "updated"
    assert len(issues) == 1
    assert len(reads) == 2
    assert [method for method, _, _ in writes] == ["POST", "PATCH"]


def test_direct_manual_publish_without_date_remains_available(publication, monkeypatch):
    news_file, _, issues, _, _ = publication
    monkeypatch.delenv("DIGEST_DATE")
    monkeypatch.delenv("GITHUB_ACTIONS")

    result = publisher.publish_issue(news_file)

    assert result["action"] == "created"
    assert issues[0]["title"] == _TITLE


def test_explicit_stale_date_is_rejected_even_for_direct_manual_publish(publication, monkeypatch):
    news_file, _, _, _, writes = publication
    monkeypatch.delenv("GITHUB_ACTIONS")
    monkeypatch.setenv("DIGEST_DATE", "2026-09-06")

    with pytest.raises(RuntimeError, match="date"):
        publisher.publish_issue(news_file)

    assert writes == []
