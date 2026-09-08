"""The collection window uses one run clock and tolerates one hour of skew."""

from datetime import datetime, timezone
from io import BytesIO

import pytest

import collector


_RUN_NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


class _FeedResponse(BytesIO):
    headers = {"Content-Type": "application/atom+xml"}

    def __init__(self, url: str, published: str | None):
        self.url = url
        date = f"<published>{published}</published>" if published else ""
        super().__init__(
            (
                '<feed xmlns="http://www.w3.org/2005/Atom">'
                '<title>Example feed</title><id>urn:example:feed</id>'
                '<entry><title>Example story</title>'
                f'<id>{url}/story</id><link href="{url}/story"/>{date}'
                '</entry></feed>'
            ).encode()
        )

    def geturl(self):
        return self.url


@pytest.mark.parametrize(
    ("published", "expected_count"),
    [
        ("2026-09-06T11:59:59Z", 0),
        ("2026-09-06T12:00:00Z", 1),
        ("2026-09-07T12:00:00Z", 1),
        ("2026-09-07T12:59:59Z", 1),
        ("2026-09-07T13:00:00Z", 1),
        ("2026-09-07T13:00:01Z", 0),
        ("2026-09-08T12:00:00Z", 0),
        ("2027-09-07T12:00:00Z", 0),
        (None, 0),
        ("2026-09-07T09:00:00-04:00", 1),
        ("2026-09-07T09:00:01-04:00", 0),
    ],
)
def test_collects_only_the_inclusive_utc_window(monkeypatch, published, expected_count):
    monkeypatch.setattr(collector, "_now", lambda: _RUN_NOW)
    monkeypatch.setattr(
        collector, "_load_feeds", lambda: {"https://feed.example/rss": {"source": "Example"}}
    )
    monkeypatch.setattr(
        collector, "urlopen", lambda request, timeout: _FeedResponse(request.full_url, published)
    )

    items, stats = collector.collect_items_with_stats()

    assert [item["id"] for item in items] == ["https://feed.example/rss/story"] * expected_count
    assert stats == {
        "feeds_total": 1,
        "feeds_succeeded": 1,
        "feeds_failed": 0,
        "items_collected": expected_count,
        "feed_errors": [],
    }


@pytest.mark.parametrize("max_workers", [1, 2])
def test_fetch_elapsed_time_does_not_move_either_window_boundary(monkeypatch, max_workers):
    clock = [_RUN_NOW]
    dates = {
        "https://first.example/rss": "2026-09-06T12:00:00Z",
        "https://second.example/rss": "2026-09-07T13:00:01Z",
    }
    monkeypatch.setattr(collector, "_now", lambda: clock[0])
    monkeypatch.setattr(collector, "RSS_MAX_WORKERS", max_workers)
    monkeypatch.setattr(
        collector, "_load_feeds", lambda: {url: {"source": url} for url in dates}
    )

    def fetch(request, timeout):
        clock[0] = datetime(2026, 9, 7, 14, 0, tzinfo=timezone.utc)
        return _FeedResponse(request.full_url, dates[request.full_url])

    monkeypatch.setattr(collector, "urlopen", fetch)

    items, stats = collector.collect_items_with_stats()

    assert [item["id"] for item in items] == ["https://first.example/rss/story"]
    assert stats == {
        "feeds_total": 2,
        "feeds_succeeded": 2,
        "feeds_failed": 0,
        "items_collected": 1,
        "feed_errors": [],
    }
