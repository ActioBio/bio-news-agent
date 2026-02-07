"""Tests for collector module."""

from datetime import datetime, timezone

import pytest
import collector
from collector import normalize_url


class TestNormalizeUrl:
    def test_removes_www_prefix(self):
        url = "https://www.example.com/article"
        assert normalize_url(url) == "https://example.com/article"

    def test_removes_trailing_slash(self):
        url = "https://example.com/article/"
        assert normalize_url(url) == "https://example.com/article"

    def test_lowercases_domain_only(self):
        url = "https://Example.COM/Article"
        # Domain lowercased, but path case preserved
        assert normalize_url(url) == "https://example.com/Article"

    def test_removes_utm_parameters(self):
        url = "https://example.com/article?utm_source=twitter&utm_medium=social"
        assert normalize_url(url) == "https://example.com/article"

    def test_preserves_non_tracking_parameters(self):
        url = "https://example.com/article?id=123"
        result = normalize_url(url)
        assert "id=" in result
        assert "example.com/article" in result

    def test_removes_fragment(self):
        url = "https://example.com/article#section"
        assert normalize_url(url) == "https://example.com/article"

    def test_handles_empty_path(self):
        url = "https://example.com"
        assert normalize_url(url) == "https://example.com"

    def test_removes_fbclid(self):
        url = "https://example.com/article?fbclid=abc123"
        assert normalize_url(url) == "https://example.com/article"

    def test_mixed_tracking_and_real_params(self):
        url = "https://example.com/article?page=2&utm_campaign=test"
        result = normalize_url(url)
        assert "utm_campaign" not in result
        assert "page" in result


def test_parse_date_uses_utc_tuple():
    entry = {"published_parsed": (2026, 2, 6, 12, 0, 0, 0, 0, 0)}
    parsed = collector._parse_date(entry)
    assert parsed == datetime(2026, 2, 6, 12, 0, tzinfo=timezone.utc)


def test_collect_items_keeps_entries_when_bozo(monkeypatch):
    class Parsed:
        bozo = True
        bozo_exception = Exception("encoding warning")
        entries = [
            {
                "published_parsed": (2026, 2, 6, 12, 0, 0, 0, 0, 0),
                "title": "Valid story",
                "link": "https://example.com/story?utm_source=newsletter",
            }
        ]

    monkeypatch.setattr(
        collector,
        "_load_feeds",
        lambda: {"https://feed.example.com/rss": {"source": "Example Feed", "category": "All"}},
    )
    monkeypatch.setattr(collector, "_fetch_with_retry", lambda _url: Parsed())
    monkeypatch.setattr(
        collector,
        "_now",
        lambda: datetime(2026, 2, 6, 13, 0, tzinfo=timezone.utc),
    )

    items = collector.collect_items()
    assert len(items) == 1
    assert items[0]["id"] == "https://example.com/story"
    assert items[0]["source"] == "Example Feed"


def test_load_feeds_defaults_missing_source(tmp_path, monkeypatch):
    feeds_path = tmp_path / "feeds.json"
    feeds_path.write_text('{"https://example.com/rss":{"category":"All"}}', encoding="utf-8")

    monkeypatch.setattr(collector, "_FEEDS_FILE", feeds_path)
    feeds = collector._load_feeds()

    assert feeds["https://example.com/rss"]["source"] == "example.com"
