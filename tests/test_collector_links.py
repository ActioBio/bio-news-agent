"""Feed URL resolution through the real parser with an in-memory HTTP response."""

from datetime import datetime, timezone
from xml.sax.saxutils import quoteattr

import pytest

import collector


_NOW = datetime(2026, 9, 7, 16, tzinfo=timezone.utc)
_FINAL_URL = "https://publisher.example/feeds/current.xml"


class _Response:
    def __init__(self, payload, url=_FINAL_URL, headers=None):
        self.payload = payload
        self.url = url
        self.headers = headers or {"Content-Type": "application/xml"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def geturl(self):
        return self.url

    def read(self, limit):
        return self.payload[:limit]


def _feed(link, kind="atom", feed_base="", entry_base="", link_base=""):
    def base(value):
        return f" xml:base={quoteattr(value)}" if value else ""

    if kind == "rss":
        return (
            f'<rss version="2.0"{base(feed_base)}><channel><title>Feed</title>'
            f'<item{base(entry_base)}><title>Story</title>'
            f'<link{base(link_base)}>{link}</link>'
            '<pubDate>Mon, 07 Sep 2026 15:00:00 GMT</pubDate>'
            '</item></channel></rss>'
        ).encode()
    return (
        f'<feed xmlns="http://www.w3.org/2005/Atom"{base(feed_base)}>'
        '<title>Feed</title><id>urn:feed</id><updated>2026-09-07T15:00:00Z</updated>'
        f'<entry{base(entry_base)}><title>Story</title><id>urn:story</id>'
        f'<link href={quoteattr(link)}{base(link_base)}/>'
        '<updated>2026-09-07T15:00:00Z</updated></entry></feed>'
    ).encode()


def _collect(monkeypatch, responses, workers=1):
    monkeypatch.setattr(collector, "_now", lambda: _NOW)
    monkeypatch.setattr(collector, "RSS_MAX_WORKERS", workers)
    monkeypatch.setattr(
        collector, "_load_feeds",
        lambda: {url: {"source": url, "category": "All"} for url in responses},
    )

    def open_response(request, timeout):
        assert timeout == collector.RSS_TIMEOUT
        return responses[request.full_url]

    monkeypatch.setattr(collector, "urlopen", open_response)
    return collector.collect_items_with_stats()


@pytest.mark.parametrize("kind", ["atom", "rss"])
@pytest.mark.parametrize(
    "link,content_location,feed_base,entry_base,link_base,expected",
    [
        ("/story", None, "", "", "", "https://publisher.example/story"),
        ("story", None, "", "", "", "https://publisher.example/feeds/story"),
        ("//other.example/story", None, "", "", "", "https://other.example/story"),
        ("https://other.example/story", None, "", "", "", "https://other.example/story"),
        ("story", "../archive/feed.xml", "", "", "", "https://publisher.example/archive/story"),
        ("story", "https://mirror.example/news/feed.xml", "", "", "", "https://mirror.example/news/story"),
        ("story", None, "../articles/", "section/", "../latest/", "https://publisher.example/articles/latest/story"),
        ("story", "https://mirror.example/news/feed.xml", "../articles/", "", "", "https://mirror.example/articles/story"),
        ("story", None, "https://canonical.example/", "", "", "https://canonical.example/story"),
    ],
    ids=[
        "root-relative", "path-relative", "scheme-relative", "absolute",
        "relative-content-location", "absolute-content-location",
        "nested-xml-base", "xml-base-over-content-location", "absolute-xml-base",
    ],
)
def test_collect_resolves_links_against_final_response(
    monkeypatch, kind, link, content_location, feed_base, entry_base, link_base, expected,
):
    headers = {"Content-Type": "application/xml"}
    if content_location:
        headers["Content-Location"] = content_location
    response = _Response(_feed(link, kind, feed_base, entry_base, link_base), headers=headers)

    items, stats = _collect(
        monkeypatch, {"https://old.example/redirected/feed.xml": response},
    )

    assert [(item["link"], item["id"]) for item in items] == [(expected, expected)]
    assert stats == {
        "feeds_total": 1, "feeds_succeeded": 1, "feeds_failed": 0,
        "items_collected": 1, "feed_errors": [],
    }


@pytest.mark.parametrize("workers", [1, 2])
def test_identical_relative_paths_on_different_origins_have_distinct_ids(monkeypatch, workers):
    responses = {
        f"https://{host}/feed.xml": _Response(_feed("/story"), f"https://{host}/feed.xml")
        for host in ("one.example", "two.example")
    }

    items, stats = _collect(monkeypatch, responses, workers)

    assert {item["id"] for item in items} == {
        "https://one.example/story", "https://two.example/story",
    }
    assert {item["link"] for item in items} == {item["id"] for item in items}
    assert stats["items_collected"] == stats["feeds_succeeded"] == 2
    assert stats["feed_errors"] == []


def test_xml_base_does_not_leak_to_sibling_entries(monkeypatch):
    payload = b'''<feed xmlns="http://www.w3.org/2005/Atom">
      <title>Feed</title><id>urn:feed</id><updated>2026-09-07T15:00:00Z</updated>
      <entry xml:base="../special/"><title>First</title><id>urn:first</id>
        <link href="story"/><updated>2026-09-07T15:00:00Z</updated></entry>
      <entry><title>Second</title><id>urn:second</id>
        <link href="story"/><updated>2026-09-07T15:00:00Z</updated></entry>
    </feed>'''

    items, _stats = _collect(monkeypatch, {"https://old.example/feed": _Response(payload)})

    assert [item["link"] for item in items] == [
        "https://publisher.example/special/story",
        "https://publisher.example/feeds/story",
    ]


def test_feed_size_limit_still_reports_a_fetch_failure(monkeypatch):
    monkeypatch.setattr(collector, "RSS_MAX_FEED_BYTES", 10)

    items, stats = _collect(
        monkeypatch, {"https://old.example/feed": _Response(_feed("/story"))},
    )

    assert items == []
    assert stats["feeds_failed"] == 1
    assert stats["feeds_succeeded"] == stats["items_collected"] == 0
    assert "exceeded max size (10 bytes)" in stats["feed_errors"][0]["error"]
