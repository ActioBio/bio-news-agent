"""Tests for filterer module."""

from datetime import datetime, timezone

from filterer import deduplicate, exclude_noise


class TestDeduplicate:
    def test_removes_duplicate_ids(self):
        items = [
            {"id": "abc", "title": "First", "published": datetime(2024, 1, 1, tzinfo=timezone.utc)},
            {"id": "abc", "title": "Second", "published": datetime(2024, 1, 2, tzinfo=timezone.utc)},
        ]
        result = deduplicate(items)
        assert len(result) == 1
        # Keeps the newest one
        assert result[0]["title"] == "Second"

    def test_keeps_unique_ids(self):
        items = [
            {"id": "abc", "title": "First", "published": datetime(2024, 1, 1, tzinfo=timezone.utc)},
            {"id": "def", "title": "Second", "published": datetime(2024, 1, 2, tzinfo=timezone.utc)},
        ]
        result = deduplicate(items)
        assert len(result) == 2

    def test_empty_list(self):
        assert deduplicate([]) == []

    def test_single_item(self):
        items = [{"id": "abc", "title": "Only", "published": datetime(2024, 1, 1, tzinfo=timezone.utc)}]
        result = deduplicate(items)
        assert len(result) == 1
        assert result[0]["title"] == "Only"

    def test_multiple_duplicates(self):
        items = [
            {"id": "abc", "title": "Oldest", "published": datetime(2024, 1, 1, tzinfo=timezone.utc)},
            {"id": "abc", "title": "Middle", "published": datetime(2024, 1, 2, tzinfo=timezone.utc)},
            {"id": "abc", "title": "Newest", "published": datetime(2024, 1, 3, tzinfo=timezone.utc)},
        ]
        result = deduplicate(items)
        assert len(result) == 1
        assert result[0]["title"] == "Newest"


def test_exclude_noise_filters_opinion_and_sponsored_links():
    items = [
        {"title": "Opinion: hospital billing and biotech", "link": "https://example.com/news/1"},
        {"title": "Clinical trial update", "link": "https://example.com/spons/paid-post"},
        {"title": "FDA expands biosimilar guidance", "link": "https://example.com/news/2"},
    ]

    result, skipped = exclude_noise(items)
    assert skipped == 2
    assert result == [{"title": "FDA expands biosimilar guidance", "link": "https://example.com/news/2"}]


def test_exclude_noise_filters_bio_roundups_and_people_moves():
    items = [
        {
            "source": "STAT Pharma",
            "title": "STAT+: Pharmalittle: We’re reading about cheap generic obesity drugs in India",
            "link": "https://example.com/news/1",
        },
        {
            "source": "STAT Pharma",
            "title": "STAT+: Up and down the ladder: The latest comings and goings",
            "link": "https://example.com/news/2",
        },
        {
            "source": "Endpoints News",
            "title": "FDA expands biosimilar guidance",
            "link": "https://example.com/news/3",
        },
    ]

    result, skipped = exclude_noise(items)
    assert skipped == 2
    assert result == [items[2]]


def test_exclude_noise_filters_semicolon_roundups_from_roundup_sources():
    items = [
        {
            "source": "Endpoints News",
            "title": "Vivatides gets $54M; Wegovy drops cold chain in EU; Gilead takes Kymera option",
            "link": "https://example.com/news/1",
        },
        {
            "source": "BioPharma Dive",
            "title": "RFK Jr. rewrites ACIP rules; Gilead, Roche dig into protein degraders",
            "link": "https://example.com/news/2",
        },
        {
            "source": "EMA",
            "title": "Meeting highlights from the PRAC committee",
            "link": "https://example.com/news/3",
        },
    ]

    result, skipped = exclude_noise(items)
    assert skipped == 2
    assert result == [items[2]]


def test_exclude_noise_filters_nih_admin_and_funding_updates():
    items = [
        {
            "source": "NIH",
            "title": "NIH awards top scientific teams for innovations linking nutrition and autoimmune disease",
            "link": "https://www.nih.gov/news-events/news-releases/example-award",
        },
        {
            "source": "NIH",
            "title": "Dr. Elisabeth Armstrong named NIH Chief of Staff",
            "link": "https://www.nih.gov/news-events/news-releases/example-chief-of-staff",
        },
        {
            "source": "NIH",
            "title": "Clinical trial results support use of weekly extended-release buprenorphine during pregnancy",
            "link": "https://www.nih.gov/news-events/news-releases/example-clinical-trial",
        },
    ]

    result, skipped = exclude_noise(items)
    assert skipped == 2
    assert result == [items[2]]


def test_exclude_noise_filters_mhra_guidance_and_keeps_news_items():
    items = [
        {
            "source": "MHRA",
            "title": "Access Consortium Promise Pilot Pathway",
            "link": "https://www.gov.uk/government/news/access-consortium-promise-pilot-pathway",
        },
        {
            "source": "MHRA",
            "title": "Register medical devices to place on the market",
            "link": "https://www.gov.uk/guidance/register-medical-devices-to-place-on-the-market",
        },
        {
            "source": "MHRA",
            "title": "MHRA approves olezarsen (Tryngolza) for the treatment of familial chylomicronemia syndrome",
            "link": "https://www.gov.uk/government/news/mhra-approves-olezarsen-tryngolza-for-the-treatment-of-familial-chylomicronemia-syndrome",
        },
    ]

    result, skipped = exclude_noise(items)
    assert skipped == 2
    assert result == [items[2]]
