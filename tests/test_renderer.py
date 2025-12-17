"""Tests for renderer module."""

from datetime import datetime, timezone

import pytest
from renderer import to_markdown


class TestToMarkdown:
    def test_empty_items(self):
        result = to_markdown([])
        assert "_No fresh biotech/pharma headlines in the last 24 h._" in result

    def test_single_item(self):
        items = [
            {
                "title": "Test Headline",
                "link": "https://example.com/article",
                "source": "Test Source",
                "category": "Regulatory & FDA",
                "published": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            }
        ]
        result = to_markdown(items)
        assert "## Daily Biotech / Pharma Headlines" in result
        assert "### Regulatory & FDA" in result
        assert "[Test Headline](https://example.com/article)" in result
        assert "Test Source" in result

    def test_multiple_categories(self):
        items = [
            {
                "title": "FDA News",
                "link": "https://example.com/1",
                "source": "Source A",
                "category": "Regulatory & FDA",
                "published": datetime(2024, 1, 1, tzinfo=timezone.utc),
            },
            {
                "title": "Trial Results",
                "link": "https://example.com/2",
                "source": "Source B",
                "category": "Clinical & Research",
                "published": datetime(2024, 1, 1, tzinfo=timezone.utc),
            },
        ]
        result = to_markdown(items)
        assert "### Regulatory & FDA" in result
        assert "### Clinical & Research" in result
        assert "FDA News" in result
        assert "Trial Results" in result

    def test_sorts_by_recency(self):
        items = [
            {
                "title": "Older Story",
                "link": "https://example.com/1",
                "source": "Source A",
                "category": "Company News",
                "published": datetime(2024, 1, 1, tzinfo=timezone.utc),
            },
            {
                "title": "Newer Story",
                "link": "https://example.com/2",
                "source": "Source A",
                "category": "Company News",
                "published": datetime(2024, 1, 2, tzinfo=timezone.utc),
            },
        ]
        result = to_markdown(items)
        # Newer story should appear first
        newer_pos = result.find("Newer Story")
        older_pos = result.find("Older Story")
        assert newer_pos < older_pos

    def test_unknown_category_maps_to_company_news(self):
        items = [
            {
                "title": "Unknown Category Item",
                "link": "https://example.com/1",
                "source": "Source A",
                "category": "Unknown Category",
                "published": datetime(2024, 1, 1, tzinfo=timezone.utc),
            },
        ]
        result = to_markdown(items)
        assert "### Company News" in result
        assert "Unknown Category Item" in result

    def test_strips_title_whitespace(self):
        items = [
            {
                "title": "  Headline with spaces  ",
                "link": "https://example.com/1",
                "source": "Source A",
                "category": "Company News",
                "published": datetime(2024, 1, 1, tzinfo=timezone.utc),
            },
        ]
        result = to_markdown(items)
        assert "[Headline with spaces]" in result
