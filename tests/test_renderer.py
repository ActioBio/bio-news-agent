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

    def test_prefers_primary_source_over_newer_commentary_when_tier_matches(self):
        items = [
            {
                "title": "Commentary take",
                "link": "https://example.com/1",
                "source": "Newsletter",
                "source_role": "commentary",
                "category": "Company News",
                "published": datetime(2024, 1, 2, tzinfo=timezone.utc),
            },
            {
                "title": "FDA notice",
                "link": "https://example.com/2",
                "source": "FDA",
                "source_role": "primary",
                "category": "Company News",
                "published": datetime(2024, 1, 1, tzinfo=timezone.utc),
            },
        ]

        result = to_markdown(items)
        assert result.find("FDA notice") < result.find("Commentary take")

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

    def test_uses_compact_lines_without_summary_text(self):
        items = [
            {
                "title": "Compact Bio Story",
                "link": "https://example.com/1",
                "source": "Source A",
                "category": "Company News",
                "summary_line": "This should not be rendered.",
                "coverage_sources": ["Source B"],
                "published": datetime(2024, 1, 1, tzinfo=timezone.utc),
            },
        ]

        result = to_markdown(items, executive_summary="This should stay hidden.")
        assert "This should not be rendered." not in result
        assert "This should stay hidden." not in result
        assert "- [Compact Bio Story](https://example.com/1) — Source A (2 sources)" in result

    def test_does_not_repeat_top_story_in_category_sections(self):
        items = [
            {
                "title": "Top Bio Story",
                "link": "https://example.com/top",
                "source": "Source A",
                "category": "Company News",
                "_prompt_id": "item-1",
                "published": datetime(2024, 1, 2, tzinfo=timezone.utc),
            },
            {
                "title": "Second Bio Story",
                "link": "https://example.com/second",
                "source": "Source B",
                "category": "Company News",
                "_prompt_id": "item-2",
                "published": datetime(2024, 1, 1, tzinfo=timezone.utc),
            },
        ]

        result = to_markdown(items, top_stories=["item-1"])
        assert result.count("Top Bio Story") == 1
        assert "Second Bio Story" in result
