"""Tests for filterer module."""

from datetime import datetime, timezone

import pytest
from filterer import deduplicate


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
