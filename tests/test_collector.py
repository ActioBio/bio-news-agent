"""Tests for collector module."""

import pytest
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
