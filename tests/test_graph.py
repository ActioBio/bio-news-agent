"""Tests for graph module."""

import pytest
from graph import _extract_keywords, _keyword_categorize


class TestExtractKeywords:
    def test_extracts_significant_words(self):
        title = "Pfizer announces new drug trial results"
        keywords = _extract_keywords(title)
        assert "pfizer" in keywords
        assert "announces" in keywords
        assert "drug" in keywords
        assert "trial" in keywords
        assert "results" in keywords

    def test_filters_stopwords(self):
        title = "The company is in the market"
        keywords = _extract_keywords(title)
        assert "the" not in keywords
        assert "is" not in keywords
        assert "in" not in keywords
        assert "company" in keywords
        assert "market" in keywords

    def test_filters_short_words(self):
        title = "FDA ok for new med"
        keywords = _extract_keywords(title)
        assert "fda" not in keywords  # 3 chars
        assert "ok" not in keywords   # 2 chars
        assert "for" not in keywords  # stopword
        assert "new" not in keywords  # stopword
        assert "med" not in keywords  # 3 chars

    def test_lowercases_words(self):
        title = "MODERNA Vaccine Shows EFFICACY"
        keywords = _extract_keywords(title)
        assert "moderna" in keywords
        assert "MODERNA" not in keywords
        assert "vaccine" in keywords
        assert "efficacy" in keywords


class TestKeywordCategorize:
    def test_regulatory_fda(self):
        assert _keyword_categorize("FDA approves new drug") == "Regulatory & FDA"
        assert _keyword_categorize("Drug gets regulatory approval") == "Regulatory & FDA"
        assert _keyword_categorize("EMA rejects application") == "Regulatory & FDA"

    def test_clinical_research(self):
        assert _keyword_categorize("Phase 3 trial shows results") == "Clinical & Research"
        assert _keyword_categorize("Study finds new efficacy data") == "Clinical & Research"
        assert _keyword_categorize("Therapy shows promise in research") == "Clinical & Research"

    def test_deals_finance(self):
        assert _keyword_categorize("Company raises $100M") == "Deals & Finance"
        assert _keyword_categorize("Merger deal announced") == "Deals & Finance"
        assert _keyword_categorize("Acquisition of startup") == "Deals & Finance"
        assert _keyword_categorize("IPO pricing announced") == "Deals & Finance"

    def test_company_news(self):
        assert _keyword_categorize("CEO steps down") == "Company News"
        assert _keyword_categorize("Company layoffs announced") == "Company News"
        assert _keyword_categorize("Executive hire at firm") == "Company News"

    def test_policy_politics(self):
        assert _keyword_categorize("Trump administration policy") == "Policy & Politics"
        assert _keyword_categorize("Congress debates Medicare") == "Policy & Politics"
        assert _keyword_categorize("New legislation proposed") == "Policy & Politics"

    def test_market_insights(self):
        assert _keyword_categorize("Market forecast for 2025") == "Market Insights"
        assert _keyword_categorize("Industry spending trends") == "Market Insights"
        assert _keyword_categorize("Billion dollar outlook") == "Market Insights"

    def test_company_name_fallback(self):
        assert _keyword_categorize("Pfizer announces something") == "Company News"
        assert _keyword_categorize("Moderna updates investors") == "Company News"

    def test_default_category(self):
        assert _keyword_categorize("Random headline here") == "Company News"
