"""Tests for graph module."""

import json
from datetime import datetime, timezone

import graph
from graph import (
    _build_candidate_groups,
    _is_high_confidence_duplicate,
    _keyword_categorize,
    build_graph,
    node_categorize,
    node_filter,
    node_render,
)


def _item(
    item_id: str,
    title: str,
    hour: int,
    *,
    source: str = "Example",
    summary: str = "",
    category: str = "All",
    source_type: str = "news",
) -> dict[str, object]:
    return {
        "id": item_id,
        "title": title,
        "original_title": title,
        "link": f"https://example.com/{item_id}",
        "source": source,
        "published": datetime(2026, 1, 1, hour, 0, tzinfo=timezone.utc),
        "category": category,
        "summary": summary,
        "source_type": source_type,
    }


def test_keyword_categorize_regulatory():
    assert _keyword_categorize("FDA approves new drug") == "Regulatory & FDA"
    assert _keyword_categorize("Drug gets regulatory approval") == "Regulatory & FDA"


def test_keyword_categorize_clinical():
    assert _keyword_categorize("Phase 3 trial shows results") == "Clinical & Research"
    assert _keyword_categorize("Therapy shows promise in research") == "Clinical & Research"


def test_keyword_categorize_deals():
    assert _keyword_categorize("Company raises $100M") == "Deals & Finance"
    assert _keyword_categorize("Merger deal announced") == "Deals & Finance"


def test_keyword_categorize_company_name_fallback():
    assert _keyword_categorize("Pfizer updates investors") == "Company News"


def test_high_confidence_duplicate_requires_strong_overlap():
    existing = {"title": "Pfizer elranatamab multiple myeloma bispecific update"}
    candidate = {"title": "Pfizer elranatamab multiple myeloma bispecific results"}
    assert _is_high_confidence_duplicate(existing, candidate)


def test_high_confidence_duplicate_avoids_unrelated_company_stories():
    existing = {"title": "Moderna wins FDA approval for RSV vaccine"}
    candidate = {"title": "Moderna announces CFO departure"}
    assert not _is_high_confidence_duplicate(existing, candidate)


def test_build_candidate_groups_clusters_possible_duplicates():
    items = [
        _item(
            "a",
            "Pfizer announces phase 3 oncology trial results",
            12,
            source="Pfizer",
            summary="Official release for the phase 3 oncology study",
        ),
        _item(
            "b",
            "Pfizer reports phase 3 oncology trial data",
            11,
            source="Endpoints News",
            summary="Coverage of the same phase 3 oncology trial",
        ),
        _item(
            "c",
            "NIH announces new grants for rare disease research",
            10,
            source="NIH",
        ),
    ]

    groups = _build_candidate_groups(items)

    assert [len(group) for group in groups] == [2, 1]
    assert {item["id"] for item in groups[0]} == {"a", "b"}


def test_node_filter_removes_noise_titles_before_source_cap(monkeypatch):
    items = [
        _item("a", "Pfizer posts phase 3 readout", 12, source="Endpoints News"),
        _item("b", "Webinar: better clinical trial enrollment", 11, source="FierceBiotech"),
        _item("c", "Opinion: the future of hospital billing", 10, source="STAT Biotech"),
        _item("d", "Roche drug fails key breast cancer study", 9, source="Endpoints News"),
    ]

    monkeypatch.setattr(graph, "MAX_ITEMS_PER_SOURCE", 1)
    result = node_filter({"items": items})

    assert [item["id"] for item in result["items"]] == ["a"]


def test_node_categorize_uses_structured_llm_response(monkeypatch):
    items = [
        _item(
            "a",
            "Pfizer announces phase 3 oncology trial results",
            12,
            source="Pfizer",
            summary="Official release for the phase 3 oncology study",
        ),
        _item(
            "b",
            "Pfizer reports phase 3 oncology trial data",
            11,
            source="Endpoints News",
            summary="Coverage of the same phase 3 oncology trial",
        ),
        _item(
            "c",
            "Hospital cafeterias debate protein shake trends",
            10,
            source="Lifestyle Weekly",
        ),
    ]
    response = {
        "groups": [
            {
                "group_id": "g1",
                "off_topic_ids": [],
                "clusters": [
                    {
                        "keep_id": "g1i1",
                        "duplicate_ids": ["g1i2"],
                        "category": "Clinical & Research",
                        "short_title": "Pfizer posts oncology trial results",
                    }
                ],
            },
            {
                "group_id": "g2",
                "off_topic_ids": ["g2i1"],
                "clusters": [],
            },
        ]
    }

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(graph, "_get_openai_client", lambda _api_key: object())
    monkeypatch.setattr(graph, "_chat_completion_text", lambda _client, _prompt: json.dumps(response))

    result = node_categorize({"items": items})

    assert len(result["items"]) == 1
    assert result["items"][0]["title"] == "Pfizer posts oncology trial results"
    assert result["items"][0]["category"] == "Clinical & Research"
    assert result["items"][0]["original_title"] == "Pfizer announces phase 3 oncology trial results"


def test_node_categorize_without_api_key_uses_local_resolution(monkeypatch):
    items = [
        _item("a", "Pfizer announces phase 3 oncology trial results", 12, source="Pfizer"),
        _item("b", "Pfizer announces phase 3 oncology trial data", 11, source="Endpoints News"),
    ]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(graph, "CONFIG_OPENAI_API_KEY", "")

    result = node_categorize({"items": items})

    assert len(result["items"]) == 1
    assert result["items"][0]["title"] == "Pfizer announces phase 3 oncology trial results"


def test_build_graph_renders_empty_digest_when_collection_is_empty(tmp_path, monkeypatch):
    output_file = tmp_path / "news.md"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(graph, "CONFIG_OPENAI_API_KEY", "")
    monkeypatch.setattr(graph, "_NEWS_FILE", output_file)
    monkeypatch.setattr(graph, "collect_items", lambda: [])

    result = build_graph().invoke({})

    assert result["items"] == []
    assert result["markdown"] == "_No fresh biotech/pharma headlines in the last 24 h._"
    assert output_file.read_text(encoding="utf-8") == result["markdown"]


def test_node_render_writes_to_configured_output(tmp_path, monkeypatch):
    output_file = tmp_path / "news.md"
    monkeypatch.setattr(graph, "_NEWS_FILE", output_file)
    state = {
        "items": [
            {
                "title": "Title",
                "link": "https://example.com",
                "source": "Example",
                "category": "Company News",
                "published": datetime(2026, 1, 1, tzinfo=timezone.utc),
            }
        ]
    }

    result = node_render(state)
    assert output_file.exists()
    assert result["markdown"]
