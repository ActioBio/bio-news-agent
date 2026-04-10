"""Tests for graph module."""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import graph
from graph import (
    _build_candidate_groups,
    _is_high_confidence_duplicate,
    _keyword_categorize,
    _should_retry_openai_error,
    apply_decisions_file,
    build_graph,
    build_candidate_snapshot,
    node_categorize,
    node_filter,
    node_render,
)
from openai import APIConnectionError
from openai import APITimeoutError


_FIXTURES_DIR = Path(__file__).with_name("fixtures")


def _load_fixture(name: str) -> dict[str, object]:
    return json.loads((_FIXTURES_DIR / name).read_text(encoding="utf-8"))


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


def test_build_candidate_snapshot_preserves_group_ids():
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
    ]

    snapshot = build_candidate_snapshot(items)

    assert snapshot["kind"] == "bio-news-agent.candidates"
    assert snapshot["groups"][0]["group_id"] == "g1"
    assert [item["item_id"] for item in snapshot["groups"][0]["items"]] == ["g1i1", "g1i2"]
    assert snapshot["groups"][0]["items"][0]["link"] == "https://example.com/a"


def test_build_candidate_snapshot_matches_contract_fixture():
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

    snapshot = build_candidate_snapshot(items)

    assert snapshot == _load_fixture("candidate_snapshot.json")


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
        "executive_summary": "Pfizer posted positive oncology data.",
        "top_stories": ["g1i1"],
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
                        "summary_line": "Phase 3 data could advance a new treatment.",
                        "tier": "high",
                    }
                ],
            },
            {
                "group_id": "g2",
                "off_topic_ids": ["g2i1"],
                "clusters": [],
            },
        ],
    }

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(graph, "_get_openai_client", lambda _api_key: object())
    monkeypatch.setattr(graph, "_chat_completion_text", lambda _client, _prompt: json.dumps(response))

    result = node_categorize({"items": items})

    assert len(result["items"]) == 1
    assert result["items"][0]["title"] == "Pfizer posts oncology trial results"
    assert result["items"][0]["category"] == "Clinical & Research"
    assert result["items"][0]["original_title"] == "Pfizer announces phase 3 oncology trial results"
    assert result["items"][0]["summary_line"] == "Phase 3 data could advance a new treatment."
    assert result["items"][0]["tier"] == "high"
    assert result["items"][0]["coverage_sources"] == ["Endpoints News"]
    assert result["executive_summary"] == "Pfizer posted positive oncology data."
    assert result["top_stories"] == ["g1i1"]


def test_node_categorize_without_api_key_uses_local_resolution(monkeypatch):
    items = [
        _item("a", "Pfizer announces phase 3 oncology trial results", 12, source="Pfizer"),
        _item("b", "Pfizer announces phase 3 oncology trial data", 11, source="Endpoints News"),
    ]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(graph, "CONFIG_OPENAI_API_KEY", "")
    monkeypatch.setattr(graph, "_get_dotenv_openai_api_key", lambda: "")

    result = node_categorize({"items": items})

    assert len(result["items"]) == 1
    assert result["items"][0]["title"] == "Pfizer announces phase 3 oncology trial results"
    assert result["items"][0]["summary_line"] == ""
    assert result["items"][0]["tier"] == "normal"
    assert result["items"][0]["coverage_sources"] == ["Endpoints News"]
    assert result.get("executive_summary") == ""
    assert result.get("top_stories") == ["g1i1"]


def test_node_categorize_without_api_key_uses_duplicate_summary_when_primary_is_blank(monkeypatch):
    items = [
        _item("a", "Pfizer announces phase 3 oncology trial results", 12, source="Pfizer"),
        _item(
            "b",
            "Pfizer announces phase 3 oncology trial data",
            11,
            source="Endpoints News",
            summary="Independent reporting explains why the phase 3 readout matters. Extra detail follows.",
        ),
    ]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(graph, "CONFIG_OPENAI_API_KEY", "")
    monkeypatch.setattr(graph, "_get_dotenv_openai_api_key", lambda: "")

    result = node_categorize({"items": items})

    assert result["items"][0]["summary_line"] == "Independent reporting explains why the phase 3 readout matters."
    assert result["items"][0]["coverage_sources"] == ["Endpoints News"]


def test_node_categorize_uses_dotenv_api_key_when_env_is_placeholder(monkeypatch):
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
            }
        ]
    }
    captured: dict[str, str] = {}

    monkeypatch.setenv("OPENAI_API_KEY", "your_api_key_here")
    monkeypatch.setattr(graph, "CONFIG_OPENAI_API_KEY", "")
    monkeypatch.setattr(graph, "_get_dotenv_openai_api_key", lambda: "test-key")

    def fake_get_openai_client(api_key: str) -> object:
        captured["api_key"] = api_key
        return object()

    monkeypatch.setattr(graph, "_get_openai_client", fake_get_openai_client)
    monkeypatch.setattr(graph, "_chat_completion_text", lambda _client, _prompt: json.dumps(response))

    result = node_categorize({"items": items})

    assert captured["api_key"] == "test-key"
    assert len(result["items"]) == 1
    assert result["items"][0]["category"] == "Clinical & Research"


def test_node_categorize_timeout_uses_local_resolution(monkeypatch):
    items = [
        _item("a", "Pfizer announces phase 3 oncology trial results", 12, source="Pfizer"),
        _item("b", "Pfizer announces phase 3 oncology trial data", 11, source="Endpoints News"),
    ]

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(graph, "_get_openai_client", lambda _api_key: object())

    def raise_timeout(_client, _prompt):
        raise APITimeoutError(
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        )

    monkeypatch.setattr(graph, "_chat_completion_text", raise_timeout)

    result = node_categorize({"items": items})

    assert len(result["items"]) == 1
    assert result["items"][0]["title"] == "Pfizer announces phase 3 oncology trial results"


def test_apply_decisions_file_renders_from_candidate_snapshot(tmp_path, monkeypatch):
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
    candidates_file = tmp_path / "digest-candidates.json"
    decisions_file = tmp_path / "digest-decisions.json"
    output_file = tmp_path / "news.md"

    candidates_file.write_text(
        json.dumps(build_candidate_snapshot(items)),
        encoding="utf-8",
    )
    decisions_file.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(graph, "_NEWS_FILE", output_file)

    result = apply_decisions_file(decisions_file, candidates_file)

    assert len(result["items"]) == 1
    assert result["items"][0]["title"] == "Pfizer posts oncology trial results"
    assert output_file.read_text(encoding="utf-8") == result["markdown"]


def test_apply_decisions_file_matches_contract_fixtures(tmp_path, monkeypatch):
    candidates_file = tmp_path / "digest-candidates.json"
    decisions_file = tmp_path / "digest-decisions.json"
    output_file = tmp_path / "news.md"

    candidates_file.write_text(
        json.dumps(_load_fixture("candidate_snapshot.json")),
        encoding="utf-8",
    )
    decisions_file.write_text(
        json.dumps(_load_fixture("decisions.json")),
        encoding="utf-8",
    )
    monkeypatch.setattr(graph, "_NEWS_FILE", output_file)

    result = apply_decisions_file(decisions_file, candidates_file)

    assert len(result["items"]) == 1
    assert result["items"][0]["title"] == "Pfizer posts oncology trial results"
    assert result["items"][0]["category"] == "Clinical & Research"
    assert result["items"][0]["summary_line"] == "Phase 3 data could advance a new oncology treatment toward approval."
    assert result["items"][0]["tier"] == "high"
    assert result["items"][0]["coverage_sources"] == ["Endpoints News"]
    assert result["executive_summary"] == "Pfizer posted positive phase 3 oncology data as biotech digest filtering removed off-topic content."
    assert result["top_stories"] == ["g1i1"]
    assert output_file.read_text(encoding="utf-8") == result["markdown"]


def test_apply_decisions_backward_compat_old_format(tmp_path, monkeypatch):
    """Old-format decisions without new fields should still work with defaults."""
    candidates_file = tmp_path / "digest-candidates.json"
    decisions_file = tmp_path / "digest-decisions.json"
    output_file = tmp_path / "news.md"

    candidates_file.write_text(
        json.dumps(_load_fixture("candidate_snapshot.json")),
        encoding="utf-8",
    )
    decisions_file.write_text(
        json.dumps(
            {
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
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(graph, "_NEWS_FILE", output_file)

    result = apply_decisions_file(decisions_file, candidates_file)

    assert result["items"][0]["summary_line"] == "Official release for the phase 3 oncology study"
    assert result["items"][0]["tier"] == "normal"
    assert result["items"][0]["coverage_sources"] == ["Endpoints News"]
    assert result.get("executive_summary") == ""
    assert result.get("top_stories") == ["g1i1", "g2i1"]


def test_apply_structured_response_uses_duplicate_summary_when_keep_summary_is_blank():
    state = {"items": [], "executive_summary": "", "top_stories": []}
    groups = [[
        _item("a", "Pfizer announces phase 3 oncology trial results", 12, source="Pfizer"),
        _item(
            "b",
            "Pfizer announces phase 3 oncology trial data",
            11,
            source="Endpoints News",
            summary="Independent reporting explains why the phase 3 readout matters. Extra detail follows.",
        ),
    ]]
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
            }
        ]
    }

    result = graph._apply_structured_response(state, groups, response, log_label="test")

    assert result["items"][0]["summary_line"] == "Independent reporting explains why the phase 3 readout matters."
    assert result["items"][0]["coverage_sources"] == ["Endpoints News"]


def test_should_retry_openai_error_retries_timeout():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    assert _should_retry_openai_error(APITimeoutError(request=request))
    assert _should_retry_openai_error(APIConnectionError(request=request))


def test_clean_summary_line_preserves_abbreviations_and_versions():
    assert (
        graph._clean_summary_line("U.S. regulators approved the therapy after review.")
        == "U.S. regulators approved the therapy after review."
    )
    assert (
        graph._clean_summary_line("Version 2.1 ships today with better coding support.")
        == "Version 2.1 ships today with better coding support."
    )
    assert (
        graph._clean_summary_line("OpenAI Inc. Launches a new coding assistant for teams.")
        == "OpenAI Inc. Launches a new coding assistant for teams."
    )
    assert (
        graph._clean_summary_line("The board met at Acme Co. Headquarters before the vote.")
        == "The board met at Acme Co. Headquarters before the vote."
    )


def test_clean_summary_line_keeps_only_first_sentence():
    assert (
        graph._clean_summary_line("No major policy change yet. Markets are watching.")
        == "No major policy change yet."
    )
    assert (
        graph._clean_summary_line("This matters. here is a lowercase second sentence.")
        == "This matters."
    )


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
