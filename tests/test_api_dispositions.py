"""API disposition integrity; no live credentials, feeds, or model calls."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import logging

import pytest

import graph


_TITLE = "Pfizer announces phase 3 oncology trial results"


def _item(index, *, feed_mode="core"):
    return {
        "id": f"article-{index}",
        "title": _TITLE,
        "original_title": _TITLE,
        "link": f"https://example.com/article-{index}",
        "source": f"Wire {index}",
        "published": datetime(2026, 9, 7, 12 - index, tzinfo=timezone.utc),
        "category": graph.CATEGORIES[0],
        "summary": "Original report summary.",
        "source_type": "news",
        "source_role": "independent_reporting",
        "feed_mode": feed_mode,
    }


@pytest.fixture(autouse=True)
def _isolate_external_state(monkeypatch, tmp_path):
    monkeypatch.setattr(graph, "_get_openai_api_key", lambda: "test-key")
    monkeypatch.setattr(graph, "_get_dotenv_openai_api_key", lambda: "")
    monkeypatch.setattr(graph, "_get_openai_client", lambda _key: object())
    monkeypatch.setattr(graph, "_NEWS_FILE", tmp_path / "news.md")

    def unexpected_call(*_args):
        raise AssertionError("Test attempted an unconfigured API call")

    monkeypatch.setattr(graph, "_chat_completion_text", unexpected_call)


def _group(clusters):
    return {"group_id": "g1", "clusters": clusters}


def _cluster(keep="g1i1", duplicates=None):
    return {"keep_id": keep, "duplicate_ids": ["g1i2"] if duplicates is None else duplicates}


@pytest.mark.parametrize("groups", [
    pytest.param(None, id="groups-not-list"),
    pytest.param([], id="missing-group"),
    pytest.param([None], id="group-not-object"),
    pytest.param([{"group_id": "g1"}], id="missing-clusters"),
    pytest.param([_group([])], id="empty-clusters"),
    pytest.param([_group(None)], id="clusters-not-list"),
    pytest.param([_group([None])], id="cluster-not-object"),
    pytest.param([_group([{"keep_id": "g1i1"}])], id="missing-duplicate-list"),
    pytest.param([_group([_cluster(duplicates="g1i2")])], id="duplicates-not-list"),
    pytest.param([_group([_cluster(duplicates=[])])], id="omitted-candidate"),
    pytest.param([_group([_cluster(duplicates=["g1i2", "g1i2"])])], id="repeated-duplicate-id"),
    pytest.param([_group([_cluster(duplicates=["g1i1", "g1i2"])])], id="keep-is-duplicate"),
    pytest.param([_group([_cluster(), _cluster("g1i2", [])])], id="cross-cluster-reuse"),
    pytest.param([_group([_cluster("unknown")])], id="unknown-keep"),
    pytest.param([_group([_cluster(duplicates=["g1i2", "unknown"])])], id="unknown-duplicate"),
    pytest.param([_group([_cluster(1)])], id="keep-not-string"),
    pytest.param([_group([_cluster(duplicates=["g1i2", None])])], id="duplicate-not-string"),
    pytest.param([_group([_cluster(" g1i1")])], id="noncanonical-keep"),
    pytest.param([_group([_cluster()]), _group([_cluster()])], id="repeated-group"),
    pytest.param([_group([_cluster()]), {"group_id": "unknown", "clusters": []}], id="unknown-group"),
    pytest.param([{"group_id": 1, "clusters": [_cluster()]}], id="group-not-string"),
    pytest.param([{"group_id": "g1", "clusters": [], "off_topic_ids": ["g1i1", "g1i2"]}], id="off-topic-in-dedupe"),
    pytest.param([{"group_id": "g1", "clusters": [_cluster()], "off_topic_ids": None}], id="off-topic-not-list"),
])
def test_dedupe_rejects_invalid_dispositions_without_mutation(groups):
    indexed = [(1, [_item(0), _item(1)])]
    before = deepcopy(indexed)

    with pytest.raises(ValueError):
        graph._apply_dedupe_response(indexed, {"groups": groups})

    assert indexed == before


def test_dedupe_validates_later_groups_before_promotion_or_seeding(monkeypatch):
    indexed = [(1, [_item(0, feed_mode="discovery_only"), _item(1)]),
               (3, [_item(2), _item(3)])]
    response = {"groups": [_group([_cluster()]),
                           {"group_id": "g3", "clusters": [
                               {"keep_id": "g3i1", "duplicate_ids": ["g1i2", "g3i2"]}
                           ]}]}
    before = deepcopy(indexed)
    promotions = []
    original_promote = graph._promote_renderable_keep_id

    def promote(*args):
        promotions.append(args[1])
        return original_promote(*args)

    monkeypatch.setattr(graph, "_promote_renderable_keep_id", promote)
    with pytest.raises(ValueError):
        graph._apply_dedupe_response(indexed, response)

    assert promotions == []
    assert indexed == before


@pytest.mark.parametrize("response", [
    pytest.param({}, id="omitted-items"),
    pytest.param({"items": []}, id="empty-items"),
    pytest.param({"items": None}, id="items-not-list"),
    pytest.param({"items": [None]}, id="item-not-object"),
    pytest.param({"items": [{}]}, id="missing-item-id"),
    pytest.param({"items": [{"item_id": 1}]}, id="item-id-not-string"),
    pytest.param({"items": [{"item_id": " g1i1"}, {"item_id": "g1i2"}]}, id="noncanonical-item-id"),
    pytest.param({"items": [{"item_id": "g1i1"}]}, id="omitted-candidate"),
    pytest.param({"items": [{"item_id": "g1i1"}, {"item_id": "g1i2"}, {"item_id": "g1i2"}]}, id="repeated-item"),
    pytest.param({"items": [{"item_id": "g1i1"}, {"item_id": "g1i2"}, {"item_id": "unknown"}]}, id="unknown-item"),
    pytest.param({"items": [{"item_id": "g1i1"}, {"item_id": "g1i2"}, None]}, id="extra-malformed-item"),
    pytest.param({"items": [{"item_id": "g1i1"}, {"item_id": "g1i2"}], "off_topic_ids": ["g1i2"]}, id="item-off-topic-conflict"),
    pytest.param({"items": [{"item_id": "g1i1"}], "off_topic_ids": ["g1i2", "g1i2"]}, id="repeated-off-topic"),
    pytest.param({"items": [{"item_id": "g1i1"}, {"item_id": "g1i2"}], "off_topic_ids": ["unknown"]}, id="unknown-off-topic"),
    pytest.param({"items": [{"item_id": "g1i1"}, {"item_id": "g1i2"}], "off_topic_ids": [None]}, id="off-topic-id-not-string"),
    pytest.param({"items": [{"item_id": "g1i1"}, {"item_id": "g1i2"}], "off_topic_ids": None}, id="off-topic-not-list"),
])
def test_enrichment_rejects_invalid_dispositions_without_mutation(response):
    items = [graph._seed_resolved_item(_item(i), f"g1i{i + 1}") for i in range(2)]
    state = {"items": items, "executive_summary": "Original overview", "top_stories": ["g1i1"]}
    before = deepcopy(state)

    with pytest.raises(ValueError):
        graph._apply_enrichment_response(
            state, items, response, skipped_items=0, log_label="test",
        )

    assert state == before
    assert items == before["items"]


def test_enrichment_validates_later_items_before_applying_earlier_fields():
    items = [graph._seed_resolved_item(_item(i), f"g1i{i + 1}") for i in range(2)]
    before = deepcopy(items)
    response = {"items": [{"item_id": "g1i1", "short_title": "API replacement"}]}
    with pytest.raises(ValueError):
        graph._apply_enrichment_response({}, items, response, skipped_items=0, log_label="test")

    assert items == before


def test_valid_noncontiguous_groups_singletons_and_promotion_preserve_accounting(caplog):
    indexed = [(1, [_item(0, feed_mode="discovery_only"), _item(1), _item(2)]),
               (3, [_item(3, feed_mode="discovery_only")])]
    response = {"groups": [
        {"group_id": "g3", "clusters": [{"keep_id": "g3i1", "duplicate_ids": []}]},
        {"group_id": "g1", "clusters": [_cluster(), _cluster("g1i3", [])], "off_topic_ids": []},
    ]}
    items, skipped = graph._apply_dedupe_response(indexed, response)

    assert [item["_prompt_id"] for item in items] == ["g1i2", "g1i3", "g3i1"]
    assert skipped == 1
    assert items[0]["coverage_sources"] == ["Wire 0"]

    with caplog.at_level(logging.INFO):
        result = graph._apply_enrichment_response(
            {}, items, {"items": [{"item_id": "g1i2"}, {"item_id": "g1i3"}, {"item_id": "g3i1"}]},
            skipped_items=skipped, log_label="integrity",
        )
    assert [item["_prompt_id"] for item in result["items"]] == ["g1i2", "g1i3"]
    assert result["items"][0]["title"] == _TITLE
    assert result["items"][0]["summary_line"] == "Original report summary."
    assert "integrity: 2 items (skipped 2 items)" in caplog.text


def test_valid_duplicate_and_off_topic_counts_accumulate_once(caplog):
    indexed = [(1, [_item(i) for i in range(5)])]
    response = {"groups": [_group([
        _cluster(duplicates=["g1i2", "g1i3"]), _cluster("g1i4", ["g1i5"]),
    ])]}
    items, skipped = graph._apply_dedupe_response(indexed, response)
    assert skipped == 3
    assert [item["_prompt_id"] for item in items] == ["g1i1", "g1i4"]

    with caplog.at_level(logging.INFO):
        result = graph._apply_enrichment_response(
            {}, items, {"items": [{"item_id": "g1i1"}], "off_topic_ids": ["g1i4"]},
            skipped_items=skipped, log_label="integrity",
        )
    assert [item["_prompt_id"] for item in result["items"]] == ["g1i1"]
    assert "integrity: 1 items (skipped 4 items)" in caplog.text


def test_enrichment_all_off_topic_is_valid(caplog):
    items = [graph._seed_resolved_item(_item(i), f"g1i{i + 1}") for i in range(2)]
    with caplog.at_level(logging.INFO):
        result = graph._apply_enrichment_response(
            {}, items, {"items": [], "off_topic_ids": ["g1i1", "g1i2"]},
            skipped_items=3, log_label="integrity",
        )
    assert result["items"] == []
    assert "integrity: 0 items (skipped 5 items)" in caplog.text


@pytest.mark.parametrize("case", ["dedupe-omitted", "dedupe-repeated", "enrichment-conflict", "enrichment-omitted"])
def test_malformed_api_response_uses_existing_local_fallback(case, monkeypatch, tmp_path, caplog):
    items = [_item(0), _item(1)]
    valid_dedupe = {"groups": [_group([_cluster()])]}
    if case == "dedupe-omitted":
        responses = [
            {"groups": [_group([])]},
            {"items": [{"item_id": "g1i1", "short_title": "API replacement"},
                       {"item_id": "g1i2", "short_title": "API replacement"}]},
        ]
    elif case == "dedupe-repeated":
        responses = [
            {"groups": [_group([_cluster(duplicates=["g1i2", "g1i2"])])]},
            {"items": [{"item_id": "g1i1", "short_title": "API replacement"}]},
        ]
    elif case == "enrichment-conflict":
        responses = [valid_dedupe, {
            "items": [{"item_id": "g1i1", "short_title": "API replacement"}],
            "off_topic_ids": ["g1i1"],
        }]
    else:
        responses = [valid_dedupe, {"items": []}]

    calls = []

    def respond(_client, prompt):
        calls.append(prompt)
        return json.dumps(responses[len(calls) - 1])

    monkeypatch.setattr(graph, "_chat_completion_text", respond)
    monkeypatch.setattr(graph, "node_collect", lambda _state: {"items": items})
    monkeypatch.setattr(graph, "node_filter", lambda state: state)
    with caplog.at_level(logging.INFO):
        result = graph.build_graph().invoke({"items": []})
    assert len(result["items"]) == 1
    assert result["items"][0]["title"] == _TITLE
    assert result["items"][0]["coverage_sources"] == ["Wire 1"]
    assert "After local categorization: 1 items (skipped 1 items)" in caplog.text
    assert "API replacement" not in result["markdown"]
    assert (tmp_path / "news.md").read_text(encoding="utf-8") == result["markdown"]
    assert len(calls) == (1 if case.startswith("dedupe-") else 2)
