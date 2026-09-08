"""Editorial fields retain source titles and count distinct publishers."""

import json
from datetime import datetime, timedelta, timezone

import graph
import publisher
import pytest
from decision_contract import build_candidate_envelope
from ranking import render_sort_key, select_top_story_ids, story_rank_key
from renderer import to_markdown


SOURCE_TITLE = "Original source headline has more than eight words for readers"


def _item(item_id, source="Endpoints News", *, age=0, **overrides):
    return {
        "id": item_id,
        "title": SOURCE_TITLE,
        "original_title": SOURCE_TITLE,
        "link": f"https://example.com/{item_id}",
        "source": source,
        "published": datetime(2026, 1, 1, 12, tzinfo=timezone.utc) - timedelta(minutes=age),
        "summary": "A clinical trial reported positive results.",
        "category": "Clinical & Research",
        "source_type": "news",
        "source_role": "independent_reporting",
        "feed_mode": "core",
        **overrides,
    }


def _cluster(group_index, count, **fields):
    return {
        "group_id": f"g{group_index}",
        "off_topic_ids": [],
        "clusters": [{
            "keep_id": f"g{group_index}i1",
            "duplicate_ids": [f"g{group_index}i{index}" for index in range(2, count + 1)],
            "category": "Clinical & Research",
            **fields,
        }],
    }


def _daily_apply(tmp_path, monkeypatch, groups, response_groups, **fields):
    snapshot = build_candidate_envelope(
        kind="bio-news-agent.candidates",
        categories=list(graph.CATEGORIES),
        groups=graph._build_candidate_snapshot_groups(groups),
    )
    candidates_file = tmp_path / "digest-candidates.json"
    decisions_file = tmp_path / "digest-decisions.json"
    candidates_file.write_text(json.dumps(snapshot), encoding="utf-8")
    decisions_file.write_text(json.dumps({
        "schema_version": 2,
        "kind": "bio-news-agent.decisions",
        "snapshot_id": snapshot["snapshot_id"],
        "groups": response_groups,
        **fields,
    }), encoding="utf-8")
    output_file = tmp_path / "news.md"
    monkeypatch.setattr(graph, "_NEWS_FILE", output_file)
    result = graph.apply_decisions_file(decisions_file, candidates_file)
    assert output_file.read_text(encoding="utf-8") == result["markdown"]
    return result


@pytest.mark.parametrize("path", ["daily", "api"])
@pytest.mark.parametrize(("fields", "expected_title"), [
    pytest.param({}, SOURCE_TITLE, id="missing"),
    pytest.param({"short_title": None}, SOURCE_TITLE, id="null"),
    pytest.param({"short_title": 42}, SOURCE_TITLE, id="number"),
    pytest.param({"short_title": False}, SOURCE_TITLE, id="boolean"),
    pytest.param({"short_title": ["fake"]}, SOURCE_TITLE, id="list"),
    pytest.param({"short_title": {"text": "fake"}}, SOURCE_TITLE, id="object"),
    pytest.param({"short_title": ""}, SOURCE_TITLE, id="empty"),
    pytest.param({"short_title": " \t\n "}, SOURCE_TITLE, id="whitespace"),
    pytest.param(
        {"short_title": " One  two\nthree four five six seven eight "},
        "One two three four five six seven eight", id="eight-words",
    ),
    pytest.param(
        {"short_title": "One two three four five six seven eight nine ten"},
        "One two three four five six seven eight", id="word-limit",
    ),
])
def test_short_titles_reach_digest_and_issue_title(
    tmp_path, monkeypatch, path, fields, expected_title,
):
    if path == "daily":
        result = _daily_apply(
            tmp_path, monkeypatch, [[_item("a")]], [_cluster(1, 1, **fields)],
        )
    else:
        monkeypatch.setattr(graph, "_chat_completion_text", lambda client, prompt: json.dumps({
            "items": [{"item_id": "g1i1", "category": "Clinical & Research", **fields}],
        }))
        result = graph._enrich_resolved_items(
            {}, object(), [graph._seed_resolved_item(_item("a"), "g1i1")], skipped_items=0,
        )
        monkeypatch.setattr(graph, "_NEWS_FILE", tmp_path / "news.md")
        result = graph.node_render(result)

    monkeypatch.delenv("DIGEST_ISSUE_TITLE_OVERRIDE", raising=False)
    monkeypatch.setattr(publisher, "DIGEST_ISSUE_TITLE_PREFIX", "Bio News")
    monkeypatch.setattr(
        publisher, "_utcnow", lambda: datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
    )
    assert result["items"][0]["title"] == expected_title
    assert result["items"][0]["original_title"] == SOURCE_TITLE
    assert f"**[{expected_title}](https://example.com/a)**" in result["markdown"]
    assert publisher._issue_title_for_body(result["markdown"]) == f"Bio News - Jan 1: {expected_title}"


@pytest.mark.parametrize("path", ["heuristic", "api", "daily"])
@pytest.mark.parametrize(("keep_source", "sources", "expected_sources"), [
    pytest.param("Endpoints News", [], [], id="no-duplicates"),
    pytest.param("Endpoints News", ["", " \t "], [], id="empty-labels"),
    pytest.param("Endpoints News", ["Endpoints News"], [], id="same-source"),
    pytest.param(" Endpoints  News ", [" endpoints\nNEWS "], [], id="kept-source-variants"),
    pytest.param("Straße News", ["STRASSE NEWS"], [], id="casefold"),
    pytest.param(
        "Endpoints News", ["STAT Biotech", "Fierce Biotech"],
        ["STAT Biotech", "Fierce Biotech"], id="distinct-sources",
    ),
    pytest.param("Endpoints News", ["STAT Biotech", "STAT Biotech"], ["STAT Biotech"], id="repeated"),
    pytest.param(
        "Endpoints News",
        [" endpoints  NEWS ", " STAT\tBiotech ", "stat biotech", "", " FIERCE  Biotech", "fierce biotech"],
        ["STAT Biotech", "FIERCE Biotech"], id="mixed-variants",
    ),
])
def test_all_resolution_paths_store_additional_distinct_sources(
    tmp_path, monkeypatch, path, keep_source, sources, expected_sources,
):
    group = [_item("keep", keep_source)] + [
        _item(f"duplicate-{index}", source, age=index)
        for index, source in enumerate(sources, start=1)
    ]
    response_group = _cluster(1, len(group))
    if path == "heuristic":
        items, skipped = graph._fallback_resolve_groups([group])
        assert skipped == len(sources)
    elif path == "api":
        items, skipped = graph._apply_dedupe_response([(1, group)], {"groups": [response_group]})
        assert skipped == len(sources)
    else:
        items = _daily_apply(tmp_path, monkeypatch, [group], [response_group])["items"]

    assert len(items) == 1
    assert items[0]["coverage_sources"] == expected_sources


@pytest.mark.parametrize("path", ["api", "daily"])
def test_coverage_excludes_promoted_core_source(tmp_path, monkeypatch, path):
    group = [
        _item("discovery", " STAT  Biotech ", feed_mode="discovery_only"),
        _item("core", "Endpoints News", age=1),
        _item("core-copy", " endpoints\tNEWS ", age=2),
        _item("discovery-copy", "stat biotech", age=3, feed_mode="discovery_only"),
        _item("independent", " Fierce\nBiotech ", age=4),
    ]
    response_group = _cluster(1, len(group))
    if path == "daily":
        result = _daily_apply(
            tmp_path, monkeypatch, [group], [response_group], top_stories=["g1i1"],
        )
        assert result["top_stories"] == ["g1i2"]
        items = result["items"]
    else:
        items, skipped = graph._apply_dedupe_response([(1, group)], {"groups": [response_group]})
        assert skipped == 4

    assert len(items) == 1
    assert items[0]["id"] == "core"
    assert items[0]["coverage_sources"] == ["Fierce Biotech", "STAT Biotech"]
    assert "(3 sources)" in to_markdown(items)


def test_repeated_api_duplicate_ids_are_rejected_before_coverage_or_skip_count():
    group = [_item("keep"), _item("duplicate", "STAT Biotech", age=1)]
    response_group = _cluster(1, 2)
    response_group["clusters"][0]["duplicate_ids"] = ["g1i2", "g1i2"]

    with pytest.raises(ValueError):
        graph._apply_dedupe_response([(1, group)], {"groups": [response_group]})


def test_daily_coverage_controls_ranking_rendering_and_api_enrichment(tmp_path, monkeypatch):
    groups = [
        [_item("same"), _item("same-copy", "Endpoints News", age=1),
         _item("same-case", "endpoints news", age=2), _item("same-space", "Endpoints  News", age=3)],
        [_item("broad"), _item("broad-stat", "STAT Biotech", age=1),
         _item("broad-fierce", "Fierce Biotech", age=2)],
    ]
    result = _daily_apply(
        tmp_path, monkeypatch, groups,
        [_cluster(1, 4, short_title="Same source story"), _cluster(2, 3, short_title="Broad coverage story")],
    )
    items = result["items"]
    assert [item["id"] for item in sorted(items, key=story_rank_key)] == ["broad", "same"]
    assert [item["id"] for item in sorted(items, key=render_sort_key)] == ["broad", "same"]
    assert result["top_stories"] == ["g2i1", "g1i1"]
    assert select_top_story_ids(items, []) == ["g2i1", "g1i1"]
    category_markdown = to_markdown(items)
    assert category_markdown.index("Broad coverage story") < category_markdown.index("Same source story")
    assert "**[Broad coverage story](https://example.com/broad)** — Endpoints News (3 sources)" in result["markdown"]
    same_line = next(line for line in category_markdown.splitlines() if "Same source story" in line)
    assert same_line.endswith("— Endpoints News")

    enrichment_inputs = []

    def respond(client, prompt):
        enrichment_inputs.extend(json.loads(prompt.split("Input JSON:\n", 1)[1])["items"])
        return json.dumps({"items": [{"item_id": "g1i1"}, {"item_id": "g2i1"}]})

    monkeypatch.setattr(graph, "_chat_completion_text", respond)
    enriched = graph._enrich_resolved_items({}, object(), items, skipped_items=0)
    assert {item["item_id"]: item["coverage_count"] for item in enrichment_inputs} == {"g1i1": 1, "g2i1": 3}
    assert enriched["top_stories"] == ["g2i1", "g1i1"]
