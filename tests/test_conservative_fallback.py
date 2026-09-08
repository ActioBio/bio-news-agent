"""Local deletion requires an exact normalized original headline within a group."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import logging

import graph
import pytest


def _item(item_id, title, *, age=0, **overrides):
    return {
        "id": item_id,
        "title": title,
        "original_title": title,
        "link": f"https://example.com/{item_id}",
        "source": "Primary Wire",
        "published": datetime(2026, 9, 8, 12, tzinfo=timezone.utc) - timedelta(minutes=age),
        "summary": "",
        "category": "Clinical & Research",
        "source_type": "news",
        "source_role": "primary",
        "feed_mode": "core",
        **overrides,
    }


@pytest.mark.parametrize(("first", "second"), [
    pytest.param("Pfizer oncology phase 2 trial meets survival endpoint",
                 "Pfizer oncology phase 3 trial meets survival endpoint", id="phase-number"),
    pytest.param("Pfizer oncology platform 1.2 launches clinical analysis",
                 "Pfizer oncology platform 1.3 launches clinical analysis", id="version-number"),
    pytest.param("Pfizer oncology trial reports melanoma survival benefit",
                 "Pfizer oncology trial reports lymphoma survival benefit", id="indication"),
    pytest.param("Pfizer oncology trial meets primary survival endpoint",
                 "Pfizer oncology trial does not meet primary survival endpoint", id="negation"),
    pytest.param("Pfizer Orion-1 oncology trial meets survival endpoint",
                 "Pfizer Orion1 oncology trial meets survival endpoint", id="punctuation"),
    pytest.param("Pfizer oncology trial compares pembrolizumab against nivolumab",
                 "Pfizer oncology trial compares nivolumab against pembrolizumab", id="word-order"),
])
def test_local_resolution_retains_distinct_original_headlines(first, second):
    group = [_item("a", first), _item("b", second, age=1)]

    items, skipped = graph._fallback_resolve_groups([group])

    assert [item["id"] for item in items] == ["a", "b"]
    assert [item["_prompt_id"] for item in items] == ["g1i1", "g1i2"]
    assert [item["coverage_sources"] for item in items] == [[], []]
    assert skipped == 0


def test_equal_display_titles_do_not_override_distinct_originals():
    group = [
        _item("a", "Pfizer oncology trial update",
              original_title="Pfizer oncology phase 2 trial meets survival endpoint"),
        _item("b", "Pfizer oncology trial update", age=1,
              original_title="Pfizer oncology phase 3 trial meets survival endpoint"),
    ]

    items, skipped = graph._fallback_resolve_groups([group])

    assert [item["id"] for item in items] == ["a", "b"]
    assert skipped == 0


def test_normalized_identical_originals_preserve_accounting_summary_and_sources():
    group = [
        _item("a", "Short display", original_title=" Straße oncology TRIAL 2: success! "),
        _item("b", "Different display", age=1,
              original_title="STRASSE\toncology\ntrial 2: success!", source=" Independent  Wire ",
              summary="The primary endpoint was met. Additional detail follows."),
        _item("c", "Third display", age=2,
              original_title="straße oncology trial 2: success!", source="independent\twire"),
        _item("d", "Fourth display", age=3,
              original_title="Straße oncology trial 2: success!", source=" primary\nWIRE "),
    ]

    items, skipped = graph._fallback_resolve_groups([group])

    assert [item["id"] for item in items] == ["a"]
    assert items[0]["_prompt_id"] == "g1i1"
    assert items[0]["summary_line"] == "The primary endpoint was met."
    assert items[0]["coverage_sources"] == ["Independent Wire"]
    assert skipped == 3


@pytest.mark.parametrize("original", [None, 42, False, ["malformed"], {"text": "malformed"}, "", " \t\n "])
def test_unusable_original_uses_usable_title_for_duplicate_key(original):
    group = [
        _item("a", "FDA approves Orion 2!", original_title=original),
        _item("b", " fda\tAPPROVES Orion 2! ", original_title=original, age=1),
    ]

    items, skipped = graph._fallback_resolve_groups([group])

    assert [item["id"] for item in items] == ["a"]
    assert skipped == 1


def test_missing_original_uses_usable_title_for_duplicate_key():
    group = [_item("a", "Trial update"), _item("b", " TRIAL\tUPDATE ", age=1)]
    for item in group:
        item.pop("original_title")

    items, skipped = graph._fallback_resolve_groups([group])

    assert [item["id"] for item in items] == ["a"]
    assert skipped == 1


@pytest.mark.parametrize("invalid", [None, 42, False, "", " \t\n ",
                                    ["Pfizer oncology clinical survival endpoint"],
                                    {"text": "Pfizer oncology clinical survival endpoint"}])
def test_unusable_titles_never_supply_a_duplicate_key(invalid):
    group = [_item("a", invalid), _item("b", invalid, age=1)]

    items, skipped = graph._fallback_resolve_groups([group])

    assert [item["id"] for item in items] == ["a", "b"]
    assert skipped == 0


def test_regular_and_broader_grouping_keep_their_existing_membership():
    items = [
        _item("a", "Pfizer oncology phase 2 trial meets survival endpoint"),
        _item("b", "Pfizer oncology phase 3 trial meets survival endpoint", age=1),
        _item("c", "Orion melanoma study update", age=2,
              summary="Acme pembrolizumab melanoma randomized endpoint"),
        _item("d", "Quarterly clinical research note", age=3,
              summary="Acme pembrolizumab melanoma randomized endpoint"),
    ]

    regular = graph._build_candidate_groups(deepcopy(items))
    broader = graph._build_fallback_candidate_groups(deepcopy(items))

    assert [[item["id"] for item in group] for group in regular] == [["a", "b"], ["c"], ["d"]]
    assert [[item["id"] for item in group] for group in broader] == [["a", "b"], ["c", "d"]]
    kept, skipped = graph._fallback_resolve_groups(broader)
    assert [item["id"] for item in kept] == ["a", "b", "c", "d"]
    assert skipped == 0


def test_matching_short_headlines_in_separate_groups_are_not_merged():
    items = [_item("a", "Trial update"), _item("b", "Trial update", age=1)]
    groups = graph._build_fallback_candidate_groups(items)

    assert [[item["id"] for item in group] for group in groups] == [["a"], ["b"]]
    kept, skipped = graph._fallback_resolve_groups(groups)
    assert [item["id"] for item in kept] == ["a", "b"]
    assert [item["_prompt_id"] for item in kept] == ["g1i1", "g2i1"]
    assert skipped == 0


@pytest.mark.parametrize("route", ["no-api-key", "attempted-api-error"])
def test_node_fallback_requires_original_title_equality(route, monkeypatch, caplog):
    items = [
        _item("a", "Pfizer oncology phase 2 trial meets survival endpoint"),
        _item("b", " PFIZER oncology\tphase 2 trial meets survival endpoint ", age=1,
              source="Independent Wire", source_role="independent_reporting",
              summary="The survival endpoint was met."),
        _item("c", "Pfizer oncology phase 3 trial meets survival endpoint", age=2),
        _item("d", "Orion melanoma study update", age=3,
              summary="Acme pembrolizumab melanoma randomized endpoint"),
        _item("e", "Quarterly clinical research note", age=4,
              summary="Acme pembrolizumab melanoma randomized endpoint"),
    ]
    monkeypatch.setattr(graph, "_get_openai_api_key", lambda: "" if route == "no-api-key" else "test-key")
    monkeypatch.setattr(graph, "_get_openai_client", lambda _key: object())
    attempted_prompts = []

    def fail_api(_client, prompt):
        attempted_prompts.append(prompt)
        raise RuntimeError("offline API failure")

    monkeypatch.setattr(graph, "_chat_completion_text", fail_api)

    with caplog.at_level(logging.INFO):
        result = graph.node_categorize({"items": items})

    assert len(attempted_prompts) == (0 if route == "no-api-key" else 1)
    assert [item["id"] for item in result["items"]] == ["a", "c", "d", "e"]
    assert [item["coverage_sources"] for item in result["items"]] == [["Independent Wire"], [], [], []]
    assert result["items"][0]["summary_line"] == "The survival endpoint was met."
    assert "4 items (skipped 1 items)" in caplog.text
    if route == "attempted-api-error":
        assert "offline API failure. Falling back to local duplicate resolution." in caplog.text
