"""URL hints propose comparisons; only explicit decisions remove coverage."""

import json
from copy import deepcopy
from datetime import datetime, timezone

import graph
import pytest


@pytest.fixture
def novartis_reports():
    """September 8 reports: two dystrophy reports and a distinct Lp(a) event."""
    reports = [
        {
            "id": "avidity",
            "title": "Novartis’ $12B Avidity acquisition hits Phase 3 speedbump as dystrophy drug disappoints",
            "summary": "While Novartis’ antibody-oligonucleotide conjugate failed to significantly improve hand function in patients with myotonic dystrophy type 1, the asset nevertheless showed signs of clinical activity in secondary and exploratory measures.",
            "link": "https://www.biospace.com/drug-development/novartis-12b-avidity-acquisition-hits-phase-3-speedbump-as-dystrophy-drug-disappoints",
            "source": "BioSpace Drug Development",
            "published": datetime(2026, 9, 8, 12, 5, 40, tzinfo=timezone.utc),
        },
        {
            "id": "lpa",
            "title": "Novartis’ cholesterol-lowering medicine fails Phase 3, clouding Lp(a) horizons",
            "summary": "The failure of Novartis’ pelacarsen to translate its Lp(a)-lowering effects to clinical benefits marks a meaningful setback for other companies taking a similar approach to cardiovascular risk, analysts say—including Eli Lilly and Amgen.",
            "link": "https://www.biospace.com/drug-development/novartis-cholesterol-lowering-medicine-fails-phase-3-clouding-lpa-horizons",
            "source": "BioSpace Drug Development",
            "published": datetime(2026, 9, 8, 11, 56, 16, tzinfo=timezone.utc),
        },
        {
            "id": "stat",
            "title": "STAT+: Neuromuscular drug from Novartis fails in key study, adding to pressure on company",
            "summary": "A Novartis drug for a rare neuromuscular condition failed in a key study, the company's second major trial collapse in a matter of days.",
            "link": "https://www.statnews.com/2026/09/08/novartis-del-desiran-myotonic-dystrophy-harbor-trial-failure-neuromuscular/?utm_campaign=rss",
            "source": "STAT Pharma",
            "published": datetime(2026, 9, 8, 7, 28, 55, tzinfo=timezone.utc),
        },
    ]
    for report in reports:
        report.update(
            original_title=report["title"],
            category="Clinical & Research",
            source_type="news",
            source_role="independent_reporting",
            feed_mode="core",
        )
    return reports


@pytest.mark.parametrize("reverse", [False, True])
def test_url_context_bridges_dystrophy_reports_without_dropping_items(novartis_reports, reverse):
    items = list(reversed(novartis_reports)) if reverse else novartis_reports
    before = deepcopy(items)

    groups = graph._build_candidate_groups(items)

    # The pre-existing STAT/Lp(a) edge makes this one comparison group, not one story.
    assert [[item["id"] for item in group] for group in groups] == [["avidity", "lpa", "stat"]]
    assert items == before
    assert not graph._is_high_confidence_duplicate(items[0], items[2])


@pytest.mark.parametrize("link", [
    "https://example.com/myotonic-dystrophy",
    "https://example.com/%6Dyotonic%2Ddystrophy/",
])
def test_article_basename_clues_can_match_summary_context(novartis_reports, link):
    left, _, right = novartis_reports
    right["link"] = link
    assert len(graph._build_candidate_groups([left, right])) == 1


@pytest.mark.parametrize("link", [
    None,
    123,
    "",
    "/myotonic-dystrophy",
    "https:///myotonic-dystrophy",
    "https://[broken/myotonic-dystrophy",
    "https://example.com:bad/myotonic-dystrophy",
    "ftp://example.com/myotonic-dystrophy",
    "https://myotonic-dystrophy.example.com/brief",
    "https://example.com/myotonic/dystrophy/brief",
    "https://example.com/brief?q=myotonic-dystrophy#myotonic-dystrophy",
    "https://example.com/20260908-20260101",
    "https://example.com/novartis-dystrophy",
    "https://example.com/myotonic-20260908",
    "https://example.com/novartis-clinical-trial-failed",
])
def test_non_article_or_weak_url_clues_do_not_create_a_group(novartis_reports, link):
    left, _, right = novartis_reports
    right["link"] = link
    assert len(graph._build_candidate_groups([left, right])) == 2


@pytest.mark.parametrize("noise", [
    "novartis",
    "bristol-myers",
    "novo-nordisk",
    "johnson-johnson",
    "20260908-september-clinical-research-article-update-failure",
])
def test_company_and_news_noise_cannot_supply_a_second_anchor(novartis_reports, noise):
    left, _, right = novartis_reports
    left.update(title="Readout overview", original_title="Readout overview", summary=f"{noise} dystrophy")
    right.update(title="Portfolio report", original_title="Portfolio report", summary="")
    left["link"] = "https://example.com/brief"
    right["link"] = f"https://example.com/{noise}-dystrophy"
    assert len(graph._build_candidate_groups([left, right])) == 2


def test_url_clues_must_match_the_other_items_context(novartis_reports):
    left, _, right = novartis_reports
    left["link"] = "https://example.com/myotonic-dystrophy"
    right["link"] = "https://example.com/brief"
    assert len(graph._build_candidate_groups([left, right])) == 2


def _response(*, merge):
    clusters = [
        {"keep_id": "g1i1", "duplicate_ids": ["g1i3"] if merge else []},
        {"keep_id": "g1i2", "duplicate_ids": []},
    ]
    if not merge:
        clusters.append({"keep_id": "g1i3", "duplicate_ids": []})
    return {"groups": [{"group_id": "g1", "clusters": clusters, "off_topic_ids": []}]}


@pytest.mark.parametrize("merge", [True, False])
def test_agent_decisions_keep_lpa_separate_and_control_dystrophy_merge(
    novartis_reports, merge, tmp_path, monkeypatch,
):
    snapshot = graph.build_candidate_snapshot(novartis_reports)
    decisions = {
        "schema_version": 2,
        "kind": "bio-news-agent.decisions",
        "snapshot_id": snapshot["snapshot_id"],
        **_response(merge=merge),
    }
    candidates_file = tmp_path / "candidates.json"
    decisions_file = tmp_path / "decisions.json"
    candidates_file.write_text(json.dumps(snapshot))
    decisions_file.write_text(json.dumps(decisions))
    monkeypatch.setattr(graph, "_NEWS_FILE", tmp_path / "news.md")

    result = graph.apply_decisions_file(decisions_file, candidates_file)

    assert [item["id"] for item in result["items"]] == (["avidity", "lpa"] if merge else ["avidity", "lpa", "stat"])
    assert result["items"][0]["coverage_sources"] == (["STAT Pharma"] if merge else [])
    assert result["items"][1]["coverage_sources"] == []
    assert "cholesterol-lowering medicine" in result["markdown"]
    assert ("(2 sources)" in result["markdown"]) is merge


def test_api_dedupe_can_compare_dystrophy_pair_and_counts_one_duplicate(novartis_reports):
    groups = graph._build_candidate_groups(novartis_reports)

    kept, skipped = graph._apply_dedupe_response(list(enumerate(groups, 1)), _response(merge=True))

    assert [item["id"] for item in kept] == ["avidity", "lpa"]
    assert skipped == 1
    assert kept[0]["coverage_sources"] == ["STAT Pharma"]
    assert kept[1]["coverage_sources"] == []


@pytest.mark.parametrize("api_error", [False, True])
def test_url_clues_do_not_change_fallback_groups_or_merge_distinct_headlines(
    novartis_reports, api_error, monkeypatch,
):
    with_hints = graph._build_fallback_candidate_groups(novartis_reports)
    without_links = [{**item, "link": ""} for item in novartis_reports]
    without_hints = graph._build_fallback_candidate_groups(without_links)
    assert [[item["id"] for item in group] for group in with_hints] == [
        [item["id"] for item in group] for group in without_hints
    ]
    monkeypatch.setattr(graph, "_get_openai_api_key", lambda: "test-key" if api_error else "")

    def unavailable(*args):
        raise RuntimeError("simulated API failure")

    monkeypatch.setattr(graph, "_categorize_with_openai", unavailable)
    result = graph.node_categorize({"items": novartis_reports})

    assert [item["id"] for item in result["items"]] == ["avidity", "lpa", "stat"]
    assert [item["coverage_sources"] for item in result["items"]] == [[], [], []]
