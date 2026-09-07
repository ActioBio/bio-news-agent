"""Execute the copyable decision examples against the current contract."""

import json
import re
from pathlib import Path
from typing import get_args

import graph
import pytest
from config import CATEGORIES
from decision_contract import build_candidate_envelope
from item_types import StoryTier


def _documented_decisions():
    document = (Path(__file__).parents[1] / "docs" / "development.md").read_text(
        encoding="utf-8"
    )
    payloads = [
        json.loads(block)
        for block in re.findall(r"^```json\s*\n(.*?)^```", document, re.MULTILINE | re.DOTALL)
    ]
    examples = [payload for payload in payloads if "groups" in payload]
    assert examples, "No copyable decision examples found"
    assert any(example["groups"] for example in examples), "No exhaustive example found"
    return examples


@pytest.mark.parametrize(
    ("field", "allowed_values"),
    [("category", CATEGORIES), ("tier", get_args(StoryTier))],
)
def test_documented_decisions_use_canonical_editorial_values(field, allowed_values):
    for example in _documented_decisions():
        for group in example["groups"]:
            for cluster in group["clusters"]:
                assert cluster[field] in allowed_values, (
                    f"{cluster['keep_id']} uses unsupported {field}: {cluster[field]!r}"
                )


def _candidate(item_id):
    return {
        "item_id": item_id,
        "id": item_id,
        "title": f"Candidate story {item_id}",
        "original_title": f"Candidate story {item_id}",
        "link": f"https://example.com/{item_id}",
        "source": f"Source {item_id}",
        "published": "2026-01-01T12:00:00+00:00",
        "summary": "",
        "category": "All",
        "source_type": "news",
        "source_role": "independent_reporting",
        "feed_mode": "core",
    }


def test_documented_exhaustive_decisions_apply_without_editorial_fallback(tmp_path, monkeypatch):
    snapshot = build_candidate_envelope(
        kind="bio-news-agent.candidates",
        categories=CATEGORIES,
        groups=[
            {"group_id": "g1", "items": [_candidate("g1i1")]},
            {"group_id": "g2", "items": [_candidate("g2i1"), _candidate("g2i2")]},
            {"group_id": "g3", "items": [_candidate("g3i1")]},
        ],
    )
    candidates_file = tmp_path / "digest-candidates.json"
    decisions_file = tmp_path / "digest-decisions.json"
    output_file = tmp_path / "news.md"
    candidates_file.write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(graph, "_NEWS_FILE", output_file)

    for example in _documented_decisions():
        if not example["groups"]:
            continue
        example["snapshot_id"] = snapshot["snapshot_id"]
        decisions_file.write_text(json.dumps(example), encoding="utf-8")

        result = graph.apply_decisions_file(decisions_file, candidates_file)

        items = {item["id"]: item for item in result["items"]}
        assert set(items) == {"g1i1", "g2i1"}
        assert output_file.read_text(encoding="utf-8") == result["markdown"]
        for group in example["groups"]:
            for cluster in group["clusters"]:
                item = items[cluster["keep_id"]]
                assert item["category"] == cluster["category"]
                assert item["tier"] == cluster["tier"]
                assert item["title"] == cluster["short_title"]
                assert cluster["short_title"] in result["markdown"]
