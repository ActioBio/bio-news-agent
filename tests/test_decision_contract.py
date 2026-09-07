from __future__ import annotations

from copy import deepcopy

import pytest

from decision_contract import (
    build_candidate_envelope,
    validate_candidate_envelope,
    validate_decision_envelope,
    validate_dispositions,
)


CANDIDATE_KIND = "bio-news-agent.candidates"
DECISION_KIND = "bio-news-agent.decisions"


def _groups() -> list[dict[str, object]]:
    return [
        {
            "group_id": "g1",
            "items": [
                {"item_id": "g1i1", "title": "Primary trial report"},
                {"item_id": "g1i2", "title": "Second trial report"},
            ],
        },
        {
            "group_id": "g2",
            "items": [{"item_id": "g2i1", "title": "Distinct company story"}],
        },
    ]


def _snapshot() -> dict[str, object]:
    return build_candidate_envelope(
        kind=CANDIDATE_KIND,
        categories=["Clinical & Research"],
        groups=_groups(),
    )


def _valid_response_groups() -> list[dict[str, object]]:
    return [
        {
            "group_id": "g1",
            "off_topic_ids": [],
            "clusters": [
                {"keep_id": "g1i1", "duplicate_ids": ["g1i2"]},
            ],
        },
        {
            "group_id": "g2",
            "off_topic_ids": [],
            "clusters": [
                {"keep_id": "g2i1", "duplicate_ids": []},
            ],
        },
    ]


def test_candidate_envelope_has_deterministic_self_hash():
    first = _snapshot()
    second = _snapshot()

    assert first == second
    assert first["schema_version"] == 5
    assert first["kind"] == CANDIDATE_KIND
    assert str(first["snapshot_id"]).startswith("sha256:")
    assert len(str(first["snapshot_id"])) == 71
    assert validate_candidate_envelope(first, expected_kind=CANDIDATE_KIND) == first["snapshot_id"]


def test_candidate_envelope_rejects_content_mutation_with_stale_hash():
    snapshot = _snapshot()
    mutated = deepcopy(snapshot)
    mutated["groups"][0]["items"][0]["title"] = "Mutated trial report"

    with pytest.raises(ValueError, match="snapshot_id"):
        validate_candidate_envelope(mutated, expected_kind=CANDIDATE_KIND)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 1, "schema version"),
        ("kind", "wrong.decisions", "kind"),
        ("snapshot_id", "sha256:wrong", "snapshot_id"),
    ],
)
def test_decision_envelope_rejects_wrong_binding(field, value, message):
    snapshot = _snapshot()
    decisions = {
        "schema_version": 2,
        "kind": DECISION_KIND,
        "snapshot_id": snapshot["snapshot_id"],
        "groups": _valid_response_groups(),
    }
    decisions[field] = value

    with pytest.raises(ValueError, match=message):
        validate_decision_envelope(
            decisions,
            expected_kind=DECISION_KIND,
            expected_snapshot_id=str(snapshot["snapshot_id"]),
        )


def test_dispositions_accept_duplicate_and_explicit_singleton():
    validate_dispositions(
        {"g1": {"g1i1", "g1i2"}, "g2": {"g2i1"}},
        _valid_response_groups(),
    )


def test_dispositions_require_explicit_duplicate_ids():
    groups = _valid_response_groups()
    del groups[1]["clusters"][0]["duplicate_ids"]

    with pytest.raises(ValueError, match="duplicate_ids"):
        validate_dispositions(
            {"g1": {"g1i1", "g1i2"}, "g2": {"g2i1"}},
            groups,
        )


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ([], "Missing decision groups"),
        ([{"group_id": "g1", "off_topic_ids": ["g1i1", "g1i2"], "clusters": []}], "Missing decision groups"),
        (
            [
                {"group_id": "g1", "off_topic_ids": ["g1i1", "g1i2"], "clusters": []},
                {"group_id": "g1", "off_topic_ids": ["g1i1", "g1i2"], "clusters": []},
                {"group_id": "g2", "off_topic_ids": ["g2i1"], "clusters": []},
            ],
            "Duplicate decision group",
        ),
        (
            [
                {"group_id": "g1", "off_topic_ids": ["g2i1"], "clusters": []},
                {"group_id": "g2", "off_topic_ids": ["g2i1"], "clusters": []},
            ],
            "Unknown item",
        ),
        (
            [
                {
                    "group_id": "g1",
                    "off_topic_ids": ["g1i1"],
                    "clusters": [{"keep_id": "g1i1", "duplicate_ids": ["g1i2"]}],
                },
                {"group_id": "g2", "off_topic_ids": ["g2i1"], "clusters": []},
            ],
            "multiple dispositions",
        ),
        (
            [
                {"group_id": "g1", "off_topic_ids": ["g1i1"], "clusters": []},
                {"group_id": "g2", "off_topic_ids": ["g2i1"], "clusters": []},
            ],
            "Missing item dispositions",
        ),
    ],
)
def test_dispositions_reject_incomplete_or_ambiguous_output(groups, message):
    with pytest.raises(ValueError, match=message):
        validate_dispositions(
            {"g1": {"g1i1", "g1i2"}, "g2": {"g2i1"}},
            groups,
        )
