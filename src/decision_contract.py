"""Versioned candidate and decision contract validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

CANDIDATE_SCHEMA_VERSION = 5
DECISION_SCHEMA_VERSION = 2


def _snapshot_id(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "snapshot_id"}
    try:
        canonical = json.dumps(
            unsigned,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Candidate snapshot is not canonical JSON") from exc
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def build_candidate_envelope(
    *,
    kind: str,
    categories: list[str],
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "kind": kind,
        "categories": list(categories),
        "groups": groups,
    }
    payload["snapshot_id"] = _snapshot_id(payload)
    return payload


def validate_candidate_envelope(
    payload: dict[str, Any],
    *,
    expected_kind: str,
) -> str:
    if payload.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise ValueError(f"Candidate snapshot schema version must be {CANDIDATE_SCHEMA_VERSION}")
    if payload.get("kind") != expected_kind:
        raise ValueError(f"Candidate snapshot kind must be {expected_kind}")
    if not isinstance(payload.get("categories"), list):
        raise ValueError("Candidate snapshot missing categories list")

    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise ValueError("Candidate snapshot missing groups list")
    for group_index, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            raise ValueError("Candidate group must be an object")
        expected_group_id = f"g{group_index}"
        if group.get("group_id") != expected_group_id:
            raise ValueError(f"Candidate group id must be {expected_group_id}")
        items = group.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError(f"Candidate group {expected_group_id} must contain items")
        for item_index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Candidate item in {expected_group_id} must be an object")
            expected_item_id = f"{expected_group_id}i{item_index}"
            if item.get("item_id") != expected_item_id:
                raise ValueError(f"Candidate item id must be {expected_item_id}")

    declared_id = payload.get("snapshot_id")
    if not isinstance(declared_id, str) or declared_id != _snapshot_id(payload):
        raise ValueError("Candidate snapshot_id does not match candidate content")
    return declared_id


def validate_decision_envelope(
    payload: dict[str, Any],
    *,
    expected_kind: str,
    expected_snapshot_id: str,
) -> None:
    if payload.get("schema_version") != DECISION_SCHEMA_VERSION:
        raise ValueError(f"Decision schema version must be {DECISION_SCHEMA_VERSION}")
    if payload.get("kind") != expected_kind:
        raise ValueError(f"Decision kind must be {expected_kind}")
    if payload.get("snapshot_id") != expected_snapshot_id:
        raise ValueError("Decision snapshot_id does not match candidate snapshot")


def validate_dispositions(
    expected_group_item_ids: dict[str, set[str]],
    response_groups: Any,
) -> None:
    if not isinstance(response_groups, list):
        raise ValueError("Decision response missing groups list")

    seen_groups: set[str] = set()
    for response_group in response_groups:
        if not isinstance(response_group, dict):
            raise ValueError("Decision group must be an object")
        group_id = response_group.get("group_id")
        if not isinstance(group_id, str) or group_id not in expected_group_item_ids:
            raise ValueError(f"Unknown decision group: {group_id}")
        if group_id in seen_groups:
            raise ValueError(f"Duplicate decision group: {group_id}")
        seen_groups.add(group_id)

        expected_ids = expected_group_item_ids[group_id]
        counts: Counter[str] = Counter()

        def record_item(raw_id: Any) -> None:
            if not isinstance(raw_id, str) or raw_id not in expected_ids:
                raise ValueError(f"Unknown item in {group_id}: {raw_id}")
            counts[raw_id] += 1

        off_topic_ids = response_group.get("off_topic_ids", [])
        clusters = response_group.get("clusters", [])
        if not isinstance(off_topic_ids, list):
            raise ValueError(f"off_topic_ids for {group_id} must be a list")
        if not isinstance(clusters, list):
            raise ValueError(f"clusters for {group_id} must be a list")
        for item_id in off_topic_ids:
            record_item(item_id)
        for cluster in clusters:
            if not isinstance(cluster, dict):
                raise ValueError(f"Cluster in {group_id} must be an object")
            record_item(cluster.get("keep_id"))
            if "duplicate_ids" not in cluster:
                raise ValueError(f"duplicate_ids for {group_id} must be present")
            duplicate_ids = cluster["duplicate_ids"]
            if not isinstance(duplicate_ids, list):
                raise ValueError(f"duplicate_ids for {group_id} must be a list")
            for item_id in duplicate_ids:
                record_item(item_id)

        repeated = sorted(item_id for item_id, count in counts.items() if count != 1)
        if repeated:
            raise ValueError(f"Items have multiple dispositions in {group_id}: {', '.join(repeated)}")
        missing = sorted(expected_ids - counts.keys())
        if missing:
            raise ValueError(f"Missing item dispositions in {group_id}: {', '.join(missing)}")

    missing_groups = sorted(set(expected_group_item_ids) - seen_groups)
    if missing_groups:
        raise ValueError(f"Missing decision groups: {', '.join(missing_groups)}")
