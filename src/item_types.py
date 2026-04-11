"""Typed dicts for the digest item lifecycle."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict

SourceRole = Literal["primary", "independent_reporting", "commentary", "community"]
FeedMode = Literal["core", "discovery_only"]
StoryTier = Literal["normal", "high"]


class CollectedItem(TypedDict):
    id: str
    title: str
    original_title: str
    link: str
    source: str
    published: datetime
    category: str
    summary: str
    source_type: str
    source_role: SourceRole
    feed_mode: FeedMode


class CandidateItem(TypedDict):
    item_id: str
    id: str
    title: str
    original_title: str
    link: str
    source: str
    published: str
    summary: str
    category: str
    source_type: str
    source_role: SourceRole
    feed_mode: FeedMode


class ResolvedItem(CollectedItem, total=False):
    _prompt_id: str
    summary_line: str
    tier: StoryTier
    coverage_sources: list[str]


class EnrichmentItem(TypedDict):
    item_id: str
    title: str
    source: str
    source_role: SourceRole
    feed_mode: FeedMode
    published: str
    summary: str
    source_type: str
    coverage_count: int
