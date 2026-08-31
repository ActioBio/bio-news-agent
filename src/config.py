"""Centralized configuration for bio-news-agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _REPO_ROOT / ".env"
load_dotenv(_ENV_FILE)


def resolve_repo_output_file(path_value: str) -> Path:
    output = Path(path_value)
    if output.is_absolute():
        return output
    return (_REPO_ROOT / output).resolve()


def _get_env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


# ── OpenAI ───────────────────────────────────────────────────────
OPENAI_API_KEY: str = _get_env_str("OPENAI_API_KEY", "")
OPENAI_MODEL: str = _get_env_str("OPENAI_MODEL", "gpt-5.6-luna")
OPENAI_TIMEOUT_SECONDS: int = _get_env_int("OPENAI_TIMEOUT_SECONDS", 60)
OPENAI_RETRIES: int = _get_env_int("OPENAI_RETRIES", 2)
DIGEST_OUTPUT_FILE: str = _get_env_str("DIGEST_OUTPUT_FILE", "news.md")
DIGEST_STATUS_FILE: str = _get_env_str("DIGEST_STATUS_FILE", "digest-run-status.json")
DIGEST_ISSUE_REPO: str = _get_env_str("DIGEST_ISSUE_REPO", "ActioBio/bio-news-agent")
DIGEST_ISSUE_LABEL: str = _get_env_str("DIGEST_ISSUE_LABEL", "ai-digest")
DIGEST_ISSUE_TITLE_PREFIX: str = _get_env_str(
    "DIGEST_ISSUE_TITLE_PREFIX",
    "Biotech / Pharma Headlines",
)
DIGEST_PUBLISH_WORKFLOW: str = _get_env_str("DIGEST_PUBLISH_WORKFLOW", "publish-digest.yml")
DIGEST_ACTIONS_REF: str = _get_env_str("DIGEST_ACTIONS_REF", "main")

# ── Pipeline limits ──────────────────────────────────────────────
PAPER_LIMIT: int = _get_env_int("PAPER_LIMIT", 7)
COMPANY_NEWS_LIMIT: int = _get_env_int("COMPANY_NEWS_LIMIT", 3)

# ── RSS settings ─────────────────────────────────────────────────
RSS_TIMEOUT: int = _get_env_int("RSS_TIMEOUT", 10)
RSS_RETRIES: int = _get_env_int("RSS_RETRIES", 2)
RSS_MAX_WORKERS: int = _get_env_int("RSS_MAX_WORKERS", 8)
RSS_MAX_FEED_BYTES: int = _get_env_int("RSS_MAX_FEED_BYTES", 5_000_000)
RSS_USER_AGENT: str = _get_env_str(
    "RSS_USER_AGENT",
    "bio-news-agent/1.0 (+https://github.com/bio-news-agent)",
)
RSS_FALLBACK_USER_AGENT: str = _get_env_str(
    "RSS_FALLBACK_USER_AGENT",
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
)

# ── Categories ───────────────────────────────────────────────────
CATEGORIES: list[str] = [
    "Regulatory & FDA",
    "Clinical & Research",
    "Deals & Finance",
    "Company News",
    "Policy & Politics",
    "Market Insights",
]

# ── Bio/pharma company names for keyword detection ───────────────
COMPANY_NAMES: list[str] = [
    "pfizer",
    "moderna",
    "gilead",
    "regeneron",
    "amgen",
    "biogen",
    "vertex",
    "abbvie",
    "novartis",
    "roche",
    "merck",
    "bms",
    "bristol-myers",
    "astrazeneca",
    "sanofi",
    "gsk",
    "glaxosmithkline",
    "lilly",
    "eli lilly",
    "johnson & johnson",
    "j&j",
    "takeda",
    "bayer",
    "novo nordisk",
    "illumina",
    "genentech",
]

# ── Source limits ──────────────────────────────────────────────
# Maximum items per source (0 = no limit)
MAX_ITEMS_PER_SOURCE: int = _get_env_int("MAX_ITEMS_PER_SOURCE", 8)

# ── Logging ─────────────────────────────────────────────────────
LOG_LEVEL: str = _get_env_str("LOG_LEVEL", "INFO")
