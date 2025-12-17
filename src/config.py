"""Centralized configuration for bio-news-agent."""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env from project root
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_FILE)

# ── OpenAI ───────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# ── Pipeline limits ──────────────────────────────────────────────
PAPER_LIMIT: int = int(os.getenv("PAPER_LIMIT", "7"))

# ── RSS settings ─────────────────────────────────────────────────
RSS_TIMEOUT: int = int(os.getenv("RSS_TIMEOUT", "10"))
RSS_RETRIES: int = int(os.getenv("RSS_RETRIES", "2"))
RSS_USER_AGENT: str = "bio-news-agent/1.0 (+https://github.com/bio-news-agent)"

# ── Categories ───────────────────────────────────────────────────
CATEGORIES: List[str] = [
    "Regulatory & FDA",
    "Clinical & Research",
    "Deals & Finance",
    "Company News",
    "Policy & Politics",
    "Market Insights",
]

# ── Bio/pharma company names for keyword detection ───────────────
COMPANY_NAMES: List[str] = [
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
MAX_ITEMS_PER_SOURCE: int = int(os.getenv("MAX_ITEMS_PER_SOURCE", "8"))
