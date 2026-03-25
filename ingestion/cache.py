"""Crawl-state persistence: content hashes and chunk IDs per tab."""

import hashlib
import json
from pathlib import Path
from typing import Any

from config.settings import CRAWL_CACHE_PATH


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_cache(path: Path = CRAWL_CACHE_PATH) -> dict[str, Any]:
    """Load cache from disk. Returns empty dict if missing."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict[str, Any], path: Path = CRAWL_CACHE_PATH) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def content_hash(rows: list[list[str]]) -> str:
    """Deterministic SHA-256 of a tab's content."""
    blob = json.dumps(rows, ensure_ascii=False, sort_keys=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def cache_key(spreadsheet_id: str, tab_name: str) -> str:
    return f"{spreadsheet_id}::{tab_name}"
