"""Recursive BFS sheet crawler with link discovery."""

from __future__ import annotations

import datetime
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

from config.settings import (
    CREDENTIALS_PATH,
    CRAWL_LOG_PATH,
    MAX_CRAWL_DEPTH,
    SCOPES,
    SEED_SPREADSHEET_IDS,
)
from ingestion.link_parser import extract_ids_from_rows
from ingestion.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class TabData:
    spreadsheet_id: str
    spreadsheet_title: str
    tab_name: str
    tab_gid: int
    rows: list[list[str]]
    headers: list[str]
    depth: int


@dataclass
class CrawlResult:
    tabs: list[TabData] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


def _open_client() -> gspread.Client:
    creds = Credentials.from_service_account_file(str(CREDENTIALS_PATH), scopes=SCOPES)
    return gspread.authorize(creds)


def _log_event(event: dict, path: Path = CRAWL_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({**event, "ts": datetime.datetime.utcnow().isoformat()}) + "\n")


def crawl(
    seed_ids: list[str] | None = None,
    max_depth: int = MAX_CRAWL_DEPTH,
) -> CrawlResult:
    """BFS crawl starting from seed spreadsheet IDs.

    Returns all tab data and any errors encountered.
    """
    seed_ids = seed_ids or SEED_SPREADSHEET_IDS
    client = _open_client()
    limiter = RateLimiter()

    visited_sheets: set[str] = set()
    queue: deque[tuple[str, int]] = deque()  # (spreadsheet_id, depth)
    result = CrawlResult()

    for sid in seed_ids:
        if sid not in visited_sheets:
            queue.append((sid, 0))
            visited_sheets.add(sid)

    while queue:
        sheet_id, depth = queue.popleft()
        logger.info("Crawling %s (depth %d)", sheet_id, depth)

        try:
            limiter.acquire()
            spreadsheet = client.open_by_key(sheet_id)
        except gspread.exceptions.APIError as exc:
            code = exc.response.status_code
            if code in (403, 404):
                logger.warning("Skipping %s — HTTP %d", sheet_id, code)
                result.errors.append({"sheet_id": sheet_id, "error": f"HTTP {code}"})
                _log_event({"event": "skip", "sheet_id": sheet_id, "code": code})
                continue
            raise
        except Exception as exc:
            logger.error("Error opening %s: %s", sheet_id, exc)
            result.errors.append({"sheet_id": sheet_id, "error": str(exc)})
            _log_event({"event": "error", "sheet_id": sheet_id, "error": str(exc)})
            continue

        for ws in spreadsheet.worksheets():
            try:
                limiter.acquire()
                all_values = ws.get_all_values()
            except Exception as exc:
                logger.error(
                    "Error reading tab '%s' in %s: %s", ws.title, sheet_id, exc
                )
                result.errors.append({
                    "sheet_id": sheet_id,
                    "tab": ws.title,
                    "error": str(exc),
                })
                continue

            headers = all_values[0] if all_values else []
            tab = TabData(
                spreadsheet_id=sheet_id,
                spreadsheet_title=spreadsheet.title,
                tab_name=ws.title,
                tab_gid=ws.id,
                rows=all_values,
                headers=headers,
                depth=depth,
            )
            result.tabs.append(tab)
            _log_event({
                "event": "tab_crawled",
                "sheet_id": sheet_id,
                "title": spreadsheet.title,
                "tab": ws.title,
                "row_count": len(all_values),
            })

            # Discover links and enqueue if within depth limit
            if depth < max_depth:
                discovered = extract_ids_from_rows(all_values)
                for new_id in discovered:
                    if new_id not in visited_sheets:
                        visited_sheets.add(new_id)
                        queue.append((new_id, depth + 1))

    logger.info(
        "Crawl complete: %d tabs from %d sheets, %d errors",
        len(result.tabs),
        len(visited_sheets),
        len(result.errors),
    )
    return result
