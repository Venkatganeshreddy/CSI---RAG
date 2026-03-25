"""Extract Google Sheets spreadsheet IDs from cell values."""

import re

# Matches spreadsheet IDs in Google Sheets URLs
_SHEET_URL_RE = re.compile(
    r"docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]{20,})"
)

# Matches =HYPERLINK("...") formulas
_HYPERLINK_FORMULA_RE = re.compile(
    r'=\s*HYPERLINK\s*\(\s*"([^"]+)"', re.IGNORECASE
)


def extract_spreadsheet_ids(cell_value: str) -> list[str]:
    """Return unique spreadsheet IDs found in a cell's text or formula."""
    if not cell_value:
        return []

    ids: list[str] = []

    # Direct URL matches
    ids.extend(_SHEET_URL_RE.findall(cell_value))

    # HYPERLINK formula — extract URL then look for spreadsheet ID
    for url in _HYPERLINK_FORMULA_RE.findall(cell_value):
        ids.extend(_SHEET_URL_RE.findall(url))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for sid in ids:
        if sid not in seen:
            seen.add(sid)
            unique.append(sid)
    return unique


def extract_ids_from_rows(rows: list[list[str]]) -> list[str]:
    """Scan all cells in a 2-D array and return unique spreadsheet IDs."""
    seen: set[str] = set()
    result: list[str] = []
    for row in rows:
        for cell in row:
            for sid in extract_spreadsheet_ids(str(cell)):
                if sid not in seen:
                    seen.add(sid)
                    result.append(sid)
    return result
