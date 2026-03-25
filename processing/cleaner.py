"""Data cleaning: whitespace, Sheets artifacts, empty rows."""

from __future__ import annotations


def clean_cell(value: str) -> str:
    """Strip whitespace and remove leading apostrophe artifacts."""
    v = value.strip()
    # Google Sheets uses a leading apostrophe to force text mode
    if v.startswith("'") and len(v) > 1:
        v = v[1:]
    return v


def clean_rows(rows: list[list[str]]) -> list[list[str]]:
    """Clean all cells and drop completely empty rows."""
    cleaned: list[list[str]] = []
    for row in rows:
        new_row = [clean_cell(str(c)) for c in row]
        if any(c != "" for c in new_row):
            cleaned.append(new_row)
    return cleaned
