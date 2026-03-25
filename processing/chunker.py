"""Row-group chunking with context headers and metadata."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from config.settings import CHUNK_ROW_OVERLAP, CHUNK_TARGET_TOKENS


@dataclass
class Chunk:
    text: str
    metadata: dict  # spreadsheet_id, title, tab, gid, start_row, end_row
    token_count: int


_enc = tiktoken.encoding_for_model("gpt-4o")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _detect_type(rows: list[list[str]], headers: list[str]) -> str:
    """Heuristic: tabular if uniform column count and >3 rows, else free-text."""
    if len(rows) < 4:
        return "free-text"
    col_counts = {len(r) for r in rows[:20]}
    if len(col_counts) == 1 and len(headers) >= 2:
        return "tabular"
    if headers and len(headers) > 15:
        return "wide-table"
    return "free-text"


def _row_to_text_tabular(row: list[str], headers: list[str]) -> str:
    """Format a row as 'Header: Value' pairs, skipping empty values."""
    parts: list[str] = []
    for i, val in enumerate(row):
        if val:
            header = headers[i] if i < len(headers) else f"Col{i+1}"
            parts.append(f"{header}: {val}")
    return " | ".join(parts)


def _row_to_text_free(row: list[str]) -> str:
    return " ".join(c for c in row if c)


def chunk_tab(
    rows: list[list[str]],
    headers: list[str],
    spreadsheet_id: str,
    spreadsheet_title: str,
    tab_name: str,
    tab_gid: int,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap: int = CHUNK_ROW_OVERLAP,
) -> list[Chunk]:
    """Split a tab's rows into token-limited chunks with context headers."""
    if not rows:
        return []

    sheet_type = _detect_type(rows, headers)
    context_header = f"Spreadsheet: {spreadsheet_title} / Tab: {tab_name}"
    if headers:
        context_header += f" / Columns: {', '.join(h for h in headers if h)}"
    header_tokens = _count_tokens(context_header + "\n\n")

    # For tabular data, skip the header row in the body (it's in context header)
    data_rows = rows[1:] if sheet_type in ("tabular", "wide-table") and len(rows) > 1 else rows

    chunks: list[Chunk] = []
    i = 0
    while i < len(data_rows):
        lines: list[str] = []
        running_tokens = header_tokens
        start_row = i + 2 if sheet_type in ("tabular", "wide-table") else i + 1  # 1-indexed, accounting for header

        j = i
        while j < len(data_rows):
            if sheet_type in ("tabular", "wide-table"):
                line = _row_to_text_tabular(data_rows[j], headers)
            else:
                line = _row_to_text_free(data_rows[j])

            if not line:
                j += 1
                continue

            line_tokens = _count_tokens(line + "\n")
            if lines and running_tokens + line_tokens > target_tokens:
                break
            lines.append(line)
            running_tokens += line_tokens
            j += 1

        if lines:
            text = context_header + "\n\n" + "\n".join(lines)
            end_row = start_row + len(lines) - 1
            chunks.append(Chunk(
                text=text,
                metadata={
                    "spreadsheet_id": spreadsheet_id,
                    "spreadsheet_title": spreadsheet_title,
                    "tab_name": tab_name,
                    "tab_gid": tab_gid,
                    "start_row": start_row,
                    "end_row": end_row,
                    "sheet_type": sheet_type,
                },
                token_count=running_tokens,
            ))

        # Advance with overlap
        advance = max(j - i - overlap, 1)
        i += advance

    return chunks
