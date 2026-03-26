import os
import re
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]
TOKEN_PATH = os.path.join(os.path.dirname(__file__), "token.json")

logger = logging.getLogger(__name__)

# Max characters to keep from a linked document
_MAX_DOC_CHARS = 10_000

# Regex to extract Google Doc/Sheet ID from URLs
_GOOGLE_URL_RE = re.compile(
    r"https?://docs\.google\.com/"
    r"(?P<type>document|spreadsheets)"
    r"/d/(?P<id>[a-zA-Z0-9_-]+)"
)


def _get_client_config():
    """Build OAuth client config from environment variables."""
    return {
        "installed": {
            "client_id": os.environ["GOOGLE_CLIENT_ID"],
            "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def authenticate():
    """Authenticate with Google APIs using OAuth2.

    Auto-detects scope mismatch and forces re-auth when needed.
    """
    creds = None

    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        # Check if token was issued with fewer scopes than we now require
        if creds and creds.scopes and not set(SCOPES).issubset(set(creds.scopes)):
            logger.info("Token scopes mismatch — deleting old token for re-auth")
            os.remove(TOKEN_PATH)
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(
                _get_client_config(), SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return creds


# ---------------------------------------------------------------------------
# Core sheet fetcher (with hyperlink extraction)
# ---------------------------------------------------------------------------

def fetch_sheet_data(spreadsheet_id: str, sheet_name: str = "Sheet1") -> list[dict]:
    """
    Fetch all data from a Google Sheet and return as list of dicts.
    First row is treated as headers.
    """
    creds = authenticate()
    service = build("sheets", "v4", credentials=creds)

    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=sheet_name)
        .execute()
    )

    values = result.get("values", [])
    if not values or len(values) < 2:
        return []

    headers = values[0]
    rows = []
    for row in values[1:]:
        # Pad row with empty strings if shorter than headers
        padded = row + [""] * (len(headers) - len(row))
        rows.append(dict(zip(headers, padded)))

    return rows


def fetch_sheet_with_hyperlinks(
    spreadsheet_id: str, sheet_name: str = "Sheet1"
) -> tuple[list[dict], list[str]]:
    """Fetch sheet rows AND extract hyperlink URLs from cells.

    Uses ``spreadsheets().get(includeGridData=True)`` so we get both
    display values and hyperlinks in a single API call.

    Returns:
        (rows, hyperlinks) where *rows* is the same list-of-dicts as
        ``fetch_sheet_data`` and *hyperlinks* is a deduplicated list of
        Google Doc/Sheet URLs found in cell hyperlinks.
    """
    creds = authenticate()
    service = build("sheets", "v4", credentials=creds)

    result = (
        service.spreadsheets()
        .get(
            spreadsheetId=spreadsheet_id,
            ranges=[sheet_name],
            includeGridData=True,
        )
        .execute()
    )

    sheets = result.get("sheets", [])
    if not sheets:
        return [], []

    grid_data = sheets[0].get("data", [])
    if not grid_data:
        return [], []

    row_data_list = grid_data[0].get("rowData", [])
    if len(row_data_list) < 2:
        return [], []

    # --- Extract headers from first row ---
    header_cells = row_data_list[0].get("values", [])
    headers = []
    for cell in header_cells:
        ev = cell.get("effectiveValue", {})
        headers.append(
            str(ev.get("stringValue", ev.get("numberValue", "")))
        )

    # --- Extract data rows + hyperlinks ---
    rows: list[dict] = []
    seen_urls: set[str] = set()
    hyperlinks: list[str] = []

    for row_data in row_data_list[1:]:
        cells = row_data.get("values", [])
        row_values: list[str] = []
        for cell in cells:
            # Display value
            fv = cell.get("formattedValue", "")
            row_values.append(fv)
            # Extract links (regular hyperlinks + smart chips)
            for link in _extract_cell_links(cell):
                if link not in seen_urls:
                    seen_urls.add(link)
                    hyperlinks.append(link)
        # Pad if fewer cells than headers
        row_values += [""] * (len(headers) - len(row_values))
        rows.append(dict(zip(headers, row_values)))

    return rows, hyperlinks


# ---------------------------------------------------------------------------
# URL / cell helpers
# ---------------------------------------------------------------------------

def _extract_cell_links(cell: dict) -> list[str]:
    """Extract all URLs from a cell.

    Checks both the ``hyperlink`` property (regular links) and
    ``chipRuns`` (smart chip / rich link embeds used by Google Sheets
    for linked Docs/Sheets).
    """
    links: list[str] = []

    # Regular hyperlink
    link = cell.get("hyperlink", "")
    if link:
        links.append(link)

    # Smart chip links (chipRuns → richLinkProperties.uri)
    for chip_run in cell.get("chipRuns", []):
        chip = chip_run.get("chip", {})
        uri = chip.get("richLinkProperties", {}).get("uri", "")
        if uri and uri not in links:
            links.append(uri)

    return links


def _parse_google_url(url: str) -> tuple[str, str] | None:
    """Extract (doc_type, document_id) from a Google Docs/Sheets URL.

    Returns ``("doc", id)`` or ``("sheet", id)``, or ``None`` for
    non-Google URLs.
    """
    m = _GOOGLE_URL_RE.search(url)
    if not m:
        return None
    raw_type = m.group("type")
    doc_type = "doc" if raw_type == "document" else "sheet"
    return doc_type, m.group("id")


# ---------------------------------------------------------------------------
# Linked-document content fetchers
# ---------------------------------------------------------------------------

def fetch_google_doc_content(doc_id: str, creds) -> str:
    """Fetch the text content of a Google Doc.

    Extracts paragraph text and simple table cell text.  Truncates at
    ``_MAX_DOC_CHARS``.
    """
    service = build("docs", "v1", credentials=creds)
    doc = service.documents().get(documentId=doc_id).execute()
    title = doc.get("title", "Untitled")

    parts: list[str] = [f"[Google Doc: {title}]"]

    for element in doc.get("body", {}).get("content", []):
        # Paragraphs
        paragraph = element.get("paragraph")
        if paragraph:
            text = ""
            for el in paragraph.get("elements", []):
                text += el.get("textRun", {}).get("content", "")
            stripped = text.strip()
            if stripped:
                parts.append(stripped)

        # Tables
        table = element.get("table")
        if table:
            for table_row in table.get("tableRows", []):
                row_texts: list[str] = []
                for cell in table_row.get("tableCells", []):
                    cell_text = ""
                    for p in cell.get("content", []):
                        for el in p.get("paragraph", {}).get("elements", []):
                            cell_text += el.get("textRun", {}).get("content", "")
                    cell_text = cell_text.strip()
                    if cell_text:
                        row_texts.append(cell_text)
                if row_texts:
                    parts.append(" | ".join(row_texts))

    full_text = "\n".join(parts)
    if len(full_text) > _MAX_DOC_CHARS:
        full_text = full_text[:_MAX_DOC_CHARS] + "\n...(truncated)"
    return full_text


def fetch_linked_sheet_content(
    sheet_id: str, creds, extract_hyperlinks: bool = True
) -> tuple[list[str], list[str]]:
    """Fetch **all tabs** of a linked spreadsheet as text documents.

    Two modes:
    - ``extract_hyperlinks=True`` (default, used for level 1): uses
      ``includeGridData=True`` on the **first tab only** to extract
      nested hyperlinks, and uses the lightweight ``values().batchGet``
      for all remaining tabs.
    - ``extract_hyperlinks=False`` (used for deeper levels): uses only
      ``values().batchGet`` — very fast, no hyperlink scanning.

    Returns:
        ``(tab_documents, hyperlinks)``
    """
    service = build("sheets", "v4", credentials=creds)

    # Step 1: lightweight metadata call (tab names + spreadsheet title)
    meta = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
    spreadsheet_title = meta.get("properties", {}).get("title", "Untitled Sheet")
    tab_names = [s["properties"]["title"] for s in meta.get("sheets", [])]

    if not tab_names:
        return [], []

    # Step 2: batch-get all tab values in ONE lightweight call
    ranges = [f"'{t}'!A:ZZ" for t in tab_names]
    batch = (
        service.spreadsheets()
        .values()
        .batchGet(spreadsheetId=sheet_id, ranges=ranges)
        .execute()
    )
    value_ranges = batch.get("valueRanges", [])

    # Build tab documents from values
    all_tab_docs: list[str] = []
    for tab_name, vr in zip(tab_names, value_ranges):
        values = vr.get("values", [])
        if not values:
            continue

        parts: list[str] = [f"[Linked Sheet: {spreadsheet_title} > {tab_name}]"]
        headers = values[0] if values else []
        if any(h.strip() for h in headers):
            parts.append(" | ".join(headers))

        for row in values[1:]:
            if not any(c.strip() for c in row):
                continue
            padded = row + [""] * (len(headers) - len(row))
            parts.append(" | ".join(padded))

        if len(parts) > 1:
            full_text = "\n".join(parts)
            if len(full_text) > _MAX_DOC_CHARS:
                full_text = full_text[:_MAX_DOC_CHARS] + "\n...(truncated)"
            all_tab_docs.append(full_text)

    # Step 3 (optional): extract hyperlinks from first tab only
    nested_links: list[str] = []
    if extract_hyperlinks and tab_names:
        try:
            grid_result = (
                service.spreadsheets()
                .get(
                    spreadsheetId=sheet_id,
                    ranges=[tab_names[0]],
                    includeGridData=True,
                )
                .execute()
            )
            seen_urls: set[str] = set()
            for sheet in grid_result.get("sheets", []):
                for gd in sheet.get("data", []):
                    for row_data in gd.get("rowData", []):
                        for cell in row_data.get("values", []):
                            for link in _extract_cell_links(cell):
                                if link not in seen_urls:
                                    seen_urls.add(link)
                                    nested_links.append(link)
        except Exception as exc:
            logger.warning("Hyperlink extraction failed for %s: %s", sheet_id, exc)

    return all_tab_docs, nested_links


# ---------------------------------------------------------------------------
# Orchestrator — fetch all linked documents
# ---------------------------------------------------------------------------

_MAX_DEPTH = 6
_MAX_WORKERS = 8  # parallel fetches per level


def _fetch_one(url, doc_type, doc_id, creds, extract_hyperlinks=True):
    """Fetch a single linked document. Returns (entries, nested_links)."""
    entries = []
    nested = []
    try:
        if doc_type == "doc":
            content = fetch_google_doc_content(doc_id, creds)
            entries.append({"url": url, "content": content, "error": None})
        else:
            tab_docs, nested_links = fetch_linked_sheet_content(
                doc_id, creds, extract_hyperlinks=extract_hyperlinks
            )
            for tab_doc in tab_docs:
                entries.append({"url": url, "content": tab_doc, "error": None})
            if not tab_docs:
                entries.append({"url": url, "content": None, "error": "No data in sheet"})
            nested = nested_links
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        entries.append({"url": url, "content": None, "error": str(exc)})
    return entries, nested


def fetch_linked_documents(
    hyperlinks: list[str],
    creds=None,
    progress_callback=None,
) -> list[dict]:
    """Fetch content for every recognised Google Doc/Sheet URL.

    Uses **parallel fetching** (up to ``_MAX_WORKERS`` threads) for
    speed.  Crawls up to ``_MAX_DEPTH`` (6) levels deep.  Documents
    are de-duplicated by ID so each is fetched at most once.

    Args:
        hyperlinks: List of URLs extracted from the main spreadsheet.
        creds: Google OAuth credentials. Obtained via ``authenticate()``
            if not provided.
        progress_callback: Optional ``callable(current, total, message)``
            for progress reporting.

    Returns:
        List of dicts ``{"url": ..., "content": ..., "error": ...}``.
    """
    if creds is None:
        creds = authenticate()

    seen_ids: set[str] = set()
    results: list[dict] = []

    # Seed the first level from the provided hyperlinks
    current_level: list[tuple[str, str, str]] = []  # (url, doc_type, doc_id)
    for url in hyperlinks:
        parsed = _parse_google_url(url)
        if parsed and parsed[1] not in seen_ids:
            seen_ids.add(parsed[1])
            current_level.append((url, *parsed))

    for depth in range(_MAX_DEPTH):
        if not current_level:
            break

        level_label = f"Level {depth + 1}"
        level_total = len(current_level)
        logger.info("%s: %d document(s) to fetch in parallel", level_label, level_total)

        next_level: list[tuple[str, str, str]] = []
        done_count = 0

        _report_progress(
            progress_callback, 0, level_total,
            f"{level_label}: fetching {level_total} docs (parallel)..."
        )

        # Only scan for hyperlinks at levels 1-2; deeper levels just get content
        scan_links = depth < 2

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {
                pool.submit(_fetch_one, url, doc_type, doc_id, creds, scan_links): url
                for url, doc_type, doc_id in current_level
            }
            for future in as_completed(futures):
                entries, nested = future.result()
                results.extend(entries)
                # Queue nested links for next level
                for nurl in nested:
                    nr = _parse_google_url(nurl)
                    if nr and nr[1] not in seen_ids:
                        seen_ids.add(nr[1])
                        next_level.append((nurl, *nr))
                done_count += 1
                _report_progress(
                    progress_callback, done_count, level_total,
                    f"{level_label}: fetched {done_count}/{level_total} docs..."
                )

        current_level = next_level

    return results


def _report_progress(callback, current, total, message=""):
    """Call progress callback, tolerating both 2-arg and 3-arg signatures."""
    if not callback:
        return
    try:
        callback(current, total, message)
    except TypeError:
        callback(current, total)


# ---------------------------------------------------------------------------
# Document builder — combines rows + linked docs
# ---------------------------------------------------------------------------

def rows_to_documents(rows: list[dict]) -> list[str]:
    """Convert sheet rows into text documents for RAG."""
    documents = []
    for i, row in enumerate(rows):
        parts = []
        for key, value in row.items():
            if value.strip():
                parts.append(f"{key}: {value}")
        if parts:
            documents.append(f"Row {i + 1}\n" + "\n".join(parts))
    return documents


def build_documents(
    rows: list[dict],
    linked_docs: list[dict] | None = None,
) -> list[str]:
    """Build the full document list for RAG indexing.

    Combines per-row documents from the main sheet with any successfully
    fetched linked document content.
    """
    documents = rows_to_documents(rows)

    if linked_docs:
        for entry in linked_docs:
            if entry["content"]:
                documents.append(entry["content"])

    return documents


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    spreadsheet_id = os.environ["SPREADSHEET_ID"]
    sheet_name = os.environ.get("SHEET_NAME", "Sheet1")

    rows, hyperlinks = fetch_sheet_with_hyperlinks(spreadsheet_id, sheet_name)
    print(f"Fetched {len(rows)} rows, found {len(hyperlinks)} hyperlinks")

    if hyperlinks:
        linked = fetch_linked_documents(hyperlinks)
        ok = sum(1 for d in linked if d["content"])
        fail = sum(1 for d in linked if d["error"])
        print(f"Linked docs: {ok} fetched, {fail} failed")
        docs = build_documents(rows, linked)
    else:
        docs = rows_to_documents(rows)

    print(f"Created {len(docs)} documents total")
    if docs:
        print("\nSample document:")
        print(docs[0])
