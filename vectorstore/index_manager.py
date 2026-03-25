"""Orchestrates full/incremental indexing pipeline."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from ingestion.cache import cache_key, content_hash, load_cache, save_cache
from ingestion.crawler import CrawlResult, TabData
from processing.chunker import Chunk, chunk_tab
from processing.cleaner import clean_rows
from vectorstore.embedder import embed_texts
from vectorstore.store import delete as store_delete
from vectorstore.store import upsert as store_upsert

logger = logging.getLogger(__name__)


def _chunk_id(spreadsheet_id: str, tab_name: str, idx: int) -> str:
    """Deterministic chunk ID."""
    raw = f"{spreadsheet_id}::{tab_name}::{idx}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _process_tab(tab: TabData) -> list[Chunk]:
    """Clean and chunk a single tab."""
    cleaned = clean_rows(tab.rows)
    headers = cleaned[0] if cleaned else []
    return chunk_tab(
        rows=cleaned,
        headers=headers,
        spreadsheet_id=tab.spreadsheet_id,
        spreadsheet_title=tab.spreadsheet_title,
        tab_name=tab.tab_name,
        tab_gid=tab.tab_gid,
    )


def index(crawl_result: CrawlResult) -> dict:
    """Run incremental indexing: skip unchanged, re-embed changed, add new, delete removed.

    Returns stats dict.
    """
    cache = load_cache()
    stats = {"skipped": 0, "updated": 0, "added": 0, "deleted": 0, "errors": 0}

    # Track which cache keys we see this crawl (to detect deletions)
    seen_keys: set[str] = set()

    # Collect all new/changed chunks for batch embedding
    pending_chunks: list[Chunk] = []
    pending_ids: list[str] = []
    old_ids_to_delete: list[str] = []

    for tab in crawl_result.tabs:
        ck = cache_key(tab.spreadsheet_id, tab.tab_name)
        seen_keys.add(ck)

        current_hash = content_hash(tab.rows)
        cached = cache.get(ck)

        if cached and cached.get("content_hash") == current_hash:
            stats["skipped"] += 1
            continue

        # Changed or new — delete old chunks if they exist
        if cached and cached.get("chunk_ids"):
            old_ids_to_delete.extend(cached["chunk_ids"])
            stats["updated"] += 1
        else:
            stats["added"] += 1

        chunks = _process_tab(tab)
        chunk_ids = [_chunk_id(tab.spreadsheet_id, tab.tab_name, i) for i in range(len(chunks))]

        for chunk, cid in zip(chunks, chunk_ids):
            pending_chunks.append(chunk)
            pending_ids.append(cid)

        # Update cache entry
        cache[ck] = {
            "content_hash": current_hash,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "chunk_ids": chunk_ids,
            "spreadsheet_id": tab.spreadsheet_id,
            "tab_name": tab.tab_name,
        }

    # Detect deleted tabs (in cache but not in crawl)
    for ck, entry in list(cache.items()):
        if ck not in seen_keys:
            if entry.get("chunk_ids"):
                old_ids_to_delete.extend(entry["chunk_ids"])
            del cache[ck]
            stats["deleted"] += 1

    # Delete old chunks
    if old_ids_to_delete:
        logger.info("Deleting %d old chunks", len(old_ids_to_delete))
        store_delete(old_ids_to_delete)

    # Embed and upsert new chunks
    if pending_chunks:
        logger.info("Embedding %d new chunks", len(pending_chunks))
        texts = [c.text for c in pending_chunks]
        embeddings = embed_texts(texts)
        documents = texts
        metadatas = [c.metadata for c in pending_chunks]
        store_upsert(pending_ids, embeddings, documents, metadatas)

    save_cache(cache)
    logger.info("Indexing complete: %s", stats)
    return stats
