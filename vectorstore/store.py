"""ChromaDB wrapper — upsert, delete, query with metadata filtering."""

from __future__ import annotations

import logging

import chromadb

from config.settings import CHROMA_COLLECTION_NAME, CHROMA_DIR

logger = logging.getLogger(__name__)


def _get_collection() -> chromadb.Collection:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """Upsert documents into ChromaDB in batches (Chroma limit ~5461)."""
    col = _get_collection()
    batch = 5000
    for i in range(0, len(ids), batch):
        col.upsert(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch],
            documents=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )
    logger.info("Upserted %d documents", len(ids))


def delete(ids: list[str]) -> None:
    """Delete documents by ID."""
    if not ids:
        return
    col = _get_collection()
    batch = 5000
    for i in range(0, len(ids), batch):
        col.delete(ids=ids[i : i + batch])
    logger.info("Deleted %d documents", len(ids))


def query(
    query_embedding: list[float],
    n_results: int = 8,
    where: dict | None = None,
) -> dict:
    """Query ChromaDB for similar documents.

    Returns dict with keys: ids, documents, metadatas, distances.
    """
    col = _get_collection()
    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    results = col.query(**kwargs)
    return {
        "ids": results["ids"][0] if results["ids"] else [],
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
    }


def count() -> int:
    return _get_collection().count()


def list_spreadsheet_titles() -> list[str]:
    """Return unique spreadsheet titles in the collection."""
    col = _get_collection()
    result = col.get(include=["metadatas"])
    titles: set[str] = set()
    if result["metadatas"]:
        for m in result["metadatas"]:
            if m and "spreadsheet_title" in m:
                titles.add(m["spreadsheet_title"])
    return sorted(titles)
