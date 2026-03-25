"""Query embeddings + ChromaDB similarity search."""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import RETRIEVER_TOP_K
from vectorstore.embedder import embed_query
from vectorstore.store import query as store_query


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    distance: float

    @property
    def source_label(self) -> str:
        m = self.metadata
        return (
            f'[Sheet: "{m.get("spreadsheet_title", "?")}" > '
            f'Tab: "{m.get("tab_name", "?")}" > '
            f'Rows: {m.get("start_row", "?")}-{m.get("end_row", "?")}]'
        )

    @property
    def sheet_url(self) -> str:
        sid = self.metadata.get("spreadsheet_id", "")
        gid = self.metadata.get("tab_gid", 0)
        return f"https://docs.google.com/spreadsheets/d/{sid}/edit#gid={gid}"


def retrieve(
    query: str,
    top_k: int = RETRIEVER_TOP_K,
    sheet_title_filter: str | None = None,
) -> list[RetrievedChunk]:
    """Embed query, search ChromaDB, return ranked chunks."""
    embedding = embed_query(query)

    where = None
    if sheet_title_filter:
        where = {"spreadsheet_title": sheet_title_filter}

    results = store_query(embedding, n_results=top_k, where=where)

    chunks: list[RetrievedChunk] = []
    for doc, meta, dist in zip(results["documents"], results["metadatas"], results["distances"]):
        chunks.append(RetrievedChunk(text=doc, metadata=meta, distance=dist))

    return chunks
