"""HuggingFace sentence-transformers embedding wrapper (runs locally, free)."""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    """Embed a list of texts locally using sentence-transformers."""
    model = _get_model()
    logger.info("Embedding %d texts (batch_size=%d)", len(texts), batch_size)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
