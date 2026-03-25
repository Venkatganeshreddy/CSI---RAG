"""OpenAI embedding wrapper with batching and retry."""

from __future__ import annotations

import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)


@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(5))
def _embed_batch(texts: list[str]) -> list[list[float]]:
    resp = _client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in resp.data]


def embed_texts(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    """Embed a list of texts, batching to stay within API limits."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info("Embedding batch %d–%d of %d", i, i + len(batch), len(texts))
        all_embeddings.extend(_embed_batch(batch))
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return _embed_batch([text])[0]
