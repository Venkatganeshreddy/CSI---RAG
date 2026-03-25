"""Citation-aware prompt construction for GPT-4o."""

from __future__ import annotations

from rag.retriever import RetrievedChunk

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions using data from Google Sheets.

RULES:
1. Answer ONLY based on the provided source chunks. If the sources don't contain the answer, say so.
2. Cite every claim using the source label exactly as given, e.g. [Source 1].
3. If multiple sources support a claim, cite all of them.
4. Be concise and direct.
"""


def build_messages(
    user_query: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
) -> list[dict]:
    """Build the full message list for the chat completion API."""
    # Format retrieved context
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"--- Source {i}: {chunk.source_label} ---\n{chunk.text}"
        )
    context_block = "\n\n".join(context_parts)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history
    if history:
        messages.extend(history)

    # User message with context
    user_content = (
        f"Use the following sources to answer my question.\n\n"
        f"{context_block}\n\n"
        f"Question: {user_query}"
    )
    messages.append({"role": "user", "content": user_content})

    return messages
