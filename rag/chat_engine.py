"""Conversation management + OpenRouter streaming."""

from __future__ import annotations

from collections.abc import Generator

from openai import OpenAI

from config.settings import (
    CHAT_MAX_HISTORY_TURNS,
    CHAT_MODEL,
    CHAT_TEMPERATURE,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from rag.prompt_builder import build_messages
from rag.retriever import RetrievedChunk, retrieve

_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)


class ChatEngine:
    def __init__(self) -> None:
        self._history: list[dict] = []

    def _trim_history(self) -> list[dict]:
        """Keep the last N turns (each turn = user + assistant)."""
        max_msgs = CHAT_MAX_HISTORY_TURNS * 2
        return self._history[-max_msgs:]

    def ask(
        self,
        query: str,
        sheet_filter: str | None = None,
    ) -> Generator[tuple[str, list[RetrievedChunk]], None, None]:
        """Stream an answer. Yields (token, sources) — sources list is populated once at start.

        Usage:
            sources = []
            for token, srcs in engine.ask("..."):
                if srcs and not sources:
                    sources = srcs
                print(token, end="")
        """
        chunks = retrieve(query, sheet_title_filter=sheet_filter)
        messages = build_messages(query, chunks, self._trim_history())

        stream = _client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=CHAT_TEMPERATURE,
            stream=True,
        )

        full_response = ""
        first = True
        for event in stream:
            delta = event.choices[0].delta
            if delta.content:
                full_response += delta.content
                yield delta.content, (chunks if first else [])
                first = False

        # Store turn in history
        self._history.append({"role": "user", "content": query})
        self._history.append({"role": "assistant", "content": full_response})

    def clear_history(self) -> None:
        self._history.clear()
