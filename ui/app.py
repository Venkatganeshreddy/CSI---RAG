"""Streamlit chat interface with sidebar controls."""

import streamlit as st

from rag.chat_engine import ChatEngine
from rag.retriever import RetrievedChunk
from vectorstore.store import count as store_count
from vectorstore.store import list_spreadsheet_titles


def _render_sources(sources: list[RetrievedChunk]) -> None:
    """Render expandable source cards below a response."""
    if not sources:
        return
    with st.expander(f"View Sources ({len(sources)})"):
        for i, src in enumerate(sources, 1):
            m = src.metadata
            st.markdown(
                f"**Source {i}**: {src.source_label}  \n"
                f"[Open in Google Sheets]({src.sheet_url})"
            )
            st.code(src.text[:500], language=None)
            if i < len(sources):
                st.divider()


def main() -> None:
    st.set_page_config(page_title="Sheets RAG Chat", page_icon="📊", layout="wide")
    st.title("Google Sheets RAG Chatbot")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Controls")

        if st.button("Re-crawl & Update Index"):
            with st.spinner("Crawling and indexing..."):
                from ingestion.crawler import crawl
                from vectorstore.index_manager import index

                result = crawl()
                stats = index(result)
                st.success(
                    f"Done! Added: {stats['added']}, Updated: {stats['updated']}, "
                    f"Skipped: {stats['skipped']}, Deleted: {stats['deleted']}"
                )

        st.divider()
        st.metric("Indexed chunks", store_count())

        titles = list_spreadsheet_titles()
        filter_options = ["All sheets"] + titles
        sheet_filter = st.selectbox("Filter by sheet", filter_options)
        if sheet_filter == "All sheets":
            sheet_filter = None

        st.divider()
        if st.button("Clear chat history"):
            st.session_state.messages = []
            if "engine" in st.session_state:
                st.session_state.engine.clear_history()
            st.rerun()

    # --- Session state ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "engine" not in st.session_state:
        st.session_state.engine = ChatEngine()

    # --- Chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    # --- Chat input ---
    if prompt := st.chat_input("Ask about your Google Sheets..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            sources: list[RetrievedChunk] = []

            for token, srcs in st.session_state.engine.ask(prompt, sheet_filter=sheet_filter):
                if srcs and not sources:
                    sources = srcs
                full_response += token
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
            _render_sources(sources)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        })


if __name__ == "__main__":
    main()
