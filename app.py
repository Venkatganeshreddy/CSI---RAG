import os
import streamlit as st
from dotenv import load_dotenv
from sheets_loader import (
    fetch_sheet_with_hyperlinks,
    fetch_linked_documents,
    build_documents,
    rows_to_documents,
    authenticate,
)
from googleapiclient.discovery import build
from rag_engine import RAGEngine

load_dotenv()

st.set_page_config(
    page_title="Google Sheets RAG Chatbot",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Google Sheets RAG Chatbot")
st.caption("Ask questions about your Google Sheet data")


@st.cache_resource
def get_rag_engine():
    """Initialize RAG engine (cached so it loads once)."""
    return RAGEngine()


@st.cache_data
def get_sheet_names():
    """Fetch all sheet names from the spreadsheet."""
    creds = authenticate()
    service = build("sheets", "v4", credentials=creds)
    spreadsheet_id = os.environ["SPREADSHEET_ID"]
    result = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    return [s["properties"]["title"] for s in result.get("sheets", [])]


def load_and_index_data(rag: RAGEngine, sheet_name: str):
    """Fetch Google Sheet data, linked documents, and index everything."""
    spreadsheet_id = os.environ["SPREADSHEET_ID"]

    with st.spinner("Fetching data from Google Sheets..."):
        rows, hyperlinks = fetch_sheet_with_hyperlinks(spreadsheet_id, sheet_name)

    if not rows:
        st.error("No data found in the sheet. Check your SPREADSHEET_ID and SHEET_NAME.")
        return False

    # Fetch linked Google Docs/Sheets (and nested links inside them)
    linked_docs = []
    if hyperlinks:
        st.info(f"Found {len(hyperlinks)} linked document(s). Fetching content (including nested links)...")
        progress_bar = st.progress(0, text="Fetching linked documents...")

        def _update_progress(current, total, message=""):
            progress_bar.progress(
                current / total if total > 0 else 1.0,
                text=message or f"Fetching document {current}/{total}...",
            )

        linked_docs = fetch_linked_documents(
            hyperlinks, progress_callback=_update_progress
        )
        progress_bar.empty()

        ok = sum(1 for d in linked_docs if d["content"])
        fail = sum(1 for d in linked_docs if d["error"])
        if fail:
            st.warning(f"Fetched {ok} doc(s), {fail} failed (permission denied or unavailable).")
        elif ok:
            st.success(f"Fetched {ok} document(s) (including nested links).")

    documents = build_documents(rows, linked_docs)

    with st.spinner(f"Indexing {len(documents)} documents..."):
        rag.index_documents(documents)

    linked_note = f" + {sum(1 for d in linked_docs if d['content'])} linked docs" if linked_docs else ""
    st.success(f"Indexed {len(rows)} rows{linked_note} from the sheet!")
    return True


# Free / low-cost models on OpenRouter (generous rate limits)
FREE_MODELS = {
    "Gemini 2.0 Flash ($0.10/M)": "google/gemini-2.0-flash-001",
    "Gemini 2.0 Flash Lite ($0.07/M)": "google/gemini-2.0-flash-lite-001",
    "Gemini 2.5 Flash ($0.30/M)": "google/gemini-2.5-flash",
    "DeepSeek V3.1 ($0.15/M)": "deepseek/deepseek-chat-v3.1",
    "DeepSeek R1 ($0.70/M)": "deepseek/deepseek-r1-0528",
}

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    # --- Sheet Selection ---
    sheet_names = get_sheet_names()
    selected_sheet = st.selectbox("Sheet", options=sheet_names, index=0)

    if "current_sheet" not in st.session_state:
        st.session_state.current_sheet = selected_sheet

    if selected_sheet != st.session_state.current_sheet:
        st.session_state.current_sheet = selected_sheet
        st.cache_resource.clear()
        st.session_state.pop("data_loaded", None)
        st.session_state.pop("messages", None)
        st.rerun()

    st.divider()

    # --- Model Selection ---
    selected_model_name = st.selectbox(
        "Model",
        options=list(FREE_MODELS.keys()),
        index=0,
    )
    selected_model = FREE_MODELS[selected_model_name]

    if st.button("Reload Sheet Data"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.pop("data_loaded", None)
        st.session_state.pop("messages", None)
        st.rerun()

    st.divider()
    st.markdown(
        f"**Sheet ID:** `{os.environ.get('SPREADSHEET_ID', 'not set')[:20]}...`"
    )

# --- Initialize ---
rag = get_rag_engine()

if "data_loaded" not in st.session_state:
    if load_and_index_data(rag, sheet_name=st.session_state.current_sheet):
        st.session_state.data_loaded = True
    else:
        st.stop()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your sheet data..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass recent chat history for context (last 10 messages)
            history = st.session_state.messages[-10:-1] if len(st.session_state.messages) > 1 else []
            response = rag.chat(prompt, chat_history=history, model=selected_model)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
