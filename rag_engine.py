import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Local embedding model (runs on your machine, no API key needed)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB persistent storage
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


class RAGEngine:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name="google_sheets_data",
            metadata={"hnsw:space": "cosine"},
        )
        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.model = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")

    def _get_collection(self):
        """Get a fresh collection reference (handles stale UUID after delete/recreate)."""
        self.collection = self.chroma_client.get_or_create_collection(
            name="google_sheets_data",
            metadata={"hnsw:space": "cosine"},
        )
        return self.collection

    def index_documents(self, documents: list[str]):
        """Embed and store documents in ChromaDB."""
        if not documents:
            return

        # Clear existing data and re-index
        try:
            self.chroma_client.delete_collection("google_sheets_data")
        except Exception:
            pass
        collection = self._get_collection()

        embeddings = self.embedder.encode(documents).tolist()
        ids = [f"doc_{i}" for i in range(len(documents))]

        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
        )
        return len(documents)

    def retrieve(self, query: str, top_k: int = 8) -> list[str]:
        """Retrieve the most relevant documents for a query."""
        # Always get a fresh reference to avoid stale UUID errors
        collection = self._get_collection()
        count = collection.count()

        if count == 0:
            return []

        query_embedding = self.embedder.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, count),
        )
        return results["documents"][0] if results["documents"] else []

    def chat(self, query: str, chat_history: list[dict] = None, model: str = None) -> str:
        """RAG-powered chat: retrieve context then generate answer."""
        relevant_docs = self.retrieve(query)

        # Trim each document to keep context small for faster LLM response
        max_per_doc = 3000
        trimmed = []
        for doc in relevant_docs:
            if len(doc) > max_per_doc:
                trimmed.append(doc[:max_per_doc] + "\n...(trimmed)")
            else:
                trimmed.append(doc)

        context = "\n\n---\n\n".join(trimmed) if trimmed else "No data found."

        system_prompt = (
            "You are a helpful assistant that answers questions based on data from a Google Sheet "
            "and any linked Google Docs or Sheets. "
            "Use the provided context to answer the user's question accurately. "
            "If the answer is not in the context, say so honestly. "
            "Be concise and direct.\n\n"
            f"## Context from Google Sheet and linked documents:\n\n{context}"
        )

        messages = [{"role": "system", "content": system_prompt}]

        if chat_history:
            messages.extend(chat_history)

        messages.append({"role": "user", "content": query})

        response = self.llm_client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=512,
        )

        return response.choices[0].message.content
