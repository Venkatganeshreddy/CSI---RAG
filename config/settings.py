import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
CRAWL_CACHE_PATH = DATA_DIR / "crawl_cache.json"
CRAWL_LOG_PATH = DATA_DIR / "crawl_log.jsonl"
CREDENTIALS_PATH = PROJECT_ROOT / "credentials" / "service_account.json"

# --- Google Sheets ---
SEED_SPREADSHEET_IDS: list[str] = [
    sid.strip()
    for sid in os.getenv("SEED_SPREADSHEET_IDS", "").split(",")
    if sid.strip()
]
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]
MAX_CRAWL_DEPTH = 5

# --- Rate Limiting ---
RATE_LIMIT_REQUESTS = 55
RATE_LIMIT_WINDOW_SECONDS = 60

# --- Chunking ---
CHUNK_TARGET_TOKENS = 512
CHUNK_ROW_OVERLAP = 2

# --- Embeddings (HuggingFace sentence-transformers, runs locally) ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 64

# --- Chat (OpenRouter API — OpenAI-compatible) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_MODEL = "openai/gpt-4o"
CHAT_TEMPERATURE = 0.1
CHAT_MAX_HISTORY_TURNS = 5
RETRIEVER_TOP_K = 8

# --- ChromaDB ---
CHROMA_COLLECTION_NAME = "gsheets"
