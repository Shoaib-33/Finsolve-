import os
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from backend.services.cache import CachedEmbeddings

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

embedding_model = CachedEmbeddings(
    SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL),
    model_cache_id=EMBEDDING_MODEL,
)

db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
    collection_name="company_docs",
    client_settings=Settings(anonymized_telemetry=False),
)

# -------------------------------
# User store (replace with DB in production)
# -------------------------------
users_db = {
    "alice": {"password": "hr123",          "role": "hr"},
    "bob":   {"password": "eng123",         "role": "engineering"},
    "carol": {"password": "fin123",         "role": "finance"},
    "admin": {"password": "admin123",       "role": "general"},
}
