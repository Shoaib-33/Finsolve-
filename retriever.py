import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "chroma_db"

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
    collection_name="company_docs"
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
