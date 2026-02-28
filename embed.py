import os
import hashlib
import shutil
import pandas as pd
from langchain_community.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# -------------------------------
# Configuration
# -------------------------------
BASE_DIR   = "resources/data"
CHROMA_DIR = "chroma_db"

# Known departments
DEPARTMENTS = ["engineering", "finance", "general", "hr", "marketing"]

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Markdown/CSV-aware splitter — respects heading boundaries
md_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "]
)

# -------------------------------
# Deduplication
# -------------------------------
seen_hashes = set()

def get_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode()).hexdigest()

def deduplicate(docs: list) -> list:
    unique = []
    for doc in docs:
        h = get_hash(doc.page_content)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(doc)
    return unique

# -------------------------------
# Main ingestion loop
# -------------------------------
all_split_docs = []

for department in DEPARTMENTS:
    dept_path = os.path.join(BASE_DIR, department)

    if not os.path.isdir(dept_path):
        print(f"⚠️  Folder not found, skipping: {dept_path}")
        continue

    print(f"\n🔍 Processing: {department}")
    dept_docs = []

    for file in sorted(os.listdir(dept_path)):
        file_path = os.path.join(dept_path, file)
        file_ext  = os.path.splitext(file)[-1].lower()

        # -------------------------------
        # Handle CSV files
        # -------------------------------
        if file_ext == ".csv":
            print(f"   📄 Loading CSV for embedding: {file}")
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    # Convert each row into a text document
                    text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": file,
                            "file_type": ".csv",
                            "role": department.lower(),
                            "category": department.lower()
                        }
                    )
                    dept_docs.append(doc)
                print(f"   ✅ Loaded {len(df)} rows from {file}")
            except Exception as e:
                print(f"   ❌ Failed to load CSV {file}: {e}")
            continue

        # -------------------------------
        # Handle Markdown files
        # -------------------------------
        if file_ext != ".md":
            print(f"   ⏭️  Skipping unsupported file type: {file}")
            continue

        try:
            try:
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load()
            except Exception:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()

            for doc in docs:
                doc.metadata["source"]    = file
                doc.metadata["file_type"] = ".md"
                doc.metadata["role"]      = department.lower()
                doc.metadata["category"]  = department.lower()

            dept_docs.extend(docs)
            print(f"   📄 Loaded: {file} ({len(docs)} doc(s))")

        except Exception as e:
            print(f"   ❌ Failed to load {file}: {e}")

    if not dept_docs:
        print(f"   ⚠️  No documents loaded for: {department}")
        continue

    # -------------------------------
    # Split large documents
    # -------------------------------
    split_docs = md_splitter.split_documents(dept_docs)

    # -------------------------------
    # Deduplicate
    # -------------------------------
    split_docs = deduplicate(split_docs)
    all_split_docs.extend(split_docs)
    print(f"   ✅ {len(split_docs)} unique chunks stored for: {department}")

# -------------------------------
# Build Chroma DB
# -------------------------------
if not all_split_docs:
    print("\n❌ No documents to embed. Check your resources/data folders.")
    exit(1)

print(f"\n⚙️  Building Chroma DB with {len(all_split_docs)} total chunks...")
shutil.rmtree(CHROMA_DIR, ignore_errors=True)

db = Chroma.from_documents(
    documents=all_split_docs,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR,
    collection_name="company_docs"
)

# -------------------------------
# Validation Summary
# -------------------------------
stored        = db._collection.get()
roles_found   = sorted({m.get("role", "?") for m in stored["metadatas"]})
sources_found = sorted({m.get("source", "?") for m in stored["metadatas"]})

print(f"\n🎉 Embedding complete!")
print(f"   Total chunks  : {len(stored['ids'])}")
print(f"   Roles indexed : {roles_found}")
print(f"   Files indexed : {sources_found}")
print(f"\n📋 Sample metadata (first 3):")
for meta in stored["metadatas"][:3]:
    print(f"   {meta}")