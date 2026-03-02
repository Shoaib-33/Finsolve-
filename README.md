# FinSolve RAG API

An enterprise-grade Retrieval-Augmented Generation (RAG) system built for FinSolve Technologies, enabling role-based intelligent Q&A over internal documents and structured employee data.

---

## Overview

FinSolve RAG API combines hybrid document retrieval with SQL-based structured querying to give employees accurate, role-scoped answers from company knowledge bases. It uses an LLM-powered query router to decide whether a question needs document retrieval (RAG) or database querying (SQL), ensuring the right data source is always used.

---

## Features

- **LLM-Based Query Router** — Automatically classifies queries as SQL (structured) or RAG (unstructured)
- **Hybrid Retrieval (BM25 + Dense)** — Combines keyword and semantic search with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking** — Re-scores retrieved chunks for maximum relevance
- **Query Rewriting** — Expands and clarifies user queries before retrieval
- **Hallucination Guardrail** — Checks answer faithfulness against retrieved context
- **Role-Based Access Control** — Filters documents and SQL access by user role
- **Conversation Memory** — Maintains per-session chat history (last 10 messages)
- **Multi-format Ingestion** — Supports Markdown and CSV files across departments

---

## Architecture

```
User Query
    │
    ▼
LLM Query Router
    │
    ├── SQL Path (HR only)
    │       └── LLM generates SQLite query → runs against employees table → returns table
    │
    └── RAG Path
            ├── Query Rewriting (LLM)
            ├── Hybrid Retrieval: BM25 + Dense (Chroma) with RRF fusion
            ├── Cross-Encoder Reranking (top 5 chunks)
            ├── LLM Answer Generation (role-scoped prompt)
            └── Hallucination Check (faithfulness verdict)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| LLM | Google Gemini 2.5 Flash Lite (via OpenRouter) |
| LLM Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Keyword Retrieval | BM25 (LangChain Community) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Structured Data | SQLite |
| Templating | Jinja2 |

---

## Project Structure

```
├── backend/
│   ├── main.py                 # FastAPI app, routes, chat logic
│   └── services/
│       ├── auth.py             # User authentication
│       ├── rag.py              # Hybrid retrieval, reranking, rewriting, guardrails
│       └── sql.py              # SQLite init and query execution
├── resources/
│   └── data/
│       ├── engineering/        # Department-specific documents
│       ├── finance/
│       ├── general/
│       ├── hr/
│       └── marketing/
├── templates/
│   └── index.html              # Frontend UI
├── static/                     # Static assets
├── chroma_db/                  # Persisted vector store (generated)
├── ingest.py                   # Document ingestion pipeline
└── .env                        # Environment variables
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/) API key

### Installation

```bash
git clone <your-repo-url>
cd finsolve-rag

pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Document Ingestion

Add your department documents (`.md` or `.csv`) to `resources/data/<department>/`, then run:

```bash
python ingest.py
```

This builds the ChromaDB vector store used for retrieval.

### Run the Server

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/login` | Authenticate and receive session token |
| `POST` | `/logout` | Invalidate session |
| `POST` | `/chat` | Submit a query (requires `Authorization` header) |
| `GET` | `/health` | Health check |

### Login

```bash
POST /login
Content-Type: application/json

{
  "username": "alice",
  "password": "password123"
}
```

Response:
```json
{
  "token": "<session-token>",
  "role": "hr",
  "username": "alice"
}
```

### Chat

```bash
POST /chat
Authorization: <session-token>
Content-Type: application/json

{
  "query": "What is the leave policy for engineering staff?"
}
```

Response (RAG):
```json
{
  "type": "text",
  "answer": "Engineering staff are entitled to...",
  "rewritten_query": "Leave entitlement policy for engineering department",
  "sources": ["engineering_hr_policy.md"],
  "faithful": true
}
```

Response (SQL — HR role only):
```json
{
  "type": "table",
  "columns": ["name", "department", "salary"],
  "rows": [["Alice", "Engineering", 95000]],
  "query": "SELECT name, department, salary FROM employees WHERE ..."
}
```

---

## Role-Based Access

| Role | RAG Access | SQL Access |
|---|---|---|
| `general` | General documents only | ❌ |
| `engineering` | Engineering + General docs | ❌ |
| `finance` | Finance + General docs | ❌ |
| `marketing` | Marketing + General docs | ❌ |
| `hr` | All documents | ✅ |

---

## Document Ingestion Details

The `ingest.py` pipeline:

1. Reads `.md` and `.csv` files from each department folder
2. Splits Markdown files into chunks (500 tokens, 50 overlap) respecting heading boundaries
3. Converts CSV rows into structured text documents
4. Deduplicates chunks via MD5 hashing
5. Embeds all chunks and persists them to ChromaDB

---

## License

MIT
