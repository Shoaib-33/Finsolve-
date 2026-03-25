# FinSolve RAG API

An enterprise-grade **Retrieval-Augmented Generation (RAG)** system built for FinSolve Technologies. Provides role-based intelligent Q&A over internal documents and structured employee data, with full observability, evaluation, and guardrails.

---

## 🚀 Features

- **LLM-Based Query Router** — Automatically classifies queries as SQL or RAG
- **Hybrid Retrieval (BM25 + Dense)** — Combines keyword and semantic search with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking** — Re-scores retrieved chunks for maximum relevance
- **Query Rewriting** — Expands and clarifies user queries before retrieval
- **Hallucination Guardrail** — Checks answer faithfulness against retrieved context
- **Role-Based Access Control** — Filters documents and SQL access by user role
- **Conversation Memory** — Maintains per-session chat history (last 10 messages)
- **Multi-format Ingestion** — Supports Markdown and CSV files across departments
- **LangSmith Tracing** — Full observability of every LLM call and pipeline step
- **RAGAS Evaluation** — Automatic RAG quality scoring (faithfulness, answer relevancy)
- **Comprehensive Guardrails** — PII detection, blocked topics, out-of-scope protection
- **Prometheus + Grafana** — Real-time API monitoring and dashboards

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Input Guardrails (PII check, blocked topics, out-of-scope)
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
            ├── Hallucination Check (faithfulness verdict)
            ├── Output Guardrails (PII scrubbing)
            └── RAGAS Evaluation (background thread → LangSmith)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| LLM | Gemini 2.5 Flash (via Google AI) |
| LLM Orchestration | LangChain |
| Tracing & Observability | LangSmith |
| RAG Evaluation | RAGAS |
| Vector Store | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 (SentenceTransformers) |
| Keyword Retrieval | BM25 (LangChain Community) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Structured Data | SQLite |
| Metrics | Prometheus + Grafana |
| Containerization | Docker + Docker Compose |
| Templating | Jinja2 |

---

## 📁 Project Structure

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
├── chroma_db/                  # Persisted vector store (generated at build)
├── embed.py                    # Document ingestion pipeline
├── retriever.py                # ChromaDB retriever setup
├── employees.db                # SQLite employee database
├── docker-compose.yml          # App + Prometheus + Grafana
├── prometheus.yml              # Prometheus scrape config
├── Dockerfile                  # Container build
└── requirements.txt            # Python dependencies
```

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- A Gemini API key
- A LangSmith API key (optional, for tracing)

### Installation

```bash
git clone https://github.com/Shoaib-33/Finsolve-.git
cd Finsolve-
```

### Environment Setup

Create a `.env` file in the project root (see `.env.example`):

```env
GEMINI_API_KEY=your_gemini_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=finsolve-rag
```

### Run with Docker Compose

```bash
docker-compose up -d --build
```

This starts 3 services:

| Service | URL |
|---|---|
| FinSolve API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## 🔄 Pipeline Flow

```
User Query
    ↓
Input Guardrails (regex + LLM)
    ↓
Query Rewriting
    ↓
Hybrid Retrieve (BM25 + Dense, RRF fusion)
    ↓
Rerank top 5 docs
    ↓
Build prompt with context + chat history
    ↓
LLM generates answer
    ↓
Hallucination check
    ↓
Output Guardrails (PII scrubbing)
    ↓
Return answer + sources + faithfulness flag
    ↓ (background)
RAGAS evaluation → LangSmith feedback
```

---

## 🛡️ Guardrails

### Input Guardrails (before RAG runs)

| Check | Method | Blocks |
|---|---|---|
| Blocked topics | Regex (free) | hack, exploit, sql injection, jailbreak... |
| PII in query | Regex (free) | email, phone, NID, passport, credit card |
| Out-of-scope | LLM check | Cross-department confidential data |

### Output Guardrails (before answer is returned)

| Role | Redaction |
|---|---|
| HR | Credit card, passport, IP address |
| All others | Full PII redaction |

---

## 📊 Observability

### LangSmith
Every LLM call is automatically traced. View full pipeline traces at [smith.langchain.com](https://smith.langchain.com) under project `finsolve-rag`.

### RAGAS Evaluation
Runs automatically in the background after every RAG response:
- `faithfulness` — Is the answer supported by retrieved docs?
- `answer_relevancy` — Does the answer address the question?

Scores are logged back to LangSmith per request.

### Prometheus + Grafana
- Metrics exposed at `/metrics`
- Prometheus scrapes every 15 seconds
- Grafana dashboard shows request rate, latency p95, error rate, endpoint breakdown

---

## 🔐 Role-Based Access

| Role | RAG Access | SQL Access |
|---|---|---|
| general | General documents only | ❌ |
| engineering | Engineering + General docs | ❌ |
| finance | Finance + General docs | ❌ |
| marketing | Marketing + General docs | ❌ |
| hr | All documents | ✅ |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/login` | Authenticate and receive session token |
| POST | `/logout` | Invalidate session |
| POST | `/chat` | Submit a query (requires Authorization header) |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

### Login
```bash
POST /login
{
  "username": "alice",
  "password": "password123"
}
```

### Chat
```bash
POST /chat
Authorization: <session-token>
{
  "query": "What is the leave policy for engineering staff?"
}
```

---

## 📹 Demo

[Watch on YouTube](https://youtu.be/mcl03F1ANpo)

---

## 📄 License

MIT