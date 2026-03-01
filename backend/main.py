import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("❌ OPENROUTER_API_KEY not found in environment. Check your .env file.")

from backend.services.auth import authenticate
from backend.services.rag import hybrid_retrieve, rerank, rewrite_query, check_faithfulness
from backend.services.sql import init_db, run_sql, get_columns

# -------------------------------
# App Setup
# -------------------------------
app = FastAPI(title="FinSolve RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
static_dir = os.path.join(BASE_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

init_db()

# In-memory sessions {token: {username, role, history}}
sessions: dict = {}

#openai_api_key=os.getenv("OPENROUTER_API_KEY")
# -------------------------------
# LLM Factory
# -------------------------------
def get_llm():
    return ChatOpenAI(
        model="google/gemini-2.5-flash-lite",
        temperature=0,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )



# -------------------------------
# LLM-Based Query Router
# -------------------------------
def is_sql_query(query: str, llm) -> bool:
    """
    Uses the LLM to classify whether a user query requires SQL (structured)
    or RAG (unstructured document retrieval).
    Returns True if SQL is needed, False otherwise.
    """
    classification_prompt = f"""You are a query router. Determine if the following user query requires structured data retrieval (SQL) or unstructured document retrieval (RAG).
A query requires SQL if it involves:
- Aggregations (count, sum, average, total, min, max)
- Filtering or comparisons on structured fields (salary > X, age < Y, department = Z)
- Listing or fetching specific records from a database table
- Sorting or ranking employees or records
- Any analytical or reporting question on tabular employee data
- Questions like "show me all employees who...", "find employees where...", "how many employees..."
A query requires RAG if it involves:
- Policy questions (leave policy, HR policies, company guidelines)
- General knowledge or explanations
- Summaries of documents or reports
- Questions that do not require querying a structured employee table
Respond with ONLY one word: "SQL" or "RAG". No explanation, no punctuation.
Query: {query}"""

    try:
        result = llm.invoke(classification_prompt).content.strip().upper()
        # Defensive: if the model returns something unexpected, default to RAG
        return result == "SQL"
    except Exception:
        return False


# -------------------------------
# Schemas
# -------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    query: str


# -------------------------------
# Routes
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/login")
async def login(req: LoginRequest):
    user = authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = str(uuid.uuid4())
    sessions[token] = {
        "username": user["username"],
        "role":     user["role"],
        "history":  []
    }
    return {"token": token, "role": user["role"], "username": user["username"]}


@app.post("/logout")
async def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization in sessions:
        del sessions[authorization]
    return {"status": "logged out"}


@app.post("/chat")
async def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
    if not authorization or authorization not in sessions:
        raise HTTPException(status_code=401, detail="Unauthorized. Please login.")

    session = sessions[authorization]
    role    = session["role"]
    history = session["history"]
    query   = req.query.strip()
    llm     = get_llm()

    # -------------------------------
    # Route: SQL or RAG (LLM-based)
    # -------------------------------
    use_sql = is_sql_query(query, llm)

    # -------------------------------
    # Structured → SQL (HR only)
    # -------------------------------
    if use_sql:
        if role.lower() != "hr":
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to run structured queries."
            )

        sql_prompt = f"""You are a SQL assistant. Translate the user query into a valid SQLite SQL statement
for the table `employees`. Return ONLY the SQL code with no explanation or markdown.
Table columns: {get_columns()}
User query: {query}"""

        raw = llm.invoke(sql_prompt).content
        sql_query = raw.strip().removeprefix("```sql").removesuffix("```").strip()

        try:
            result = run_sql(sql_query)
            return {
                "type":    "table",
                "columns": result["columns"],
                "rows":    result["rows"],
                "query":   sql_query
            }
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SQL error: {str(e)}")

    # -------------------------------
    # RAG Flow
    # -------------------------------
    # Step 1 — Query Rewriting
    rewritten = rewrite_query(query, llm)

    # Step 2 — Hybrid Retrieve + RRF
    docs = hybrid_retrieve(rewritten, role, top_k=50)

    # Step 3 — Rerank
    docs = rerank(rewritten, docs, top_k=5)

    if not docs:
        return {
            "type":    "text",
            "answer":  "I don't have relevant information to answer that.",
            "sources": [],
            "faithful": True,
            "rewritten_query": rewritten
        }

    context = "\n\n".join([d.page_content for d in docs])

    # Step 4 — Format history (last 10 messages)
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Bot'}: {m['content']}"
        for m in history[-10:]
    )

    # Step 5 — Generate Answer
    prompt = f"""You are an AI assistant at FinSolve Technologies. The user has the role: {role}.
Do not answer questions outside of the user's role scope.
Conversation History:
{history_text}
Instructions:
1) If the context does not contain relevant information, respond with "I don't have that information."
2) If the question is outside your role, respond with "I'm not authorized to answer that."
3) Always keep answers concise and to the point.
4) Do not make up answers outside of the provided context.
Context:
{context}
Question: {query}"""

    answer = llm.invoke(prompt).content

    # Step 6 — Hallucination Check
    faithful = check_faithfulness(context, answer, llm)

    # Step 7 — Sources
    sources = list({d.metadata.get("source", "Unknown") for d in docs})

    # Step 8 — Save to memory
    session["history"].append({"role": "user",    "content": query})
    session["history"].append({"role": "bot", "content": answer})

    return {
        "type":            "text",
        "answer":          answer,
        "rewritten_query": rewritten,
        "sources":         sources,
        "faithful":        faithful
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
