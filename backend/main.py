import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("❌ GEMINI_API_KEY not found in environment. Check your .env file.")

# LangSmith Setup
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "finsolve-rag")

from langsmith import traceable, Client
ls_client = Client()

from backend.services.auth import authenticate
from backend.services.rag import (
    hybrid_retrieve, rerank, rewrite_query,
    check_faithfulness, run_input_guardrails, run_output_guardrails,
)
from backend.services.sql import init_db, run_sql, get_columns
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI(title="FinSolve RAG API")
Instrumentator().instrument(app).expose(app)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

init_db()
sessions: dict = {}

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0, google_api_key=os.getenv("GEMINI_API_KEY"))

@traceable(name="Query Router")
def is_sql_query(query: str, llm) -> bool:
    prompt = f"""You are a query router. Respond with ONLY one word: "SQL" or "RAG".
SQL: aggregations, filtering, listing/sorting records from employee table.
RAG: policies, general knowledge, document summaries.
Query: {query}"""
    try:
        return llm.invoke(prompt).content.strip().upper() == "SQL"
    except Exception:
        return False

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/login")
async def login(req: LoginRequest):
    user = authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = str(uuid.uuid4())
    sessions[token] = {"username": user["username"], "role": user["role"], "history": []}
    return {"token": token, "role": user["role"], "username": user["username"]}

@app.post("/logout")
async def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization in sessions:
        del sessions[authorization]
    return {"status": "logged out"}

@app.post("/chat")
@traceable(name="FinSolve Chat")
async def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
    if not authorization or authorization not in sessions:
        raise HTTPException(status_code=401, detail="Unauthorized. Please login.")

    session = sessions[authorization]
    role    = session["role"]
    history = session["history"]
    query   = req.query.strip()
    llm     = get_llm()
    run_id  = str(uuid.uuid4())

    use_sql = is_sql_query(query, llm)

    if use_sql:
        if role.lower() != "hr":
            raise HTTPException(status_code=403, detail="You do not have permission to run structured queries.")
        sql_prompt = f"""You are a SQL assistant. Translate the user query into a valid SQLite SQL statement
for the table `employees`. Return ONLY the SQL code with no explanation or markdown.
Table columns: {get_columns()}
User query: {query}"""
        raw = llm.invoke(sql_prompt).content
        sql_query = raw.strip().removeprefix("```sql").removesuffix("```").strip()
        try:
            result = run_sql(sql_query)
            return {"type": "table", "columns": result["columns"], "rows": result["rows"], "query": sql_query}
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SQL error: {str(e)}")

    guard = run_input_guardrails(query, role, llm)
    if guard.blocked:
        return {"type": "text", "answer": guard.reason, "sources": [], "faithful": True, "rewritten_query": query, "blocked": True}

    rewritten = rewrite_query(query, llm)
    docs = hybrid_retrieve(rewritten, role, top_k=50)
    docs = rerank(rewritten, docs, top_k=5)

    if not docs:
        return {"type": "text", "answer": "I don't have relevant information to answer that.", "sources": [], "faithful": True, "rewritten_query": rewritten}

    context = "\n\n".join([d.page_content for d in docs])
    history_text = "\n".join(f"{'User' if m['role']=='user' else 'Bot'}: {m['content']}" for m in history[-10:])

    prompt = f"""You are a helpful AI assistant at FinSolve Technologies. The user has the role: {role}.

Conversation History:
{history_text}

Instructions:
1) Answer using ONLY the provided context below.
2) If the context contains the answer, always answer it — regardless of the user's role.
3) Only say "I'm not authorized to answer that" if the question asks for another department's CONFIDENTIAL data.
4) If the context does not contain relevant information, respond with "I don't have that information."
5) Always keep answers concise and to the point.

Context:
{context}

Question: {query}"""

    answer = llm.invoke(prompt).content
    faithful = check_faithfulness(context, answer, llm)
    answer = run_output_guardrails(answer, role)
    sources = list({d.metadata.get("source", "Unknown") for d in docs})

    session["history"].append({"role": "user", "content": query})
    session["history"].append({"role": "bot", "content": answer})

    return {"type": "text", "answer": answer, "rewritten_query": rewritten, "sources": sources, "faithful": faithful, "run_id": run_id}

@app.get("/health")
async def health():
    return {"status": "ok"}