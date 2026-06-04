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
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found in environment. Check your .env file.")

from backend.services.auth import authenticate
from backend.services.rag import (
    run_input_guardrails, run_self_rag_answer,
)
from backend.services.cache import (
    TTL_INTENT_ROUTER, TTL_RAG_ANSWER,
    get_json, set_json,
)
from backend.services.sql import init_db
from backend.services.sql_pipeline import (
    build_sql_context, execute_approved_sql, generate_sql, reject_sql_approval,
    request_sql_approval, validate_sql,
)
from backend.services.security import run_request_security_pipeline
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

def is_sql_query(query: str, llm) -> bool:
    cached = get_json("intent_router", query)
    if cached is not None:
        return bool(cached)

    prompt = f"""You are a query router. Respond with ONLY one word: "SQL" or "RAG".
SQL: aggregations, filtering, listing/sorting records from employee table.
RAG: policies, general knowledge, document summaries.
Query: {query}"""
    try:
        result = llm.invoke(prompt).content.strip().upper() == "SQL"
        set_json("intent_router", result, TTL_INTENT_ROUTER, query)
        return result
    except Exception:
        return False

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=12_000)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query cannot be blank.")
        return value

class SqlApprovalRequest(BaseModel):
    approval_id: str
    approved: bool

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/login")
async def login(req: LoginRequest):
    user = authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = str(uuid.uuid4())
    sessions[token] = {"username": user["username"], "role": user["role"], "history": [], "last_sql_context": None}
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
    original_query = req.query.strip()
    user_key = f"{session['username']}:{role}"
    try:
        security = run_request_security_pipeline(original_query, user_key)
    except PermissionError as e:
        raise HTTPException(status_code=429, detail=str(e))

    if security.blocked:
        return {
            "type": "text",
            "answer": security.reason,
            "sources": [],
            "faithful": True,
            "rewritten_query": original_query,
            "blocked": True,
            "security": {"warnings": security.warnings, "estimated_tokens": security.estimated_tokens},
        }

    query = security.query
    llm     = get_llm()
    run_id  = str(uuid.uuid4())

    use_sql = is_sql_query(query, llm)

    if use_sql:
        if role.lower() != "hr":
            raise HTTPException(status_code=403, detail="You do not have permission to run structured queries.")
        try:
            sql_query = validate_sql(generate_sql(query, llm, session.get("last_sql_context")))
            response = request_sql_approval(query, sql_query, session["username"], role)
            response["security"] = {"warnings": security.warnings, "estimated_tokens": security.estimated_tokens}
            return response
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text2SQL error: {str(e)}")

    guard = run_input_guardrails(query, role, llm)
    if guard.blocked:
        return {
            "type": "text",
            "answer": guard.reason,
            "sources": [],
            "faithful": True,
            "rewritten_query": query,
            "blocked": True,
            "security": {"warnings": security.warnings, "estimated_tokens": security.estimated_tokens},
        }

    history_cache_key = history[-10:]
    cached_answer = get_json("rag_answer:v2", role, query, history_cache_key)
    if cached_answer is not None:
        cached_answer["run_id"] = run_id
        cached_answer["cached"] = True
        return cached_answer

    response = run_self_rag_answer(query, role, history, llm)
    response["run_id"] = run_id
    response["security"] = {"warnings": security.warnings, "estimated_tokens": security.estimated_tokens}

    session["history"].append({"role": "user", "content": query})
    session["history"].append({"role": "bot", "content": response["answer"]})

    set_json("rag_answer:v2", {k: v for k, v in response.items() if k != "run_id"}, TTL_RAG_ANSWER, role, query, history_cache_key)
    return response

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/sql/approve")
async def approve_sql(req: SqlApprovalRequest, authorization: Optional[str] = Header(None)):
    if not authorization or authorization not in sessions:
        raise HTTPException(status_code=401, detail="Unauthorized. Please login.")

    session = sessions[authorization]
    if session["role"].lower() != "hr":
        raise HTTPException(status_code=403, detail="You do not have permission to approve SQL execution.")

    try:
        if req.approved:
            response = execute_approved_sql(req.approval_id, session["username"], session["role"])
            session["last_sql_context"] = build_sql_context(response.get("user_query", ""), response)
            return response
        return reject_sql_approval(req.approval_id, session["username"], session["role"])
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL execution error: {str(e)}")
