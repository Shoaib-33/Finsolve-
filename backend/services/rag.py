import re
from pathlib import Path
from typing import Any

import pandas as pd
from retriever import db
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from sentence_transformers import CrossEncoder


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESOURCES_DIR = PROJECT_DIR / "resources" / "data"
DEPARTMENTS = ["engineering", "finance", "general", "hr", "marketing"]
md_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "],
)

# -------------------------------
# Reranker
# -------------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, docs: list, top_k: int = 5) -> list:
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


# -------------------------------
# BM25 per role — built once at startup
# -------------------------------
def load_resource_docs() -> list:
    docs = []
    for department in DEPARTMENTS:
        dept_path = RESOURCES_DIR / department
        if not dept_path.is_dir():
            continue

        for path in sorted(dept_path.iterdir()):
            file_ext = path.suffix.lower()
            if file_ext == ".csv":
                try:
                    df = pd.read_csv(path)
                    for _, row in df.iterrows():
                        text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "source": path.name,
                                "file_type": ".csv",
                                "role": department,
                                "category": department,
                            },
                        ))
                except Exception:
                    continue
            elif file_ext == ".md":
                text = path.read_text(encoding="utf-8")
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "file_type": ".md",
                        "role": department,
                        "category": department,
                    },
                ))

    return md_splitter.split_documents(docs)


def init_bm25_by_role() -> dict:
    stored = db.get()
    role_docs = {}
    for text, meta in zip(stored["documents"], stored["metadatas"]):
        role = meta.get("role", "general")
        if role not in role_docs:
            role_docs[role] = []
        role_docs[role].append(Document(page_content=text, metadata=meta))

    if not role_docs:
        for doc in load_resource_docs():
            role = doc.metadata.get("role", "general")
            if role not in role_docs:
                role_docs[role] = []
            role_docs[role].append(doc)

    return {
        role: BM25Retriever.from_documents(docs)
        for role, docs in role_docs.items()
    }


bm25_by_role = init_bm25_by_role()


# -------------------------------
# Hybrid Retrieval with RRF
# -------------------------------
def hybrid_retrieve(query: str, role: str, top_k: int = 50, k: int = 60) -> list:
    retrieval_query = expand_retrieval_query(query)
    bm25_docs = []
    accessible_roles = DEPARTMENTS if role.lower() == "hr" else [role, "general"]
    for accessible_role in accessible_roles:
        if accessible_role in bm25_by_role:
            bm25_docs.extend(bm25_by_role[accessible_role].get_relevant_documents(retrieval_query))

    dense_docs = []
    try:
        for accessible_role in accessible_roles:
            dense_docs.extend(db.as_retriever(
                search_kwargs={"filter": {"role": accessible_role}, "k": top_k}
            ).get_relevant_documents(retrieval_query))
    except Exception:
        dense_docs = []

    rrf_scores = {}

    def add_to_rrf(docs_list, weight=1.0):
        for rank, doc in enumerate(docs_list):
            key = doc.page_content
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "doc": doc}
            rrf_scores[key]["score"] += weight * (1 / (k + rank + 1))

    add_to_rrf(bm25_docs, weight=1.0)
    add_to_rrf(dense_docs, weight=1.0)

    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [entry["doc"] for entry in sorted_docs[:top_k]]


def expand_retrieval_query(query: str) -> str:
    query_lower = query.lower()
    if "architecture" in query_lower and re.search(r"\b(layer|layers|component|components)\b", query_lower):
        return (
            f"{query} High-Level Architecture Client Apps API Gateway "
            "Microservices Layer Data Layer Infrastructure Layer"
        )
    return query


# -------------------------------
# Query Rewriting
# -------------------------------
def rewrite_query(query: str, llm) -> str:
    prompt = f"""Rewrite this query to be more specific and retrieval-friendly 
for a corporate knowledge base. Return only the rewritten query, nothing else.
Query: {query}"""
    return llm.invoke(prompt).content.strip()


# -------------------------------
# Hallucination Guardrail (existing)
# -------------------------------
def check_faithfulness(context: str, answer: str, llm) -> bool:
    prompt = f"""You are a factual consistency checker.
Given the context and an answer, determine if the answer is fully supported by the context.
Reply with only YES or NO.

Context: {context}
Answer: {answer}"""
    verdict = llm.invoke(prompt).content.strip().upper()
    return verdict == "YES"


# -------------------------------
# Self-RAG
# -------------------------------
def grade_retrieved_docs(query: str, docs: list, llm, max_docs: int = 8) -> list:
    """
    Self-RAG retrieval grading. Keeps only docs the LLM judges useful for the
    user question. Fails open so retrieval still works if the grader errors.
    """
    if not docs:
        return []

    candidates = docs[:max_docs]
    snippets = []
    for index, doc in enumerate(candidates, start=1):
        snippet = doc.page_content[:900].replace("\n", " ")
        source = doc.metadata.get("source", "Unknown")
        snippets.append(f"[{index}] source={source}\n{snippet}")

    prompt = f"""You are a retrieval grader for a corporate RAG system.

Question:
{query}

Candidate document snippets:
{chr(10).join(snippets)}

Return only the numbers of snippets that contain information useful for answering the question.
Use comma-separated numbers like: 1,3,5
If none are useful, return NONE."""

    try:
        verdict = llm.invoke(prompt).content.strip().upper()
        if verdict == "NONE":
            return []
        selected = {int(match) for match in re.findall(r"\d+", verdict)}
        filtered = [doc for index, doc in enumerate(candidates, start=1) if index in selected]
        return filtered or candidates
    except Exception:
        return candidates


def generate_answer_from_docs(query: str, role: str, history: list, docs: list, llm) -> str:
    context = "\n\n".join([d.page_content for d in docs])
    history_text = "\n".join(
        f"{'User' if m['role']=='user' else 'Bot'}: {m['content']}"
        for m in history[-10:]
    )

    prompt = f"""You are a helpful AI assistant at FinSolve Technologies. The user has the role: {role}.

Conversation History:
{history_text}

Instructions:
1) Answer using ONLY the provided context below.
2) If the context contains the answer, always answer it, regardless of the user's role.
3) Only say "I'm not authorized to answer that" if the question asks for another department's CONFIDENTIAL data.
4) If the context does not contain relevant information, respond with "I don't have that information."
5) Always keep answers concise and to the point.

Context:
{context}

Question: {query}"""

    return llm.invoke(prompt).content


def grade_answer_usefulness(query: str, answer: str, llm) -> bool:
    prompt = f"""You are an answer grader.
Determine whether the answer directly addresses the user's question.
Reply with only YES or NO.

Question: {query}
Answer: {answer}"""
    try:
        return llm.invoke(prompt).content.strip().upper() == "YES"
    except Exception:
        return True


def rewrite_after_failed_answer(query: str, previous_query: str, answer: str, llm) -> str:
    prompt = f"""Rewrite the user's question for a better corporate document search.
The previous retrieval attempt did not produce a sufficiently supported answer.
Return only the rewritten search query.

Original question: {query}
Previous search query: {previous_query}
Previous answer: {answer}"""
    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return previous_query


def run_self_rag_answer(query: str, role: str, history: list, llm, max_retries: int = 1) -> dict[str, Any]:
    search_query = rewrite_query(query, llm)
    last_answer = ""
    last_docs = []
    attempts = 0

    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        docs = hybrid_retrieve(search_query, role, top_k=50)
        docs = rerank(search_query, docs, top_k=8)
        docs = grade_retrieved_docs(query, docs, llm, max_docs=8)
        docs = rerank(search_query, docs, top_k=5)
        last_docs = docs

        if not docs:
            last_answer = "I don't have relevant information to answer that."
            if attempt < max_retries:
                search_query = rewrite_after_failed_answer(query, search_query, last_answer, llm)
                continue
            break

        context = "\n\n".join([d.page_content for d in docs])
        answer = generate_answer_from_docs(query, role, history, docs, llm)
        faithful = check_faithfulness(context, answer, llm)
        useful = grade_answer_usefulness(query, answer, llm)
        last_answer = answer

        if faithful and useful:
            sources = list({d.metadata.get("source", "Unknown") for d in docs})
            return {
                "type": "text",
                "answer": run_output_guardrails(answer, role),
                "rewritten_query": search_query,
                "sources": sources,
                "faithful": faithful,
                "self_rag": {
                    "attempts": attempts,
                    "docs_used": len(docs),
                    "answer_useful": useful,
                },
            }

        if attempt < max_retries:
            search_query = rewrite_after_failed_answer(query, search_query, answer, llm)

    sources = list({d.metadata.get("source", "Unknown") for d in last_docs})
    context = "\n\n".join([d.page_content for d in last_docs])
    faithful = check_faithfulness(context, last_answer, llm) if context and last_answer else True
    return {
        "type": "text",
        "answer": run_output_guardrails(last_answer, role),
        "rewritten_query": search_query,
        "sources": sources,
        "faithful": faithful,
        "self_rag": {
            "attempts": attempts,
            "docs_used": len(last_docs),
            "answer_useful": False,
        },
    }


# ================================================================
# GUARDRAILS
# ================================================================

# -------------------------------
# 1. PII Patterns — personal data only
# -------------------------------
PII_PATTERNS = {
    "email":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone":       r"\b(\+?\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
    "nid":         r"\b\d{13,17}\b",           # National ID — tightened to 13-17 digits
    "passport":    r"\b[A-Z]{2}\d{7}\b",       # Passport — tightened format
    "credit_card": r"\b(?:\d[ -]?){15,16}\b",  # Credit card
    "ip_address":  r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "dob":         r"\b(0?[1-9]|[12]\d|3[01])[\/\-](0?[1-9]|1[0-2])[\/\-](\d{2}|\d{4})\b",
}


def detect_pii(text: str) -> dict:
    found = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found[pii_type] = matches
    return found


def redact_pii(text: str) -> str:
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED-{pii_type.upper()}]", text)
    return text


def check_query_for_pii(query: str) -> tuple[bool, str]:
    found = detect_pii(query)
    if found:
        types = ", ".join(found.keys())
        return True, f"Your query contains sensitive personal data ({types}). Please avoid sharing personal information."
    return False, ""


# -------------------------------
# 2. Blocked Topics — only truly harmful content
#    General company questions are ALWAYS allowed
# -------------------------------
BLOCKED_TOPICS = [
    "how to hack",
    "how to exploit",
    "jailbreak",
    "bypass security",
    "sql injection",
    "insider trading",
    "confidential merger",
]


def check_blocked_topics(query: str) -> tuple[bool, str]:
    """
    Only blocks clearly harmful or malicious queries.
    Uses full phrase matching to avoid false positives.
    """
    query_lower = query.lower()
    for topic in BLOCKED_TOPICS:
        if re.search(r'\b' + re.escape(topic) + r'\b', query_lower):
            return True, f"This type of query is not permitted."
    return False, ""


# -------------------------------
# 3. Role-Based SQL Access Guard
#    (RAG questions are open to all roles)
# -------------------------------
ROLE_CONFIDENTIAL_DATA = {
    "finance":     ["hr salary data", "employee personal records"],
    "engineering": ["finance budget details", "confidential financial reports"],
    "marketing":   ["finance budget details", "hr salary data"],
    "general":     ["finance budget details", "hr salary data", "confidential reports"],
}


def check_out_of_scope(query: str, role: str, llm) -> tuple[bool, str]:
    """
    Only blocks cross-department CONFIDENTIAL data access.
    General company info (headquarters, policies, holidays, etc.)
    is always IN SCOPE for everyone.
    """
    prompt = f"""You are a query scope validator for a corporate chatbot at FinSolve Technologies.

The user has the role: "{role}"

IMPORTANT RULES:
- General company information (headquarters, mission, vision, holidays, office locations, company history, general policies) is ALWAYS IN SCOPE for ALL roles.
- Department-specific documents are IN SCOPE for that department AND for HR.
- Only mark OUT OF SCOPE if the query explicitly asks for another department's CONFIDENTIAL financial or personal data.

Examples that are ALWAYS IN SCOPE (regardless of role):
- "Where is the headquarters?"
- "What are the company holidays?"
- "What is FinSolve's mission?"
- "What is the leave policy?"
- "Who is the CEO?"

Examples that are OUT OF SCOPE:
- A marketing user asking "show me the finance department's salary budget breakdown"
- An engineering user asking "give me the personal salary details of all HR employees"

Respond with ONLY:
- "IN_SCOPE" if the query is appropriate
- "OUT_OF_SCOPE: <one line reason>" if it truly crosses confidential data boundaries

Query: {query}"""

    try:
        result = llm.invoke(prompt).content.strip()
        if result.startswith("OUT_OF_SCOPE"):
            reason = result.replace("OUT_OF_SCOPE:", "").strip()
            return True, reason
        return False, ""
    except Exception:
        return False, ""  # fail open — never block on LLM error


# -------------------------------
# 4. Response PII Scrubber
# -------------------------------
def scrub_response(answer: str, role: str) -> str:
    """
    Scrubs PII from answers before returning to user.
    HR role: only redact credit card, passport, IP
    All others: full PII redaction
    """
    if role == "hr":
        sensitive_only = ["credit_card", "passport", "ip_address"]
        for pii_type in sensitive_only:
            pattern = PII_PATTERNS[pii_type]
            answer = re.sub(pattern, f"[REDACTED-{pii_type.upper()}]", answer)
        return answer
    return redact_pii(answer)


# -------------------------------
# 5. Master Guardrail Runner
# -------------------------------
class GuardrailResult:
    def __init__(self, blocked: bool, reason: str = ""):
        self.blocked = blocked
        self.reason  = reason


def run_input_guardrails(query: str, role: str, llm) -> GuardrailResult:
    """
    Runs all input guardrails cheapest first:
    1. Blocked topics (regex — free)
    2. PII in query (regex — free)
    3. Out-of-scope check (LLM — only for cross-dept confidential data)
    """
    # Step 1 — Blocked topics (harmful content only)
    blocked, reason = check_blocked_topics(query)
    if blocked:
        return GuardrailResult(blocked=True, reason=reason)

    # Step 2 — PII in query
    has_pii, warning = check_query_for_pii(query)
    if has_pii:
        return GuardrailResult(blocked=True, reason=warning)

    # Step 3 — Out-of-scope (confidential cross-dept data only)
    out_of_scope, reason = check_out_of_scope(query, role, llm)
    if out_of_scope:
        return GuardrailResult(blocked=True, reason=f"Access denied: {reason}")

    return GuardrailResult(blocked=False)


def run_output_guardrails(answer: str, role: str) -> str:
    return scrub_response(answer, role)
