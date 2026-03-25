import re
from retriever import db
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from sentence_transformers import CrossEncoder

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
def init_bm25_by_role() -> dict:
    stored = db.get()
    role_docs = {}
    for text, meta in zip(stored["documents"], stored["metadatas"]):
        role = meta.get("role", "general")
        if role not in role_docs:
            role_docs[role] = []
        role_docs[role].append(Document(page_content=text, metadata=meta))

    return {
        role: BM25Retriever.from_documents(docs)
        for role, docs in role_docs.items()
    }


bm25_by_role = init_bm25_by_role()


# -------------------------------
# Hybrid Retrieval with RRF
# -------------------------------
def hybrid_retrieve(query: str, role: str, top_k: int = 50, k: int = 60) -> list:
    bm25_docs = []
    if role in bm25_by_role:
        bm25_docs = bm25_by_role[role].get_relevant_documents(query)

    dense_role = db.as_retriever(
        search_kwargs={"filter": {"role": role}, "k": top_k}
    ).get_relevant_documents(query)

    dense_general = db.as_retriever(
        search_kwargs={"filter": {"role": "general"}, "k": top_k}
    ).get_relevant_documents(query)

    rrf_scores = {}

    def add_to_rrf(docs_list, weight=1.0):
        for rank, doc in enumerate(docs_list):
            key = doc.page_content
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "doc": doc}
            rrf_scores[key]["score"] += weight * (1 / (k + rank + 1))

    add_to_rrf(bm25_docs,     weight=1.0)
    add_to_rrf(dense_role,    weight=1.0)
    add_to_rrf(dense_general, weight=0.8)

    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [entry["doc"] for entry in sorted_docs[:top_k]]


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