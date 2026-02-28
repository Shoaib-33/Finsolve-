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

    # Also fetch general docs accessible to all roles
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
    add_to_rrf(dense_general, weight=0.8)  # slight downweight for general docs

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
# Hallucination Guardrail
# -------------------------------
def check_faithfulness(context: str, answer: str, llm) -> bool:
    prompt = f"""You are a factual consistency checker.
Given the context and an answer, determine if the answer is fully supported by the context.
Reply with only YES or NO.

Context: {context}
Answer: {answer}"""
    verdict = llm.invoke(prompt).content.strip().upper()
    return verdict == "YES"
