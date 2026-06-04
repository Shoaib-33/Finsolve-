import hashlib
import json
import os
from typing import Any


TTL_EMBEDDING = 7 * 24 * 60 * 60
TTL_INTENT_ROUTER = 24 * 60 * 60
TTL_SQL_GEN = 24 * 60 * 60
TTL_SQL_RESULT = 15 * 60
TTL_RAG_ANSWER = 60 * 60

CACHE_PREFIX = os.getenv("CACHE_PREFIX", "finsolve")

_client = None
_client_checked = False


def _redis():
    global _client, _client_checked
    if _client_checked:
        return _client

    _client_checked = True
    url = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    if not url or not token:
        return None

    try:
        from upstash_redis import Redis

        _client = Redis(url=url, token=token)
    except Exception as exc:
        print(f"Upstash Redis cache disabled: {exc}")
        _client = None

    return _client


def _cache_key(namespace: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{CACHE_PREFIX}:{namespace}:{digest}"


def get_json(namespace: str, *parts: Any) -> Any | None:
    redis = _redis()
    if redis is None:
        return None

    key = _cache_key(namespace, *parts)
    try:
        value = redis.get(key)
    except Exception as exc:
        print(f"Cache read failed for {namespace}: {exc}")
        return None

    if value is None:
        return None

    try:
        return json.loads(value)
    except Exception:
        return None


def set_json(namespace: str, value: Any, ttl_seconds: int, *parts: Any) -> None:
    redis = _redis()
    if redis is None:
        return

    key = _cache_key(namespace, *parts)
    try:
        redis.set(key, json.dumps(value, default=str, ensure_ascii=True), ex=ttl_seconds)
    except Exception as exc:
        print(f"Cache write failed for {namespace}: {exc}")


class CachedEmbeddings:
    def __init__(self, base_embeddings: Any, model_cache_id: str):
        self.base_embeddings = base_embeddings
        self.model_cache_id = model_cache_id

    def embed_query(self, text: str) -> list[float]:
        cached = get_json("embedding", self.model_cache_id, "query", text)
        if cached is not None:
            return cached

        vector = self.base_embeddings.embed_query(text)
        set_json("embedding", vector, TTL_EMBEDDING, self.model_cache_id, "query", text)
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = [None] * len(texts)
        missing_indexes = []
        missing_texts = []

        for index, text in enumerate(texts):
            cached = get_json("embedding", self.model_cache_id, "document", text)
            if cached is None:
                missing_indexes.append(index)
                missing_texts.append(text)
            else:
                vectors[index] = cached

        if missing_texts:
            fresh_vectors = self.base_embeddings.embed_documents(missing_texts)
            for index, text, vector in zip(missing_indexes, missing_texts, fresh_vectors):
                vectors[index] = vector
                set_json("embedding", vector, TTL_EMBEDDING, self.model_cache_id, "document", text)

        return [vector for vector in vectors if vector is not None]
