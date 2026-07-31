"""
Semantic answer cache.

Exact-match caching barely helps a chat product, because two users almost never
type a question the same way. This cache falls back to comparing question
embeddings, so "how many vacation days do I get?" hits the entry stored for
"what is the vacation allowance?".

In-memory and per-process, which is fine for a single container and wrong for a
horizontally scaled deployment. Redis is the drop-in replacement; see README.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from core.logger import logger

# Cosine similarity above which two questions are treated as the same question.
# 0.92 is deliberately strict: a false hit serves a confidently wrong answer,
# which costs far more than the API call it saved.
DEFAULT_SIMILARITY_THRESHOLD = 0.92


@dataclass
class CacheEntry:
    key: str
    query: str
    response: Any
    embedding: list[float] | None
    created_at: float
    ttl: int
    tenant_id: str
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl


class SemanticCache:
    """LRU cache with exact-match and embedding-similarity lookup."""

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl: int = 3600,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        embedding_func=None,
    ) -> None:
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._threshold = similarity_threshold
        self._embed = embedding_func
        self._stats = {"hits": 0, "semantic_hits": 0, "misses": 0, "evictions": 0}

    @staticmethod
    def _exact_key(query: str, tenant_id: str) -> str:
        return hashlib.sha256(f"{tenant_id}:{query.strip().lower()}".encode()).hexdigest()

    def get(self, query: str, tenant_id: str = "default") -> Any | None:
        key = self._exact_key(query, tenant_id)
        entry = self._cache.get(key)

        if entry and entry.is_expired:
            del self._cache[key]
            entry = None

        if entry:
            entry.hit_count += 1
            self._stats["hits"] += 1
            self._cache.move_to_end(key)
            return entry.response

        semantic = self._semantic_lookup(query, tenant_id)
        if semantic is not None:
            self._stats["hits"] += 1
            self._stats["semantic_hits"] += 1
            return semantic

        self._stats["misses"] += 1
        return None

    def put(
        self,
        query: str,
        response: Any,
        tenant_id: str = "default",
        ttl: int | None = None,
    ) -> None:
        if len(self._cache) >= self._max_entries:
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1

        embedding = None
        if self._embed:
            try:
                embedding = self._embed(query)
            except Exception as e:  # noqa: BLE001 - cache must never break a request
                logger.debug("Cache embedding failed, storing exact-match only: %s", e)

        key = self._exact_key(query, tenant_id)
        self._cache[key] = CacheEntry(
            key=key,
            query=query,
            response=response,
            embedding=embedding,
            created_at=time.time(),
            ttl=ttl or self._default_ttl,
            tenant_id=tenant_id,
        )

    def invalidate(self, tenant_id: str) -> int:
        keys = [k for k, v in self._cache.items() if v.tenant_id == tenant_id]
        for k in keys:
            del self._cache[k]
        logger.info("Invalidated %d cache entries for tenant=%s", len(keys), tenant_id)
        return len(keys)

    def invalidate_all(self) -> int:
        """Called whenever the document set changes, which invalidates every answer."""
        count = len(self._cache)
        self._cache.clear()
        logger.info("Invalidated all %d cache entries (document set changed)", count)
        return count

    def get_stats(self) -> dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total * 100) if total else 0.0
        return {
            **self._stats,
            "total_queries": total,
            "hit_rate_pct": round(hit_rate, 1),
            "entries": len(self._cache),
            "max_entries": self._max_entries,
        }

    def _semantic_lookup(self, query: str, tenant_id: str) -> Any | None:
        if not self._embed:
            return None
        try:
            query_vec = self._embed(query)
        except Exception:  # noqa: BLE001
            return None

        best_entry, best_score = None, 0.0
        for entry in self._cache.values():
            if entry.tenant_id != tenant_id or entry.is_expired or entry.embedding is None:
                continue
            score = self._cosine(query_vec, entry.embedding)
            if score > best_score:
                best_entry, best_score = entry, score

        if best_entry and best_score >= self._threshold:
            best_entry.hit_count += 1
            logger.debug("Semantic cache hit (%.3f): '%s'", best_score, best_entry.query[:60])
            return best_entry.response
        return None

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return dot / norm if norm else 0.0
