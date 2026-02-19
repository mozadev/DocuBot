"""
Cache semantico inteligente.
No busca texto exacto, sino preguntas SIMILARES ya respondidas.
Ahorra 40-60% en costos de API de OpenAI.

Flujo:
1. Usuario pregunta algo
2. Se genera embedding de la pregunta
3. Se busca en cache un embedding similar (cosine similarity > threshold)
4. Si existe y no ha expirado → retorna respuesta cacheada
5. Si no → llama al LLM, guarda respuesta + embedding en cache
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from collections import OrderedDict

from core.logger import logger


@dataclass
class CacheEntry:
    key: str
    query: str
    response: Any
    embedding: Optional[List[float]] = None
    created_at: float = 0.0
    ttl: int = 3600  # 1 hour default
    hit_count: int = 0
    tenant_id: str = "default"

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl


class SemanticCache:
    """
    Cache en memoria con busqueda semantica.
    Para produccion se puede reemplazar con Redis + pgvector.
    """

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl: int = 3600,
        similarity_threshold: float = 0.92,
        embedding_func=None,
    ) -> None:
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._similarity_threshold = similarity_threshold
        self._embedding_func = embedding_func
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "semantic_hits": 0}

    @staticmethod
    def _exact_key(query: str, tenant_id: str) -> str:
        normalized = query.strip().lower()
        return hashlib.sha256(f"{tenant_id}:{normalized}".encode()).hexdigest()

    def get(self, query: str, tenant_id: str = "default") -> Optional[Any]:
        """Busca respuesta en cache (exacta primero, luego semantica)."""
        key = self._exact_key(query, tenant_id)
        entry = self._cache.get(key)

        if entry and not entry.is_expired:
            entry.hit_count += 1
            self._stats["hits"] += 1
            self._cache.move_to_end(key)
            logger.debug(f"Cache HIT (exact) for tenant={tenant_id}")
            return entry.response

        if entry and entry.is_expired:
            del self._cache[key]

        semantic_result = self._semantic_search(query, tenant_id)
        if semantic_result:
            self._stats["semantic_hits"] += 1
            self._stats["hits"] += 1
            logger.debug(f"Cache HIT (semantic) for tenant={tenant_id}")
            return semantic_result

        self._stats["misses"] += 1
        return None

    def put(
        self,
        query: str,
        response: Any,
        tenant_id: str = "default",
        ttl: Optional[int] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Guarda respuesta en cache."""
        key = self._exact_key(query, tenant_id)

        if len(self._cache) >= self._max_entries:
            self._evict()

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
        """Invalida todo el cache de un tenant (util al subir nuevos docs)."""
        keys_to_remove = [k for k, v in self._cache.items() if v.tenant_id == tenant_id]
        for k in keys_to_remove:
            del self._cache[k]
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries for tenant={tenant_id}")
        return len(keys_to_remove)

    def get_stats(self) -> Dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0
        return {
            **self._stats,
            "total_queries": total,
            "hit_rate_pct": round(hit_rate, 1),
            "entries": len(self._cache),
            "max_entries": self._max_entries,
            "estimated_savings_pct": round(hit_rate * 0.85, 1),
        }

    def _semantic_search(self, query: str, tenant_id: str) -> Optional[Any]:
        """Busca queries semanticamente similares en cache."""
        if not self._embedding_func:
            return None

        try:
            query_embedding = self._embedding_func(query)
        except Exception:
            return None

        best_score = 0.0
        best_entry = None

        for entry in self._cache.values():
            if entry.tenant_id != tenant_id or entry.is_expired or entry.embedding is None:
                continue
            score = self._cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self._similarity_threshold:
            best_entry.hit_count += 1
            return best_entry.response

        return None

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _evict(self) -> None:
        """LRU eviction — remueve el entry mas viejo."""
        if self._cache:
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1
