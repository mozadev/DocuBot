"""Observability and cache endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from api.deps import get_container
from api.rate_limiter import get_rate_limiter

router = APIRouter(prefix="/api/v1", tags=["Observability"])

SESSION_HEADER = Header("default", alias="X-Session-ID")


@router.get("/observability/traces", summary="Recent request traces for a session")
async def list_traces(
    session_id: str = SESSION_HEADER,
    limit: int = Query(20, ge=1, le=100),
    container=Depends(get_container),
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "traces": container.tracer.get_tenant_traces(session_id, limit),
    }


@router.get("/observability/traces/{trace_id}", summary="Full span breakdown for one request")
async def get_trace(trace_id: str, container=Depends(get_container)) -> dict[str, Any]:
    """
    Every span recorded for a single answer: guardrail verdicts, cache hit or
    miss, each vector search with its top score, and total latency.
    """
    trace = container.tracer.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"No trace with id '{trace_id}'.")
    return trace


@router.get("/observability/analytics", summary="Aggregate cost, latency and error rate")
async def analytics(
    session_id: str = SESSION_HEADER, container=Depends(get_container)
) -> dict[str, Any]:
    return {"session_id": session_id, **container.tracer.get_analytics(session_id)}


@router.get("/cache/stats", tags=["Cache"], summary="Cache hit rate")
async def cache_stats(container=Depends(get_container)) -> dict[str, Any]:
    return container.cache.get_stats()


@router.post("/cache/invalidate", tags=["Cache"], summary="Drop all cached answers")
async def invalidate_cache(container=Depends(get_container)) -> dict[str, Any]:
    return {"entries_removed": container.cache.invalidate_all()}


@router.get("/rate-limit/usage", tags=["Rate Limiting"], summary="Current usage against limits")
async def rate_limit_usage(session_id: str = SESSION_HEADER) -> dict[str, Any]:
    return get_rate_limiter().get_usage(session_id)
