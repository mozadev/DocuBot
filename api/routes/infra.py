"""Infrastructure endpoints: guardrails, cache, observability, rate limiting."""

from __future__ import annotations
from typing import List

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from domain.guardrails import ContentGuardrails
from adapters.cache.semantic_cache import SemanticCache
from adapters.observability.tracer import AgentTracer
from api.rate_limiter import get_rate_limiter
from api.schemas.seo import GuardrailCheckRequest, CampaignGuardrailRequest, SetTierRequest

router = APIRouter(prefix="/api/v1", tags=["Infrastructure"])


def create_infra_routes(services: dict) -> APIRouter:

    # Guardrails
    @router.post("/guardrails/check", tags=["Guardrails"])
    async def check_content_safety(body: GuardrailCheckRequest, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        guardrails = ContentGuardrails(brand_never_include=body.brand_never_include, industry=body.industry, strict_mode=body.strict_mode)
        result = guardrails.validate(body.content)
        return {"tenant_id": x_tenant_id, **result.to_dict(), "sanitized_content": result.sanitized_content}

    @router.post("/guardrails/check-campaign", tags=["Guardrails"])
    async def check_campaign_safety(body: CampaignGuardrailRequest, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        guardrails = ContentGuardrails(brand_never_include=body.brand_never_include, industry=body.industry)
        result = guardrails.validate_campaign(body.campaign)
        return {"tenant_id": x_tenant_id, **result}

    # Cache
    @router.get("/cache/stats", tags=["Cache"])
    async def cache_stats(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        cache: SemanticCache = services.get("cache")
        if not cache:
            return {"error": "Cache not initialized"}
        return {"tenant_id": x_tenant_id, **cache.get_stats()}

    @router.post("/cache/invalidate", tags=["Cache"])
    async def invalidate_cache(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        cache: SemanticCache = services.get("cache")
        if not cache:
            return {"error": "Cache not initialized"}
        return {"tenant_id": x_tenant_id, "entries_removed": cache.invalidate(x_tenant_id)}

    # Observability
    @router.get("/observability/traces", tags=["Observability"])
    async def list_traces(x_tenant_id: str = Header("default", alias="X-Tenant-ID"), limit: int = Query(20, ge=1, le=100)):
        tracer: AgentTracer = services.get("tracer")
        if not tracer:
            return {"error": "Tracer not initialized"}
        return {"tenant_id": x_tenant_id, "traces": tracer.get_tenant_traces(x_tenant_id, limit)}

    @router.get("/observability/traces/{trace_id}", tags=["Observability"])
    async def get_trace_detail(trace_id: str, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        tracer: AgentTracer = services.get("tracer")
        if not tracer:
            return {"error": "Tracer not initialized"}
        trace = tracer.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace

    @router.get("/observability/analytics", tags=["Observability"])
    async def get_analytics(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        tracer: AgentTracer = services.get("tracer")
        if not tracer:
            return {"error": "Tracer not initialized"}
        return {"tenant_id": x_tenant_id, **tracer.get_analytics(x_tenant_id)}

    # Rate Limiting
    @router.get("/rate-limit/usage", tags=["Rate Limiting"])
    async def rate_limit_usage(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        return get_rate_limiter().get_usage(x_tenant_id)

    @router.post("/rate-limit/set-tier", tags=["Rate Limiting"])
    async def set_tenant_tier(body: SetTierRequest):
        get_rate_limiter().set_tenant_tier(body.tenant_id, body.tier)
        return {"tenant_id": body.tenant_id, "tier": body.tier, "message": "Tier updated"}

    return router
