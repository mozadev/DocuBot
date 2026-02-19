"""
Rate limiting per-tenant.
Evita que un tenant abuse de la API y consuma todo el budget de OpenAI.
Configurable por tier: free, pro, enterprise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from collections import defaultdict

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response, JSONResponse

from core.logger import logger


@dataclass
class TenantLimits:
    """Limites por tier de tenant."""
    requests_per_minute: int = 20
    requests_per_hour: int = 200
    requests_per_day: int = 1000
    max_campaigns_per_day: int = 10
    max_images_per_day: int = 20
    max_tokens_per_day: int = 500_000


TIER_LIMITS: Dict[str, TenantLimits] = {
    "free": TenantLimits(
        requests_per_minute=10,
        requests_per_hour=60,
        requests_per_day=200,
        max_campaigns_per_day=3,
        max_images_per_day=5,
        max_tokens_per_day=100_000,
    ),
    "pro": TenantLimits(
        requests_per_minute=30,
        requests_per_hour=500,
        requests_per_day=5000,
        max_campaigns_per_day=50,
        max_images_per_day=100,
        max_tokens_per_day=2_000_000,
    ),
    "enterprise": TenantLimits(
        requests_per_minute=100,
        requests_per_hour=3000,
        requests_per_day=50000,
        max_campaigns_per_day=500,
        max_images_per_day=1000,
        max_tokens_per_day=20_000_000,
    ),
}


@dataclass
class _BucketState:
    """Sliding window counter."""
    timestamps: list = field(default_factory=list)

    def count_in_window(self, window_seconds: int) -> int:
        now = time.time()
        cutoff = now - window_seconds
        self.timestamps = [t for t in self.timestamps if t > cutoff]
        return len(self.timestamps)

    def add(self) -> None:
        self.timestamps.append(time.time())


class RateLimiter:
    """Rate limiter in-memory per-tenant con sliding window."""

    def __init__(self, default_tier: str = "free") -> None:
        self._buckets: Dict[str, _BucketState] = defaultdict(_BucketState)
        self._tenant_tiers: Dict[str, str] = {}
        self._default_tier = default_tier

    def set_tenant_tier(self, tenant_id: str, tier: str) -> None:
        """NestJS puede configurar el tier de cada tenant."""
        if tier not in TIER_LIMITS:
            tier = self._default_tier
        self._tenant_tiers[tenant_id] = tier

    def check(self, tenant_id: str) -> Tuple[bool, Optional[Dict]]:
        """
        Retorna (allowed, info).
        Si allowed=False, info contiene detalles del limite excedido.
        """
        tier = self._tenant_tiers.get(tenant_id, self._default_tier)
        limits = TIER_LIMITS[tier]
        bucket = self._buckets[tenant_id]

        per_min = bucket.count_in_window(60)
        if per_min >= limits.requests_per_minute:
            return False, {
                "error": "rate_limit_exceeded",
                "limit": "requests_per_minute",
                "current": per_min,
                "max": limits.requests_per_minute,
                "tier": tier,
                "retry_after_seconds": 60,
            }

        per_hour = bucket.count_in_window(3600)
        if per_hour >= limits.requests_per_hour:
            return False, {
                "error": "rate_limit_exceeded",
                "limit": "requests_per_hour",
                "current": per_hour,
                "max": limits.requests_per_hour,
                "tier": tier,
                "retry_after_seconds": 3600,
            }

        per_day = bucket.count_in_window(86400)
        if per_day >= limits.requests_per_day:
            return False, {
                "error": "rate_limit_exceeded",
                "limit": "requests_per_day",
                "current": per_day,
                "max": limits.requests_per_day,
                "tier": tier,
                "retry_after_seconds": 86400,
            }

        bucket.add()
        return True, {
            "tier": tier,
            "remaining_minute": limits.requests_per_minute - per_min - 1,
            "remaining_hour": limits.requests_per_hour - per_hour - 1,
            "remaining_day": limits.requests_per_day - per_day - 1,
        }

    def get_usage(self, tenant_id: str) -> Dict:
        tier = self._tenant_tiers.get(tenant_id, self._default_tier)
        limits = TIER_LIMITS[tier]
        bucket = self._buckets[tenant_id]
        return {
            "tenant_id": tenant_id,
            "tier": tier,
            "usage_minute": bucket.count_in_window(60),
            "limit_minute": limits.requests_per_minute,
            "usage_hour": bucket.count_in_window(3600),
            "limit_hour": limits.requests_per_hour,
            "usage_day": bucket.count_in_window(86400),
            "limit_day": limits.requests_per_day,
        }


_global_limiter = RateLimiter(default_tier="free")


def get_rate_limiter() -> RateLimiter:
    return _global_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware de FastAPI que aplica rate limiting automaticamente."""

    EXEMPT_PATHS = {"/api/v1/health", "/api/v1/status", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        if path in self.EXEMPT_PATHS or not path.startswith("/api/"):
            return await call_next(request)

        tenant_id = request.headers.get("X-Tenant-ID", "default")
        limiter = get_rate_limiter()

        allowed, info = limiter.check(tenant_id)

        if not allowed:
            logger.warning(f"Rate limit exceeded for tenant={tenant_id}: {info['limit']}")
            return JSONResponse(
                status_code=429,
                content=info,
                headers={"Retry-After": str(info.get("retry_after_seconds", 60))},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining-Minute"] = str(info.get("remaining_minute", 0))
        response.headers["X-RateLimit-Tier"] = info.get("tier", "free")
        return response
