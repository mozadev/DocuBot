"""
Per-session rate limiting.

The thing being protected is the OpenAI bill: one runaway client looping on
/chat can spend real money in minutes. A sliding window over request timestamps
is enough for that and needs no dependencies.

In-memory and per-process, so with N replicas the effective limit is N times the
configured one. Correct behind a single container; Redis is the fix at scale,
noted in the README.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from core.logger import logger


@dataclass(frozen=True)
class Limits:
    per_minute: int = 20
    per_hour: int = 200
    per_day: int = 1000


DEFAULT_LIMITS = Limits()

# Probes and docs must never be rate limited: the orchestrator's health check
# would start failing under exactly the load where you need it to be accurate.
EXEMPT_PATHS = frozenset(
    {"/api/v1/health", "/api/v1/status", "/docs", "/redoc", "/openapi.json"}
)


class RateLimiter:
    """Sliding-window request counter keyed by session."""

    def __init__(self, limits: Limits = DEFAULT_LIMITS) -> None:
        self._limits = limits
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, key: str) -> deque[float]:
        window = self._windows[key]
        cutoff = time.time() - 86400
        while window and window[0] < cutoff:
            window.popleft()
        return window

    @staticmethod
    def _count_since(window: deque[float], seconds: int) -> int:
        cutoff = time.time() - seconds
        return sum(1 for t in window if t > cutoff)

    def check(self, session_id: str) -> tuple[bool, dict]:
        """Record a request and report whether it is allowed."""
        window = self._prune(session_id)

        for seconds, limit, label in (
            (60, self._limits.per_minute, "per_minute"),
            (3600, self._limits.per_hour, "per_hour"),
            (86400, self._limits.per_day, "per_day"),
        ):
            used = self._count_since(window, seconds)
            if used >= limit:
                return False, {
                    "error": "rate_limit_exceeded",
                    "limit": label,
                    "current": used,
                    "max": limit,
                    "retry_after_seconds": seconds,
                }

        window.append(time.time())
        return True, {
            "remaining_minute": self._limits.per_minute - self._count_since(window, 60),
            "remaining_hour": self._limits.per_hour - self._count_since(window, 3600),
            "remaining_day": self._limits.per_day - self._count_since(window, 86400),
        }

    def get_usage(self, session_id: str) -> dict:
        window = self._prune(session_id)
        return {
            "session_id": session_id,
            "usage_minute": self._count_since(window, 60),
            "limit_minute": self._limits.per_minute,
            "usage_hour": self._count_since(window, 3600),
            "limit_hour": self._limits.per_hour,
            "usage_day": self._count_since(window, 86400),
            "limit_day": self._limits.per_day,
        }


_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Applies the rate limiter to every /api/ route except the exempt ones."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if path in EXEMPT_PATHS or not path.startswith("/api/"):
            return await call_next(request)

        session_id = request.headers.get("X-Session-ID", "default")
        allowed, info = get_rate_limiter().check(session_id)

        if not allowed:
            logger.warning("Rate limit hit for session=%s (%s)", session_id, info["limit"])
            return JSONResponse(
                status_code=429,
                content=info,
                headers={"Retry-After": str(info["retry_after_seconds"])},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining-Minute"] = str(info["remaining_minute"])
        return response
