"""
DocuBot AI - FastAPI application.

Entry point only: mounts routers and middleware. All behaviour lives in
domain/services; all wiring lives in api/factory.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.factory import build_container
from api.rate_limiter import RateLimitMiddleware
from api.routes import chat, documents, health, infra, streaming
from config.settings import settings
from core.logger import logger

DESCRIPTION = """
Ask questions about your own documents and get answers with citations.

Upload PDF, DOCX, TXT or Markdown files to `/documents/upload`, then ask
questions at `/chat`. Answers are generated only from the indexed documents; if
the documents do not contain the answer, the API says so rather than guessing.

Every response carries a `trace_id` that can be replayed against
`/observability/traces/{trace_id}` to see the retrieval scores, guardrail
verdicts and latency behind it.

Conversation state and rate limits are keyed by the optional `X-Session-ID`
header.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build the container at startup rather than on the first request.

    This makes a bad API key or an unreadable database fail immediately and
    visibly, instead of turning into a confusing 500 for whoever happens to send
    the first question.
    """
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    build_container()
    logger.info("Startup complete")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="DocuBot AI",
    description=DESCRIPTION,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Session-ID"],
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(streaming.router)
app.include_router(infra.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Log the traceback, return a generic message.

    Stack traces in an HTTP response leak file paths and dependency versions to
    anyone who can trigger an error.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check the server logs for details."},
    )
