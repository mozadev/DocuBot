"""
DocuBot AI — FastAPI REST API.
Punto de entrada. Solo monta routers, middleware, y lifespan.
Toda la logica vive en api/routes/.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.factory import create_all_services
from adapters.cache.semantic_cache import SemanticCache
from adapters.observability.tracer import AgentTracer
from domain.guardrails import ContentGuardrails
from api.rate_limiter import RateLimitMiddleware
from api.routes.health import create_health_routes
from api.routes.documents import create_document_routes
from api.routes.chat import create_chat_routes
from api.routes.marketing import create_marketing_routes
from api.routes.seo import create_seo_routes
from api.routes.infra import create_infra_routes
from core.logger import logger


_services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa servicios al arrancar, limpia al apagar."""
    logger.info("FastAPI: inicializando servicios...")
    doc_svc, chat_svc, mkt_svc, dalle = create_all_services(session_id="fastapi")
    _services["doc"] = doc_svc
    _services["chat"] = chat_svc
    _services["mkt"] = mkt_svc
    _services["dalle"] = dalle
    _services["cache"] = SemanticCache(max_entries=1000, default_ttl=3600)
    _services["tracer"] = AgentTracer(max_traces=500)
    _services["guardrails"] = ContentGuardrails()

    from api.streaming import create_streaming_routes
    from api.webhooks import create_webhook_routes

    app.include_router(create_health_routes(_services))
    app.include_router(create_document_routes(_services))
    app.include_router(create_chat_routes(_services))
    app.include_router(create_marketing_routes(_services))
    app.include_router(create_seo_routes(_services))
    app.include_router(create_infra_routes(_services))
    app.include_router(create_streaming_routes(_services))
    app.include_router(create_webhook_routes(_services))

    logger.info("FastAPI: 8 routers montados, servicios listos")
    yield
    _services.clear()
    logger.info("FastAPI: servicios liberados")


app = FastAPI(
    title="DocuBot AI API",
    description=(
        "API REST de DocuBot AI — Agente LangGraph + RAG Multimodal + MCP + Marketing.\n\n"
        "Diseñada para integrarse con backends NestJS, clientes Flutter/React, etc.\n"
        "Soporta multi-tenancy mediante el header `X-Tenant-ID`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
