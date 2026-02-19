"""Health & status endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from config.settings import settings
from api.schemas.common import HealthResponse, StatusResponse

router = APIRouter(prefix="/api/v1", tags=["Health"])


def create_health_routes(services: dict) -> APIRouter:

    @router.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok", service="DocuBot AI", version="1.0.0")

    @router.get("/status", response_model=StatusResponse)
    async def status():
        doc_svc = services.get("doc")
        chat_svc = services.get("chat")
        mcp_info = chat_svc.get_mcp_status() if chat_svc else None
        return StatusResponse(
            service="DocuBot AI", version="1.0.0",
            model=settings.openai_model,
            embedding_model=settings.embedding_model,
            multimodal=settings.enable_multimodal,
            vision_model=settings.vision_model,
            document_count=doc_svc.get_document_count() if doc_svc else 0,
            mcp={
                "initialized": mcp_info.initialized if mcp_info else False,
                "connected_servers": mcp_info.connected_servers if mcp_info else 0,
                "tools": mcp_info.tool_names if mcp_info else [],
            },
        )

    return router
