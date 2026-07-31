"""Health and status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_container
from api.schemas.common import HealthResponse, StatusResponse
from config.settings import settings

router = APIRouter(prefix="/api/v1", tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """
    Cheap liveness check for the container orchestrator.

    Deliberately does not touch the vector store or OpenAI: a liveness probe that
    depends on a third party will restart a healthy container during their
    outage. Dependency state belongs in /status.
    """
    return HealthResponse(status="ok", service=settings.app_name, version=settings.app_version)


@router.get(
    "/status", response_model=StatusResponse, summary="Effective configuration and index size"
)
async def status(container=Depends(get_container)) -> StatusResponse:
    return StatusResponse(
        service=settings.app_name,
        version=settings.app_version,
        model=settings.openai_model,
        embedding_model=settings.embedding_model,
        multimodal=settings.enable_multimodal,
        vision_model=settings.vision_model if settings.enable_multimodal else "disabled",
        chunk_size=settings.chunk_size,
        document_count=container.documents.get_document_count(),
        supported_formats=container.documents.supported_extensions,
    )
