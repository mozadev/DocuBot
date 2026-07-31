"""Document upload and management endpoints."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.deps import get_container
from api.schemas.common import (
    DocumentStatsResponse,
    MessageResponse,
    UploadResponse,
    UploadResult,
)
from config.settings import settings
from core.logger import logger

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


@router.post("/upload", response_model=UploadResponse, summary="Upload and index documents")
async def upload_documents(
    files: list[UploadFile] = File(...), container=Depends(get_container)
) -> UploadResponse:
    """
    Index one or more documents.

    Each file is processed independently: one bad file reports its error and the
    rest still land, which is what you want when a user drags in a folder.
    """
    doc_svc = container.documents
    supported = set(doc_svc.supported_extensions)
    results: list[UploadResult] = []

    for upload in files:
        name = upload.filename or "unnamed"
        ext = Path(name).suffix.lower()

        if ext not in supported:
            results.append(
                UploadResult(
                    filename=name,
                    error=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(supported))}",
                )
            )
            continue

        content = await upload.read()
        if len(content) > settings.max_upload_bytes:
            results.append(
                UploadResult(
                    filename=name,
                    error=f"File exceeds the {settings.max_upload_mb} MB limit.",
                )
            )
            continue

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # The original name is passed through so citations show the file the
            # user recognises, not the temp file we happened to write.
            stats = doc_svc.process_and_index(tmp_path, original_filename=name)
            results.append(
                UploadResult(
                    filename=name,
                    text_chunks=stats["text_chunks"],
                    image_chunks=stats["image_chunks"],
                    total_chunks=stats["total"],
                )
            )
        except Exception as e:  # noqa: BLE001 - reported per file, never fatal
            logger.exception("Failed to index %s", name)
            results.append(UploadResult(filename=name, error=str(e)))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return UploadResponse(
        files_processed=len(results),
        total_chunks_indexed=sum(r.total_chunks for r in results),
        details=results,
    )


@router.get("/stats", response_model=DocumentStatsResponse)
async def document_stats(container=Depends(get_container)) -> DocumentStatsResponse:
    doc_svc = container.documents
    count = doc_svc.get_document_count()
    return DocumentStatsResponse(
        total_chunks=count,
        status="active" if count > 0 else "empty",
        supported_formats=doc_svc.supported_extensions,
        documents=doc_svc.list_documents(),
    )


@router.delete("", response_model=MessageResponse, summary="Delete every indexed document")
async def clear_documents(container=Depends(get_container)) -> MessageResponse:
    if container.documents.get_document_count() == 0:
        raise HTTPException(status_code=404, detail="There are no indexed documents to delete.")
    container.documents.clear_database()
    return MessageResponse(message="All documents removed from the index.")
