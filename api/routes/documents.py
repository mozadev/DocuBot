"""Document upload & management endpoints."""

from __future__ import annotations

import os
import tempfile
from typing import List

from fastapi import APIRouter, UploadFile, File, Header

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


def create_document_routes(services: dict) -> APIRouter:

    @router.post("/upload")
    async def upload_documents(
        files: List[UploadFile] = File(...),
        x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
    ):
        doc_svc = services["doc"]
        results = []
        for f in files:
            ext = f"'.{f.filename.split('.')[-1].lower()}" if f.filename else ""
            if ext.lstrip("'") not in [e for e in doc_svc.supported_extensions]:
                results.append({"filename": f.filename, "error": f"Formato no soportado: {ext}"})
                continue
            try:
                suffix = f".{f.filename.split('.')[-1].lower()}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    content = await f.read()
                    tmp.write(content)
                    tmp_path = tmp.name
                stats = doc_svc.process_and_index(tmp_path)
                os.unlink(tmp_path)
                results.append({
                    "filename": f.filename,
                    "text_chunks": stats["text_chunks"],
                    "image_chunks": stats["image_chunks"],
                    "total_chunks": stats["total"],
                })
            except Exception as e:
                results.append({"filename": f.filename, "error": str(e)})
        return {
            "tenant_id": x_tenant_id,
            "files_processed": len(results),
            "total_chunks_indexed": sum(r.get("total_chunks", 0) for r in results),
            "details": results,
        }

    @router.get("/stats")
    async def document_stats(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        doc_svc = services["doc"]
        count = doc_svc.get_document_count()
        return {
            "tenant_id": x_tenant_id, "total_chunks": count,
            "status": "active" if count > 0 else "empty",
            "supported_formats": doc_svc.supported_extensions,
        }

    @router.delete("/")
    async def clear_documents(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        services["doc"].clear_database()
        return {"tenant_id": x_tenant_id, "message": "Base de datos limpiada"}

    return router
