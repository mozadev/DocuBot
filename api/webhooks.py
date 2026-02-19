"""
Sistema de Webhooks para generacion asincrona.
Las campanas toman tiempo (30-90s), asi que:
1. NestJS envia request → DocuBot responde con job_id inmediatamente
2. DocuBot genera en background
3. Al terminar, hace POST al webhook de NestJS con el resultado
"""

from __future__ import annotations

import uuid
import asyncio
import httpx
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field, asdict

from fastapi import APIRouter, Header, BackgroundTasks
from pydantic import BaseModel, Field, HttpUrl

from core.logger import logger

router = APIRouter(prefix="/api/v1/jobs", tags=["Async Jobs"])


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    tenant_id: str
    job_type: str
    status: JobStatus
    created_at: str
    updated_at: str
    webhook_url: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0
    phases_completed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


_jobs: Dict[str, Job] = {}


class AsyncCampaignRequest(BaseModel):
    business_description: str = Field(..., min_length=10)
    target_audience: str = ""
    channels: str = "instagram,facebook"
    goals: str = ""
    tone: str = "profesional"
    webhook_url: Optional[str] = Field(None, description="URL donde NestJS recibira el resultado")
    priority: str = Field("normal", pattern="^(low|normal|high)$")


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    estimated_seconds: int


async def _notify_webhook(webhook_url: str, payload: Dict[str, Any]) -> None:
    """Envia el resultado al webhook de NestJS."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json", "X-Source": "docubot-ai"},
            )
            logger.info(f"Webhook delivered to {webhook_url}: {response.status_code}")
    except Exception as e:
        logger.error(f"Webhook delivery failed to {webhook_url}: {e}")


async def _process_campaign_job(job_id: str, request: AsyncCampaignRequest, services: dict) -> None:
    """Background task que ejecuta la generacion de campana."""
    job = _jobs.get(job_id)
    if not job:
        return

    job.status = JobStatus.PROCESSING
    job.updated_at = datetime.now(timezone.utc).isoformat()

    try:
        job.phases_completed.append("research")
        job.progress = 0.25

        from domain.models import CampaignRequest
        campaign_req = CampaignRequest(
            business_description=request.business_description,
            target_audience=request.target_audience,
            channels=[c.strip() for c in request.channels.split(",")],
            goals=[g.strip() for g in request.goals.split(",") if g.strip()],
            tone=request.tone,
        )

        mkt_svc = services["mkt"]
        result = await asyncio.get_event_loop().run_in_executor(
            None, mkt_svc.generate_campaign, campaign_req
        )

        job.phases_completed.extend(["strategy", "content", "review"])
        job.progress = 1.0
        job.status = JobStatus.COMPLETED
        job.result = {
            "strategy_summary": result.strategy_summary,
            "content_pieces": [
                {"channel": p.channel, "title": p.title, "body": p.body, "cta": p.cta}
                for p in result.content_pieces
            ],
            "budget_recommendation": result.budget_recommendation,
            "kpis": result.kpis,
        }
        job.updated_at = datetime.now(timezone.utc).isoformat()

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.updated_at = datetime.now(timezone.utc).isoformat()
        logger.error(f"Job {job_id} failed: {e}")

    if job.webhook_url:
        await _notify_webhook(job.webhook_url, {
            "event": "job.completed" if job.status == JobStatus.COMPLETED else "job.failed",
            "job": job.to_dict(),
        })


def create_webhook_routes(services: dict) -> APIRouter:
    """Crea las rutas de webhooks inyectando servicios."""

    @router.post("/campaign", response_model=JobResponse)
    async def create_campaign_job(
        body: AsyncCampaignRequest,
        background_tasks: BackgroundTasks,
        x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
    ):
        """Encola generacion de campana. Retorna job_id inmediatamente."""
        now = datetime.now(timezone.utc).isoformat()
        job = Job(
            id=str(uuid.uuid4()),
            tenant_id=x_tenant_id,
            job_type="campaign_generation",
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            webhook_url=body.webhook_url,
        )
        _jobs[job.id] = job

        background_tasks.add_task(_process_campaign_job, job.id, body, services)

        return JobResponse(
            job_id=job.id,
            status="queued",
            message="Campana en generacion. Consulta el status o espera el webhook.",
            estimated_seconds=45,
        )

    @router.get("/{job_id}")
    async def get_job_status(
        job_id: str,
        x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
    ):
        """Consulta el estado de un job — polling alternativo al webhook."""
        job = _jobs.get(job_id)
        if not job:
            return {"error": "Job not found", "job_id": job_id}
        if job.tenant_id != x_tenant_id:
            return {"error": "Unauthorized"}
        return job.to_dict()

    @router.get("/")
    async def list_jobs(
        x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
        status: Optional[str] = None,
    ):
        """Lista todos los jobs de un tenant."""
        tenant_jobs = [j for j in _jobs.values() if j.tenant_id == x_tenant_id]
        if status:
            tenant_jobs = [j for j in tenant_jobs if j.status.value == status]
        return {
            "jobs": [j.to_dict() for j in sorted(tenant_jobs, key=lambda x: x.created_at, reverse=True)],
            "total": len(tenant_jobs),
        }

    return router
