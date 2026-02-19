"""
Streaming SSE (Server-Sent Events) para generacion en tiempo real.
NestJS/Flutter reciben tokens del agente mientras genera, no al final.
"""

from __future__ import annotations

import json
import asyncio
from typing import AsyncGenerator, Dict, Any

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.logger import logger

router = APIRouter(prefix="/api/v1/stream", tags=["Streaming"])


class StreamChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)


class StreamCampaignRequest(BaseModel):
    business_description: str = Field(..., min_length=10)
    target_audience: str = ""
    channels: str = "instagram,facebook"
    goals: str = ""
    tone: str = "profesional"


def _sse_event(event: str, data: Any) -> str:
    """Formatea un evento SSE."""
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_agent_execution(graph, initial_state: dict) -> AsyncGenerator[str, None]:
    """Ejecuta el agente LangGraph y emite eventos SSE por cada paso."""
    yield _sse_event("start", {"message": "Iniciando agente..."})

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: graph.invoke(initial_state))

        messages = result.get("messages", [])
        for msg in messages:
            msg_type = type(msg).__name__

            if msg_type == "AIMessage" and msg.content:
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        yield _sse_event("tool_call", {
                            "tool": tc.get("name", "unknown"),
                            "args": str(tc.get("args", {}))[:200],
                        })
                else:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    chunks = [content[i:i+100] for i in range(0, len(content), 100)]
                    for chunk in chunks:
                        yield _sse_event("token", {"content": chunk})
                        await asyncio.sleep(0.02)

            elif msg_type == "ToolMessage" and msg.content:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                yield _sse_event("tool_result", {
                    "tool": getattr(msg, "name", "unknown"),
                    "preview": content[:200],
                })

        phase = result.get("phase", "")
        sources = result.get("sources", [])
        confidence = result.get("confidence", 0.0)

        yield _sse_event("done", {
            "phase": phase,
            "sources_count": len(sources),
            "confidence": confidence,
        })

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield _sse_event("error", {"message": str(e)})


def create_streaming_routes(services: dict) -> APIRouter:
    """Crea rutas de streaming inyectando los servicios."""

    @router.post("/chat")
    async def stream_chat(
        body: StreamChatRequest,
        x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
    ):
        """Chat con streaming SSE — el cliente recibe tokens en tiempo real."""
        from langchain_core.messages import HumanMessage

        chat_svc = services["chat"]
        graph = chat_svc._graph

        initial_state = {
            "messages": [HumanMessage(content=body.question)],
            "sources": [],
            "confidence": 0.0,
            "route": "",
        }

        return StreamingResponse(
            _stream_agent_execution(graph, initial_state),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Tenant-ID": x_tenant_id,
            },
        )

    @router.post("/campaign")
    async def stream_campaign(
        body: StreamCampaignRequest,
        x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
    ):
        """Generacion de campana con streaming SSE — ve cada fase en tiempo real."""
        from langchain_core.messages import HumanMessage

        mkt_svc = services["mkt"]
        graph = mkt_svc._graph

        if not graph:
            return StreamingResponse(
                iter([_sse_event("error", {"message": "Marketing agent not available"})]),
                media_type="text/event-stream",
            )

        user_message = (
            f"Genera campana de marketing:\n"
            f"Negocio: {body.business_description}\n"
            f"Audiencia: {body.target_audience}\n"
            f"Canales: {body.channels}\n"
            f"Objetivos: {body.goals}\n"
            f"Tono: {body.tone}"
        )

        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "phase": "research",
            "business_context_summary": "",
            "research_findings": "",
            "strategy": "",
            "content_pieces_json": "",
            "iteration_count": 0,
        }

        return StreamingResponse(
            _stream_agent_execution(graph, initial_state),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Tenant-ID": x_tenant_id,
            },
        )

    return router
