"""Chat endpoints."""

from __future__ import annotations
from fastapi import APIRouter, Header
from api.schemas.marketing import ChatRequest, ChatResponseSchema

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


def create_chat_routes(services: dict) -> APIRouter:

    @router.post("/", response_model=ChatResponseSchema)
    async def chat(body: ChatRequest, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        chat_svc = services["chat"]
        response = chat_svc.ask_question(body.question)
        return ChatResponseSchema(
            answer=response.answer,
            sources=[s.__dict__ if hasattr(s, "__dict__") else s for s in response.sources],
            confidence=response.confidence, question=response.question,
        )

    @router.get("/history")
    async def chat_history(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        history = services["chat"].get_chat_history()
        return {"tenant_id": x_tenant_id, "messages": [{"role": m.role, "content": m.content} for m in history]}

    @router.post("/summary")
    async def chat_summary(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        return {"tenant_id": x_tenant_id, "summary": services["chat"].get_conversation_summary()}

    @router.delete("/memory")
    async def clear_memory(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        services["chat"].clear_memory()
        return {"tenant_id": x_tenant_id, "message": "Memoria del chat limpiada"}

    return router
