"""Chat endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header

from api.deps import get_container
from api.schemas.common import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    MessageResponse,
    SourceSchema,
    SummaryResponse,
)

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])

# Conversation history is keyed by this header. There is no auth in this build,
# so it scopes state and is not a security boundary -- see README.
SESSION_HEADER = Header("default", alias="X-Session-ID")


@router.post("", response_model=ChatResponse, summary="Ask a question about your documents")
async def chat(
    body: ChatRequest,
    session_id: str = SESSION_HEADER,
    container=Depends(get_container),
) -> ChatResponse:
    """
    Answer a question using only the indexed documents.

    The response carries the citations retrieval actually returned, a confidence
    score, and the trace id for this request, which can be replayed against
    /api/v1/observability/traces/{trace_id}.
    """
    result = container.chat.ask_question(body.question, session_id=session_id)
    return ChatResponse(
        answer=result.answer,
        question=result.question,
        sources=[SourceSchema(**s.to_dict()) for s in result.sources],
        confidence=result.confidence,
        trace_id=result.trace_id,
        cached=result.cached,
        guardrail=result.guardrail,
    )


@router.get("/history", response_model=ChatHistoryResponse)
async def chat_history(
    session_id: str = SESSION_HEADER, container=Depends(get_container)
) -> ChatHistoryResponse:
    history = container.chat.get_chat_history(session_id)
    return ChatHistoryResponse(
        session_id=session_id,
        messages=[{"role": m.role, "content": m.content} for m in history],
    )


@router.post("/summary", response_model=SummaryResponse)
async def chat_summary(
    session_id: str = SESSION_HEADER, container=Depends(get_container)
) -> SummaryResponse:
    return SummaryResponse(
        session_id=session_id, summary=container.chat.get_conversation_summary(session_id)
    )


@router.delete("/memory", response_model=MessageResponse)
async def clear_memory(
    session_id: str = SESSION_HEADER, container=Depends(get_container)
) -> MessageResponse:
    container.chat.clear_memory(session_id)
    return MessageResponse(message=f"Conversation history cleared for session '{session_id}'.")
