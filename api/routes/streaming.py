"""
Server-sent events for token-level streaming.

A grounded answer takes several seconds: the agent searches, reads the passages,
then writes. Streaming turns that into visible progress instead of a spinner,
and the tool_call events let the UI show "searching your documents..." while it
happens.

This streams from the graph directly rather than through ChatService, so it
deliberately bypasses the cache and the output guardrail: you cannot retract a
token you have already sent. The UI uses the non-streaming endpoint whenever the
grounding verdict matters. Noted as a known trade-off in the README.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, Header
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage

from api.deps import get_container
from api.schemas.common import ChatRequest
from core.logger import logger

router = APIRouter(prefix="/api/v1/stream", tags=["Streaming"])

SESSION_HEADER = Header("default", alias="X-Session-ID")


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_answer(container, question: str, session_id: str) -> AsyncGenerator[str, None]:
    guard = container.guardrails.check_input(question)
    if not guard.passed:
        yield _sse("error", {"message": " ".join(guard.violations)})
        yield _sse("done", {"blocked": True})
        return

    graph = container.chat.graph
    state: dict[str, Any] = {
        "messages": [HumanMessage(content=guard.content)],
        "sources": [],
        "confidence": 0.0,
    }

    yield _sse("start", {"question": guard.content})
    answer_parts: list[str] = []

    try:
        # stream_mode="messages" yields (chunk, metadata) as the model produces
        # tokens; "updates" yields each node's output once it completes. Asking
        # for both means one pass gives token text and tool activity.
        async for mode, payload in graph.astream(
            state, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                chunk, _meta = payload
                # "messages" also yields the ToolMessage carrying raw retrieval
                # output. Emitting that as tokens would dump the retrieved
                # passages into the answer, so only model output is streamed.
                if not isinstance(chunk, AIMessageChunk):
                    continue
                text = chunk.content
                if isinstance(text, str) and text:
                    answer_parts.append(text)
                    yield _sse("token", {"content": text})

            elif mode == "updates":
                for node, update in (payload or {}).items():
                    if node == "extract_sources":
                        yield _sse(
                            "sources",
                            {
                                "sources": update.get("sources", []),
                                "confidence": update.get("confidence", 0.0),
                            },
                        )
                    elif node == "agent":
                        for msg in update.get("messages", []):
                            for call in getattr(msg, "tool_calls", None) or []:
                                yield _sse(
                                    "tool_call",
                                    {
                                        "tool": call.get("name", "unknown"),
                                        "query": str(call.get("args", {}).get("query", ""))[:200],
                                    },
                                )

        yield _sse("done", {"chars": sum(len(p) for p in answer_parts)})

    except Exception as e:  # noqa: BLE001 - the stream must close cleanly
        logger.exception("Streaming failed")
        yield _sse("error", {"message": str(e)})
        yield _sse("done", {"error": True})


@router.post("/chat", summary="Stream an answer as it is generated (SSE)")
async def stream_chat(
    body: ChatRequest,
    session_id: str = SESSION_HEADER,
    container=Depends(get_container),
) -> StreamingResponse:
    """
    Emits `start`, `tool_call`, `token`, `sources` and `done` events.

    Streamed turns are not written to conversation history; use POST /chat for
    stateful conversation.
    """
    return StreamingResponse(
        _stream_answer(container, body.question, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
