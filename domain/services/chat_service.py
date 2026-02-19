"""Servicio de dominio: chat con agente LangGraph + MCP."""

from __future__ import annotations

import asyncio
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage

from domain.models import ChatResponse, Source, ChatMessage, MCPStatus
from core.logger import logger, log_function_call


def _run_async(coro):
    """Ejecuta una corrutina desde código síncrono."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


class ChatService:
    """Orquesta preguntas, historial y resúmenes usando el agente LangGraph."""

    def __init__(self, vector_store, llm_adapter, mcp_manager, graph_builder, session_id: str = "default") -> None:
        self._vector_store = vector_store
        self._llm = llm_adapter
        self._mcp_manager = mcp_manager
        self._session_id = session_id
        self._chat_history: Dict[str, List] = {}

        # Inicializar MCP
        self._mcp_initialized = False
        self._init_mcp()

        # Construir grafo
        self._graph = graph_builder(
            vector_store=vector_store,
            llm_adapter=llm_adapter,
            mcp_manager=mcp_manager,
        )
        logger.info("ChatService inicializado (LangGraph + MCP)")

    def _init_mcp(self) -> None:
        try:
            _run_async(self._mcp_manager.initialize())
            self._mcp_initialized = True
            tool_count = len(self._mcp_manager.tools)
            if tool_count:
                logger.info(f"MCP: {tool_count} herramientas externas cargadas")
            else:
                logger.info("MCP: sin servidores externos conectados")
        except Exception as e:
            logger.warning(f"MCP no pudo inicializarse: {e}")

    def _get_history(self, session_id: str) -> List:
        if session_id not in self._chat_history:
            self._chat_history[session_id] = []
        return self._chat_history[session_id]

    @log_function_call
    def ask_question(self, question: str) -> ChatResponse:
        if not question or not question.strip():
            return ChatResponse(answer="Por favor, proporciona una pregunta válida.")

        history = self._get_history(self._session_id)
        messages = list(history) + [HumanMessage(content=question)]

        result = self._graph.invoke({
            "messages": messages,
            "sources": [],
            "confidence": 0.0,
            "route": "",
        })

        # Extraer respuesta del último mensaje AI
        all_messages = result.get("messages", [])
        answer_text = ""
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                answer_text = msg.content
                break

        # Actualizar historial
        history.append(HumanMessage(content=question))
        if answer_text:
            history.append(AIMessage(content=answer_text))

        # Construir fuentes
        raw_sources = result.get("sources", [])
        sources = [Source(**s) if isinstance(s, dict) else s for s in raw_sources]
        confidence = result.get("confidence", 0.0)

        # Fallback si no se extrajeron fuentes
        if not sources and answer_text:
            try:
                scored = self._vector_store.similarity_search(question, k=4)
                for sr in scored:
                    src = Source(
                        filename=sr.chunk.filename,
                        content=(sr.chunk.content[:200] + "...") if sr.chunk.content else "",
                        score=sr.score,
                        content_type=sr.chunk.content_type,
                        image_path=sr.chunk.metadata.get("image_path", ""),
                        page_number=sr.chunk.metadata.get("page_number", 0),
                    )
                    sources.append(src)
                if sources:
                    confidence = sum(s.score for s in sources) / len(sources)
            except Exception:
                pass

        return ChatResponse(
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            question=question,
        )

    @log_function_call
    def get_chat_history(self) -> List[ChatMessage]:
        history = self._get_history(self._session_id)
        return [
            ChatMessage(
                role="human" if isinstance(m, HumanMessage) else "ai",
                content=m.content,
            )
            for m in history
        ]

    @log_function_call
    def clear_memory(self) -> None:
        self._chat_history.pop(self._session_id, None)
        logger.info("Memoria limpiada")

    @log_function_call
    def get_conversation_summary(self) -> str:
        history = self._get_history(self._session_id)
        if not history:
            return "No hay historial de conversación."
        messages_repr = [
            {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
            for m in history
        ]
        return self._llm.invoke([
            {"role": "system", "content": "Resume la conversación de forma breve. Responde en español."},
            {"role": "user", "content": str(messages_repr)},
        ])

    def get_mcp_status(self) -> MCPStatus:
        return self._mcp_manager.get_status()
