"""Grafo LangGraph del agente DocuBot AI."""

from __future__ import annotations

from typing import Annotated, List, Dict, Any, Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from core.logger import logger
from agents.chat_tools import build_tools


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    sources: List[Dict[str, Any]]
    confidence: float
    route: str


def _build_system_prompt(mcp_tool_names: List[str]) -> str:
    base = (
        "Eres DocuBot AI, un asistente empresarial inteligente para análisis de documentos "
        "y generación de contenido de marketing.\n\n"
        "CAPACIDADES INTERNAS:\n"
        "- Buscar y responder preguntas sobre documentos indexados (texto e imágenes)\n"
        "- Proporcionar estadísticas de la base de datos documental\n"
        "- Resumir temas específicos encontrados en los documentos\n"
        "- Sugerir contenido de marketing basado en productos/servicios del negocio\n"
        "- Consultar el catálogo de productos/servicios para campañas\n"
    )
    if mcp_tool_names:
        mcp_list = "\n".join(f"  - {n}" for n in mcp_tool_names)
        base += f"\nCAPACIDADES EXTERNAS (MCP):\n{mcp_list}\n"
    base += (
        "\nREGLAS:\n"
        "- Usa las herramientas disponibles para buscar información antes de responder\n"
        "- Cuando uses información de una imagen, menciónalo\n"
        "- Si usas una herramienta MCP externa, indica la fuente\n"
        "- Si no encuentras información relevante, dilo explícitamente\n"
        "- Responde siempre en español\n"
        "- Sé conciso pero completo\n"
        "- Cita las fuentes cuando sea posible"
    )
    return base


def build_agent_graph(vector_store, llm_adapter, mcp_manager=None):
    """Construye el grafo del agente combinando tools internas + MCP."""

    internal_tools = build_tools(vector_store)
    mcp_tools = mcp_manager.tools if mcp_manager else []
    mcp_tool_names = [t.name for t in mcp_tools]
    all_tools = internal_tools + mcp_tools

    system_prompt = _build_system_prompt(mcp_tool_names)
    llm = llm_adapter.get_langchain_llm().bind_tools(all_tools)

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        return {"messages": [llm.invoke(messages)]}

    def extract_sources(state: AgentState) -> dict:
        sources: List[Dict[str, Any]] = []
        for msg in state["messages"]:
            if not isinstance(msg, ToolMessage):
                continue
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            if msg.name == "search_documents" and "No se encontraron" not in content:
                for block in content.split("\n\n---\n\n"):
                    if block.startswith("[Fuente"):
                        header_end = block.find("]\n")
                        if header_end > 0:
                            header = block[1:header_end]
                            parts = [p.strip() for p in header.split("|")]
                            score_str = [p for p in parts if p.startswith("score=")]
                            score = float(score_str[0].split("=")[1]) if score_str else 0.0
                            sources.append({
                                "filename": parts[2] if len(parts) > 2 else "Desconocido",
                                "content": (block[header_end+2:][:200] + "..."),
                                "score": score,
                                "content_type": "image" if "IMAGEN" in header else "text",
                            })
            elif msg.name in mcp_tool_names and content:
                sources.append({
                    "filename": f"MCP: {msg.name}",
                    "content": (content[:200] + "...") if len(content) > 200 else content,
                    "score": 0.8,
                    "content_type": "mcp",
                })

        confidence = sum(s["score"] for s in sources) / len(sources) if sources else 0.0
        return {"sources": sources, "confidence": confidence}

    def should_continue(state: AgentState) -> Literal["tools", "extract_sources"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "extract_sources"

    tool_node = ToolNode(all_tools)
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("extract_sources", extract_sources)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "extract_sources": "extract_sources"})
    graph.add_edge("tools", "agent")
    graph.add_edge("extract_sources", END)

    compiled = graph.compile()
    logger.info(f"Grafo compilado: {len(internal_tools)} internas + {len(mcp_tools)} MCP")
    return compiled
