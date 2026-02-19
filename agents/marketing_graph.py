"""
Grafo LangGraph del agente de marketing DocuBot AI.

Flujo del agente:
  START → research → strategize → create_content → review → END

Cada nodo puede usar herramientas y el agente decide autonomamente
cuando necesita mas informacion, iterar o finalizar.
"""

from __future__ import annotations

import json
from typing import Annotated, List, Dict, Any, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from core.logger import logger
from agents.marketing_tools import build_marketing_tools


# ──────────────────────────── State ────────────────────────────

class MarketingAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    phase: str  # "research" | "strategy" | "content" | "review" | "done"
    business_context_summary: str
    research_findings: str
    strategy: str
    content_pieces_json: str
    iteration_count: int


# ──────────────────────────── System Prompts por Fase ────────────────────────────

RESEARCH_PROMPT = """\
Eres el INVESTIGADOR de un equipo de marketing digital de elite.

FASE ACTUAL: INVESTIGACION
Tu trabajo es recopilar toda la informacion necesaria antes de crear una campana.

TAREAS:
1. Busca en los documentos del negocio: productos, precios, descripciones, diferenciadores
2. Analiza los datos de negocio proporcionados (ventas, WhatsApp, ads anteriores)
3. Investiga la audiencia ideal para este tipo de negocio
4. Si hay datos de ads anteriores, identifica que funciono y que no

USA LAS HERRAMIENTAS para investigar. No inventes datos.
Cuando hayas recopilado suficiente informacion, responde con un RESUMEN DE INVESTIGACION
que incluya:
- Productos/servicios clave encontrados
- Insights de datos de ventas y WhatsApp
- Audiencia recomendada
- Oportunidades detectadas
- Riesgos o limitaciones

Responde en espanol.
"""

STRATEGY_PROMPT = """\
Eres el ESTRATEGA de un equipo de marketing digital de elite.

FASE ACTUAL: ESTRATEGIA
Ya tienes la investigacion del equipo. Ahora crea la estrategia.

INVESTIGACION PREVIA:
{research_findings}

TAREAS:
1. Define el objetivo principal de la campana (awareness/traffic/conversion/leads)
2. Usa la herramienta calculate_budget para definir presupuesto basado en datos reales
3. Usa plan_content_calendar para definir el calendario
4. Define los KPIs con metas numericas basadas en datos reales si los tienes

USA LAS HERRAMIENTAS. Basa todo en los datos de la investigacion.
Cuando termines, responde con la ESTRATEGIA COMPLETA incluyendo:
- Objetivo de campana
- Presupuesto recomendado (con calculo)
- Calendario de publicacion
- Audiencia segmentada
- KPIs con metas

Responde en espanol.
"""

CONTENT_PROMPT = """\
Eres el DIRECTOR CREATIVO de un equipo de marketing digital de elite.

FASE ACTUAL: CREACION DE CONTENIDO
Ya tienes la investigacion y la estrategia. Ahora crea el contenido.

INVESTIGACION:
{research_findings}

ESTRATEGIA:
{strategy}

TAREAS:
1. Usa generate_ad_copy para crear cada pieza de contenido
2. Crea al menos 1 pieza por canal solicitado
3. Incluye variedad de formatos (feed post, story, carousel, reel caption, email)
4. Cada pieza debe tener: titulo, body, hashtags, CTA, descripcion de imagen
5. Adapta el tono y largo a cada canal

USA LAS HERRAMIENTAS para generar cada pieza.
Cuando termines, responde con TODAS las piezas en JSON:
```json
{{
  "campaign_name": "nombre creativo",
  "strategy_summary": "resumen basado en datos",
  "data_insights": "que dicen los numeros",
  "audience_recommendation": "segmentacion detallada",
  "budget_recommendation": "presupuesto con calculo",
  "content_pieces": [
    {{
      "channel": "facebook|instagram|whatsapp|email",
      "content_type": "post|ad_copy|story|reel_caption|carousel|email",
      "title": "titulo",
      "body": "contenido completo",
      "hashtags": ["h1", "h2"],
      "call_to_action": "CTA",
      "suggested_image_prompt": "descripcion de imagen detallada con colores de marca",
      "target_audience_detail": "segmento especifico",
      "suggested_budget_daily": 5.0,
      "suggested_duration_days": 7,
      "placement": ["facebook_feed", "instagram_stories"]
    }}
  ],
  "calendar_suggestion": "calendario detallado",
  "kpi_suggestions": ["KPI con meta numerica"],
  "estimated_reach": "estimacion basada en datos"
}}
```

Responde en espanol.
"""

REVIEW_PROMPT = """\
Eres el DIRECTOR DE CALIDAD de un equipo de marketing digital de elite.

FASE ACTUAL: REVISION DE CALIDAD
Revisa la campana generada y asegura que cumple estandares profesionales.

USA la herramienta review_campaign_quality para evaluar.

Si la campana es buena, responde "APROBADA" seguido del JSON final de la campana.
Si necesita mejoras, indica que cambiar y responde con el JSON corregido.

El JSON final DEBE seguir exactamente el schema de content_pieces.
Responde en espanol.
"""


# ──────────────────────────── Graph Builder ────────────────────────────

def build_marketing_agent_graph(
    vector_store, llm_adapter, mcp_manager=None, tavily_adapter=None, dalle_adapter=None,
):
    """Construye el grafo del agente de marketing con 4 fases."""

    internal_tools = build_marketing_tools(
        vector_store, tavily_adapter=tavily_adapter, dalle_adapter=dalle_adapter,
    )
    mcp_tools = mcp_manager.tools if mcp_manager else []
    all_tools = internal_tools + mcp_tools

    llm_with_tools = llm_adapter.get_langchain_llm().bind_tools(all_tools)

    tool_node = ToolNode(all_tools)

    # ---- Nodos ----

    def research_node(state: MarketingAgentState) -> dict:
        biz_ctx = state.get("business_context_summary", "")
        system = RESEARCH_PROMPT
        if biz_ctx:
            system += f"\n\nDATOS DE NEGOCIO DEL TENANT:\n{biz_ctx}"
        msgs = [SystemMessage(content=system)] + list(state["messages"])
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response], "phase": "research"}

    def strategy_node(state: MarketingAgentState) -> dict:
        research = state.get("research_findings", "")
        system = STRATEGY_PROMPT.format(research_findings=research)
        if state.get("business_context_summary"):
            system += f"\n\nDATOS DE NEGOCIO:\n{state['business_context_summary']}"
        msgs = [SystemMessage(content=system)] + list(state["messages"])
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response], "phase": "strategy"}

    def content_node(state: MarketingAgentState) -> dict:
        research = state.get("research_findings", "")
        strategy = state.get("strategy", "")
        system = CONTENT_PROMPT.format(research_findings=research, strategy=strategy)
        msgs = [SystemMessage(content=system)] + list(state["messages"])
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response], "phase": "content"}

    def review_node(state: MarketingAgentState) -> dict:
        msgs = [SystemMessage(content=REVIEW_PROMPT)] + list(state["messages"])
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response], "phase": "review"}

    def extract_phase_output(state: MarketingAgentState) -> dict:
        last_msg = state["messages"][-1]
        content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        phase = state.get("phase", "research")
        iteration = state.get("iteration_count", 0)
        updates: Dict[str, Any] = {"iteration_count": iteration + 1}
        if phase == "research":
            updates["research_findings"] = content
        elif phase == "strategy":
            updates["strategy"] = content
        elif phase in ("content", "review"):
            updates["content_pieces_json"] = content
        return updates

    # ---- Router helpers ----

    def should_use_tools_research(state: MarketingAgentState) -> Literal["tools_research", "extract"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools_research"
        return "extract"

    def should_use_tools_strategy(state: MarketingAgentState) -> Literal["tools_strategy", "extract"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools_strategy"
        return "extract"

    def should_use_tools_content(state: MarketingAgentState) -> Literal["tools_content", "extract"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools_content"
        return "extract"

    def should_use_tools_review(state: MarketingAgentState) -> Literal["tools_review", "extract"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools_review"
        return "extract"

    def next_phase(state: MarketingAgentState) -> Literal[
        "strategy_node", "content_node", "review_node", "end"
    ]:
        phase = state.get("phase", "research")
        if phase == "research":
            return "strategy_node"
        elif phase == "strategy":
            return "content_node"
        elif phase == "content":
            return "review_node"
        return "end"

    # ---- Build Graph ----

    graph = StateGraph(MarketingAgentState)

    graph.add_node("research_node", research_node)
    graph.add_node("strategy_node", strategy_node)
    graph.add_node("content_node", content_node)
    graph.add_node("review_node", review_node)
    graph.add_node("tools_research", tool_node)
    graph.add_node("tools_strategy", tool_node)
    graph.add_node("tools_content", tool_node)
    graph.add_node("tools_review", tool_node)
    graph.add_node("extract", extract_phase_output)

    graph.add_edge(START, "research_node")

    graph.add_conditional_edges("research_node", should_use_tools_research,
                                {"tools_research": "tools_research", "extract": "extract"})
    graph.add_conditional_edges("strategy_node", should_use_tools_strategy,
                                {"tools_strategy": "tools_strategy", "extract": "extract"})
    graph.add_conditional_edges("content_node", should_use_tools_content,
                                {"tools_content": "tools_content", "extract": "extract"})
    graph.add_conditional_edges("review_node", should_use_tools_review,
                                {"tools_review": "tools_review", "extract": "extract"})

    graph.add_edge("tools_research", "research_node")
    graph.add_edge("tools_strategy", "strategy_node")
    graph.add_edge("tools_content", "content_node")
    graph.add_edge("tools_review", "review_node")

    graph.add_conditional_edges("extract", next_phase, {
        "strategy_node": "strategy_node",
        "content_node": "content_node",
        "review_node": "review_node",
        "end": END,
    })

    compiled = graph.compile()
    logger.info(
        f"Marketing agent compilado: {len(internal_tools)} internas + {len(mcp_tools)} MCP, "
        f"4 fases (research -> strategy -> content -> review)"
    )
    return compiled
