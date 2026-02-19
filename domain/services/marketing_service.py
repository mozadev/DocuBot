"""Servicio de dominio: generacion de campanas y contenido de marketing con agente LangGraph."""

from __future__ import annotations

import json
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage

from domain.models import (
    CampaignRequest, CampaignResponse, ContentPiece,
    ContentRequest, BusinessContext,
)
from core.logger import logger, log_function_call


def _build_business_context_block(ctx: Optional[BusinessContext]) -> str:
    """Arma el bloque de contexto de negocio real para el prompt del agente."""
    if not ctx:
        return "Sin datos de negocio disponibles. Usa conocimiento general."

    parts = []

    if ctx.business_name:
        parts.append(f"NEGOCIO: {ctx.business_name}")
    if ctx.industry:
        parts.append(f"INDUSTRIA: {ctx.industry}")
    if ctx.location:
        parts.append(f"UBICACION: {ctx.location}")
    if ctx.brand_voice:
        parts.append(f"VOZ DE MARCA: {ctx.brand_voice}")
    if ctx.brand_colors:
        parts.append(f"COLORES DE MARCA: {', '.join(ctx.brand_colors)}")

    if ctx.products:
        lines = []
        for p in ctx.products[:10]:
            line = f"  - {p.name}"
            if p.price:
                line += f" (${p.price:.2f} {p.currency})"
            if p.category:
                line += f" [{p.category}]"
            if p.is_top_seller:
                line += " TOP SELLER"
            if p.description:
                line += f": {p.description[:100]}"
            lines.append(line)
        parts.append(f"CATALOGO DE PRODUCTOS ({len(ctx.products)} total):\n" + "\n".join(lines))

    if ctx.whatsapp_metrics:
        wm = ctx.whatsapp_metrics
        wa_parts = [f"  - Conversaciones totales: {wm.total_conversations}"]
        wa_parts.append(f"  - Mensajes diarios promedio: {wm.avg_daily_messages}")
        if wm.conversion_rate:
            wa_parts.append(f"  - Tasa de conversion: {wm.conversion_rate:.1%}")
        if wm.peak_hours:
            wa_parts.append(f"  - Horas pico: {', '.join(f'{h}:00' for h in wm.peak_hours)}")
        if wm.top_questions:
            wa_parts.append(f"  - Preguntas frecuentes: {'; '.join(wm.top_questions[:5])}")
        if wm.avg_response_time_seconds:
            wa_parts.append(f"  - Tiempo respuesta promedio: {wm.avg_response_time_seconds}s")
        parts.append("METRICAS WHATSAPP (datos reales):\n" + "\n".join(wa_parts))

    if ctx.sales_data:
        sd = ctx.sales_data
        s_parts = []
        if sd.total_sales_last_30d:
            s_parts.append(f"  - Ventas 30d: ${sd.total_sales_last_30d:,.2f} {sd.currency}")
        if sd.total_orders_last_30d:
            s_parts.append(f"  - Ordenes 30d: {sd.total_orders_last_30d}")
        if sd.avg_ticket:
            s_parts.append(f"  - Ticket promedio: ${sd.avg_ticket:,.2f} {sd.currency}")
        if sd.top_products:
            s_parts.append(f"  - Mas vendidos: {', '.join(sd.top_products[:5])}")
        if s_parts:
            parts.append("DATOS DE VENTAS (datos reales):\n" + "\n".join(s_parts))

    if ctx.previous_ads:
        pa = ctx.previous_ads
        a_parts = []
        if pa.avg_cpc:
            a_parts.append(f"  - CPC promedio: ${pa.avg_cpc:.2f} {pa.currency}")
        if pa.avg_ctr:
            a_parts.append(f"  - CTR promedio: {pa.avg_ctr:.2%}")
        if pa.avg_cpm:
            a_parts.append(f"  - CPM promedio: ${pa.avg_cpm:.2f} {pa.currency}")
        if pa.best_performing_ad:
            a_parts.append(f"  - Mejor ad: {pa.best_performing_ad}")
        if pa.best_audience_segment:
            a_parts.append(f"  - Mejor segmento: {pa.best_audience_segment}")
        if pa.total_spend_last_30d:
            a_parts.append(f"  - Gasto 30d: ${pa.total_spend_last_30d:,.2f} {pa.currency}")
        if pa.total_conversions_last_30d:
            a_parts.append(f"  - Conversiones 30d: {pa.total_conversions_last_30d}")
        if a_parts:
            parts.append("ADS ANTERIORES (datos reales de Meta):\n" + "\n".join(a_parts))

    if ctx.competitor_names:
        parts.append(f"COMPETIDORES: {', '.join(ctx.competitor_names)}")

    return "\n\n".join(parts)


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """Extrae JSON de la respuesta del agente (puede venir con markdown)."""
    cleaned = text.strip()
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        candidate = cleaned[json_start:json_end]
        return json.loads(candidate)
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return json.loads(cleaned)


class MarketingService:
    """Genera campanas y contenido de marketing usando agente LangGraph multi-fase."""

    def __init__(self, llm_adapter, vector_store, marketing_graph=None, mcp_manager=None) -> None:
        self._llm = llm_adapter
        self._vector_store = vector_store
        self._graph = marketing_graph
        self._mcp_manager = mcp_manager

    @log_function_call
    def generate_campaign(self, request: CampaignRequest) -> CampaignResponse:
        """Genera una campana de marketing usando el agente LangGraph de 4 fases."""
        biz_context = _build_business_context_block(request.business_context)

        user_message = (
            f"Genera una campana de marketing digital completa.\n\n"
            f"BRIEFING:\n"
            f"- Negocio: {request.business_description}\n"
            f"- Publico objetivo: {request.target_audience or 'General'}\n"
            f"- Canales: {', '.join(request.channels)}\n"
            f"- Presupuesto: {request.budget_range or 'No especificado'}\n"
            f"- Objetivos: {request.goals or 'Aumentar ventas y visibilidad'}\n"
            f"- Tono: {request.tone}\n"
            f"- Idioma: {request.language}\n"
        )

        if self._graph:
            return self._run_agent_campaign(user_message, biz_context, request)
        else:
            return self._run_simple_campaign(user_message, biz_context, request)

    def _run_agent_campaign(
        self, user_message: str, biz_context: str, request: CampaignRequest,
    ) -> CampaignResponse:
        """Ejecuta el agente LangGraph multi-fase para generar la campana."""
        logger.info("Marketing agent: iniciando pipeline de 4 fases")

        result = self._graph.invoke({
            "messages": [HumanMessage(content=user_message)],
            "phase": "research",
            "business_context_summary": biz_context,
            "research_findings": "",
            "strategy": "",
            "content_pieces_json": "",
            "iteration_count": 0,
        })

        raw_json = result.get("content_pieces_json", "")

        final_text = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                final_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        content_to_parse = raw_json or final_text

        try:
            data = _parse_json_from_text(content_to_parse)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Marketing agent: no retorno JSON valido, usando texto crudo")
            return CampaignResponse(
                campaign_name="Campana Generada por Agente",
                strategy_summary=final_text or raw_json,
                data_insights=result.get("research_findings", "")[:500],
            )

        pieces = [
            ContentPiece(
                channel=p.get("channel", ""),
                content_type=p.get("content_type", "post"),
                title=p.get("title", ""),
                body=p.get("body", ""),
                hashtags=p.get("hashtags", []),
                call_to_action=p.get("call_to_action", ""),
                suggested_image_prompt=p.get("suggested_image_prompt", ""),
                target_audience_detail=p.get("target_audience_detail", ""),
                suggested_budget_daily=float(p.get("suggested_budget_daily", 0)),
                suggested_duration_days=int(p.get("suggested_duration_days", 7)),
                placement=p.get("placement", []),
            )
            for p in data.get("content_pieces", [])
        ]

        campaign = CampaignResponse(
            campaign_name=data.get("campaign_name", "Campana"),
            strategy_summary=data.get("strategy_summary", ""),
            content_pieces=pieces,
            calendar_suggestion=data.get("calendar_suggestion", ""),
            kpi_suggestions=data.get("kpi_suggestions", []),
            estimated_reach=data.get("estimated_reach", ""),
            audience_recommendation=data.get("audience_recommendation", ""),
            budget_recommendation=data.get("budget_recommendation", ""),
            data_insights=data.get("data_insights", result.get("research_findings", "")[:500]),
        )

        logger.info(
            f"Marketing agent completado: {campaign.campaign_name} "
            f"({len(pieces)} piezas, fases ejecutadas: "
            f"research->strategy->content->review)"
        )
        return campaign

    def _run_simple_campaign(
        self, user_message: str, biz_context: str, request: CampaignRequest,
    ) -> CampaignResponse:
        """Fallback: genera campana con LLM directo (sin agente)."""
        logger.info("Marketing: usando LLM directo (agente no disponible)")

        prompt = (
            f"{user_message}\n\n"
            f"DATOS DEL NEGOCIO:\n{biz_context}\n\n"
            "Responde en JSON con: campaign_name, strategy_summary, data_insights, "
            "audience_recommendation, budget_recommendation, content_pieces[], "
            "calendar_suggestion, kpi_suggestions[], estimated_reach"
        )

        raw = self._llm.invoke([
            {"role": "system", "content": (
                "Eres un estratega de marketing digital experto en Meta Ads para PYMEs LATAM. "
                "Genera campanas basadas en datos reales cuando esten disponibles. "
                "Responde en JSON valido. Responde en espanol."
            )},
            {"role": "user", "content": prompt},
        ])

        try:
            data = _parse_json_from_text(raw)
        except (json.JSONDecodeError, ValueError):
            return CampaignResponse(campaign_name="Campana Generada", strategy_summary=raw)

        pieces = [
            ContentPiece(
                channel=p.get("channel", ""),
                content_type=p.get("content_type", "post"),
                title=p.get("title", ""),
                body=p.get("body", ""),
                hashtags=p.get("hashtags", []),
                call_to_action=p.get("call_to_action", ""),
                suggested_image_prompt=p.get("suggested_image_prompt", ""),
                target_audience_detail=p.get("target_audience_detail", ""),
                suggested_budget_daily=float(p.get("suggested_budget_daily", 0)),
                suggested_duration_days=int(p.get("suggested_duration_days", 7)),
                placement=p.get("placement", []),
            )
            for p in data.get("content_pieces", [])
        ]

        return CampaignResponse(
            campaign_name=data.get("campaign_name", "Campana"),
            strategy_summary=data.get("strategy_summary", ""),
            content_pieces=pieces,
            calendar_suggestion=data.get("calendar_suggestion", ""),
            kpi_suggestions=data.get("kpi_suggestions", []),
            estimated_reach=data.get("estimated_reach", ""),
            audience_recommendation=data.get("audience_recommendation", ""),
            budget_recommendation=data.get("budget_recommendation", ""),
            data_insights=data.get("data_insights", ""),
        )

    @log_function_call
    def generate_content(self, request: ContentRequest) -> ContentPiece:
        """Genera una pieza individual de contenido."""
        biz_context = _build_business_context_block(request.business_context)

        doc_context = "Sin documentos."
        try:
            results = self._vector_store.similarity_search(request.topic, k=4)
            if results:
                doc_context = "\n".join(
                    f"[{sr.chunk.filename}]: {sr.chunk.content[:400]}"
                    for sr in results
                )
        except Exception:
            pass

        raw = self._llm.invoke([
            {"role": "system", "content": (
                "Eres un copywriter experto en marketing digital para PYMEs LATAM. "
                "Genera contenido especifico, creativo y orientado a conversion. "
                "Responde en JSON con: title, body, hashtags, call_to_action, suggested_image_prompt"
            )},
            {"role": "user", "content": (
                f"Tipo: {request.content_type}\n"
                f"Tema: {request.topic}\n"
                f"Tono: {request.tone}\n"
                f"Max: {request.max_length} chars\n"
                f"Hashtags: {'Si' if request.include_hashtags else 'No'}\n"
                f"CTA: {'Si' if request.include_cta else 'No'}\n"
                f"Idioma: {request.language}\n\n"
                f"DATOS NEGOCIO:\n{biz_context}\n\n"
                f"DOCUMENTOS:\n{doc_context}"
            )},
        ])

        try:
            data = _parse_json_from_text(raw)
        except (json.JSONDecodeError, ValueError):
            return ContentPiece(
                channel=request.content_type,
                content_type=request.content_type,
                title="Contenido Generado",
                body=raw,
            )

        return ContentPiece(
            channel=request.content_type,
            content_type=request.content_type,
            title=data.get("title", ""),
            body=data.get("body", ""),
            hashtags=data.get("hashtags", []),
            call_to_action=data.get("call_to_action", ""),
            suggested_image_prompt=data.get("suggested_image_prompt", ""),
        )

    @log_function_call
    def analyze_market(self, query: str, business_context: Optional[BusinessContext] = None) -> str:
        """Analiza informacion de mercado usando documentos + datos reales."""
        biz_context = _build_business_context_block(business_context)

        doc_context = "Sin documentos."
        try:
            results = self._vector_store.similarity_search(query, k=6)
            if results:
                doc_context = "\n".join(
                    f"[{sr.chunk.filename}]: {sr.chunk.content[:400]}"
                    for sr in results
                )
        except Exception:
            pass

        return self._llm.invoke([
            {"role": "system", "content": (
                "Eres un analista de marketing con acceso a datos reales del negocio. "
                "Diferencia entre DATOS REALES y ESTIMACIONES. Responde en espanol."
            )},
            {"role": "user", "content": (
                f"Analiza: {query}\n\n"
                f"DATOS NEGOCIO:\n{biz_context}\n\n"
                f"DOCUMENTOS:\n{doc_context}"
            )},
        ])
