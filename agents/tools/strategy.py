"""Tools de estrategia: presupuesto, calendario, calidad."""

from __future__ import annotations
from langchain_core.tools import tool


def build_strategy_tools() -> list:

    @tool
    def calculate_budget(
        objective: str,
        avg_cpc: float = 0.0,
        avg_cpm: float = 0.0,
        target_conversions: int = 0,
        duration_days: int = 7,
        currency: str = "USD",
    ) -> str:
        """Calcula el presupuesto recomendado para una campana de Meta Ads."""
        if avg_cpc > 0 and target_conversions > 0:
            estimated_budget = avg_cpc * target_conversions
            daily = estimated_budget / duration_days if duration_days > 0 else estimated_budget
            return (
                f"CALCULO BASADO EN DATOS REALES:\n"
                f"CPC promedio: ${avg_cpc:.2f} {currency}\n"
                f"Conversiones objetivo: {target_conversions}\n"
                f"Presupuesto total estimado: ${estimated_budget:,.2f} {currency}\n"
                f"Presupuesto diario: ${daily:,.2f} {currency}\n"
                f"Duracion: {duration_days} dias\n"
                f"Nota: Basado en rendimiento historico real del tenant."
            )
        elif avg_cpm > 0:
            impressions = 50000
            estimated_budget = (avg_cpm / 1000) * impressions
            daily = estimated_budget / duration_days if duration_days > 0 else estimated_budget
            return (
                f"CALCULO BASADO EN CPM REAL:\n"
                f"CPM promedio: ${avg_cpm:.2f} {currency}\n"
                f"Impresiones estimadas: {impressions:,}\n"
                f"Presupuesto total: ${estimated_budget:,.2f} {currency}\n"
                f"Presupuesto diario: ${daily:,.2f} {currency}\n"
                f"Duracion: {duration_days} dias"
            )
        else:
            return (
                f"ESTIMACION GENERICA (sin datos historicos):\n"
                f"Objetivo: {objective}\n"
                f"Duracion: {duration_days} dias\n"
                f"Rango recomendado LATAM:\n"
                f"  - Awareness: $3-5 {currency}/dia\n"
                f"  - Traffic: $5-10 {currency}/dia\n"
                f"  - Conversiones: $10-25 {currency}/dia\n"
                f"  - Lead generation: $8-15 {currency}/dia\n"
                f"IMPORTANTE: Estos son estimados genericos. "
                f"Se recomienda conectar Meta Ads para datos reales de CPC/CPM."
            )

    @tool
    def plan_content_calendar(
        campaign_duration_days: int,
        channels: str,
        num_pieces: int = 10,
    ) -> str:
        """Planifica un calendario de contenido para la campana."""
        return (
            f"PLANIFICA CALENDARIO:\n"
            f"Duracion: {campaign_duration_days} dias\n"
            f"Canales: {channels}\n"
            f"Piezas a distribuir: {num_pieces}\n\n"
            "REGLAS DE CALENDARIO PARA LATAM:\n"
            "- Facebook Feed: mejor entre 12-14h y 19-21h\n"
            "- Instagram Feed: mejor entre 11-13h y 18-20h\n"
            "- Instagram Stories: mejor entre 8-10h y 20-22h\n"
            "- Email: mejor martes a jueves 9-11h\n"
            "- WhatsApp broadcast: respetar horario comercial 9-18h\n"
            "- Frecuencia Facebook/IG: 1 post/dia max, 3-5 stories/dia\n"
            "- No publicar los mismos dias en todos los canales (distribuir)\n"
            "- Si tienes horas pico de WhatsApp del tenant, priorizar esos horarios"
        )

    @tool
    def review_campaign_quality(campaign_summary: str) -> str:
        """Revisa la calidad de la campana generada y sugiere mejoras."""
        return (
            f"REVISA ESTA CAMPANA:\n{campaign_summary}\n\n"
            "CHECKLIST DE CALIDAD:\n"
            "1. Cada pieza tiene headline < 40 chars?\n"
            "2. Los CTAs son claros y accionables?\n"
            "3. Los hashtags son relevantes (no genericos)?\n"
            "4. El tono es consistente en todas las piezas?\n"
            "5. Hay variedad de formatos (feed, stories, carousel, reel)?\n"
            "6. El presupuesto es realista para el mercado LATAM?\n"
            "7. El calendario tiene buena distribucion?\n"
            "8. Los KPIs son medibles y alcanzables?\n"
            "9. Las imagenes sugeridas son coherentes con la marca?\n"
            "10. Hay al menos un CTA hacia WhatsApp?\n"
            "Si algo falla, corrige y mejora."
        )

    return [calculate_budget, plan_content_calendar, review_campaign_quality]
