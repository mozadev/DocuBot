"""Tools multi-plataforma: Google Ads, TikTok, LinkedIn + multi-idioma."""

from __future__ import annotations
from langchain_core.tools import tool


def build_platform_tools() -> list:

    @tool
    def adapt_for_google_ads(content: str, campaign_type: str = "search") -> str:
        """Adapta contenido para Google Ads (Search, Display, Shopping, YouTube)."""
        limits = {
            "search": "Headlines: max 30 chars (x15). Descriptions: max 90 chars (x4).",
            "display": "Headline: max 30 chars. Long headline: max 90 chars. Description: max 90 chars.",
            "shopping": "Title: max 150 chars. Description: max 5000 chars.",
            "youtube": "Headline: max 15 chars. Long headline: max 90 chars. Description: max 70 chars.",
        }
        return (
            f"ADAPTA PARA GOOGLE ADS ({campaign_type.upper()}):\n"
            f"Contenido original (Meta):\n{content}\n\n"
            f"LIMITES: {limits.get(campaign_type, limits['search'])}\n\n"
            f"REGLAS GOOGLE ADS:\n"
            f"- Google Ads es mas directo/informativo que Meta\n"
            f"- Incluir keywords del negocio en los headlines\n"
            f"- Search: foco en intencion de compra, no engagement\n"
            f"- Display: visual + brand awareness\n"
            f"- Incluir extensiones: sitelinks, callouts, precio\n"
            f"- Generar multiples variaciones de headlines"
        )

    @tool
    def adapt_for_tiktok(content: str, trend_style: str = "storytelling") -> str:
        """Adapta contenido para TikTok Ads. trend_style: storytelling | ugc | educational | meme | challenge."""
        return (
            f"ADAPTA PARA TIKTOK ADS:\nContenido original:\n{content}\n\n"
            f"Estilo trending: {trend_style}\n\n"
            f"REGLAS TIKTOK:\n"
            f"- NO parece publicidad. Contenido organico de creator\n"
            f"- Hook en primeros 2 segundos\n- Formato vertical 9:16\n"
            f"- Duracion ideal: 15-30 seg\n- Trending sounds/music\n"
            f"- Texto en pantalla OBLIGATORIO\n"
            f"- Storytelling: problema → solucion → resultado\n"
            f"- UGC style: grabado con celular\n"
            f"- CTA: 'Link en bio' o 'Comentame X'\n"
            f"- Trending hashtags, NO corporativos"
        )

    @tool
    def adapt_for_linkedin(
        content: str, objective: str = "lead_generation", ad_format: str = "single_image",
    ) -> str:
        """Adapta contenido para LinkedIn Ads (B2B).
        objective: lead_generation | brand_awareness | website_visits | engagement.
        ad_format: single_image | carousel | video | text_ad | message_ad | document_ad."""
        return (
            f"ADAPTA PARA LINKEDIN ADS (B2B):\nContenido original:\n{content}\n\n"
            f"Objetivo: {objective}\nFormato: {ad_format}\n\n"
            f"REGLAS LINKEDIN:\n"
            f"- Tono PROFESIONAL. Sin emojis excesivos\n"
            f"- Enfocate en ROI, eficiencia, resultados medibles\n"
            f"- Headline: max 70 chars, directo al pain point\n"
            f"- Body: datos, estadisticas, social proof\n"
            f"- CTA: 'Solicitar demo' / 'Descargar whitepaper' / 'Agendar reunion'\n\n"
            f"TARGETING:\n- Job titles relevantes\n- Industrias objetivo\n"
            f"- Tamano empresa (1-50, 51-200, 201-1000, 1000+)\n\n"
            f"METRICAS LINKEDIN:\n- CTR: 0.4-0.6%\n- CPC: $5-12 USD\n"
            f"- Mejor dia: martes-jueves 8-10am"
        )

    @tool
    def translate_campaign_content(
        content: str, target_language: str = "en", adapt_culturally: bool = True,
    ) -> str:
        """Traduce y adapta culturalmente contenido de marketing."""
        return (
            f"TRADUCE Y ADAPTA:\nIdioma destino: {target_language}\n"
            f"Adaptacion cultural: {'Si' if adapt_culturally else 'No (literal)'}\n"
            f"Contenido:\n{content}\n\n"
            f"REGLAS:\n- ADAPTAR al mercado destino, no traducir literal\n"
            f"- Adaptar modismos, humor, hashtags\n- Mantener tono e intencion\n"
            f"- Mantener emojis y formato"
        )

    return [adapt_for_google_ads, adapt_for_tiktok, adapt_for_linkedin, translate_campaign_content]
