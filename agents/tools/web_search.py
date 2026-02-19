"""Tools de busqueda web via Tavily."""

from __future__ import annotations
from langchain_core.tools import tool


def build_web_search_tools(tavily_adapter=None) -> list:

    def _not_available():
        return "Busqueda web no disponible (Tavily no configurado)."

    def _format_results(result: dict) -> str:
        if result.get("error"):
            return f"Error en busqueda: {result['error']}"
        parts = []
        if result.get("answer"):
            parts.append(f"RESUMEN: {result['answer']}")
        for r in result.get("results", []):
            parts.append(f"[{r['title']}] ({r['url']})\n{r['content'][:300]}")
        return "\n\n---\n\n".join(parts) if parts else "Sin resultados."

    @tool
    def search_market_trends(industry: str, location: str = "Latinoamerica") -> str:
        """Busca tendencias REALES de marketing digital en internet."""
        if not tavily_adapter or not tavily_adapter.is_enabled:
            return _not_available()
        return _format_results(tavily_adapter.search_market_trends(industry, location))

    @tool
    def search_competitors(business_description: str, location: str = "") -> str:
        """Busca informacion REAL de competidores en internet."""
        if not tavily_adapter or not tavily_adapter.is_enabled:
            return _not_available()
        return _format_results(tavily_adapter.search_competitors(business_description, location))

    @tool
    def search_ad_benchmarks(industry: str, platform: str = "Facebook Ads") -> str:
        """Busca benchmarks REALES de CPC, CTR, CPM para una industria."""
        if not tavily_adapter or not tavily_adapter.is_enabled:
            return (
                "Busqueda web no disponible. Benchmarks LATAM genericos:\n"
                "- CPC Facebook: $0.20-0.80 USD\n- CTR Facebook: 1.5-3.0%\n"
                "- CPM Facebook: $3-8 USD\n- CPC Instagram: $0.30-1.00 USD"
            )
        return _format_results(tavily_adapter.search_ad_benchmarks(industry, platform))

    @tool
    def search_content_ideas(topic: str, channel: str = "Instagram") -> str:
        """Busca ideas de contenido VIRAL y TRENDING en internet."""
        if not tavily_adapter or not tavily_adapter.is_enabled:
            return _not_available()
        return _format_results(tavily_adapter.search_content_ideas(topic, channel))

    return [search_market_trends, search_competitors, search_ad_benchmarks, search_content_ideas]
