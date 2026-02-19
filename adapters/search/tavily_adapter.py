"""Adapter: busqueda web con Tavily API — disenada para agentes AI."""

from __future__ import annotations

from typing import List, Dict, Any, Optional

from core.logger import logger


class TavilyAdapter:
    """Wrapper sobre Tavily que expone metodos de busqueda orientados a marketing."""

    def __init__(self, api_key: str) -> None:
        self._enabled = bool(api_key)
        self._client = None

        if self._enabled:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=api_key)
                logger.info("TavilyAdapter: conectado (busqueda web habilitada)")
            except ImportError:
                logger.warning("tavily-python no instalado, busqueda web deshabilitada")
                self._enabled = False
            except Exception as e:
                logger.warning(f"Tavily no pudo inicializarse: {e}")
                self._enabled = False
        else:
            logger.info("TavilyAdapter: sin API key, busqueda web deshabilitada")

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._client is not None

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
    ) -> Dict[str, Any]:
        """Busqueda web general. Retorna resultados + respuesta resumida."""
        if not self.is_enabled:
            return {"answer": "", "results": [], "error": "Tavily no configurado"}

        try:
            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
            )
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                }
                for r in response.get("results", [])
            ]
            return {
                "answer": response.get("answer", ""),
                "results": results,
            }
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return {"answer": "", "results": [], "error": str(e)}

    def search_market_trends(self, industry: str, location: str = "Latinoamerica") -> Dict[str, Any]:
        """Busca tendencias de mercado para una industria."""
        query = f"tendencias marketing digital {industry} {location} 2026"
        return self.search(query, max_results=5, search_depth="advanced")

    def search_competitors(self, business: str, location: str = "") -> Dict[str, Any]:
        """Busca informacion de competidores."""
        query = f"competidores {business} {location} precios marketing"
        return self.search(query, max_results=5, search_depth="advanced")

    def search_ad_benchmarks(self, industry: str, platform: str = "Facebook Ads") -> Dict[str, Any]:
        """Busca benchmarks de ads para la industria."""
        query = f"{platform} benchmarks CPC CTR {industry} 2026 Latinoamerica"
        return self.search(query, max_results=3, search_depth="advanced")

    def search_content_ideas(self, topic: str, channel: str = "Instagram") -> Dict[str, Any]:
        """Busca ideas de contenido viral/trending para un tema."""
        query = f"ideas contenido viral {channel} {topic} 2026 tendencias"
        return self.search(query, max_results=5, search_depth="basic")
