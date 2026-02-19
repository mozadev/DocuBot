"""
Marketing agent tools — punto de ensamblaje.
Cada categoria vive en su propio modulo en agents/tools/.
"""

from __future__ import annotations

from agents.tools.research import build_research_tools
from agents.tools.strategy import build_strategy_tools
from agents.tools.creation import build_creation_tools
from agents.tools.platforms import build_platform_tools
from agents.tools.performance import build_performance_tools
from agents.tools.seo import build_seo_tools
from agents.tools.quality import build_quality_tools
from agents.tools.web_search import build_web_search_tools


def build_marketing_tools(vector_store, tavily_adapter=None, dalle_adapter=None) -> list:
    """Construye las 27 herramientas de marketing inyectando dependencias."""
    tools = []
    tools.extend(build_research_tools(vector_store))
    tools.extend(build_strategy_tools())
    tools.extend(build_creation_tools(vector_store, dalle_adapter))
    tools.extend(build_platform_tools())
    tools.extend(build_performance_tools())
    tools.extend(build_seo_tools(tavily_adapter))
    tools.extend(build_quality_tools(tavily_adapter))
    tools.extend(build_web_search_tools(tavily_adapter))
    return tools
