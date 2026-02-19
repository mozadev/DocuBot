"""Tools de performance learning: analisis de resultados y optimizacion."""

from __future__ import annotations
from langchain_core.tools import tool


def build_performance_tools() -> list:

    @tool
    def analyze_past_performance(performance_data: str) -> str:
        """Analiza rendimiento de campanas/ads anteriores para aprender que funciono."""
        return (
            f"ANALIZA RENDIMIENTO PASADO:\n{performance_data}\n\n"
            f"EXTRAE:\n"
            f"1. Que contenido tuvo mejor CTR? (copy, formato, canal)\n"
            f"2. Que audiencia convirtio mejor?\n"
            f"3. Que horarios funcionaron?\n"
            f"4. Que CPC/CPM es realista para este negocio?\n"
            f"5. Que NO funciono y debemos evitar?\n"
            f"6. Recomendaciones concretas para la proxima campana\n\n"
            f"APLICA ESTAS LECCIONES en todo el contenido que generes."
        )

    @tool
    def suggest_optimization(current_metrics: str, target_kpi: str = "CTR") -> str:
        """Sugiere optimizaciones para mejorar un KPI basandose en metricas actuales."""
        return (
            f"OPTIMIZA PARA MEJORAR {target_kpi}:\n"
            f"Metricas actuales:\n{current_metrics}\n\n"
            f"SUGIERE CAMBIOS EN:\n"
            f"1. Copy: nuevo headline o body\n"
            f"2. Audiencia: ajustar targeting\n"
            f"3. Presupuesto: redistribuir entre ad sets\n"
            f"4. Creative: nueva imagen o formato\n"
            f"5. Schedule: cambiar horarios\n"
            f"6. CTA: probar otro call to action\n"
            f"Prioriza los cambios de mayor impacto primero."
        )

    return [analyze_past_performance, suggest_optimization]
