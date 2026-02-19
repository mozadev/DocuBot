"""Tools de investigacion: catalogo, datos de negocio, audiencia."""

from __future__ import annotations
from typing import List
from langchain_core.tools import tool
from core.logger import logger


def build_research_tools(vector_store) -> list:

    @tool
    def search_product_catalog(query: str) -> str:
        """Busca productos, servicios, precios y descripciones en los documentos del negocio.
        Usa esta herramienta PRIMERO para entender que vende el negocio."""
        try:
            queries = [query, f"productos {query}", f"precios {query}", f"catalogo {query}"]
            all_results = []
            seen = set()
            for q in queries:
                results = vector_store.similarity_search(q, k=3)
                for sr in results:
                    key = f"{sr.chunk.filename}:{sr.chunk.content[:50]}"
                    if key not in seen:
                        seen.add(key)
                        all_results.append(sr)
            if not all_results:
                return f"No se encontraron productos o servicios relacionados con '{query}'."
            parts: List[str] = []
            for i, sr in enumerate(all_results[:8], 1):
                parts.append(
                    f"[Producto/Info {i} | {sr.chunk.filename} | score={sr.score:.3f}]\n"
                    f"{sr.chunk.content[:500]}"
                )
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            logger.error(f"Error en search_product_catalog: {e}")
            return f"Error: {e}"

    @tool
    def analyze_business_data(data_description: str) -> str:
        """Analiza los datos de negocio proporcionados (ventas, WhatsApp, ads anteriores).
        Usa esta herramienta para extraer insights accionables de los datos del tenant."""
        return (
            f"Datos para analizar: {data_description}\n\n"
            "INSTRUCCIONES: Con estos datos, identifica:\n"
            "1. Productos estrella (top sellers)\n"
            "2. Horarios de mayor actividad\n"
            "3. Preguntas frecuentes que indican necesidades no cubiertas\n"
            "4. Ticket promedio para definir presupuesto de ads\n"
            "5. Tasa de conversion actual para establecer KPIs realistas\n"
            "6. Oportunidades de mejora vs ads anteriores"
        )

    @tool
    def research_audience(business_type: str, location: str = "", current_audience: str = "") -> str:
        """Investiga y define el publico objetivo ideal basandose en el tipo de negocio."""
        try:
            results = vector_store.similarity_search(
                f"clientes publico objetivo {business_type}", k=4
            )
            doc_context = ""
            if results:
                doc_context = "\n".join(
                    f"- [{sr.chunk.filename}]: {sr.chunk.content[:300]}"
                    for sr in results
                )
            return (
                f"INVESTIGACION DE AUDIENCIA:\n"
                f"Tipo de negocio: {business_type}\n"
                f"Ubicacion: {location or 'No especificada'}\n"
                f"Audiencia actual conocida: {current_audience or 'No especificada'}\n"
                f"Informacion de documentos:\n{doc_context}\n\n"
                "DEFINE:\n"
                "1. Segmento primario (edad, genero, ubicacion, intereses)\n"
                "2. Segmento secundario (lookalike)\n"
                "3. Segmento de retargeting\n"
                "4. Intereses especificos para Meta Ads targeting\n"
                "5. Comportamientos de compra relevantes"
            )
        except Exception as e:
            return f"Error investigando audiencia: {e}"

    return [search_product_catalog, analyze_business_data, research_audience]
