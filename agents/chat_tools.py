"""Herramientas del agente LangGraph para DocuBot AI."""

from __future__ import annotations

from typing import List

from langchain_core.tools import tool

from core.logger import logger


def build_tools(vector_store) -> list:
    """Construye herramientas internas inyectando el vector store."""

    @tool
    def search_documents(query: str) -> str:
        """Busca información relevante en los documentos indexados.
        Usa esta herramienta cuando el usuario pregunte sobre contenido de documentos."""
        try:
            results = vector_store.similarity_search(query, k=4)
            if not results:
                return "No se encontraron documentos relevantes para esta consulta."
            parts: List[str] = []
            for i, sr in enumerate(results, 1):
                label = "IMAGEN" if sr.chunk.is_image else "TEXTO"
                parts.append(
                    f"[Fuente {i} | {label} | {sr.chunk.filename} | score={sr.score:.3f}]\n"
                    f"{sr.chunk.content[:500]}"
                )
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            logger.error(f"Error en search_documents: {e}")
            return f"Error buscando documentos: {e}"

    @tool
    def get_document_stats() -> str:
        """Devuelve estadísticas sobre los documentos indexados.
        Usa esta herramienta cuando el usuario pregunte cuántos documentos hay."""
        try:
            count = vector_store.get_document_count()
            return (
                f"Base de datos vectorial:\n"
                f"- Chunks indexados: {count}\n"
                f"- Estado: {'Activa' if count > 0 else 'Vacía'}"
            )
        except Exception as e:
            return f"Error: {e}"

    @tool
    def summarize_topic(topic: str) -> str:
        """Resume información sobre un tema en los documentos indexados.
        Usa esta herramienta cuando el usuario pida un resumen sobre un tema."""
        try:
            results = vector_store.similarity_search(topic, k=6)
            if not results:
                return f"No se encontró información sobre '{topic}'."
            combined = "\n\n".join(sr.chunk.content[:400] for sr in results)
            return f"Información sobre '{topic}' ({len(results)} fuentes):\n\n{combined}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def suggest_marketing_content(product_or_service: str) -> str:
        """Busca información del producto/servicio en los documentos para sugerir contenido de marketing.
        Usa esta herramienta cuando el usuario pida ayuda con marketing, publicidad o campañas."""
        try:
            results = vector_store.similarity_search(product_or_service, k=6)
            if not results:
                return f"No se encontró información sobre '{product_or_service}' para generar contenido de marketing."
            context = "\n\n".join(
                f"- [{sr.chunk.filename}]: {sr.chunk.content[:300]}"
                for sr in results
            )
            return (
                f"Información disponible sobre '{product_or_service}' para marketing "
                f"({len(results)} fuentes):\n\n{context}\n\n"
                "Usa esta información para generar contenido de marketing relevante."
            )
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_product_catalog() -> str:
        """Obtiene un resumen del catálogo de productos/servicios disponibles en los documentos.
        Usa cuando necesites saber qué productos/servicios tiene el negocio."""
        try:
            queries = ["productos", "servicios", "precios", "catálogo", "oferta"]
            all_content: List[str] = []
            seen_files = set()
            for q in queries:
                results = vector_store.similarity_search(q, k=3)
                for sr in results:
                    key = f"{sr.chunk.filename}:{sr.chunk.content[:50]}"
                    if key not in seen_files:
                        seen_files.add(key)
                        all_content.append(f"[{sr.chunk.filename}]: {sr.chunk.content[:300]}")
            if not all_content:
                return "No se encontró información de productos o servicios en los documentos."
            return f"Catálogo encontrado ({len(all_content)} items):\n\n" + "\n\n".join(all_content)
        except Exception as e:
            return f"Error: {e}"

    return [
        search_documents,
        get_document_stats,
        summarize_topic,
        suggest_marketing_content,
        get_product_catalog,
    ]
