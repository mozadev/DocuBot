"""MCP Server propio de DocuBot AI.
Expone búsqueda documental como herramientas MCP.

Ejecutar: python -m adapters.mcp.docubot_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from config.settings import settings
from adapters.vectordb.lancedb_adapter import LanceDBAdapter
from core.logger import logger

mcp = FastMCP("DocuBot AI")

_vs: LanceDBAdapter | None = None


def _get_vs() -> LanceDBAdapter:
    global _vs
    if _vs is None:
        _vs = LanceDBAdapter(
            db_path=settings.lancedb_path,
            table_name=settings.lancedb_table,
            embedding_model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    return _vs


@mcp.tool()
def search_documents(query: str, num_results: int = 4) -> str:
    """Busca información en los documentos indexados de la empresa."""
    vs = _get_vs()
    results = vs.similarity_search(query, k=num_results)
    if not results:
        return "No se encontraron documentos relevantes."
    parts = []
    for i, sr in enumerate(results, 1):
        label = "IMAGEN" if sr.chunk.is_image else "TEXTO"
        parts.append(
            f"[Fuente {i} | {label} | {sr.chunk.filename} | score={sr.score:.3f}]\n"
            f"{sr.chunk.content[:600]}"
        )
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def get_stats() -> str:
    """Estadísticas de la base documental."""
    vs = _get_vs()
    count = vs.get_document_count()
    return f"Chunks indexados: {count}\nEstado: {'Activa' if count > 0 else 'Vacía'}"


@mcp.resource("docubot://status")
def get_status() -> str:
    """Estado del servidor DocuBot MCP."""
    vs = _get_vs()
    return f"DocuBot AI MCP - Operativo - {vs.get_document_count()} docs"


if __name__ == "__main__":
    mcp.run()
