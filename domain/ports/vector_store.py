"""Port: contrato para cualquier vector store."""

from __future__ import annotations

from typing import Protocol, List, Sequence, runtime_checkable

from domain.models import DocumentChunk, SearchResult


@runtime_checkable
class VectorStorePort(Protocol):
    """Interfaz que debe cumplir cualquier vector store (LanceDB, Pinecone, Qdrant, etc.)."""

    def add_documents(self, docs: Sequence[DocumentChunk]) -> int:
        """Indexa chunks. Retorna cuántos se añadieron."""
        ...

    def similarity_search(self, query: str, k: int = 4) -> List[SearchResult]:
        """Búsqueda semántica con scores."""
        ...

    def as_retriever(self, k: int = 4):
        """Retorna un retriever compatible con LangChain."""
        ...

    def get_document_count(self) -> int:
        """Cantidad de chunks indexados."""
        ...

    def clear(self) -> None:
        """Elimina todos los documentos."""
        ...
