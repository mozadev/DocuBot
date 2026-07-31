"""Port: the contract any vector store must satisfy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from domain.models import DocumentChunk, SearchResult


@runtime_checkable
class VectorStorePort(Protocol):
    """
    Implemented by LanceDBAdapter today; pgvector, Qdrant or Pinecone would
    satisfy the same interface without any change above this line.
    """

    def add_documents(self, docs: Sequence[DocumentChunk]) -> int:
        """Index chunks. Returns how many were added."""
        ...

    def similarity_search(self, query: str, k: int = 4) -> list[SearchResult]:
        """Return the k most similar chunks, each with a relevance score."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a string into the same space as the indexed documents."""
        ...

    def get_document_count(self) -> int:
        """Total number of indexed chunks."""
        ...

    def list_sources(self) -> list[dict[str, Any]]:
        """Distinct source files with their chunk counts."""
        ...

    def clear(self) -> None:
        """Delete the whole index."""
        ...
