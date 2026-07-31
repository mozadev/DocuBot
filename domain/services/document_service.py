"""Domain service: document ingestion and indexing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.logger import log_function_call, logger


class DocumentService:
    """Loads a file with the right loader and indexes the resulting chunks."""

    def __init__(self, loaders: dict[str, Any], vector_store, cache=None) -> None:
        self._loaders = loaders  # {".pdf": PDFLoader(), ".docx": DOCXLoader(), ...}
        self._vector_store = vector_store
        self._cache = cache

    @log_function_call
    def process_and_index(
        self, file_path: str, original_filename: str | None = None
    ) -> dict[str, Any]:
        """
        Process one file and index it.

        original_filename matters: uploads arrive as temp files, and without it
        every citation would read 'tmp9rdv1gq9.pdf' instead of the name the user
        recognises.
        """
        name = original_filename or Path(file_path).name
        ext = Path(name).suffix.lower()
        loader = self._loaders.get(ext)

        if loader is None:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(self._loaders))}"
            )

        chunks = loader.load(file_path, original_filename=name)
        if not chunks:
            return {"filename": name, "text_chunks": 0, "image_chunks": 0, "total": 0}

        added = self._vector_store.add_documents(chunks)

        # New documents change what retrieval can return, so previously cached
        # answers are now potentially stale.
        if self._cache:
            self._cache.invalidate_all()

        text_count = sum(1 for c in chunks if c.content_type == "text")
        image_count = sum(1 for c in chunks if c.content_type == "image")
        logger.info("Indexed %s: %d text + %d image chunks", name, text_count, image_count)

        return {
            "filename": name,
            "text_chunks": text_count,
            "image_chunks": image_count,
            "total": added,
        }

    def get_document_count(self) -> int:
        return self._vector_store.get_document_count()

    def list_documents(self) -> list[dict[str, Any]]:
        """Distinct source files currently indexed, with their chunk counts."""
        return self._vector_store.list_sources()

    def clear_database(self) -> None:
        self._vector_store.clear()
        if self._cache:
            self._cache.invalidate_all()
        logger.info("Vector store cleared")

    @property
    def supported_extensions(self) -> list[str]:
        return sorted(self._loaders.keys())
