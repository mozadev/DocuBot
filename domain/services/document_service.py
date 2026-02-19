"""Servicio de dominio: procesamiento e indexación de documentos."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from domain.models import DocumentChunk
from core.logger import logger, log_function_call


class DocumentService:
    """Orquesta la carga, procesamiento e indexación de documentos."""

    def __init__(self, loaders: Dict[str, Any], vector_store) -> None:
        self._loaders = loaders  # {".pdf": PDFLoader, ".docx": DOCXLoader}
        self._vector_store = vector_store

    @log_function_call
    def process_and_index(self, file_path: str) -> Dict[str, Any]:
        """
        Procesa un archivo y lo indexa en el vector store.
        Retorna estadísticas del procesamiento.
        """
        ext = Path(file_path).suffix.lower()
        loader = self._loaders.get(ext)

        if loader is None:
            raise ValueError(f"Formato no soportado: {ext}. Soportados: {list(self._loaders.keys())}")

        chunks = loader.load(file_path)

        if not chunks:
            return {"filename": Path(file_path).name, "text_chunks": 0, "image_chunks": 0, "total": 0}

        added = self._vector_store.add_documents(chunks)

        text_count = sum(1 for c in chunks if c.content_type == "text")
        image_count = sum(1 for c in chunks if c.content_type == "image")

        logger.info(
            f"Documento indexado: {Path(file_path).name} → "
            f"{text_count} texto + {image_count} imágenes"
        )
        return {
            "filename": Path(file_path).name,
            "text_chunks": text_count,
            "image_chunks": image_count,
            "total": added,
        }

    def get_document_count(self) -> int:
        return self._vector_store.get_document_count()

    def clear_database(self) -> None:
        self._vector_store.clear()
        logger.info("Base de datos limpiada")

    @property
    def supported_extensions(self) -> List[str]:
        return list(self._loaders.keys())
