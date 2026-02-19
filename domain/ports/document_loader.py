"""Port: contrato para cualquier cargador de documentos."""

from __future__ import annotations

from typing import Protocol, List, runtime_checkable

from domain.models import DocumentChunk


@runtime_checkable
class DocumentLoaderPort(Protocol):
    """Interfaz que debe cumplir cualquier loader de documentos."""

    def load(self, file_path: str) -> List[DocumentChunk]:
        """Extrae contenido de un archivo y lo divide en chunks."""
        ...

    @property
    def supported_extensions(self) -> List[str]:
        """Extensiones de archivo que soporta este loader."""
        ...
