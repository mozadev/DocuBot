"""Port: the contract any document loader must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from domain.models import DocumentChunk


@runtime_checkable
class DocumentLoaderPort(Protocol):
    """One implementation per file format. DocumentService dispatches by extension."""

    supported_extensions: list[str]

    def load(self, file_path: str, original_filename: str | None = None) -> list[DocumentChunk]:
        """
        Read a file and split it into indexable chunks.

        original_filename overrides the name recorded in each chunk's metadata,
        which matters for uploads that arrive as temp files.
        """
        ...
