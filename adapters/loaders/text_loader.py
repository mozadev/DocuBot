"""Adapter: plain text and Markdown loader."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.logger import logger
from domain.models import DocumentChunk


class TextLoader:
    """Loads .txt and .md files."""

    supported_extensions = [".txt", ".md"]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        # Markdown headings first: splitting on '\n## ' keeps a section's heading
        # attached to its body, which matters because the heading is often the
        # only place the section's subject is named.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )

    def load(self, file_path: str, original_filename: str | None = None) -> list[DocumentChunk]:
        name = original_filename or Path(file_path).name
        text = Path(file_path).read_text(encoding="utf-8", errors="replace").strip()

        if not text:
            logger.warning("Text file is empty: %s", name)
            return []

        metadata = {
            "source": name,
            "filename": name,
            "file_type": Path(name).suffix.lower(),
            "file_size": os.path.getsize(file_path),
            "content_type": "text",
        }

        chunks = [
            DocumentChunk(content=s, metadata={**metadata})
            for s in self._splitter.split_text(text)
        ]
        logger.info("Text file processed: %s -> %d chunks", name, len(chunks))
        return chunks
