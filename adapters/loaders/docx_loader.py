"""Adapter: cargador de documentos DOCX."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from domain.models import DocumentChunk
from core.logger import logger


class DOCXLoader:
    """Extrae texto de documentos Word (.docx)."""

    supported_extensions = [".docx"]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def load(self, file_path: str) -> List[DocumentChunk]:
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs).strip()

        if not text:
            return []

        metadata = {
            "source": file_path,
            "filename": Path(file_path).name,
            "file_type": ".docx",
            "file_size": os.path.getsize(file_path),
            "content_type": "text",
        }

        splits = self._splitter.split_text(text)
        chunks = [DocumentChunk(content=s, metadata={**metadata}) for s in splits]
        logger.info(f"DOCX procesado: {Path(file_path).name} → {len(chunks)} chunks")
        return chunks
