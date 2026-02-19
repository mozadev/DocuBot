"""Adapter: cargador de documentos PDF (texto + imágenes)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional

import PyPDF2
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from domain.models import DocumentChunk
from core.logger import logger


class PDFLoader:
    """Extrae texto e imágenes de PDFs."""

    supported_extensions = [".pdf"]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        enable_images: bool = True,
        images_dir: str = "./data/images",
        min_image_size: int = 100,
        describe_image_fn=None,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        self._enable_images = enable_images
        self._images_dir = Path(images_dir)
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._min_image_size = min_image_size
        self._describe_image = describe_image_fn

    def load(self, file_path: str) -> List[DocumentChunk]:
        text_chunks = self._extract_text_chunks(file_path)
        image_chunks = []
        if self._enable_images and self._describe_image:
            image_chunks = self._extract_image_chunks(file_path)

        logger.info(
            f"PDF procesado: {Path(file_path).name} → "
            f"{len(text_chunks)} texto + {len(image_chunks)} imágenes"
        )
        return text_chunks + image_chunks

    def _extract_text_chunks(self, file_path: str) -> List[DocumentChunk]:
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"

        text = text.strip()
        if not text:
            return []

        metadata = {
            "source": file_path,
            "filename": Path(file_path).name,
            "file_type": ".pdf",
            "file_size": os.path.getsize(file_path),
            "content_type": "text",
        }

        splits = self._splitter.split_text(text)
        return [
            DocumentChunk(content=chunk, metadata={**metadata})
            for chunk in splits
        ]

    def _extract_image_chunks(self, file_path: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        pdf_name = Path(file_path).stem

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.warning(f"No se pudo abrir PDF para imágenes: {e}")
            return chunks

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    continue

                w, h = base_image.get("width", 0), base_image.get("height", 0)
                if w < self._min_image_size or h < self._min_image_size:
                    continue

                img_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                img_hash = hashlib.md5(img_bytes).hexdigest()[:10]
                filename = f"{pdf_name}_p{page_idx+1}_img{img_idx+1}_{img_hash}.{ext}"
                save_path = self._images_dir / filename

                with open(save_path, "wb") as f:
                    f.write(img_bytes)

                description = self._describe_image(
                    str(save_path),
                    context=f"Página {page_idx+1} de '{Path(file_path).name}'",
                )
                if description.startswith("[Imagen no procesada"):
                    continue

                chunks.append(DocumentChunk(
                    content=f"[IMAGEN - Página {page_idx+1}]\n{description}",
                    metadata={
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "file_type": ".pdf",
                        "content_type": "image",
                        "image_path": str(save_path),
                        "page_number": page_idx + 1,
                        "image_dimensions": f"{w}x{h}",
                    },
                ))

        doc.close()
        return chunks
