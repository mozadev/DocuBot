"""
Adapter: PDF loader, text plus embedded images.

Uses PyMuPDF for both passes. Text extraction and image extraction need the same
document open anyway, and PyMuPDF's layout-aware extraction preserves reading
order on multi-column pages better than pypdf does.

When multimodal mode is on, each embedded image is described by a vision model
and the description is indexed as its own chunk. That is what makes a question
like "what does the architecture diagram show?" answerable at all -- the text
layer of a PDF says nothing about its figures.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.logger import logger
from domain.models import DocumentChunk


class PDFLoader:
    """Extracts text and image descriptions from PDF files."""

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
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._enable_images = enable_images
        self._images_dir = Path(images_dir)
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._min_image_size = min_image_size
        self._describe_image = describe_image_fn

    def load(self, file_path: str, original_filename: str | None = None) -> list[DocumentChunk]:
        name = original_filename or Path(file_path).name

        try:
            doc = fitz.open(file_path)
        except Exception as e:  # noqa: BLE001
            logger.error("Could not open PDF %s: %s", name, e)
            raise ValueError(f"Could not read PDF: {e}") from e

        try:
            text_chunks = self._extract_text(doc, file_path, name)
            image_chunks = (
                self._extract_images(doc, name)
                if self._enable_images and self._describe_image
                else []
            )
        finally:
            doc.close()

        logger.info(
            "PDF processed: %s -> %d text + %d image chunks",
            name, len(text_chunks), len(image_chunks),
        )
        return text_chunks + image_chunks

    def _extract_text(self, doc, file_path: str, name: str) -> list[DocumentChunk]:
        """
        Chunk page by page so page_number stays accurate on every citation.

        Concatenating the whole document first would be simpler, but then a chunk
        spanning a page break could not be attributed to a page, and page numbers
        are the single most useful thing to show a user verifying an answer.
        """
        chunks: list[DocumentChunk] = []
        file_size = os.path.getsize(file_path)

        for page_idx in range(len(doc)):
            page_text = doc[page_idx].get_text().strip()
            if not page_text:
                continue

            for split in self._splitter.split_text(page_text):
                chunks.append(
                    DocumentChunk(
                        content=split,
                        metadata={
                            "source": name,
                            "filename": name,
                            "file_type": ".pdf",
                            "file_size": file_size,
                            "content_type": "text",
                            "page_number": page_idx + 1,
                        },
                    )
                )

        if not chunks:
            logger.warning(
                "No extractable text in %s. It may be a scanned PDF; OCR is not enabled.", name
            )
        return chunks

    def _extract_images(self, doc, name: str) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        stem = Path(name).stem
        seen_hashes: set = set()

        for page_idx in range(len(doc)):
            for img_info in doc[page_idx].get_images(full=True):
                try:
                    base_image = doc.extract_image(img_info[0])
                except Exception:  # noqa: BLE001 - a single bad image must not fail the upload
                    continue

                width, height = base_image.get("width", 0), base_image.get("height", 0)
                if width < self._min_image_size or height < self._min_image_size:
                    continue  # icons, bullets, logos

                img_bytes = base_image["image"]
                img_hash = hashlib.md5(img_bytes).hexdigest()[:10]
                if img_hash in seen_hashes:
                    continue  # headers/footers repeated on every page
                seen_hashes.add(img_hash)

                ext = base_image.get("ext", "png")
                save_path = self._images_dir / f"{stem}_p{page_idx + 1}_{img_hash}.{ext}"
                save_path.write_bytes(img_bytes)

                try:
                    description = self._describe_image(
                        str(save_path), context=f"Page {page_idx + 1} of '{name}'"
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("Vision description failed for %s: %s", save_path.name, e)
                    continue

                if not description or description.startswith("[image not processed"):
                    continue

                chunks.append(
                    DocumentChunk(
                        content=f"[FIGURE - page {page_idx + 1}]\n{description}",
                        metadata={
                            "source": name,
                            "filename": name,
                            "file_type": ".pdf",
                            "content_type": "image",
                            "image_path": str(save_path),
                            "page_number": page_idx + 1,
                            "image_dimensions": f"{width}x{height}",
                        },
                    )
                )

        return chunks
