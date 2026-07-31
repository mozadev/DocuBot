"""Document loading, chunking and indexing."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from adapters.loaders.pdf_loader import PDFLoader
from adapters.loaders.text_loader import TextLoader
from domain.services.document_service import DocumentService


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((60, 80), "Vacation Policy")
    page1.insert_text((60, 110), "Full-time employees accrue 20 days per year.")
    page2 = doc.new_page()
    page2.insert_text((60, 80), "Remote Work")
    page2.insert_text((60, 110), "Up to 3 days per week from home.")
    path = tmp_path / "handbook.pdf"
    doc.save(path)
    doc.close()
    return path


class TestTextLoader:
    def test_splits_and_records_metadata(self, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("## Section A\n" + ("alpha " * 400) + "\n## Section B\nbeta")

        chunks = TextLoader(chunk_size=200, chunk_overlap=20).load(str(path))

        assert len(chunks) > 1
        assert all(c.metadata["filename"] == "notes.md" for c in chunks)
        assert all(c.metadata["file_type"] == ".md" for c in chunks)

    def test_empty_file_yields_no_chunks(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("   ")
        assert TextLoader().load(str(path)) == []

    def test_original_filename_overrides_the_temp_name(self, tmp_path):
        # The upload path writes to a temp file; citations must not show it.
        path = tmp_path / "tmp9rdv1gq9.txt"
        path.write_text("Some content about vacations.")

        chunks = TextLoader().load(str(path), original_filename="handbook.txt")

        assert chunks[0].metadata["filename"] == "handbook.txt"


class TestPDFLoader:
    def test_extracts_text_with_page_numbers(self, pdf_path):
        chunks = PDFLoader(enable_images=False).load(str(pdf_path))

        assert chunks
        pages = {c.metadata["page_number"] for c in chunks}
        assert pages == {1, 2}
        assert any("20 days" in c.content for c in chunks)

    def test_uses_the_original_filename(self, pdf_path):
        chunks = PDFLoader(enable_images=False).load(
            str(pdf_path), original_filename="Employee Handbook.pdf"
        )
        assert all(c.metadata["filename"] == "Employee Handbook.pdf" for c in chunks)

    def test_unreadable_file_raises_a_clear_error(self, tmp_path):
        broken = tmp_path / "broken.pdf"
        broken.write_text("this is not a pdf")

        with pytest.raises(ValueError, match="Could not read PDF"):
            PDFLoader(enable_images=False).load(str(broken))

    def test_image_extraction_is_skipped_without_a_vision_function(self, pdf_path):
        chunks = PDFLoader(enable_images=True, describe_image_fn=None).load(str(pdf_path))
        assert all(c.content_type == "text" for c in chunks)


class TestDocumentService:
    def test_dispatches_to_the_loader_for_the_extension(self, tmp_path, empty_vector_store):
        path = tmp_path / "notes.txt"
        path.write_text("Vacation policy details.")
        service = DocumentService(
            loaders={".txt": TextLoader()}, vector_store=empty_vector_store
        )

        stats = service.process_and_index(str(path))

        assert stats["total"] == 1
        assert stats["text_chunks"] == 1
        assert empty_vector_store.get_document_count() == 1

    def test_rejects_an_unsupported_extension(self, tmp_path, empty_vector_store):
        path = tmp_path / "data.xlsx"
        path.write_text("x")
        service = DocumentService(loaders={".txt": TextLoader()}, vector_store=empty_vector_store)

        with pytest.raises(ValueError, match="Unsupported file type"):
            service.process_and_index(str(path))

    def test_indexing_invalidates_the_cache(self, tmp_path, empty_vector_store):
        class SpyCache:
            invalidated = False

            def invalidate_all(self):
                self.invalidated = True

        cache = SpyCache()
        path = tmp_path / "notes.txt"
        path.write_text("New content that changes what retrieval can return.")
        service = DocumentService(
            loaders={".txt": TextLoader()}, vector_store=empty_vector_store, cache=cache
        )

        service.process_and_index(str(path))

        assert cache.invalidated

    def test_clear_empties_the_index(self, vector_store):
        service = DocumentService(loaders={}, vector_store=vector_store)
        assert service.get_document_count() > 0

        service.clear_database()

        assert service.get_document_count() == 0
