"""
LanceDB adapter, exercised against a real on-disk database.

Embeddings are injected rather than called, so these run offline. The store
itself is genuine: table creation, persistence and search all happen for real,
which is the only way this layer's bugs surface.
"""

from __future__ import annotations

import math

import pytest

from adapters.vectordb.lancedb_adapter import LanceDBAdapter
from domain.models import DocumentChunk

DIMENSIONS = 8


class HashEmbeddings:
    """Deterministic unit-norm embeddings derived from token overlap."""

    VOCAB = ["vacation", "days", "remote", "work", "policy", "chart", "team", "office"]

    def _vector(self, text: str) -> list[float]:
        words = text.lower().split()
        raw = [float(sum(1 for w in words if term in w)) for term in self.VOCAB]
        if not any(raw):
            raw = [1.0] + [0.0] * (DIMENSIONS - 1)
        norm = math.sqrt(sum(v * v for v in raw))
        return [v / norm for v in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def chunk(text: str, filename: str = "handbook.pdf", page: int = 1, kind: str = "text"):
    return DocumentChunk(
        content=text,
        metadata={
            "filename": filename,
            "source": filename,
            "file_type": ".pdf",
            "content_type": kind,
            "page_number": page,
        },
    )


@pytest.fixture
def store(tmp_path) -> LanceDBAdapter:
    return LanceDBAdapter(
        db_path=str(tmp_path / "db"),
        table_name="documents",
        embedding_model="test",
        api_key="unused",
        embeddings=HashEmbeddings(),
        dimensions=DIMENSIONS,
    )


@pytest.fixture
def populated(store) -> LanceDBAdapter:
    store.add_documents(
        [
            chunk("Employees accrue 20 vacation days per year."),
            chunk("Remote work is allowed three days per week.", page=2),
            chunk("[FIGURE] Org chart of each team.", page=3, kind="image"),
            chunk("Office opening hours.", filename="facilities.docx"),
        ]
    )
    return store


class TestIndexing:
    def test_empty_input_creates_nothing(self, store):
        assert store.add_documents([]) == 0
        assert store.get_document_count() == 0

    def test_first_index_creates_the_table(self, store):
        assert store.add_documents([chunk("Vacation policy.")]) == 1
        assert store.get_document_count() == 1

    def test_second_index_appends_rather_than_replacing(self, store):
        store.add_documents([chunk("Vacation policy.")])
        store.add_documents([chunk("Remote work policy.")])
        assert store.get_document_count() == 2


class TestSearch:
    def test_search_returns_the_relevant_chunk(self, populated):
        # Regression: an existence check that silently evaluated false made every
        # search return nothing while indexing still reported success.
        results = populated.similarity_search("vacation days", k=3)

        assert results, "indexed documents must be searchable"
        assert "vacation" in results[0].chunk.content.lower()

    def test_scores_are_similarities_in_range(self, populated):
        results = populated.similarity_search("vacation days", k=4)
        assert all(0.0 <= r.score <= 1.0 for r in results)
        assert results == sorted(results, key=lambda r: r.score, reverse=True)

    def test_metadata_survives_the_roundtrip(self, populated):
        result = populated.similarity_search("remote work", k=1)[0]
        assert result.chunk.filename == "handbook.pdf"
        assert result.chunk.metadata["page_number"] == 2

    def test_image_chunks_keep_their_content_type(self, populated):
        results = populated.similarity_search("chart team", k=4)
        assert any(r.chunk.is_image for r in results)

    def test_k_limits_the_result_count(self, populated):
        assert len(populated.similarity_search("vacation", k=2)) <= 2

    def test_search_on_an_empty_store_returns_nothing(self, store):
        assert store.similarity_search("anything") == []


class TestSources:
    def test_lists_distinct_files_with_counts(self, populated):
        sources = {s["filename"]: s for s in populated.list_sources()}

        assert sources["handbook.pdf"]["chunks"] == 3
        assert sources["handbook.pdf"]["images"] == 1
        assert sources["facilities.docx"]["chunks"] == 1

    def test_empty_store_lists_nothing(self, store):
        assert store.list_sources() == []


class TestClear:
    def test_clear_removes_everything(self, populated):
        populated.clear()
        assert populated.get_document_count() == 0
        assert populated.similarity_search("vacation") == []

    def test_clear_on_an_empty_store_is_a_no_op(self, store):
        store.clear()
        assert store.get_document_count() == 0

    def test_store_is_reusable_after_clear(self, populated):
        populated.clear()
        populated.add_documents([chunk("New vacation policy.")])
        assert populated.get_document_count() == 1
        assert populated.similarity_search("vacation")


class TestPersistence:
    def test_data_survives_reopening_the_database(self, tmp_path):
        path = str(tmp_path / "db")
        kwargs = dict(
            db_path=path,
            table_name="documents",
            embedding_model="test",
            api_key="unused",
            embeddings=HashEmbeddings(),
            dimensions=DIMENSIONS,
        )
        LanceDBAdapter(**kwargs).add_documents([chunk("Vacation policy.")])

        reopened = LanceDBAdapter(**kwargs)

        assert reopened.get_document_count() == 1
        assert reopened.similarity_search("vacation")
