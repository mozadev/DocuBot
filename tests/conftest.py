"""
Shared fixtures.

The whole suite runs without an OpenAI key and without network access. Anything
that would call out is replaced with a fake that implements the same port, which
is the practical payoff of defining ports at all: the tests exercise real
service logic rather than mocks of it.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

# Must be set before config.settings is imported anywhere.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")

from domain.models import DocumentChunk, SearchResult  # noqa: E402


class FakeVectorStore:
    """In-memory VectorStorePort. Scores by keyword overlap instead of cosine."""

    def __init__(self, chunks: list[DocumentChunk] | None = None) -> None:
        self._chunks: list[DocumentChunk] = list(chunks or [])

    def add_documents(self, docs) -> int:
        self._chunks.extend(docs)
        return len(docs)

    def similarity_search(self, query: str, k: int = 4) -> list[SearchResult]:
        terms = {t for t in query.lower().split() if len(t) > 2}
        scored = []
        for chunk in self._chunks:
            words = set(chunk.content.lower().split())
            overlap = len(terms & words)
            if overlap:
                scored.append(SearchResult(chunk=chunk, score=min(overlap / max(len(terms), 1), 1.0)))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def embed_query(self, text: str) -> list[float]:
        # Deterministic pseudo-embedding: identical text gives identical vectors,
        # which is all the cache tests need.
        return [float(ord(c) % 17) for c in text.ljust(32)[:32]]

    def get_document_count(self) -> int:
        return len(self._chunks)

    def list_sources(self) -> list[dict[str, Any]]:
        names: dict[str, dict[str, Any]] = {}
        for chunk in self._chunks:
            entry = names.setdefault(
                chunk.filename, {"filename": chunk.filename, "chunks": 0, "images": 0}
            )
            entry["chunks"] += 1
            if chunk.is_image:
                entry["images"] += 1
        return list(names.values())

    def clear(self) -> None:
        self._chunks.clear()


class FakeLLM:
    """LLMPort stub. Returns canned text; records what it was asked."""

    def __init__(self, reply: str = "A canned reply.") -> None:
        self.reply = reply
        self.calls: list[Any] = []

    def invoke(self, messages) -> str:
        self.calls.append(messages)
        return self.reply

    def describe_image(self, image_path: str, context: str = "") -> str:
        return f"Description of {image_path}"

    def get_langchain_llm(self):
        raise NotImplementedError("Graph tests inject a fake graph instead.")


class FakeGraph:
    """Stands in for the compiled LangGraph agent."""

    def __init__(self, answer: str = "The answer.", sources=None, confidence: float = 0.8) -> None:
        self.answer = answer
        self.sources = sources if sources is not None else [
            {"filename": "handbook.pdf", "content": "...", "score": 0.8, "content_type": "text"}
        ]
        self.confidence = confidence
        self.invocations: list[dict] = []

    def invoke(self, state: dict) -> dict:
        from langchain_core.messages import AIMessage

        self.invocations.append(state)
        return {
            "messages": list(state["messages"]) + [AIMessage(content=self.answer)],
            "sources": self.sources,
            "confidence": self.confidence,
        }


@pytest.fixture
def chunks() -> list[DocumentChunk]:
    def chunk(text: str, filename: str = "handbook.pdf", page: int = 1, kind: str = "text"):
        return DocumentChunk(
            content=text,
            metadata={
                "filename": filename,
                "source": filename,
                "page_number": page,
                "content_type": kind,
            },
        )

    return [
        chunk("Full-time employees accrue 20 days of paid vacation per year.", page=1),
        chunk("Unused vacation days carry over, up to a maximum of 5 days.", page=1),
        chunk("Employees may work remotely up to 3 days per week.", page=2),
        chunk("[FIGURE - page 3] Org chart showing four engineering teams.", page=3, kind="image"),
    ]


@pytest.fixture
def vector_store(chunks) -> FakeVectorStore:
    return FakeVectorStore(chunks)


@pytest.fixture
def empty_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
