"""
Domain entities for DocuBot AI.

Pure data structures with no framework dependencies — these are the types that
cross layer boundaries (adapters -> services -> API), so keeping them free of
LangChain/FastAPI/LanceDB imports is what lets each adapter be swapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """A fragment of a source document, ready to be embedded and indexed."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return self.metadata.get("filename", "unknown")

    @property
    def content_type(self) -> str:
        return self.metadata.get("content_type", "text")

    @property
    def is_image(self) -> bool:
        return self.content_type == "image"


@dataclass
class SearchResult:
    """A retrieved chunk together with its relevance score (0.0 - 1.0)."""

    chunk: DocumentChunk
    score: float


@dataclass
class Source:
    """A citation attached to an answer, surfaced to the user."""

    filename: str
    content: str
    score: float
    content_type: str = "text"
    image_path: str = ""
    page_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "content": self.content,
            "score": self.score,
            "content_type": self.content_type,
            "image_path": self.image_path,
            "page_number": self.page_number,
        }


@dataclass
class ChatResponse:
    """The agent's answer to a single user question."""

    answer: str
    sources: list[Source] = field(default_factory=list)
    confidence: float = 0.0
    question: str = ""
    trace_id: str = ""
    cached: bool = False
    guardrail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence,
            "question": self.question,
            "trace_id": self.trace_id,
            "cached": self.cached,
            "guardrail": self.guardrail,
        }


@dataclass
class ChatMessage:
    """One turn in a conversation history."""

    role: str  # "human" | "ai"
    content: str
