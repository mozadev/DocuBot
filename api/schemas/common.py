"""Request and response schemas for the public API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class StatusResponse(BaseModel):
    service: str
    version: str
    model: str
    embedding_model: str
    multimodal: bool
    vision_model: str
    chunk_size: int
    document_count: int
    supported_formats: list[str]


class SourceSchema(BaseModel):
    filename: str
    content: str
    score: float
    content_type: str = "text"
    image_path: str = ""
    page_number: int = 0


class ChatRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=2000, examples=["What is the vacation policy?"]
    )


class ChatResponse(BaseModel):
    answer: str
    question: str
    sources: list[SourceSchema] = Field(default_factory=list)
    confidence: float = 0.0
    trace_id: str = ""
    cached: bool = False
    guardrail: dict[str, Any] = Field(default_factory=dict)


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[dict[str, str]]


class SummaryResponse(BaseModel):
    session_id: str
    summary: str


class UploadResult(BaseModel):
    filename: str
    text_chunks: int = 0
    image_chunks: int = 0
    total_chunks: int = 0
    error: str = ""


class UploadResponse(BaseModel):
    files_processed: int
    total_chunks_indexed: int
    details: list[UploadResult]


class DocumentStatsResponse(BaseModel):
    total_chunks: int
    status: str
    supported_formats: list[str]
    documents: list[dict[str, Any]] = Field(default_factory=list)


class MessageResponse(BaseModel):
    message: str
