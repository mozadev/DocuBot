"""
Composition root.

The one place that knows which concrete adapter implements each port. Every
other module depends on the interface, so swapping LanceDB for pgvector or
OpenAI for a local model is an edit here and nowhere else.

Built once per process and memoised: the adapters hold network clients and an
open database handle, and rebuilding them per request would be wasteful.
"""

from __future__ import annotations

from dataclasses import dataclass

from adapters.cache.semantic_cache import SemanticCache
from adapters.llm.openai_adapter import OpenAIAdapter
from adapters.loaders import DOCXLoader, PDFLoader, TextLoader
from adapters.observability.tracer import AgentTracer
from adapters.vectordb.lancedb_adapter import LanceDBAdapter
from agents.graph import build_agent_graph
from config.settings import settings
from core.logger import logger
from domain.guardrails import RagGuardrails
from domain.services.chat_service import ChatService
from domain.services.document_service import DocumentService


@dataclass
class Container:
    """Everything the API and the UI need, wired together."""

    documents: DocumentService
    chat: ChatService
    cache: SemanticCache
    tracer: AgentTracer
    guardrails: RagGuardrails


_container: Container | None = None


def build_container() -> Container:
    """Assemble the application. Idempotent -- returns the same instance."""
    global _container
    if _container is not None:
        return _container

    vector_store = LanceDBAdapter(
        db_path=settings.lancedb_path,
        table_name=settings.lancedb_table,
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    llm = OpenAIAdapter(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        vision_model=settings.vision_model,
        temperature=settings.openai_temperature,
    )

    # The cache embeds questions with the same model used for documents, so
    # similarity thresholds mean the same thing in both places.
    cache = SemanticCache(
        max_entries=settings.cache_max_entries,
        default_ttl=settings.cache_ttl_seconds,
        embedding_func=vector_store.embed_query,
    )
    tracer = AgentTracer(max_traces=settings.max_traces)
    guardrails = RagGuardrails(min_grounding_score=settings.min_grounding_score)

    loaders = {
        ".pdf": PDFLoader(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            enable_images=settings.enable_multimodal,
            images_dir=settings.images_path,
            min_image_size=settings.min_image_size,
            describe_image_fn=llm.describe_image if settings.enable_multimodal else None,
        ),
        ".docx": DOCXLoader(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        ),
    }
    text_loader = TextLoader(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    loaders[".txt"] = text_loader
    loaders[".md"] = text_loader

    _container = Container(
        documents=DocumentService(loaders=loaders, vector_store=vector_store, cache=cache),
        chat=ChatService(
            vector_store=vector_store,
            llm_adapter=llm,
            graph_builder=build_agent_graph,
            guardrails=guardrails,
            tracer=tracer,
            cache=cache,
        ),
        cache=cache,
        tracer=tracer,
        guardrails=guardrails,
    )

    logger.info(
        "Container ready: model=%s embeddings=%s multimodal=%s",
        settings.openai_model, settings.embedding_model, settings.enable_multimodal,
    )
    return _container


def reset_container() -> None:
    """Drop the memoised container. Used by tests."""
    global _container
    _container = None
