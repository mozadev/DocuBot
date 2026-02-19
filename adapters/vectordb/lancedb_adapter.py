"""Adapter: implementación de VectorStorePort con LanceDB."""

from __future__ import annotations

from typing import List, Optional, Sequence

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import LanceDB
import lancedb

from domain.models import DocumentChunk, SearchResult
from core.logger import logger


class LanceDBAdapter:
    """Vector store backed by LanceDB en disco."""

    def __init__(self, db_path: str, table_name: str, embedding_model: str, api_key: str) -> None:
        self._emb = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
        self._db = lancedb.connect(db_path)
        self._table_name = table_name
        self._vs: Optional[LanceDB] = None

        try:
            self._db.open_table(self._table_name)
            self._vs = self._init_vs()
            logger.info(f"LanceDB: tabla '{self._table_name}' abierta.")
        except Exception:
            logger.info(f"LanceDB: tabla '{self._table_name}' se creará al indexar.")

    def _init_vs(self) -> LanceDB:
        try:
            return LanceDB(connection=self._db, embedding=self._emb, table_name=self._table_name)
        except TypeError:
            return LanceDB(connection=self._db, embedding=self._emb)

    def _ensure_vs(self) -> None:
        if self._vs is None:
            self._vs = self._init_vs()

    # --- VectorStorePort ---

    def add_documents(self, docs: Sequence[DocumentChunk]) -> int:
        self._ensure_vs()
        if not docs:
            return 0
        lc_docs = [
            Document(page_content=d.content, metadata=d.metadata)
            for d in docs
        ]
        self._vs.add_documents(lc_docs)
        logger.info(f"{len(lc_docs)} chunks indexados en LanceDB.")
        return len(lc_docs)

    def similarity_search(self, query: str, k: int = 4) -> List[SearchResult]:
        self._ensure_vs()
        results = self._vs.similarity_search_with_relevance_scores(query, k=k)
        return [
            SearchResult(
                chunk=DocumentChunk(content=doc.page_content, metadata=doc.metadata),
                score=float(score),
            )
            for doc, score in results
        ]

    def as_retriever(self, k: int = 4):
        self._ensure_vs()
        return self._vs.as_retriever(search_kwargs={"k": k})

    def get_document_count(self) -> int:
        try:
            tbl = self._db.open_table(self._table_name)
            if hasattr(tbl, "count_rows"):
                return int(tbl.count_rows())
            return int(len(tbl))
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            self._db.drop_table(self._table_name)
            logger.info(f"LanceDB: tabla '{self._table_name}' eliminada.")
        except Exception as e:
            logger.warning(f"No se pudo eliminar tabla: {e}")
        self._vs = None
