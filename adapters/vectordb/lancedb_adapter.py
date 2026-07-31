"""
Adapter: VectorStorePort backed by LanceDB.

Talks to LanceDB directly rather than through langchain-community's wrapper. The
wrapper is now sunset, and it hid the two things worth controlling here: how
metadata is stored, and how distance becomes the relevance score users see.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import lancedb
import pyarrow as pa
from langchain_openai import OpenAIEmbeddings

from core.logger import logger
from domain.models import DocumentChunk, SearchResult

# text-embedding-3-small produces 1536-dimensional vectors.
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
DEFAULT_DIMENSIONS = 1536

# Embed in batches: one request per chunk is slow and rate-limits quickly on a
# large PDF, while OpenAI accepts many inputs per call.
EMBED_BATCH_SIZE = 96


class LanceDBAdapter:
    """
    On-disk vector store.

    LanceDB is embedded: no server, no extra container, the index is a directory.
    That is what makes `git clone && docker compose up` actually work, which is
    worth more in a reviewable project than the marginal recall a managed store
    would add. VectorStorePort keeps the swap cheap if that changes.
    """

    def __init__(
        self,
        db_path: str,
        table_name: str,
        embedding_model: str,
        api_key: str,
        embeddings=None,
        dimensions: int | None = None,
    ) -> None:
        # embeddings/dimensions are injectable so the store can be exercised
        # offline in tests. Production always uses the OpenAI default.
        self._emb = embeddings or OpenAIEmbeddings(model=embedding_model, api_key=api_key)
        self._dim = dimensions or EMBEDDING_DIMENSIONS.get(embedding_model, DEFAULT_DIMENSIONS)
        self._db = lancedb.connect(db_path)
        self._table_name = table_name

        if self._table_exists():
            logger.info("LanceDB: opened table '%s'", table_name)
        else:
            logger.info("LanceDB: table '%s' will be created on first index", table_name)

    def _table_exists(self) -> bool:
        """
        Whether the table has been created yet.

        lancedb 0.36 returns a paginated ListTablesResponse from list_tables(),
        while older versions return a plain list. A membership test against the
        response object silently evaluates false, which made every search return
        nothing while indexing appeared to succeed -- so normalise explicitly
        rather than trusting the return type.
        """
        listing = self._db.list_tables()
        names = getattr(listing, "tables", listing)
        return self._table_name in list(names)

    @property
    def _schema(self) -> pa.Schema:
        # Metadata is stored as flat columns rather than a nested struct so that
        # filtering by filename or page is a plain SQL predicate later.
        return pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), self._dim)),
                pa.field("content", pa.string()),
                pa.field("filename", pa.string()),
                pa.field("source", pa.string()),
                pa.field("file_type", pa.string()),
                pa.field("content_type", pa.string()),
                pa.field("page_number", pa.int32()),
                pa.field("image_path", pa.string()),
            ]
        )

    def _table(self) -> Any | None:
        if not self._table_exists():
            return None
        return self._db.open_table(self._table_name)

    def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            vectors.extend(self._emb.embed_documents(texts[i:i + EMBED_BATCH_SIZE]))
        return vectors

    @staticmethod
    def _row(chunk: DocumentChunk, vector: list[float]) -> dict[str, Any]:
        meta = chunk.metadata
        return {
            "vector": vector,
            "content": chunk.content,
            "filename": meta.get("filename", "unknown"),
            "source": meta.get("source", ""),
            "file_type": meta.get("file_type", ""),
            "content_type": meta.get("content_type", "text"),
            "page_number": int(meta.get("page_number", 0) or 0),
            "image_path": meta.get("image_path", ""),
        }

    # ---- VectorStorePort ----

    def add_documents(self, docs: Sequence[DocumentChunk]) -> int:
        if not docs:
            return 0

        vectors = self._embed_batched([d.content for d in docs])
        rows = [self._row(d, v) for d, v in zip(docs, vectors, strict=True)]

        table = self._table()
        if table is None:
            self._db.create_table(self._table_name, data=rows, schema=self._schema)
        else:
            table.add(rows)

        logger.info("Indexed %d chunks into LanceDB", len(rows))
        return len(rows)

    def similarity_search(self, query: str, k: int = 4) -> list[SearchResult]:
        table = self._table()
        if table is None:
            return []

        rows = (
            table.search(self._emb.embed_query(query))
            .metric("cosine")
            .limit(k)
            .to_list()
        )

        results = []
        for row in rows:
            # LanceDB returns cosine distance in [0, 2]; users expect a
            # similarity where higher is better.
            score = 1.0 - float(row.get("_distance", 1.0))
            results.append(
                SearchResult(
                    chunk=DocumentChunk(
                        content=row["content"],
                        metadata={
                            "filename": row["filename"],
                            "source": row["source"],
                            "file_type": row["file_type"],
                            "content_type": row["content_type"],
                            "page_number": row["page_number"],
                            "image_path": row["image_path"],
                        },
                    ),
                    score=max(0.0, min(score, 1.0)),
                )
            )
        return results

    def embed_query(self, text: str) -> list[float]:
        """Exposed so the semantic cache shares the document embedding space."""
        return self._emb.embed_query(text)

    def get_document_count(self) -> int:
        table = self._table()
        return int(table.count_rows()) if table is not None else 0

    def list_sources(self) -> list[dict[str, Any]]:
        """Distinct indexed files with chunk counts, for the UI's document list."""
        table = self._table()
        if table is None:
            return []

        rows = table.to_arrow().select(["filename", "file_type", "content_type"]).to_pylist()
        summary: dict[str, dict[str, Any]] = {}
        for row in rows:
            entry = summary.setdefault(
                row["filename"],
                {"filename": row["filename"], "file_type": row["file_type"], "chunks": 0, "images": 0},
            )
            entry["chunks"] += 1
            if row["content_type"] == "image":
                entry["images"] += 1

        return sorted(summary.values(), key=lambda d: d["filename"])

    def clear(self) -> None:
        if self._table_exists():
            self._db.drop_table(self._table_name)
            logger.info("LanceDB: dropped table '%s'", self._table_name)
