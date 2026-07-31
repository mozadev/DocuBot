"""Tools exposed to the LangGraph agent."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import tool

from core.logger import logger

# Retrieval depth. 4 chunks at ~1000 chars each keeps the prompt well inside the
# context window while giving the model enough material to cross-reference two
# or three passages. Raising it mostly adds noise; see README for the trade-off.
DEFAULT_K = 4

# How much of each chunk to show the model. The full chunk is ~1000 chars, and
# truncating below that loses the tail of a passage mid-sentence.
SNIPPET_CHARS = 1000


def format_results(results: list) -> str:
    """
    Render retrieved chunks for the model.

    The header is parsed back out in agents/graph.py to build citations, so the
    field order here and the parser there are one contract. Keeping page numbers
    in the header is what lets a user jump to the page an answer came from.
    """
    blocks = []
    for i, sr in enumerate(results, 1):
        label = "IMAGE" if sr.chunk.is_image else "TEXT"
        page = sr.chunk.metadata.get("page_number", 0) or 0
        blocks.append(
            f"[Source {i} | {label} | {sr.chunk.filename} | page={page} | score={sr.score:.3f}]\n"
            f"{sr.chunk.content[:SNIPPET_CHARS]}"
        )
    return "\n\n---\n\n".join(blocks)


def build_tools(vector_store, on_retrieval: Callable | None = None) -> list:
    """
    Build the agent's tools with the vector store injected.

    on_retrieval is an optional callback invoked with (query, results) after each
    search. The graph uses it to record retrieval spans, so the tools themselves
    never need to know that a tracer exists.
    """

    def _search(query: str, k: int) -> list:
        results = vector_store.similarity_search(query, k=k)
        if on_retrieval:
            on_retrieval(query, results)
        return results

    @tool
    def search_documents(query: str) -> str:
        """Search the indexed documents for passages relevant to a query.

        Use this for any question about document content. Pass a focused query
        describing the information you need, not the user's full sentence."""
        try:
            results = _search(query, DEFAULT_K)
            if not results:
                return "NO_RESULTS: no indexed document matched this query."
            return format_results(results)
        except Exception as e:  # noqa: BLE001 - surfaced to the model as text
            logger.error("search_documents failed: %s", e)
            return f"ERROR: document search failed ({e})."

    @tool
    def get_document_stats() -> str:
        """Report how many chunks are currently indexed.

        Use this when the user asks what documents are loaded or whether the
        knowledge base is empty."""
        try:
            count = vector_store.get_document_count()
            state = "active" if count > 0 else "empty"
            return f"Vector store: {count} chunks indexed, status {state}."
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    @tool
    def summarize_topic(topic: str) -> str:
        """Gather broader context about a topic across the document collection.

        Use this instead of search_documents when the user asks for a summary or
        an overview, since it retrieves more passages."""
        try:
            results = _search(topic, 6)
            if not results:
                return f"NO_RESULTS: nothing indexed about '{topic}'."
            return f"{len(results)} passages about '{topic}':\n\n" + format_results(results)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    return [search_documents, get_document_stats, summarize_topic]
