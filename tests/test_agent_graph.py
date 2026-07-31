"""
Agent graph: citation extraction.

Citations are rebuilt from the tool output rather than from what the model says
it used, which is what stops a hallucinated filename reaching the user. The
formatter and the parser are one contract, so they are tested together.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.chat_tools import format_results
from agents.graph import _parse_source_block, build_agent_graph
from domain.models import DocumentChunk, SearchResult


def result(text: str, filename="handbook.pdf", page=1, score=0.8, kind="text") -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            content=text,
            metadata={"filename": filename, "page_number": page, "content_type": kind},
        ),
        score=score,
    )


class TestFormatterParserRoundTrip:
    def test_a_formatted_block_parses_back_to_its_fields(self):
        formatted = format_results([result("Employees get 20 days.", page=7, score=0.63)])

        parsed = _parse_source_block(formatted)

        assert parsed["filename"] == "handbook.pdf"
        assert parsed["page_number"] == 7
        assert parsed["score"] == 0.63
        assert parsed["content_type"] == "text"
        assert "20 days" in parsed["content"]

    def test_image_chunks_round_trip_as_figures(self):
        formatted = format_results([result("[FIGURE] A chart.", kind="image")])
        assert _parse_source_block(formatted)["content_type"] == "image"

    def test_multiple_results_split_on_the_separator(self):
        formatted = format_results(
            [result("First passage.", page=1), result("Second passage.", page=2)]
        )
        blocks = formatted.split("\n\n---\n\n")

        assert len(blocks) == 2
        assert [_parse_source_block(b)["page_number"] for b in blocks] == [1, 2]

    def test_non_source_text_is_ignored(self):
        assert _parse_source_block("NO_RESULTS: nothing found.") is None
        assert _parse_source_block("") is None

    def test_a_malformed_header_does_not_raise(self):
        assert _parse_source_block("[Source 1 | TEXT | f.pdf | score=oops]\nbody") is not None


class FakeChatModel:
    """Minimal LangChain-model stand-in: emits a scripted sequence of messages."""

    def __init__(self, replies):
        self._replies = list(replies)

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return self._replies.pop(0)


class FakeLLMAdapter:
    def __init__(self, model):
        self._model = model

    def get_langchain_llm(self):
        return self._model


class StubVectorStore:
    def __init__(self, results):
        self._results = results
        self.queries = []

    def similarity_search(self, query, k=4):
        self.queries.append(query)
        return self._results

    def get_document_count(self):
        return len(self._results)


class TestGraphExecution:
    def test_tool_calling_turn_produces_citations(self):
        store = StubVectorStore([result("Employees get 20 days.", page=3, score=0.71)])
        tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "search_documents", "args": {"query": "vacation"}, "id": "1"}],
        )
        final = AIMessage(content="You get 20 days.")
        graph = build_agent_graph(store, FakeLLMAdapter(FakeChatModel([tool_call, final])))

        state = graph.invoke(
            {"messages": [HumanMessage(content="How many vacation days?")],
             "sources": [], "confidence": 0.0}
        )

        assert state["sources"][0]["filename"] == "handbook.pdf"
        assert state["sources"][0]["page_number"] == 3
        assert state["confidence"] == 0.71
        assert store.queries == ["vacation"]

    def test_turn_without_retrieval_has_no_sources(self):
        store = StubVectorStore([])
        graph = build_agent_graph(
            store, FakeLLMAdapter(FakeChatModel([AIMessage(content="Hello.")]))
        )

        state = graph.invoke(
            {"messages": [HumanMessage(content="Hi")], "sources": [], "confidence": 0.0}
        )

        assert state["sources"] == []
        assert state["confidence"] == 0.0

    def test_empty_retrieval_yields_no_citations(self):
        store = StubVectorStore([])
        tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "search_documents", "args": {"query": "x"}, "id": "1"}],
        )
        graph = build_agent_graph(
            store,
            FakeLLMAdapter(FakeChatModel([tool_call, AIMessage(content="Not in the docs.")])),
        )

        state = graph.invoke(
            {"messages": [HumanMessage(content="?")], "sources": [], "confidence": 0.0}
        )

        assert state["sources"] == []


class TestDeduplication:
    def test_the_same_passage_retrieved_twice_is_cited_once(self):
        from agents.graph import build_agent_graph as _build  # noqa: F401

        duplicate = format_results([result("Same passage.", page=1)])
        messages = [
            HumanMessage(content="q"),
            ToolMessage(content=duplicate, name="search_documents", tool_call_id="1"),
            ToolMessage(content=duplicate, name="search_documents", tool_call_id="2"),
            AIMessage(content="answer"),
        ]

        store = StubVectorStore([])
        graph = _build(store, FakeLLMAdapter(FakeChatModel([AIMessage(content="answer")])))
        state = graph.invoke({"messages": messages, "sources": [], "confidence": 0.0})

        assert len(state["sources"]) == 1

    def test_on_retrieval_callback_receives_each_search(self):
        seen = []
        store = StubVectorStore([result("Passage.")])
        tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "search_documents", "args": {"query": "vacation"}, "id": "1"}],
        )
        graph = build_agent_graph(
            store,
            FakeLLMAdapter(FakeChatModel([tool_call, AIMessage(content="done")])),
            on_retrieval=lambda q, r: seen.append((q, len(r))),
        )

        graph.invoke({"messages": [HumanMessage(content="q")], "sources": [], "confidence": 0.0})

        assert seen == [("vacation", 1)]
