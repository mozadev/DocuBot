"""
Domain service: question answering over the indexed document collection.

This is where the cross-cutting concerns are composed into one request path:

    guardrail(in) -> cache lookup -> agent graph -> guardrail(out) -> cache store

Each step opens a span on the tracer, so every answer is explainable after the
fact: which tools ran, how long retrieval took, what the model cost, and whether
a guardrail fired. Tracing lives here rather than in the API layer because the
Streamlit UI calls the same service and should produce the same traces.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from adapters.observability.tracer import SpanType
from core.logger import log_function_call, logger
from domain.guardrails import RagGuardrails
from domain.models import ChatMessage, ChatResponse, Source

# How many prior turns to replay into the prompt. Full history would grow the
# prompt without bound; 6 turns covers the follow-up questions users actually
# ask ("what about the second one?") while keeping token cost flat.
HISTORY_WINDOW = 6


class ChatService:
    """Answers questions about indexed documents, scoped per session."""

    def __init__(
        self,
        vector_store,
        llm_adapter,
        graph_builder,
        guardrails: RagGuardrails | None = None,
        tracer=None,
        cache=None,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm_adapter
        self._guardrails = guardrails or RagGuardrails()
        self._tracer = tracer
        self._cache = cache
        self._histories: dict[str, list] = {}

        # Retrieval spans are recorded through this callback, which the graph
        # passes down to the search tools. _current_trace is set per request.
        self._current_trace: str | None = None
        self._graph = graph_builder(
            vector_store=vector_store,
            llm_adapter=llm_adapter,
            on_retrieval=self._record_retrieval,
        )
        logger.info("ChatService ready")

    @property
    def graph(self):
        """The compiled agent graph, exposed for the streaming endpoint."""
        return self._graph

    # ---- tracing helpers ----

    def _record_retrieval(self, query: str, results: list) -> None:
        if not self._tracer or not self._current_trace:
            return
        span = self._tracer.start_span(
            self._current_trace,
            name="vector_search",
            span_type=SpanType.RETRIEVAL,
            input_data={"query": query[:200]},
        )
        if span:
            top = max((r.score for r in results), default=0.0)
            self._tracer.finish_span(
                span,
                output={
                    "results": len(results),
                    "top_score": round(top, 4),
                    "files": sorted({r.chunk.filename for r in results}),
                },
            )

    def _span(self, name: str, span_type: SpanType, **input_data):
        if not self._tracer or not self._current_trace:
            return None
        return self._tracer.start_span(
            self._current_trace, name=name, span_type=span_type, input_data=input_data
        )

    def _finish(self, span, output: Any = None, error: str | None = None) -> None:
        if span and self._tracer:
            self._tracer.finish_span(span, output=output, error=error)

    # ---- public API ----

    @log_function_call
    def ask_question(self, question: str, session_id: str = "default") -> ChatResponse:
        """Answer one question within a session's conversation history."""
        trace_id = ""
        if self._tracer:
            trace_id = self._tracer.start_trace(
                tenant_id=session_id, session_type="chat", metadata={"question": question[:200]}
            )
        self._current_trace = trace_id

        try:
            return self._ask(question, session_id, trace_id)
        finally:
            if self._tracer and trace_id:
                self._tracer.finish_trace(trace_id)
            self._current_trace = None

    def _ask(self, question: str, session_id: str, trace_id: str) -> ChatResponse:
        # 1. Input guardrail - cheapest possible rejection, before any token spend.
        span = self._span("guardrail_input", SpanType.GUARDRAIL, question=question[:200])
        check = self._guardrails.check_input(question)
        self._finish(span, output=check.to_dict())

        if not check.passed:
            return ChatResponse(
                answer="I can't process that request: " + " ".join(check.violations),
                question=question,
                trace_id=trace_id,
                guardrail={"stage": "input", **check.to_dict()},
            )

        question = check.content

        # 2. Cache lookup, scoped per session so one user's answers never leak
        #    into another's.
        if self._cache:
            span = self._span("cache_lookup", SpanType.CACHE_LOOKUP, question=question[:200])
            hit = self._cache.get(question, tenant_id=session_id)
            self._finish(span, output={"hit": hit is not None})
            if hit is not None:
                cached = ChatResponse(**{**hit, "sources": [Source(**s) for s in hit["sources"]]})
                cached.cached = True
                cached.trace_id = trace_id
                return cached

        # 3. Run the agent.
        history = self._histories.setdefault(session_id, [])
        messages = history[-HISTORY_WINDOW * 2:] + [HumanMessage(content=question)]

        span = self._span("agent_graph", SpanType.AGENT_STEP, turns=len(messages))
        try:
            result = self._graph.invoke(
                {"messages": messages, "sources": [], "confidence": 0.0}
            )
        except Exception as e:  # noqa: BLE001 - one bad turn must not kill the session
            logger.exception("Agent graph failed")
            self._finish(span, error=str(e))
            return ChatResponse(
                answer="Something went wrong while answering. Please try again.",
                question=question,
                trace_id=trace_id,
            )

        answer_text = self._last_answer(result.get("messages", []))
        sources = [Source(**s) for s in result.get("sources", [])]
        confidence = result.get("confidence", 0.0)
        self._finish(
            span, output={"answer_chars": len(answer_text), "sources": len(sources)}
        )

        # 4. Output guardrail - the grounding check that makes this a document
        #    assistant rather than a chatbot with documents attached.
        span = self._span("guardrail_output", SpanType.GUARDRAIL, sources=len(sources))
        out = self._guardrails.check_output(answer_text, sources)
        self._finish(span, output=out.to_dict())

        if not out.passed:
            logger.warning("Output guardrail blocked an answer: %s", out.violations)
            return ChatResponse(
                answer=(
                    "I couldn't find anything in your documents that answers this. "
                    "Try rephrasing, or upload a document that covers it."
                ),
                question=question,
                confidence=0.0,
                trace_id=trace_id,
                guardrail={"stage": "output", **out.to_dict()},
            )

        answer_text = out.content

        # 5. Commit the turn to history and cache.
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer_text))

        response = ChatResponse(
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            question=question,
            trace_id=trace_id,
            guardrail={"stage": "output", **out.to_dict()},
        )

        if self._cache:
            self._cache.put(question, response.to_dict(), tenant_id=session_id)

        return response

    @staticmethod
    def _last_answer(messages: list) -> str:
        """The final AI message that is prose rather than a tool call."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                return msg.content if isinstance(msg.content, str) else str(msg.content)
        return ""

    @log_function_call
    def get_chat_history(self, session_id: str = "default") -> list[ChatMessage]:
        return [
            ChatMessage(role="human" if isinstance(m, HumanMessage) else "ai", content=m.content)
            for m in self._histories.get(session_id, [])
        ]

    @log_function_call
    def clear_memory(self, session_id: str = "default") -> None:
        self._histories.pop(session_id, None)
        logger.info("Cleared history for session=%s", session_id)

    @log_function_call
    def get_conversation_summary(self, session_id: str = "default") -> str:
        history = self._histories.get(session_id, [])
        if not history:
            return "No conversation history yet."
        transcript = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in history
        )
        return self._llm.invoke(
            [
                {
                    "role": "system",
                    "content": "Summarize this conversation in three sentences or fewer.",
                },
                {"role": "user", "content": transcript},
            ]
        )
