"""
Observability: tracing completo de cada decision del agente.
Compatible con LangSmith, pero funciona standalone.
Registra: cada LLM call, tool call, latencia, tokens, costos estimados.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from enum import Enum

from core.logger import logger


class SpanType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    AGENT_STEP = "agent_step"
    GUARDRAIL = "guardrail"
    CACHE_LOOKUP = "cache_lookup"
    WEBHOOK = "webhook"
    IMAGE_GEN = "image_generation"


@dataclass
class Span:
    """Un span individual dentro de un trace."""
    id: str
    trace_id: str
    name: str
    span_type: SpanType
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    parent_span_id: Optional[str] = None

    def finish(self, output: Any = None, error: str = None) -> None:
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        if output:
            self.output_data = output if isinstance(output, dict) else {"result": str(output)[:500]}
        if error:
            self.error = error

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["span_type"] = self.span_type.value
        return d


@dataclass
class Trace:
    """Un trace completo de un request."""
    id: str
    tenant_id: str
    session_type: str  # "chat", "campaign", "content"
    started_at: str
    spans: List[Span] = field(default_factory=list)
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "session_type": self.session_type,
            "started_at": self.started_at,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "span_count": len(self.spans),
            "metadata": self.metadata,
        }

    def to_detailed_dict(self) -> Dict[str, Any]:
        d = self.to_dict()
        d["spans"] = [s.to_dict() for s in self.spans]
        return d


# Costos aproximados por modelo (USD por 1K tokens)
MODEL_COSTS = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}


class AgentTracer:
    """Tracing centralizado para todos los agentes."""

    def __init__(self, max_traces: int = 500) -> None:
        self._traces: Dict[str, Trace] = {}
        self._max_traces = max_traces

    def start_trace(
        self,
        tenant_id: str,
        session_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Inicia un nuevo trace. Retorna trace_id."""
        if len(self._traces) >= self._max_traces:
            oldest_key = next(iter(self._traces))
            del self._traces[oldest_key]

        trace_id = str(uuid.uuid4())
        self._traces[trace_id] = Trace(
            id=trace_id,
            tenant_id=tenant_id,
            session_type=session_type,
            started_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        return trace_id

    def start_span(
        self,
        trace_id: str,
        name: str,
        span_type: SpanType,
        input_data: Optional[Dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
    ) -> Optional[Span]:
        """Crea y registra un span dentro de un trace."""
        trace = self._traces.get(trace_id)
        if not trace:
            return None

        span = Span(
            id=str(uuid.uuid4()),
            trace_id=trace_id,
            name=name,
            span_type=span_type,
            start_time=time.time(),
            input_data=input_data or {},
            parent_span_id=parent_span_id,
        )
        trace.spans.append(span)
        return span

    def finish_span(
        self,
        span: Span,
        output: Any = None,
        error: str = None,
        tokens: int = 0,
        model: str = "gpt-4o-mini",
    ) -> None:
        """Finaliza un span con su resultado."""
        span.finish(output=output, error=error)
        span.tokens_used = tokens
        if tokens > 0:
            costs = MODEL_COSTS.get(model, MODEL_COSTS["gpt-4o-mini"])
            span.estimated_cost_usd = (tokens / 1000) * (costs["input"] + costs["output"]) / 2

        trace = self._traces.get(span.trace_id)
        if trace:
            trace.total_tokens += tokens
            trace.total_cost_usd += span.estimated_cost_usd

    def finish_trace(self, trace_id: str) -> Optional[Trace]:
        """Finaliza un trace y calcula totales."""
        trace = self._traces.get(trace_id)
        if not trace or not trace.spans:
            return trace

        first = min(s.start_time for s in trace.spans)
        last = max(s.end_time for s in trace.spans if s.end_time > 0)
        trace.total_duration_ms = (last - first) * 1000 if last > first else 0

        logger.info(
            f"Trace {trace_id} finished: "
            f"{len(trace.spans)} spans, "
            f"{trace.total_duration_ms:.0f}ms, "
            f"{trace.total_tokens} tokens, "
            f"${trace.total_cost_usd:.4f}"
        )
        return trace

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        trace = self._traces.get(trace_id)
        return trace.to_detailed_dict() if trace else None

    def get_tenant_traces(self, tenant_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        tenant_traces = [t for t in self._traces.values() if t.tenant_id == tenant_id]
        tenant_traces.sort(key=lambda t: t.started_at, reverse=True)
        return [t.to_dict() for t in tenant_traces[:limit]]

    def get_analytics(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Analitica de uso — cuanto cuesta, que tan rapido, donde se gasta mas."""
        traces = list(self._traces.values())
        if tenant_id:
            traces = [t for t in traces if t.tenant_id == tenant_id]

        if not traces:
            return {"message": "No traces found"}

        total_cost = sum(t.total_cost_usd for t in traces)
        total_tokens = sum(t.total_tokens for t in traces)
        avg_duration = sum(t.total_duration_ms for t in traces) / len(traces)

        by_type: Dict[str, int] = {}
        for t in traces:
            by_type[t.session_type] = by_type.get(t.session_type, 0) + 1

        all_spans = [s for t in traces for s in t.spans]
        tool_usage: Dict[str, int] = {}
        for s in all_spans:
            if s.span_type == SpanType.TOOL_CALL:
                tool_usage[s.name] = tool_usage.get(s.name, 0) + 1

        error_count = sum(1 for s in all_spans if s.error)

        return {
            "total_traces": len(traces),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "avg_duration_ms": round(avg_duration, 0),
            "by_session_type": by_type,
            "top_tools": dict(sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:10]),
            "error_count": error_count,
            "error_rate_pct": round(error_count / max(len(all_spans), 1) * 100, 1),
        }
