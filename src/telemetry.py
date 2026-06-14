"""
Observability layer: per-node timing, token usage, and cost tracking.

Usage pattern (in each agent node):
    response = client.chat(...)
    telemetry.record_tokens(response.usage_input, response.usage_output)

Usage pattern (in graph.py, wrapping each node):
    builder.add_node("planner", timed("planner", planner_node))

The span for the current node is stored in thread-local storage so that
concurrent runs (each in their own ThreadPoolExecutor thread) stay isolated.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# Per-token costs in USD for known models.
# (input_per_token, output_per_token)
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-sonnet-4-6":       (3.00e-6, 15.00e-6),
    "claude-opus-4-8":         (5.00e-6, 25.00e-6),
    "claude-haiku-4-5":        (1.00e-6,  5.00e-6),
    # OpenAI
    "gpt-4o":                  (2.50e-6, 10.00e-6),
    "gpt-4o-mini":             (0.15e-6,  0.60e-6),
    "gpt-4.1":                 (2.00e-6,  8.00e-6),
    "gpt-4.1-mini":            (0.40e-6,  1.60e-6),
    "gpt-4.1-nano":            (0.10e-6,  0.40e-6),
    # Google Gemini
    "gemini-2.0-flash":        (0.10e-6,  0.40e-6),
    "gemini-2.5-flash":        (0.15e-6,  0.60e-6),
    "gemini-1.5-pro":          (1.25e-6,  5.00e-6),
    "gemini-1.5-flash":        (0.075e-6, 0.30e-6),
    # xAI Grok
    "grok-3":                  (3.00e-6, 15.00e-6),
    "grok-3-mini":             (0.30e-6,  0.50e-6),
    "grok-3-fast":             (5.00e-6, 25.00e-6),
}


@dataclass
class NodeSpan:
    run_id: str
    node: str
    started_at: datetime
    model: str = ""
    ended_at: datetime | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds() * 1000

    @property
    def cost_usd(self) -> float:
        in_cost, out_cost = _MODEL_COSTS.get(self.model, (0.0, 0.0))
        return self.input_tokens * in_cost + self.output_tokens * out_cost


@dataclass
class RunTrace:
    run_id: str
    feature_request: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    final_status: str = "running"
    pr_url: str | None = None
    spans: list[NodeSpan] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.spans)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.spans)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "feature_request": self.feature_request,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "final_status": self.final_status,
            "pr_url": self.pr_url,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "spans": [
                {
                    "node": s.node,
                    "model": s.model,
                    "started_at": s.started_at.isoformat(),
                    "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                    "duration_ms": round(s.duration_ms, 1) if s.duration_ms is not None else None,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "cost_usd": round(s.cost_usd, 6),
                }
                for s in self.spans
            ],
        }


# ---------------------------------------------------------------------------
# Global trace store
# ---------------------------------------------------------------------------
_traces: dict[str, RunTrace] = {}
_lock = threading.Lock()
_local = threading.local()


def start_run(run_id: str, feature_request: str) -> RunTrace:
    trace = RunTrace(run_id=run_id, feature_request=feature_request)
    with _lock:
        _traces[run_id] = trace
    return trace


def finish_run(run_id: str, final_status: str, pr_url: str | None = None) -> None:
    with _lock:
        if run_id in _traces:
            _traces[run_id].ended_at = datetime.now(UTC)
            _traces[run_id].final_status = final_status
            _traces[run_id].pr_url = pr_url


def get_trace(run_id: str) -> RunTrace | None:
    return _traces.get(run_id)


def get_all_traces() -> list[RunTrace]:
    with _lock:
        return list(_traces.values())


# ---------------------------------------------------------------------------
# Per-node instrumentation
# ---------------------------------------------------------------------------

def _open_span(run_id: str, node: str) -> None:
    # Lazy import avoids circular dependency (agents import telemetry + _client)
    from src.agents._client import MODEL
    _local.span = NodeSpan(run_id=run_id, node=node, started_at=datetime.now(UTC), model=MODEL)


def _close_span() -> None:
    span: NodeSpan | None = getattr(_local, "span", None)
    if span is None:
        return
    span.ended_at = datetime.now(UTC)
    with _lock:
        if span.run_id in _traces:
            _traces[span.run_id].spans.append(span)
    _local.span = None


def record_tokens(input_tokens: int, output_tokens: int) -> None:
    """Call after every LLM call with usage counts from ChatResponse."""
    span: NodeSpan | None = getattr(_local, "span", None)
    if span is None:
        return
    span.input_tokens += input_tokens
    span.output_tokens += output_tokens


def timed(node_name: str, fn):
    """Wrap a LangGraph node function to open/close a telemetry span."""
    def wrapper(state):
        run_id = state.get("run_id", "")
        _open_span(run_id, node_name)
        try:
            return fn(state)
        finally:
            _close_span()
    wrapper.__name__ = fn.__name__
    return wrapper


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def aggregate_stats() -> dict:
    with _lock:
        traces = list(_traces.values())

    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    model = os.environ.get("LLM_MODEL", "")

    if not traces:
        return {"total_runs": 0, "provider": provider, "model": model}

    finished = [t for t in traces if t.ended_at is not None]
    successful = [t for t in finished if t.final_status == "pr_created"]

    avg_duration = (
        sum(t.duration_seconds for t in finished if t.duration_seconds) / len(finished)
        if finished else 0.0
    )

    return {
        "total_runs": len(traces),
        "finished_runs": len(finished),
        "pr_created": len(successful),
        "success_rate": round(len(successful) / len(finished), 3) if finished else 0.0,
        "avg_cost_usd": round(sum(t.total_cost_usd for t in finished) / len(finished), 4) if finished else 0.0,
        "avg_duration_seconds": round(avg_duration, 1),
        "total_cost_usd": round(sum(t.total_cost_usd for t in traces), 4),
        "total_tokens": {
            "input": sum(t.total_input_tokens for t in traces),
            "output": sum(t.total_output_tokens for t in traces),
        },
        "provider": provider,
        "model": model,
    }
