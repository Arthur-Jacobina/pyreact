"""Boundary metrics through the OpenTelemetry metrics API.

Instruments (``meter = "pyreact_otel"``):

======================================  =========  ==========================================
name                                    unit       what
======================================  =========  ==========================================
``gen_ai.client.token.usage``           ``{token}`` histogram per LLM call, ``gen_ai.token.type`` =
                                                    ``input`` / ``output`` (GenAI semconv)
``gen_ai.client.operation.duration``    ``s``       histogram, ``llm`` (chat) and ``tool``
                                                    (execute_tool) calls (GenAI semconv)
``pyreact.llm.cost``                    ``USD``     counter, summed ``attrs["cost"]``
``pyreact.turn.duration``               ``s``       histogram, one per ``turn``
``pyreact.boundary.duration``           ``s``       histogram, every boundary (any kind)
``pyreact.boundary.errors``             ``{error}`` counter, failed / blocked boundaries
======================================  =========  ==========================================

GenAI semconv defines token usage as a *histogram* (its sum is the total
counter); backends show both distribution and totals from it.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from opentelemetry import metrics as otel_metrics

from pyreact.core.harness import Blocked, Boundary

from . import attributes as A
from .tracing import _usage, provider_of

__all__ = ["BoundaryMetrics"]

# GenAI semconv advisory buckets
_TOKEN_BUCKETS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576,
                  4194304, 16777216, 67108864]
_DURATION_BUCKETS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12,
                     10.24, 20.48, 40.96, 81.92]


def _is_tool(kind: str) -> bool:
    return kind == "tool" or kind.startswith("tool.")


class BoundaryMetrics:
    """Records metrics for one finished boundary (``record(b, seconds, error)``)."""

    def __init__(self, meter_provider: Any = None) -> None:
        from . import __version__

        meter = otel_metrics.get_meter("pyreact_otel", __version__, meter_provider=meter_provider)
        self.token_usage = meter.create_histogram(
            "gen_ai.client.token.usage", unit="{token}",
            description="Number of input and output tokens used.",
            explicit_bucket_boundaries_advisory=_TOKEN_BUCKETS)
        self.operation_duration = meter.create_histogram(
            "gen_ai.client.operation.duration", unit="s",
            description="GenAI operation duration.",
            explicit_bucket_boundaries_advisory=_DURATION_BUCKETS)
        self.cost = meter.create_counter(
            "pyreact.llm.cost", unit="USD", description="LLM cost reported by the call owner.")
        self.turn_duration = meter.create_histogram(
            "pyreact.turn.duration", unit="s", description="Duration of a pyreact turn.",
            explicit_bucket_boundaries_advisory=_DURATION_BUCKETS)
        self.boundary_duration = meter.create_histogram(
            "pyreact.boundary.duration", unit="s", description="Duration of any boundary call.",
            explicit_bucket_boundaries_advisory=_DURATION_BUCKETS)
        self.errors = meter.create_counter(
            "pyreact.boundary.errors", unit="{error}",
            description="Boundary calls that raised or were blocked.")

    def record(self, b: Boundary, seconds: float, error: Optional[BaseException] = None) -> None:
        cancelled = isinstance(error, asyncio.CancelledError)
        err_type = None
        if error is not None and not cancelled:
            err_type = "Blocked" if isinstance(error, Blocked) else type(error).__qualname__

        base: Dict[str, Any] = {A.BOUNDARY_KIND: b.kind}
        if err_type:
            base[A.ERROR_TYPE] = err_type
        self.boundary_duration.record(seconds, {**base, A.BOUNDARY_NAME: b.name})

        if b.kind == "llm":
            model = b.attrs.get("model")
            genai: Dict[str, Any] = {
                A.GEN_AI_OPERATION_NAME: A.OP_CHAT,
                A.GEN_AI_PROVIDER_NAME: provider_of(model, b.attrs.get("provider")),
            }
            if model:
                genai[A.GEN_AI_REQUEST_MODEL] = str(model)
            self.operation_duration.record(seconds, {**genai, **({A.ERROR_TYPE: err_type} if err_type else {})})
            usage = _usage(b.attrs)
            for key, token_type in (("input", "input"), ("output", "output")):
                if key in usage:
                    self.token_usage.record(usage[key], {**genai, A.GEN_AI_TOKEN_TYPE: token_type})
            cost = b.attrs.get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost > 0:
                extra = {A.MODULE_KEY: str(b.attrs["module_key"])} if b.attrs.get("module_key") else {}
                self.cost.add(float(cost), {**genai, **extra})
        elif _is_tool(b.kind):
            attrs = {A.GEN_AI_OPERATION_NAME: A.OP_EXECUTE_TOOL, A.GEN_AI_TOOL_NAME: b.name}
            if err_type:
                attrs[A.ERROR_TYPE] = err_type
            self.operation_duration.record(seconds, attrs)
        elif b.kind == "turn":
            attrs = {A.COMPONENT: b.component} if b.component else {}
            if b.attrs.get("status"):
                attrs[A.TURN_STATUS] = str(b.attrs["status"])
            if err_type:
                attrs[A.ERROR_TYPE] = err_type
            self.turn_duration.record(seconds, attrs)

        if err_type:
            self.errors.add(1, {**base, A.BOUNDARY_NAME: b.name})
