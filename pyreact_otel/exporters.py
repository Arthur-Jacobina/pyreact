"""Exporter registry: one name -> how to ship spans/metrics to a backend.

Built in: ``langfuse``, ``otlp``, ``console``, ``memory``. Any OTLP vendor
(Phoenix, Jaeger, Honeycomb, Grafana, Datadog agent, ...) works through
``otlp`` with its endpoint/headers; register a name for convenience or for
vendor-specific attributes::

    from pyreact_otel import ExporterConfig, register_exporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    def honeycomb(api_key=None, dataset=None):
        return ExporterConfig(span_exporter=OTLPSpanExporter(
            endpoint="https://api.honeycomb.io/v1/traces",
            headers={"x-honeycomb-team": api_key or os.environ["HONEYCOMB_API_KEY"]}))

    register_exporter("honeycomb", honeycomb)
    pyreact_otel.setup(exporter="honeycomb")

A factory takes the extra keyword arguments given to ``setup`` and returns an
:class:`ExporterConfig`.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .tracing import Enricher, SpanInfo

__all__ = [
    "ExporterConfig",
    "ExporterFactory",
    "available_exporters",
    "get_exporter",
    "langfuse_enricher",
    "langfuse_event_attributes",
    "register_exporter",
    "unregister_exporter",
]


@dataclass
class ExporterConfig:
    """What a backend needs. Every field is optional except, usually, ``span_exporter``."""

    span_exporter: Any = None  # opentelemetry.sdk.trace.export.SpanExporter
    metric_reader: Any = None  # opentelemetry.sdk.metrics.export.MetricReader (already wrapping an exporter)
    enrichers: Sequence[Enricher] = ()  # vendor attributes on boundary spans
    event_attributes: Optional[Callable[..., Optional[Mapping[str, Any]]]] = None  # same, for bridge events
    bridge_mode: str = "event"  # "span" for backends that do not display span events
    batch: bool = True  # BatchSpanProcessor (False: SimpleSpanProcessor, synchronous)
    resource_attributes: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)  # exposed on the setup handle


ExporterFactory = Callable[..., ExporterConfig]

_registry: Dict[str, ExporterFactory] = {}
_lock = threading.Lock()


def register_exporter(name: str, factory: ExporterFactory, *, replace: bool = False) -> None:
    """Register ``factory`` under ``name`` for ``setup(exporter=name)``."""
    with _lock:
        if name in _registry and not replace:
            raise ValueError(f"exporter {name!r} is already registered (pass replace=True)")
        _registry[name] = factory


def unregister_exporter(name: str) -> None:
    with _lock:
        _registry.pop(name, None)


def get_exporter(name: str) -> ExporterFactory:
    with _lock:
        try:
            return _registry[name]
        except KeyError:
            raise ValueError(
                f"unknown exporter {name!r}; registered: {sorted(_registry)}. "
                "Use register_exporter(name, factory) to add one."
            ) from None


def available_exporters() -> List[str]:
    with _lock:
        return sorted(_registry)


# ---------------------------------------------------------------------------
# Built-ins
# ---------------------------------------------------------------------------
def _signal_url(base: str, signal: str) -> str:
    return base.rstrip("/") + f"/v1/{signal}"


def _parse_headers(headers: Any) -> Optional[Dict[str, str]]:
    if headers is None or isinstance(headers, Mapping):
        return dict(headers) if headers is not None else None
    out = {}
    for part in str(headers).split(","):  # OTEL_EXPORTER_OTLP_HEADERS syntax
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def otlp_exporter(
    endpoint: Optional[str] = None,
    headers: Any = None,
    metrics: bool = True,
    metric_export_interval: float = 60.0,
    timeout: Optional[float] = None,
) -> ExporterConfig:
    """OTLP/HTTP (protobuf). Without arguments, the standard environment
    variables apply (``OTEL_EXPORTER_OTLP_ENDPOINT``, ``..._TRACES_ENDPOINT``,
    ``..._HEADERS``, ``..._TIMEOUT``, ...). ``endpoint`` is the *base* URL
    (``/v1/traces`` and ``/v1/metrics`` are appended)."""
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    hdrs = _parse_headers(headers)
    kw: Dict[str, Any] = {}
    if timeout is not None:
        kw["timeout"] = timeout
    spans = OTLPSpanExporter(
        endpoint=_signal_url(endpoint, "traces") if endpoint else None, headers=hdrs, **kw)
    reader = None
    if metrics:
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=_signal_url(endpoint, "metrics") if endpoint else None,
                               headers=hdrs, **kw),
            export_interval_millis=metric_export_interval * 1000)
    return ExporterConfig(span_exporter=spans, metric_reader=reader)


def console_exporter(metrics: bool = False, metric_export_interval: float = 60.0) -> ExporterConfig:
    """Print spans (and optionally metrics) to stdout. For debugging."""
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter

    reader = None
    if metrics:
        reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(), export_interval_millis=metric_export_interval * 1000)
    return ExporterConfig(span_exporter=ConsoleSpanExporter(), metric_reader=reader, batch=False)


def memory_exporter() -> ExporterConfig:
    """In-memory spans and metrics (tests): ``handle.span_exporter.get_finished_spans()``,
    ``handle.metric_reader.get_metrics_data()``."""
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    spans, reader = InMemorySpanExporter(), InMemoryMetricReader()
    return ExporterConfig(span_exporter=spans, metric_reader=reader, batch=False,
                          extras={"span_exporter": spans, "metric_reader": reader})


# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------
# https://langfuse.com/integrations/native/opentelemetry -- OTLP/HTTP only (no
# gRPC), Basic auth with the project keys. Langfuse v4 queries observations
# directly, so trace-level attributes (trace name, session, user, ...) are
# copied onto *every* span, not just the root.
LANGFUSE_DEFAULT_HOST = "https://cloud.langfuse.com"

_LF_TYPES = {"turn": "agent", "llm": "generation", "tool": "tool"}


def _lf_type(kind: str) -> str:
    if kind.startswith("tool."):
        return "tool"
    if kind.startswith("guard"):
        return "guardrail"
    return _LF_TYPES.get(kind, "span")


def _lf_messages(lm_calls: Any) -> Any:
    """lm_calls -> ([{role, content}] input, completion output) for the Langfuse UI."""
    if not isinstance(lm_calls, (list, tuple)) or not lm_calls:
        return None, None
    msgs, completions = [], []
    for call in lm_calls:
        if not isinstance(call, Mapping):
            continue
        for m in call.get("messages") or []:
            if isinstance(m, Mapping):
                msgs.append({"role": m.get("role"), "content": m.get("content")})
        if call.get("completion") is not None:
            completions.append(call.get("completion"))
    out = completions[0] if len(completions) == 1 else (completions or None)
    return msgs or None, out


def langfuse_enricher(
    *,
    environment: Optional[str] = None,
    release: Optional[str] = None,
    tags: Sequence[str] = (),
    max_content_length: int = 8192,
) -> Enricher:
    """``langfuse.*`` attributes for boundary spans (types, trace name/session/user
    on every span, usage/cost details, level, and -- only when content capture
    is on -- observation input/output)."""

    def _cut(s: str) -> str:
        return s if len(s) <= max_content_length else s[:max_content_length] + "...[truncated]"

    def enrich(info: SpanInfo) -> Dict[str, Any]:
        b, a = info.boundary, info.attributes
        out: Dict[str, Any] = {"langfuse.observation.type": _lf_type(b.kind)}
        if info.trace_name:
            out["langfuse.trace.name"] = info.trace_name
        if a.get("pyreact.session_id"):
            out["langfuse.session.id"] = a["pyreact.session_id"]
        if a.get("user.id"):
            out["langfuse.user.id"] = a["user.id"]
        if environment:
            out["langfuse.environment"] = environment
        if release:
            out["langfuse.release"] = release
        if tags:
            out["langfuse.trace.tags"] = list(tags)
        if b.component:
            out["langfuse.trace.metadata.component"] = b.component
            out["langfuse.observation.metadata.component"] = b.component
        if a.get("pyreact.turn_id"):
            out["langfuse.trace.metadata.turn_id"] = a["pyreact.turn_id"]
        if b.kind == "llm":
            if a.get("gen_ai.request.model"):
                out["langfuse.observation.model.name"] = a["gen_ai.request.model"]
            usage = {k: a[src] for k, src in (("input", "gen_ai.usage.input_tokens"),
                                              ("output", "gen_ai.usage.output_tokens"),
                                              ("total", "pyreact.usage.total_tokens"))
                     if isinstance(a.get(src), int)}
            if usage:
                out["langfuse.observation.usage_details"] = json.dumps(usage)
            if isinstance(a.get("pyreact.cost"), float):
                out["langfuse.observation.cost_details"] = json.dumps({"total": a["pyreact.cost"]})
        if info.blocked:
            out["langfuse.observation.level"] = "WARNING"
            out["langfuse.observation.status_message"] = f"blocked: {info.error.reason}"
        elif "error.type" in a and not a.get("pyreact.cancelled"):
            out["langfuse.observation.level"] = "ERROR"
            out["langfuse.observation.status_message"] = (
                f"{type(info.error).__name__}: {info.error}" if info.error is not None
                else "turn ended with errors")
        if info.content:  # capture_content=True
            inp = info.content.get("pyreact.inputs")
            outp = info.content.get("pyreact.outputs")
            if b.kind == "llm":
                msgs, completion = _lf_messages(b.attrs.get("lm_calls"))
                if msgs:
                    inp = _cut(json.dumps(msgs, ensure_ascii=False, default=str))
                if completion is not None and outp is None:
                    outp = _cut(json.dumps(completion, ensure_ascii=False, default=str))
            elif b.kind == "tool" or b.kind.startswith("tool."):
                inp = info.content.get("gen_ai.tool.call.arguments", inp)
                outp = info.content.get("gen_ai.tool.call.result", outp)
            if inp is not None:
                out["langfuse.observation.input"] = inp
            if outp is not None:
                out["langfuse.observation.output"] = outp
        return out

    return enrich


def langfuse_event_attributes(*, environment: Optional[str] = None):
    """``langfuse.*`` attributes for bridge events (shown as ``event`` observations)."""

    def attrs(ev, span, root_name) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "langfuse.observation.type": "event",
            "langfuse.trace.name": root_name or f"pyreact.event {ev.kind}",
        }
        parent_attrs = getattr(span, "attributes", None) or {}
        session = parent_attrs.get("pyreact.session_id")
        if session:
            out["langfuse.session.id"] = session
        if parent_attrs.get("user.id"):
            out["langfuse.user.id"] = parent_attrs["user.id"]
        if environment:
            out["langfuse.environment"] = environment
        if ev.kind == "error" or ev.kind.endswith(".error"):
            out["langfuse.observation.level"] = "ERROR"
            err = (ev.payload or {}).get("error")
            if err:
                out["langfuse.observation.status_message"] = str(err)[:2000]
        return out

    return attrs


def langfuse_exporter(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None,
    environment: Optional[str] = None,
    release: Optional[str] = None,
    tags: Sequence[str] = (),
    ingestion_version: Optional[str] = "4",
    timeout: Optional[float] = 30.0,
) -> ExporterConfig:
    """Langfuse over OTLP/HTTP: ``{host}/api/public/otel/v1/traces`` with Basic
    auth ``public_key:secret_key``. Defaults: ``LANGFUSE_PUBLIC_KEY``,
    ``LANGFUSE_SECRET_KEY``, ``LANGFUSE_HOST`` (or ``LANGFUSE_BASE_URL``,
    else Langfuse Cloud EU). Langfuse ingests traces only: metrics are not
    exported unless you add a ``metric_readers=[...]`` to ``setup``.

    ``timeout`` (seconds per export) is 30 rather than the OTLP default 10:
    a cold self-hosted Langfuse can take longer than 10 s to answer its
    first OTLP request, and a timed-out batch is dropped."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = (host or os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
            or LANGFUSE_DEFAULT_HOST).rstrip("/")
    if not public_key or not secret_key:
        raise ValueError("Langfuse exporter needs public_key/secret_key "
                         "(or LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)")
    environment = environment or os.getenv("LANGFUSE_TRACING_ENVIRONMENT")
    release = release or os.getenv("LANGFUSE_RELEASE")
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    headers = {"Authorization": f"Basic {token}"}
    if ingestion_version:
        headers["x-langfuse-ingestion-version"] = str(ingestion_version)
    kw: Dict[str, Any] = {"timeout": timeout} if timeout is not None else {}
    endpoint = f"{host}/api/public/otel/v1/traces"
    return ExporterConfig(
        span_exporter=OTLPSpanExporter(endpoint=endpoint, headers=headers, **kw),
        enrichers=[langfuse_enricher(environment=environment, release=release, tags=tags)],
        event_attributes=langfuse_event_attributes(environment=environment),
        bridge_mode="span",
        extras={"endpoint": endpoint, "host": host},
    )


register_exporter("otlp", otlp_exporter)
register_exporter("console", console_exporter)
register_exporter("memory", memory_exporter)
register_exporter("langfuse", langfuse_exporter)
