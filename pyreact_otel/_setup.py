"""One call to export a pyreact app's telemetry.

    import pyreact_otel
    otel = pyreact_otel.setup(exporter="langfuse", service_name="store")
    otel.attach_app(app)          # bridge message/error/feedback events of its sessions
    ...
    otel.shutdown()               # flush + uninstall

``setup`` builds a TracerProvider (+ MeterProvider when the exporter has a
metric reader), installs :func:`~pyreact_otel.otel_middleware` process-wide
in the pyreact harness, and bridges the global EventBus. Providers stay
private to pyreact unless ``set_global=True``.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from opentelemetry import metrics as otel_metrics
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

from pyreact.core.events import EventBus
from pyreact.core.harness import install as install_middleware

from .bridge import DEFAULT_KINDS, EventBridge
from .exporters import ExporterConfig, get_exporter
from .metrics import BoundaryMetrics
from .tracing import otel_middleware

__all__ = ["OtelHandle", "setup", "active"]

_TRUTHY = {"1", "true", "yes", "on"}
_active: Optional["OtelHandle"] = None
_active_lock = threading.Lock()


def _env_capture() -> bool:
    for var in ("PYREACT_OTEL_CAPTURE_CONTENT", "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"):
        if os.getenv(var, "").strip().lower() in _TRUTHY:
            return True
    return False


class OtelHandle:
    """What :func:`setup` returns: providers, the middleware and the bridge."""

    def __init__(self, *, tracer_provider: TracerProvider, meter_provider: Optional[MeterProvider],
                 middleware: Any, bridge: Optional[EventBridge], config: ExporterConfig,
                 uninstall: Optional[Callable[[], None]]) -> None:
        self.tracer_provider = tracer_provider
        self.meter_provider = meter_provider
        self.middleware = middleware
        self.bridge = bridge
        self.config = config
        self._uninstall = uninstall
        self._closed = False
        for k, v in config.extras.items():  # e.g. memory: span_exporter / metric_reader
            setattr(self, k, v)

    def attach_bus(self, bus: EventBus) -> Callable[[], None]:
        """Bridge events of one more EventBus (e.g. a session's)."""
        if self.bridge is None:
            return lambda: None
        return self.bridge.attach(bus)

    def attach_app(self, app: Any) -> None:
        """Bridge the event buses of every currently open session of ``app``
        (an ``AppRunner`` from ``pyreact.boot.bootstrap``)."""
        sessions = getattr(app, "sessions", None)
        buses = [s.event_bus for s in (sessions() if callable(sessions) else [])]
        if not buses and getattr(app, "event_bus", None) is not None:
            buses = [app.event_bus]
        for bus in buses:
            self.attach_bus(bus)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        ok = self.tracer_provider.force_flush(timeout_millis)
        if self.meter_provider is not None:
            ok = self.meter_provider.force_flush(timeout_millis) and ok
        return ok

    def shutdown(self) -> None:
        """Uninstall the middleware, detach the bridge, flush and shut down."""
        global _active
        if self._closed:
            return
        self._closed = True
        if self._uninstall is not None:
            self._uninstall()
        if self.bridge is not None:
            self.bridge.detach()
        try:
            self.tracer_provider.force_flush()
        finally:
            self.tracer_provider.shutdown()
            if self.meter_provider is not None:
                self.meter_provider.shutdown()
        with _active_lock:
            if _active is self:
                _active = None

    def __enter__(self) -> "OtelHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()


def active() -> Optional[OtelHandle]:
    """The handle of the last :func:`setup` that is still running."""
    return _active


def setup(
    exporter: Any = "otlp",
    *,
    service_name: str = "pyreact-app",
    service_version: Optional[str] = None,
    capture_content: Optional[bool] = None,
    metrics: bool = True,
    events: Optional[Iterable[str]] = DEFAULT_KINDS,
    event_buses: Sequence[EventBus] = (),
    bridge_global_bus: bool = True,
    resource_attributes: Optional[Mapping[str, Any]] = None,
    install: bool = True,
    set_global: bool = False,
    sampler: Any = None,
    span_processors: Sequence[Any] = (),
    metric_readers: Sequence[Any] = (),
    max_content_length: int = 8192,
    **opts: Any,
) -> OtelHandle:
    """Wire pyreact to an OpenTelemetry backend.

    ``exporter``: a registered name (``langfuse``, ``otlp``, ``console``,
    ``memory``, or yours) or an :class:`ExporterConfig`. Extra keyword
    arguments go to the exporter factory (e.g. ``host=``, ``endpoint=``).

    ``capture_content`` (default: ``$PYREACT_OTEL_CAPTURE_CONTENT`` or
    ``$OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``) records prompts,
    completions and tool arguments/results. ``events``: EventBus kinds to
    bridge (``None`` = all, ``()`` = none). Calling ``setup`` again shuts the
    previous handle down first.
    """
    global _active
    if isinstance(exporter, ExporterConfig):
        config = exporter
        if opts:
            raise TypeError(f"unexpected options for an ExporterConfig: {sorted(opts)}")
    else:
        config = get_exporter(str(exporter))(**opts)
    if capture_content is None:
        capture_content = _env_capture()

    res: Dict[str, Any] = {"service.name": service_name, "telemetry.sdk.name": "opentelemetry",
                           "pyreact.otel.exporter": exporter if isinstance(exporter, str) else "custom"}
    if service_version:
        res["service.version"] = service_version
    res.update(config.resource_attributes)
    res.update(resource_attributes or {})
    resource = Resource.create(res)

    tp_kw: Dict[str, Any] = {"resource": resource}
    if sampler is not None:
        tp_kw["sampler"] = sampler
    tracer_provider = TracerProvider(**tp_kw)
    if config.span_exporter is not None:
        proc = BatchSpanProcessor if config.batch else SimpleSpanProcessor
        tracer_provider.add_span_processor(proc(config.span_exporter))
    for p in span_processors:
        tracer_provider.add_span_processor(p)

    meter_provider = None
    readers: List[Any] = [r for r in (config.metric_reader, *metric_readers) if r is not None]
    if metrics and readers:
        meter_provider = MeterProvider(metric_readers=readers, resource=resource)

    mw = otel_middleware(
        tracer_provider,
        capture_content=capture_content,
        max_content_length=max_content_length,
        enrichers=config.enrichers,
        metrics=BoundaryMetrics(meter_provider) if meter_provider is not None else None,
    )

    bridge = None
    if events is None or tuple(events):
        bridge = EventBridge(
            tracer_provider,
            kinds=events,
            mode=config.bridge_mode,
            capture_content=capture_content,
            extra_attributes=config.event_attributes,
        )
        if bridge_global_bus:
            from pyreact.core.hook import HookContext

            bridge.attach(HookContext.get_service("event_bus", EventBus))
        for bus in event_buses:
            bridge.attach(bus)

    if set_global:
        trace.set_tracer_provider(tracer_provider)
        if meter_provider is not None:
            otel_metrics.set_meter_provider(meter_provider)

    with _active_lock:
        previous = _active
    if previous is not None:
        previous.shutdown()

    uninstall = install_middleware(mw) if install else None
    handle = OtelHandle(tracer_provider=tracer_provider, meter_provider=meter_provider,
                        middleware=mw, bridge=bridge, config=config, uninstall=uninstall)
    with _active_lock:
        _active = handle
    return handle
