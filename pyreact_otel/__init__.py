"""OpenTelemetry for pyreact: traces, metrics and events, any OTLP backend.

A separate library at the border, like ``pyreact_durable``: nothing in
``pyreact`` imports it. It plugs in through the harness (process-wide
middleware around every boundary call) and the EventBus.

    import pyreact_otel
    otel = pyreact_otel.setup(exporter="langfuse", service_name="store")
    ...
    otel.shutdown()

Requires the ``otel`` extra (``pip install 'pyreact[otel]'``).
"""

from __future__ import annotations

from ._deps import INSTALL_HINT, require_otel

require_otel()

__version__ = "0.1.0"

from . import attributes  # noqa: E402
from ._setup import OtelHandle, active, setup  # noqa: E402
from .bridge import DEFAULT_KINDS, EventBridge  # noqa: E402
from .exporters import (  # noqa: E402
    ExporterConfig,
    ExporterFactory,
    available_exporters,
    get_exporter,
    langfuse_enricher,
    register_exporter,
    unregister_exporter,
)
from .metrics import BoundaryMetrics  # noqa: E402
from .tracing import Enricher, SpanInfo, otel_middleware, provider_of, turn_span  # noqa: E402

__all__ = [
    "BoundaryMetrics",
    "DEFAULT_KINDS",
    "Enricher",
    "EventBridge",
    "ExporterConfig",
    "ExporterFactory",
    "INSTALL_HINT",
    "OtelHandle",
    "SpanInfo",
    "__version__",
    "active",
    "attributes",
    "available_exporters",
    "get_exporter",
    "langfuse_enricher",
    "otel_middleware",
    "provider_of",
    "register_exporter",
    "setup",
    "turn_span",
    "unregister_exporter",
]
