"""Import guard: ``pyreact_otel`` needs the optional ``otel`` extra."""

from __future__ import annotations

INSTALL_HINT = (
    "pyreact_otel needs the OpenTelemetry packages (opentelemetry-api, "
    "opentelemetry-sdk, opentelemetry-exporter-otlp-proto-http). Install the "
    "extra: `pip install 'pyreact[otel]'` or `uv sync --extra otel`."
)


def require_otel() -> None:
    try:
        import opentelemetry.sdk  # noqa: F401
        import opentelemetry.trace  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
        raise ImportError(INSTALL_HINT) from exc
