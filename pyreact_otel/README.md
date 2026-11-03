# pyreact_otel: OpenTelemetry for pyreact

Traces, metrics and events from a pyreact app to any OpenTelemetry backend.
**Langfuse** is supported out of the box. Phoenix, Jaeger, Tempo, Honeycomb,
Datadog, Grafana Cloud and others are OTLP endpoints, or a few lines through
the exporter registry.

Like `pyreact_durable`, it is a separate library at the border. Nothing in
`pyreact` imports it (a test enforces this), and the core runs without the
OpenTelemetry packages. It plugs in at two points:

- **Harness middleware** (`pyreact.core.harness.install`): one span per
  boundary call (`turn`, `llm`, `tool`, `sandbox.exec`, ...), plus metrics.
- **EventBus bridge**: events that are not boundaries (`message`, `error`,
  `feedback`, `optimize.*`) are attached to the trace they belong to.

```bash
pip install 'pyreact[otel]'      # or: uv sync --extra otel
```

## Quick start: Langfuse

```python
import pyreact_otel

otel = pyreact_otel.setup(exporter="langfuse", service_name="store")
# LANGFUSE_HOST (default https://cloud.langfuse.com), LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
app = bootstrap(Root)
otel.attach_app(app)              # bridge message/error/feedback events of its sessions
...
otel.shutdown()                   # flush, uninstall
```

In Langfuse each question becomes one trace, `pyreact.turn <component>`. The
turn is an *agent* observation. Its LLM calls are *generations* with model,
token usage and cost, the tools they call are *tool* observations, and app
events are *event* observations. All turns of a pyreact session land in one
Langfuse **session**. A local Langfuse for development is in
[`deploy/langfuse/`](../deploy/langfuse/README.md), with an end-to-end check
in `scripts/e2e_otel_langfuse.py`. For any OTLP backend there is a
one-container collector in [`deploy/otel-collector/`](../deploy/otel-collector/README.md)
and `scripts/e2e_otel_collector.py`.

Options for `exporter="langfuse"`: `public_key`, `secret_key`, `host`,
`environment` (or `LANGFUSE_TRACING_ENVIRONMENT`), `release`, `tags`,
`timeout` (30 s), `ingestion_version` ("4").

## Quick start: any OTLP backend

```python
pyreact_otel.setup(exporter="otlp")   # standard OTEL_EXPORTER_OTLP_* env vars
pyreact_otel.setup(exporter="otlp", endpoint="http://localhost:4318",
                   headers={"x-api-key": "..."})        # base URL; /v1/traces, /v1/metrics appended
```

| Backend | How |
|---|---|
| Jaeger / Tempo / OTel Collector | `endpoint="http://host:4318"` |
| Arize Phoenix | `endpoint="http://localhost:6006"` (Phoenix serves `/v1/traces`) |
| Honeycomb | `endpoint="https://api.honeycomb.io"`, `headers={"x-honeycomb-team": KEY}` |
| Grafana Cloud | its OTLP gateway URL + `Authorization: Basic ...` |
| Datadog | the agent's OTLP receiver, `endpoint="http://localhost:4318"` |

Also `exporter="console"` (prints spans) and `exporter="memory"` (tests:
`handle.span_exporter`, `handle.metric_reader`).

## `setup()`

```python
handle = pyreact_otel.setup(
    exporter="langfuse",          # registered name or an ExporterConfig
    service_name="pyreact-app",
    capture_content=None,         # default: $PYREACT_OTEL_CAPTURE_CONTENT / $OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
    metrics=True,                 # when the exporter takes metrics (Langfuse doesn't)
    events=("message", "error", "feedback", "optimize.*"),   # None = all, () = none
    event_buses=(),               # extra EventBus objects to bridge
    set_global=False,             # also make the providers OTel-global
    span_processors=(), metric_readers=(), sampler=None, resource_attributes=None,
    **exporter_options,
)
handle.attach_bus(bus); handle.attach_app(app)
handle.force_flush(); handle.shutdown()      # also a context manager
```

`setup` installs the middleware **process-wide** (outermost, before any
`HarnessProvider`). This is required: turn boundaries see only process-wide
middleware. Calling `setup` again shuts the previous handle down first.
Without `setup`, the building blocks are `otel_middleware(...)`,
`BoundaryMetrics(meter_provider)` and `EventBridge(...)`.

## Span reference

| Boundary kind | Span name | Kind | Key attributes |
|---|---|---|---|
| `turn` | `pyreact.turn` (trace root) | INTERNAL | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.name`, `pyreact.turn.status/index/source` |
| `llm` | `chat <model>` | CLIENT | `gen_ai.operation.name=chat`, `gen_ai.provider.name` (+ deprecated `gen_ai.system`), `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `pyreact.usage.total_tokens`, `pyreact.cost`, `pyreact.call_id`, `pyreact.module_key`, `pyreact.arch`, `pyreact.lm_calls` |
| `tool` / `tool.*` | `execute_tool <name>` | INTERNAL | `gen_ai.operation.name=execute_tool`, `gen_ai.tool.name`, `gen_ai.tool.type=function`, `gen_ai.tool.call.id` |
| `sandbox.exec`, other | `<kind> <name>` | INTERNAL | common attributes |

Every span carries `pyreact.boundary.kind/name/id/parent_id`,
`pyreact.component`, `pyreact.session_id` (+ `session.id`),
`pyreact.turn_id`, `pyreact.attempt` (final attempt after retries), and
`user.id` when `attrs["user_id"]` is set. Session, user and turn are
inherited from the trace when a boundary does not carry them.

**Parenting.** A span's parent is, in order: the span of `Boundary.parent_id`;
the current OTel context, which flows through `asyncio` tasks and
`asyncio.to_thread`; the live `turn` span of the same session and turn. If
none applies, the span starts a new trace.

**Status.**
- An exception sets status ERROR, adds an `exception` event and sets `error.type`.
- `Blocked` sets status ERROR with `error.type=Blocked`, `pyreact.blocked.reason` and `pyreact.blocked.by`, and records no exception event.
- A turn whose spawned work failed (`status="error"`) sets ERROR with `error.type=TurnError`.
- Cancellation leaves the status UNSET and sets `pyreact.cancelled=true`.

**Content** is recorded only with `capture_content=True`. Each value is
truncated to `max_content_length` (8192 characters):
- `llm`: `gen_ai.input.messages` and `gen_ai.output.messages` (semconv JSON parts, from `attrs["lm_calls"]`).
- `tool`: `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result`.
- Any kind: `pyreact.inputs` and `pyreact.outputs`.

**Langfuse attributes**, added by the `langfuse` exporter's enricher:
- `langfuse.observation.type`: turn → `agent`, llm → `generation`, tool → `tool`, other → `span`.
- Copied to **every** span, because Langfuse v4 queries observations directly: `langfuse.trace.name`, `langfuse.session.id`, `langfuse.user.id`, `langfuse.environment`, `langfuse.release` and `langfuse.trace.tags`.
- `langfuse.observation.model.name`, `usage_details` and `cost_details`.
- `langfuse.observation.level`: `WARNING` for Blocked, `ERROR` for errors, with `status_message`.
- `langfuse.observation.input`/`output`, only with content capture.

## Metrics

| Instrument | Type, unit | Attributes |
|---|---|---|
| `gen_ai.client.token.usage` | histogram, `{token}` | `gen_ai.token.type` (input/output), operation, provider, model |
| `gen_ai.client.operation.duration` | histogram, `s` | `chat` and `execute_tool`; `error.type` |
| `pyreact.llm.cost` | counter, `USD` | provider, model, `pyreact.module_key` |
| `pyreact.turn.duration` | histogram, `s` | `pyreact.component`, `pyreact.turn.status`, `error.type` |
| `pyreact.boundary.duration` | histogram, `s` | kind, name, `error.type` |
| `pyreact.boundary.errors` | counter | kind, name, `error.type` (Blocked included, cancellation not) |

Token usage is a histogram, as the GenAI conventions specify. Its sum is the
token counter.

## Events (EventBus bridge)

Each matching event is attached where it belongs: the publisher's current
span first, then the live turn span of `event.turn_id`. Otherwise it becomes
a standalone `pyreact.event <kind>` span, or is dropped with
`orphans="drop"`. With `mode="event"` (the default) it is added as a span
event. With `mode="span"` it becomes a zero-length child span; the Langfuse
exporter uses this mode so events show up as observations. Payload scalars
become `pyreact.event.<key>`. Content-like keys (`text`, `inputs`,
`outputs`, ...) are included only with content capture.

## Adding a vendor

A vendor is a factory that returns an `ExporterConfig`:

```python
import os
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from pyreact_otel import ExporterConfig, register_exporter

def phoenix(endpoint=None, project=None):
    def enrich(info):                                  # optional vendor attributes
        return {"openinference.span.kind": {"llm": "LLM", "tool": "TOOL"}.get(info.boundary.kind, "CHAIN")}
    return ExporterConfig(
        span_exporter=OTLPSpanExporter(endpoint=(endpoint or "http://localhost:6006") + "/v1/traces"),
        enrichers=[enrich],
        resource_attributes={"openinference.project.name": project or "pyreact"},
    )

register_exporter("phoenix", phoenix)
pyreact_otel.setup(exporter="phoenix", project="store")
```

`ExporterConfig` fields:
- `span_exporter`
- `metric_reader`: for example a `PeriodicExportingMetricReader`
- `enrichers`: `(SpanInfo) -> dict`, called when a span ends. `SpanInfo` has `boundary`, `attributes`, `error`, `blocked`, `content`, `trace_name` and `is_root`.
- `event_attributes`: the same idea, for bridge events
- `bridge_mode`
- `batch`
- `resource_attributes`
- `extras`: values exposed as attributes of the handle

## Known gaps

- `attach_app` bridges the sessions open at the time of the call. pyreact has no session-opened hook yet, so call `attach_bus(session.event_bus)` for sessions opened later.
- Events are not exported as OTel log records; the Python logs API is still experimental.
- Only HTTP/protobuf OTLP is bundled. For gRPC, add `opentelemetry-exporter-otlp-proto-grpc` and register it. Langfuse accepts HTTP only.
