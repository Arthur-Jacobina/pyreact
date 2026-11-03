import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

import pyreact_otel  # noqa: E402
from pyreact.core.events import EventBus  # noqa: E402
from pyreact.core.harness import Blocked, through  # noqa: E402
from pyreact_otel import ExporterConfig, register_exporter, unregister_exporter  # noqa: E402
from pyreact_otel.exporters import langfuse_exporter  # noqa: E402

USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


async def llm(b):
    b.attrs.update(model="openai/gpt-4o-mini", usage=USAGE, cost=0.001,
                   lm_calls=[{"messages": [{"role": "user", "content": "hello?"}],
                              "completion": "hi!"}])
    return "hi!"


async def tool(b):
    return 3


async def turn(b):
    await through("llm", "supervisor", {"q": "hello?"}, llm)
    await through("tool", "lookup_price", {"sku": "X"}, tool)
    with pytest.raises(Blocked):
        await through("tool", "rm", {}, _blocked)


async def _blocked(b):
    raise Blocked("not allowed", by="guard")


def _turn(session="sess-1"):
    from pyreact.core.harness import Boundary, resolve_harness

    async def main():
        b = Boundary(kind="turn", name="StoreHome", session_id=session, turn_id=f"{session}:t1",
                     component="Root/StoreHome")
        await resolve_harness().run(b, turn)

    asyncio.run(main())


def _spans(handle):
    return {s.name: s for s in handle.span_exporter.get_finished_spans()}


def test_memory_setup_installs_process_wide_and_shutdown_uninstalls():
    bus = EventBus()
    h = pyreact_otel.setup("memory", service_name="unit", event_buses=[bus])
    try:
        assert pyreact_otel.active() is h

        async def with_event(b):
            bus.publish("feedback", {"score": 1.0})
            return await turn(b)

        from pyreact.core.harness import Boundary, resolve_harness

        asyncio.run(resolve_harness().run(Boundary(kind="turn", name="App"), with_event))
        s = _spans(h)
        assert {"pyreact.turn", "chat openai/gpt-4o-mini", "execute_tool lookup_price",
                "execute_tool rm"} <= set(s)
        assert s["pyreact.turn"].resource.attributes["service.name"] == "unit"
        assert [e.name for e in s["pyreact.turn"].events] == ["feedback"]
        names = {m.name for rm in h.metric_reader.get_metrics_data().resource_metrics
                 for sm in rm.scope_metrics for m in sm.metrics}
        assert {"gen_ai.client.token.usage", "gen_ai.client.operation.duration",
                "pyreact.turn.duration", "pyreact.boundary.errors", "pyreact.llm.cost"} <= names
    finally:
        h.shutdown()
    assert pyreact_otel.active() is None
    exporter = h.span_exporter
    n = len(exporter.get_finished_spans())
    asyncio.run(through("tool", "after", {}, tool))  # middleware gone: no new span
    assert len(exporter.get_finished_spans()) == n


def test_setup_twice_replaces_previous_handle():
    first = pyreact_otel.setup("memory")
    second = pyreact_otel.setup("memory")
    try:
        asyncio.run(through("tool", "t", {}, tool))
        assert len(first.span_exporter.get_finished_spans()) == 0
        assert len(second.span_exporter.get_finished_spans()) == 1
    finally:
        second.shutdown()


def test_registry():
    made = {}

    def mine(flavor="x"):
        made["flavor"] = flavor
        return ExporterConfig(span_exporter=InMemorySpanExporter(), batch=False)

    register_exporter("mine", mine)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_exporter("mine", mine)
        assert "mine" in pyreact_otel.available_exporters()
        pyreact_otel.setup("mine", flavor="y", events=()).shutdown()
        assert made == {"flavor": "y"}
        with pytest.raises(TypeError):
            pyreact_otel.setup("mine", nope=1)
    finally:
        unregister_exporter("mine")
    with pytest.raises(ValueError, match="unknown exporter 'mine'.*langfuse"):
        pyreact_otel.setup("mine")


def test_capture_content_from_env(monkeypatch):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
    h = pyreact_otel.setup("memory", events=())
    try:
        asyncio.run(through("llm", "m", {"q": "hello?"}, llm))
        (s,) = h.span_exporter.get_finished_spans()
        assert "hello?" in s.attributes["gen_ai.input.messages"]
    finally:
        h.shutdown()


def test_otlp_endpoint_and_env(monkeypatch):
    from pyreact_otel.exporters import otlp_exporter

    cfg = otlp_exporter(endpoint="http://collector:4318/", headers="api-key=abc,x=1")
    assert cfg.span_exporter._endpoint == "http://collector:4318/v1/traces"
    cfg.metric_reader.shutdown()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://env-host:4318")
    cfg = otlp_exporter(metrics=False)
    assert cfg.span_exporter._endpoint == "http://env-host:4318/v1/traces"
    assert cfg.metric_reader is None


def _langfuse_handle(monkeypatch, **kw):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://lf.local:3000/")
    cfg = langfuse_exporter(environment="test", tags=["pyreact"])
    assert cfg.span_exporter._endpoint == "http://lf.local:3000/api/public/otel/v1/traces"
    mem = InMemorySpanExporter()
    cfg.span_exporter, cfg.batch, cfg.extras = mem, False, {"span_exporter": mem}
    return pyreact_otel.setup(cfg, **kw)


def test_langfuse_attribute_mapping(monkeypatch):
    h = _langfuse_handle(monkeypatch, events=())
    try:
        _turn()
        s = _spans(h)
    finally:
        h.shutdown()
    t, g, ok, rm = (s["pyreact.turn"], s["chat openai/gpt-4o-mini"],
                    s["execute_tool lookup_price"], s["execute_tool rm"])
    assert t.attributes["langfuse.observation.type"] == "agent"
    assert g.attributes["langfuse.observation.type"] == "generation"
    assert ok.attributes["langfuse.observation.type"] == "tool"
    for span in (t, g, ok, rm):  # trace-level attributes on every span (Langfuse v4)
        assert span.attributes["langfuse.trace.name"] == "pyreact.turn StoreHome"
        assert span.attributes["langfuse.session.id"] == "sess-1"
        assert span.attributes["langfuse.environment"] == "test"
        assert tuple(span.attributes["langfuse.trace.tags"]) == ("pyreact",)
    assert g.attributes["langfuse.observation.model.name"] == "openai/gpt-4o-mini"
    assert json.loads(g.attributes["langfuse.observation.usage_details"]) == {
        "input": 10, "output": 5, "total": 15}
    assert json.loads(g.attributes["langfuse.observation.cost_details"]) == {"total": 0.001}
    assert rm.attributes["langfuse.observation.level"] == "WARNING"
    assert "not allowed" in rm.attributes["langfuse.observation.status_message"]
    assert not any(k.endswith((".input", ".output")) for sp in s.values() for k in sp.attributes)


def test_langfuse_content_and_events(monkeypatch):
    bus = EventBus()
    h = _langfuse_handle(monkeypatch, capture_content=True, event_buses=[bus])
    try:
        async def with_msg(b):
            bus.publish("message", {"text": "olá", "sender": "assistant"})
            return await turn(b)

        from pyreact.core.harness import Boundary, resolve_harness

        asyncio.run(resolve_harness().run(
            Boundary(kind="turn", name="StoreHome", session_id="s2", turn_id="s2:t1"), with_msg))
        s = _spans(h)
    finally:
        h.shutdown()
    g = s["chat openai/gpt-4o-mini"]
    assert json.loads(g.attributes["langfuse.observation.input"]) == [{"role": "user", "content": "hello?"}]
    assert g.attributes["langfuse.observation.output"] == "hi!"
    tool_span = s["execute_tool lookup_price"]
    assert tool_span.attributes["langfuse.observation.input"] == '{"sku": "X"}'
    assert tool_span.attributes["langfuse.observation.output"] == "3"
    ev = s["pyreact.event message"]  # Langfuse: events as event observations
    assert ev.attributes["langfuse.observation.type"] == "event"
    assert ev.attributes["langfuse.session.id"] == "s2"
    assert ev.attributes["langfuse.trace.name"] == "pyreact.turn StoreHome"
    assert ev.parent.span_id == s["pyreact.turn"].context.span_id


def test_langfuse_requires_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
        langfuse_exporter()


def test_langfuse_wire_format_against_a_local_receiver(monkeypatch):
    """The real OTLP/HTTP exporter posts protobuf spans to the Langfuse path
    with Basic auth (a localhost HTTP server stands in for Langfuse)."""
    got = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            got.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.end_headers()

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        h = pyreact_otel.setup("langfuse", public_key="pk-lf-1", secret_key="sk-lf-1",
                               host=f"http://127.0.0.1:{server.server_port}", events=())
        _turn()
        h.shutdown()
    finally:
        server.shutdown()
    assert got, "no OTLP request received"
    path, headers, body = got[0]
    assert path == "/api/public/otel/v1/traces"
    assert headers["authorization"] == "Basic " + base64.b64encode(b"pk-lf-1:sk-lf-1").decode()
    assert headers["x-langfuse-ingestion-version"] == "4"
    assert headers["content-type"] == "application/x-protobuf"
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    req = ExportTraceServiceRequest.FromString(body)
    names = {s.name for rs in req.resource_spans for ss in rs.scope_spans for s in ss.spans}
    assert {"pyreact.turn", "chat openai/gpt-4o-mini", "execute_tool lookup_price"} <= names


def test_parse_otlp_headers():
    from pyreact_otel.exporters import _parse_headers

    assert _parse_headers("api-key=abc, x=1") == {"api-key": "abc", "x": "1"}
    assert _parse_headers({"a": "b"}) == {"a": "b"} and _parse_headers(None) is None
