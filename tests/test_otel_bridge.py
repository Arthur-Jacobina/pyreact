import asyncio

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from pyreact.core.events import EventBus  # noqa: E402
from pyreact.core.harness import Boundary, Harness, through  # noqa: E402
from pyreact_otel import EventBridge, otel_middleware  # noqa: E402


@pytest.fixture
def setup():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter, EventBus()
    provider.shutdown()


def spans(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def test_events_become_span_events_on_the_current_span(setup):
    provider, exporter, bus = setup
    bridge = EventBridge(provider)
    bridge.attach(bus)
    h = Harness([otel_middleware(provider)])

    async def turn(b):
        bus.publish("message", {"text": "olá cliente", "sender": "assistant"}, turn_id="t1")
        bus.publish("feedback", {"score": 0.5, "text": "meh", "target": "llm-3"}, turn_id="t1")
        bus.publish("llm.end", {"usage": {}}, turn_id="t1")  # not in the default kinds
        bus.publish("optimize.end", {"score": 0.9, "swapped": True})

    asyncio.run(through("turn", "App", {}, turn, harness=h))
    (t,) = exporter.get_finished_spans()
    assert [e.name for e in t.events] == ["message", "feedback", "optimize.end"]
    msg, fb, opt = t.events
    assert msg.attributes["pyreact.event.sender"] == "assistant"
    assert "pyreact.event.text" not in msg.attributes  # content is opt-in
    assert fb.attributes["pyreact.event.score"] == 0.5 and fb.attributes["pyreact.event.target"] == "llm-3"
    assert opt.attributes["pyreact.event.swapped"] is True
    assert msg.attributes["pyreact.turn_id"] == "t1"
    bridge.detach()
    bus.publish("message", {"text": "after detach"})
    assert len(exporter.get_finished_spans()) == 1


def test_falls_back_to_the_turn_span_then_orphan_span(setup):
    provider, exporter, bus = setup
    EventBridge(provider, kinds=("error",), capture_content=True).attach(bus)
    h = Harness([otel_middleware(provider)])

    async def main():
        started, release = asyncio.Event(), asyncio.Event()

        async def turn(b):
            started.set()
            await release.wait()

        async def other():  # a task without the turn's context
            await started.wait()
            bus.publish("error", {"phase": "effect", "component": "App", "error": "boom"},
                        turn_id="t9")
            release.set()

        import contextvars

        task = asyncio.create_task(other(), context=contextvars.Context())
        await h.run(Boundary(kind="turn", name="App", turn_id="t9"), turn)
        await task

    asyncio.run(main())
    bus.publish("error", {"error": "no turn"})  # nothing live: orphan span
    s = spans(exporter)
    (ev,) = s["pyreact.turn"].events
    assert ev.name == "error" and ev.attributes["pyreact.event.error"] == "boom"
    orphan = s["pyreact.event error"]
    assert orphan.parent is None and orphan.events[0].attributes["pyreact.event.error"] == "no turn"


def test_span_mode_and_drop_orphans(setup):
    provider, exporter, bus = setup
    extra = []
    bridge = EventBridge(provider, mode="span", orphans="drop", capture_content=True,
                         extra_attributes=lambda ev, span, root: extra.append(root) or {"x.y": 1})
    bridge.attach(bus)
    h = Harness([otel_middleware(provider)])

    async def turn(b):
        bus.publish("message", {"text": "hi"})

    asyncio.run(through("turn", "App", {}, turn, harness=h))
    bus.publish("message", {"text": "orphan"})  # dropped
    s = spans(exporter)
    child = s["pyreact.event message"]
    assert child.parent.span_id == s["pyreact.turn"].context.span_id
    assert child.attributes["pyreact.event.text"] == "hi" and child.attributes["x.y"] == 1
    assert extra == ["pyreact.turn App"]
    assert len(s) == 2


def test_bridge_errors_never_reach_the_publisher(setup):
    provider, exporter, bus = setup

    def bad(ev, span, root):
        raise RuntimeError("enricher bug")

    EventBridge(provider, extra_attributes=bad).attach(bus)
    bus.publish("message", {"text": "x"})  # must not raise


def test_invalid_options():
    with pytest.raises(ValueError):
        EventBridge(mode="log")
    with pytest.raises(ValueError):
        EventBridge(orphans="keep")
