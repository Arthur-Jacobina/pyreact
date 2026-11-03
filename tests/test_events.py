from pyreact.core import runtime
from pyreact.core.events import EventBus


def test_publish_filters_by_kind_and_prefix():
    bus, got = EventBus(), []
    bus.subscribe(lambda e: got.append(("all", e.kind)))
    bus.subscribe(lambda e: got.append(("llm", e.kind)), kinds=["llm.*"])
    bus.subscribe(lambda e: got.append(("msg", e.kind)), kinds=["message"])
    bus.publish("llm.end", {"tokens": 3}, turn_id="t1")
    bus.publish("message", {"text": "hi"})
    assert got == [("all", "llm.end"), ("llm", "llm.end"), ("all", "message"), ("msg", "message")]


def test_listener_error_is_isolated():
    bus, got, errs = EventBus(), [], []
    runtime.set_error_handler(lambda exc, ctx, phase: errs.append(phase))
    bus.subscribe(lambda e: 1 / 0)
    bus.subscribe(lambda e: got.append(e.payload))
    unsub = bus.subscribe(lambda e: got.append("removed"))
    unsub()
    bus.publish("x", {"a": 1})
    assert got == [{"a": 1}] and errs == ["event-listener"]
