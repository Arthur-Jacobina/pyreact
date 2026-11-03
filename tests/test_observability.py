import asyncio
import io
import json

import pytest

from pyreact.boot import bootstrap
from pyreact.boot.terminal import handle_command
from pyreact.components.message import (
    Message,
    parse_legacy_line,
    set_legacy_message_bridge,
)
from pyreact.core import HookContext, MessageBuffer, component, hooks, runtime
from pyreact.core.events import Event, EventBus
from pyreact.core.runtime import run_renders, schedule_rerender
from pyreact.observability import (
    ConsoleSink,
    EventLog,
    JSONLSink,
    TurnTimeline,
    UsageAggregator,
    install_error_events,
    normalize_usage,
    read_jsonl,
)


async def _mount(fn):
    root = HookContext(fn.__name__, fn)
    schedule_rerender(root)
    for _ in range(3):
        await asyncio.sleep(0)
        await run_renders()
    return root


@pytest.fixture
def legacy_capture():
    buf = MessageBuffer()
    lines = []
    buf.subscribe(lines.append)
    yield lines
    buf.unsubscribe(lines.append)
    set_legacy_message_bridge(True)


# --------------------------------------------------------------- Message
def test_message_publishes_typed_event_and_legacy_line(legacy_capture):
    bus = HookContext.get_service("event_bus", EventBus)
    got = []
    bus.subscribe(got.append, kinds=["message"])

    @component
    def App():
        return [Message(key="m", text="hello", sender="assistant", message_type="info")]

    asyncio.run(_mount(App))
    assert len(got) == 1
    ev = got[0]
    assert ev.payload["text"] == "hello"
    assert ev.payload["sender"] == "assistant"
    assert ev.payload["message_type"] == "info"
    assert ev.source.endswith("Message[m]") and ev.source.startswith("App")
    legacy = [parse_legacy_line(x) for x in legacy_capture]
    legacy = [x for x in legacy if x]
    assert legacy and legacy[-1]["text"] == "hello" and legacy[-1]["type"] == "message"


def test_legacy_bridge_can_be_disabled(legacy_capture):
    set_legacy_message_bridge(False)
    bus = HookContext.get_service("event_bus", EventBus)
    got = []
    bus.subscribe(got.append, kinds=["message"])

    @component
    def App():
        return [Message(key="m", text="quiet")]

    asyncio.run(_mount(App))
    assert [e.payload["text"] for e in got] == ["quiet"]
    assert not any(x.startswith("__MESSAGE__:") for x in legacy_capture)


def test_examples_message_is_reexport():
    from examples.message import Message as M2

    assert M2 is Message


# ---------------------------------------------------------------- errors
def test_error_handler_publishes_error_event_and_prints(capsys):
    bus = EventBus()
    got = []
    bus.subscribe(got.append, kinds=["error"])
    uninstall = install_error_events(bus)

    @component
    def Boom():
        raise ValueError("kaboom")

    @component
    def App():
        return [Boom(key="b")]

    asyncio.run(_mount(App))
    uninstall()
    assert runtime._error_handler is None
    assert len(got) == 1
    p = got[0].payload
    assert p["phase"] == "render" and p["component"] == "App/Boom[b]"
    assert "ValueError: kaboom" in p["error"] and "Traceback" in p["traceback"]
    assert "kaboom" in capsys.readouterr().err


def test_failing_error_listener_does_not_recurse(capsys):
    bus = EventBus()
    calls = []

    def bad(ev):
        calls.append(ev)
        raise RuntimeError("listener broke")

    bus.subscribe(bad, kinds=["error"])
    uninstall = install_error_events(bus)
    try:
        runtime.report_error(ValueError("x"), None, "effect")
    finally:
        uninstall()
    assert len(calls) == 1  # the listener's own failure is not re-published
    err = capsys.readouterr().err
    assert "ValueError: x" in err and "listener broke" in err


def test_app_runner_routes_component_errors_to_events():
    @component
    def App():
        v, setv = hooks.use_state(0)

        def boom():
            if v == 0:
                raise RuntimeError("effect failed")

        hooks.use_effect(boom, [v])
        return [Message(key="hi", text="hi", sender="assistant", legacy=False)]

    app = bootstrap(App)
    try:
        app.wait_idle(timeout=5)
        kinds = [e.kind for e in app.events.recent(None)]
        assert "message" in kinds and "error" in kinds
        err = app.events.recent(1, ["error"])[0]
        assert err.payload["phase"] == "effect" and err.payload["component"] == "App"
        assert app.timeline.get("-").errors
    finally:
        app.shutdown()
    assert runtime._error_handler is None


# ----------------------------------------------------------------- sinks
def test_jsonl_sink_appends_and_rotates(tmp_path):
    path = tmp_path / "logs" / "events.jsonl"
    bus = EventBus()
    sink = JSONLSink(str(path), max_bytes=400, backups=2)
    sink.attach(bus)
    for i in range(20):
        bus.publish("message", {"text": f"m{i}", "obj": object()}, turn_id="t1")
    sink.close()
    bus.publish("message", {"text": "after close"})
    files = sorted(p.name for p in path.parent.iterdir())
    assert files == ["events.jsonl", "events.jsonl.1", "events.jsonl.2"]
    rows = read_jsonl(str(path))
    assert rows and all(r["kind"] == "message" for r in rows)
    assert rows[-1]["payload"]["text"] == "m19" and rows[-1]["turn_id"] == "t1"
    assert all(p.stat().st_size <= 400 for p in path.parent.iterdir())
    # append mode: reopening keeps existing lines
    s2 = JSONLSink(str(path), max_bytes=0)
    s2(Event("x"))
    s2.close()
    assert read_jsonl(str(path))[-1]["kind"] == "x"


def test_console_sink_formats_and_filters():
    out = io.StringIO()
    sink = ConsoleSink(stream=out, color=False)  # default: message + error
    sink(Event("message", {"text": "hi", "sender": "assistant", "message_type": "chat"}))
    sink(Event("llm.end", {"usage": {}}))
    sink(Event("error", {"phase": "render", "component": "App/X", "error": "E: bad"}))
    assert out.getvalue().splitlines() == [
        "[ASSISTANT] hi",
        "[ERROR] render <App/X>: E: bad",
    ]
    out2 = io.StringIO()
    sink2 = ConsoleSink(kinds=None, exclude=["message"], stream=out2, color=False)
    sink2(Event("message", {"text": "hidden"}))
    sink2(Event("llm.end", {"name": "sup", "latency": 1.25, "usage": {"total_tokens": 7}}))
    sink2(Event("tool.end", {"name": "lookup_price", "result": 29.9}))
    lines = out2.getvalue().splitlines()
    assert lines[0] == "· llm.end sup 1.2s 7 tok" and "lookup_price → 29.9" in lines[1]


def test_event_log_replays_without_gaps_or_duplicates():
    bus, log = EventBus(), EventLog(maxlen=3)
    log.attach(bus)
    for i in range(5):
        bus.publish("k", {"i": i})
    got = []
    log.subscribe(lambda e: got.append(e.payload["i"]), replay=True)
    bus.publish("k", {"i": 5})
    assert got == [2, 3, 4, 5]
    assert [e.payload["i"] for e in log.recent(2)] == [4, 5]


# ----------------------------------------------------------------- usage
def test_normalize_usage_variants():
    assert normalize_usage({}) is None
    assert normalize_usage(None) is None
    assert normalize_usage({"input_tokens": 3, "output_tokens": 4})["total_tokens"] == 7
    nested = {"m1": {"prompt_tokens": 1, "completion_tokens": 2}, "m2": {"total_tokens": 10}}
    assert normalize_usage(nested)["total_tokens"] == 13
    assert normalize_usage({"total_tokens": 5, "estimated": True})["estimated"] is True


def test_usage_aggregator_per_turn_and_session_tolerates_missing_usage():
    agg, bus = UsageAggregator(), EventBus()
    agg.attach(bus)
    bus.publish("turn.start", {}, turn_id="t1")
    bus.publish("llm.end", {"usage": {"prompt_tokens": 10, "completion_tokens": 5}, "cost": 0.01, "latency": 1.0})
    bus.publish("llm.end", {"usage": {}, "latency": 2.0}, turn_id="t1")  # CLI LM: empty usage
    bus.publish("llm.end", {})  # nothing at all
    bus.publish("turn.end", {}, turn_id="t1")
    bus.publish("llm.error", {"error": "x"}, turn_id="t2")
    t1 = agg.turn("t1")
    assert (t1.calls, t1.calls_with_usage, t1.calls_without_usage) == (3, 1, 2)
    assert t1.total_tokens == 15 and t1.cost == pytest.approx(0.01)
    assert t1.latency_total == 3.0 and t1.latency_max == 2.0
    assert agg.turn("t2").errors == 1 and agg.session.errors == 1
    assert agg.session.calls == 3
    text = agg.render()
    assert "session: 3 calls" in text and "2 without usage" in text and "turn t1" in text
    json.dumps(agg.to_dict())


# -------------------------------------------------------------- timeline
def test_turn_timeline_groups_and_renders():
    tl, bus = TurnTimeline(), EventBus()
    tl.attach(bus)
    bus.publish("message", {"text": "welcome", "sender": "assistant"})  # before any turn
    bus.publish("turn.start", {"input": "price of shirt?"}, turn_id="t1")
    bus.publish("llm.start", {"name": "supervisor"}, turn_id="t1")
    bus.publish("llm.end", {"name": "supervisor", "model": "fast", "latency": 1.5, "usage": {"total_tokens": 42, "estimated": True}}, turn_id="t1")
    bus.publish("tool.start", {"name": "lookup_price"})  # no turn_id: open turn
    bus.publish("tool.end", {"name": "lookup_price", "result": 29.9, "latency": 0.01})
    bus.publish("message", {"text": "R$ 29.90", "sender": "assistant", "message_type": "chat"})
    bus.publish("error", {"phase": "effect", "component": "App/X", "error": "E: x"})
    bus.publish("turn.end", {}, turn_id="t1")
    bus.publish("llm.start", {"name": "late"}, turn_id="t2")  # partial turn, missing fields
    t1 = tl.get("t1")
    assert t1.closed and t1.input == "price of shirt?"
    assert [m["text"] for m in t1.messages] == ["R$ 29.90"]
    assert len(t1.llm_calls) == 1 and t1.llm_calls[0]["tokens"] == 42
    assert t1.tools[0]["ok"] and len(t1.errors) == 1
    assert [m["text"] for m in tl.get("-").messages] == ["welcome"]
    text = tl.render()
    assert "turn t1" in text and "1 msg, 1 llm, 1 tool, 1 err" in text
    assert "supervisor model=fast 1.5s 42 tok~est" in text
    assert "lookup_price ok" in text and "[ASSISTANT] R$ 29.90" in text
    assert "turn t2 (open)" in text and "+1 pending" in text
    assert "(no turn)" in text and "(no turn)" not in tl.render(include_untracked=False)
    json.dumps(tl.to_dict())


# ----------------------------------------------------------- trace events
def test_trace_render_events_when_enabled():
    from pyreact.core import debug

    bus = HookContext.get_service("event_bus", EventBus)
    got = []
    bus.subscribe(got.append, kinds=["trace.render"])

    @component
    def Child():
        return []

    @component
    def App():
        return [Child(key="c")]

    prev = (debug._TRACE_ENABLED, debug._TRACE_EVENTS)
    try:
        debug.enable_tracing(events=True)
        asyncio.run(_mount(App))
    finally:
        debug._TRACE_ENABLED, debug._TRACE_EVENTS = prev
    assert got and got[0].payload["root"] == "App"
    assert [r["name"] for r in got[0].payload["rendered"]] == ["App", "Child"]


# ------------------------------------------------------- terminal commands
def test_terminal_commands_print_events_usage_turns(capsys):
    @component
    def App():
        return [Message(key="w", text="welcome!", sender="assistant", legacy=False)]

    app = bootstrap(App)
    try:
        app.wait_idle(timeout=5)
        app.event_bus.publish("turn.start", {"input": "q"}, turn_id="t1")
        app.event_bus.publish("llm.end", {"name": "sup", "usage": {}, "latency": 0.5})
        app.event_bus.publish("turn.end", {}, turn_id="t1")
        assert handle_command(app, "events", "message")
        assert handle_command(app, "usage")
        assert handle_command(app, "turns")
        assert not handle_command(app, "nope")
        out = capsys.readouterr().out
        assert "[ASSISTANT] welcome!" in out
        assert "1 without usage" in out
        assert "turn t1" in out and "usage n/a" in out
    finally:
        app.shutdown()


def test_claude_cli_lm_reports_estimated_usage():
    from integrations.claude_cli_lm import ClaudeCLILM, _wrap, estimate_usage

    u = estimate_usage("sys", "hello world!", "ok")
    assert u["estimated"] and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] > 0
    assert normalize_usage(dict(_wrap("ok", "m", u).usage))["estimated"]
    assert ClaudeCLILM("haiku", approx_usage=False)._usage("s", "u", "o") == {}


def test_terminal_commands_accept_a_root_or_a_session(capsys, monkeypatch):
    from pyreact.boot import create_root, read_terminal_and_invoke, register_command
    from pyreact.boot.terminal import unregister_command
    from pyreact.core import use_on_turn

    got = []

    @component
    def App():
        async def on_turn(turn):
            got.append(turn.text)

        use_on_turn(on_turn)
        return [Message(key="w", text="welcome!", sender="assistant", legacy=False)]

    seen = []
    register_command("who", lambda target, args: seen.append(target), "test")
    root = create_root(App, session_id="cmd")
    try:
        root.wait_idle(timeout=5)
        root.send("hi", wait=True, timeout=5)
        for target in (root, root.session):
            # :trace used to raise on a Session (no print_render_trace)
            for cmd, args in [("events", "message"), ("usage", ""), ("turns", ""),
                              ("tree", ""), ("trace", ""), ("route", ""), ("nav", "/x"),
                              ("who", "")]:
                assert handle_command(target, cmd, args), cmd
        out = capsys.readouterr().out
        assert "[ASSISTANT] welcome!" in out and "Usage" in out and "Turns" in out
        assert "Route" in out
        assert seen == [root.session, root.session]  # registered: the root's session

        lines = iter(["typed", ":usage", ":q"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
        asyncio.run(read_terminal_and_invoke(root, console=False))
        assert got == ["hi", "typed"]
    finally:
        unregister_command("who")
        root.unmount()
