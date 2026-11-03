"""use_input: the latest raw input of the component's session."""

import pytest

from pyreact.boot import AppRunner, bootstrap
from pyreact.components.keystroke import Keystroke
from pyreact.core import component, hooks, use_input


@pytest.fixture
def run_app():
    apps = []

    def _start(fn, **kw):
        app = bootstrap(fn, **kw)
        apps.append(app)
        return app

    yield _start
    for app in apps:
        app.shutdown()


def test_use_input_rerenders_on_every_input_even_repeated(run_app):
    seen = []

    @component
    def Probe():
        ev = use_input(types="submit")
        seen.append(None if ev is None else (ev["type"], ev["value"], ev["seq"], ev["turn_id"]))
        return []

    app = run_app(Probe)
    assert seen == [None]
    ids = [app.invoke(t, wait=True, timeout=5) for t in ("yes", "yes")]
    assert seen[1:] == [("submit", "yes", 1, ids[0]), ("submit", "yes", 2, ids[1])]


def test_use_input_type_filter_and_all_types(run_app):
    seen = []

    @component
    def Probe():
        ev, text = use_input(), use_input(types=("text",))
        seen.append((ev and (ev["type"], ev["seq"]), text and (text["type"], text["seq"])))
        return []

    app = run_app(Probe)
    app.invoke("a", wait=True, timeout=5)
    # a turn emits text then submit; renders batch, the last value wins
    assert seen[-1] == (("submit", 2), ("text", 1))


def test_use_input_is_session_scoped():
    got = []

    @component
    def Probe():
        ev = use_input(types="submit")
        if ev is not None:
            got.append((hooks.get_service("session", lambda: None).id, ev["value"]))
        return []

    r = AppRunner(Probe, default_session=False)
    try:
        a, b = r.open_session("a"), r.open_session("b")
        a.invoke("to-a", wait=True, timeout=5)
        b.invoke("to-b", wait=True, timeout=5)
        assert got == [("a", "to-a"), ("b", "to-b")]
    finally:
        r.shutdown()


def test_keystroke_built_on_use_input_passes_turn(run_app):
    calls = []

    @component
    def App():
        return [Keystroke(key="k", on_submit=lambda text, turn: calls.append((text, turn.id)))]

    app = run_app(App)
    ids = [app.invoke(t, wait=True, timeout=5) for t in ("x", "x")]
    assert calls == [("x", ids[0]), ("x", ids[1])]


def test_keystroke_one_arg_callback_and_latest_props(run_app):
    calls = []

    @component
    def App():
        n, set_n = hooks.use_state(0)
        return [Keystroke(key="k", on_submit=lambda text: (calls.append((n, text)), set_n(n + 1)))]

    app = run_app(App)
    app.invoke("a", wait=True, timeout=5)
    app.invoke("b", wait=True, timeout=5)
    assert calls == [(0, "a"), (1, "b")]


# ---------------------------------------------------------------- events
def test_use_emit_publishes_with_component_path_and_turn(run_app):
    from pyreact.core import use_emit, use_on_turn

    @component
    def Child():
        emit = use_emit()
        use_on_turn(lambda turn: emit("probe", {"text": turn.text}))
        return []

    @component
    def App():
        return [Child(key="c")]

    app = run_app(App)
    tid = app.invoke("hi", wait=True, timeout=5)
    (ev,) = app.events.recent(None, ["probe"])
    assert (ev.payload, ev.source, ev.turn_id) == ({"text": "hi"}, "App/Child[c]", tid)


def test_use_emit_is_stable_and_overridable(run_app):
    from pyreact.core import use_emit

    emits, out = [], []

    @component
    def App():
        emit = use_emit()
        emits.append(emit)
        n, set_n = hooks.use_state(0)

        def eff():
            if n < 1:
                out.append(emit("x", turn_id="t-1", source="custom"))
                set_n(1)

        hooks.use_effect(eff, [n])
        return []

    run_app(App)
    assert len(emits) >= 2 and all(e is emits[0] for e in emits)
    assert (out[0].kind, out[0].turn_id, out[0].source) == ("x", "t-1", "custom")


def test_use_event_subscribes_in_effect_sees_latest_props_and_unsubscribes(run_app):
    from pyreact.core import use_event, use_on_turn

    got = []

    @component
    def Listener(label):
        use_event(["ping", "tool.*"], lambda ev: got.append((label, ev.kind)))
        return []

    @component
    def App():
        label, set_label = hooks.use_state("a")
        use_on_turn(lambda turn: set_label(turn.text))
        return [] if label == "hide" else [Listener(key="l", label=label)]

    app = run_app(App)
    bus = app.event_bus
    n_subs = len(bus._subs)
    bus.publish("ping")
    bus.publish("tool.start")
    bus.publish("other")
    assert got == [("a", "ping"), ("a", "tool.start")]

    app.invoke("b", wait=True, timeout=5)  # new props, same subscription
    assert len(bus._subs) == n_subs
    bus.publish("ping")
    assert got[-1] == ("b", "ping")

    app.invoke("hide", wait=True, timeout=5)  # unmount -> cleanup unsubscribes
    assert len(bus._subs) == n_subs - 1
    bus.publish("ping")
    assert got[-1] == ("b", "ping")


def test_message_publishes_through_use_emit(run_app):
    from pyreact.components.message import Message

    @component
    def App():
        return [Message(key="m", text="hello", sender="assistant", legacy=False)]

    app = run_app(App)
    (ev,) = app.events.recent(None, ["message"])
    assert ev.source == "App/Message[m]" and ev.payload["text"] == "hello"
