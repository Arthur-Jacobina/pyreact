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
