"""ErrorBoundary: React error boundaries for render / effect / task errors."""

import asyncio

from pyreact.boot import bootstrap
from pyreact.core import ErrorBoundary, component, hooks, use_turn
from pyreact.core.runtime import spawn


def _app(App, **kw):
    app = bootstrap(App, **kw)
    events = []
    app.event_bus.subscribe(events.append, ["error", "error.caught"])
    return app, events


@component
def Note(text, log=None):
    log = log if log is not None else []
    log.append(text)
    return []


def test_render_error_renders_fallback_and_is_handled(capfd):
    shown, errors = [], []

    @component
    def Widget():
        turn = use_turn()
        if turn is not None and "boom" in turn.text:
            raise ValueError(f"cannot render {turn.text!r}")
        shown.append(("widget", turn and turn.text))
        return []

    @component
    def App():
        return [ErrorBoundary(
            key="eb",
            fallback=lambda error: Note(text=f"fallback: {error}", log=shown),
            on_error=lambda error, info: errors.append((str(error), info)),
            children=[Widget(key="w")],
        )]

    app, events = _app(App)
    try:
        t1 = app.invoke("boom please", wait=True, timeout=5)
        assert app.wait_turn(t1, timeout=5)["status"] == "done"  # handled
        assert shown[-1] == "fallback: cannot render 'boom please'"
        # transparent like a provider: no ErrorBoundary segment in the path
        assert errors[0][1]["component"] == "App/Widget[w]"
        assert errors[0][1]["phase"] == "render" and errors[0][1]["turn_id"] == t1
        assert [e.kind for e in events] == ["error.caught"]
        assert events[0].payload["boundary"] == "App/ErrorBoundary[eb]"
        assert "Traceback" in events[0].payload["traceback"]

        app.invoke("fine", wait=True, timeout=5)  # reset_on="turn": re-mounted
        app.wait_idle(timeout=5)
        assert shown[-1] == ("widget", "fine")
    finally:
        app.shutdown()
    assert "Traceback" not in capfd.readouterr().err


def test_without_a_boundary_behavior_is_unchanged():
    @component
    def Widget():
        turn = use_turn()
        if turn is not None:
            raise ValueError("bad")
        return []

    @component
    def App():
        return [Widget(key="w")]

    app, events = _app(App)
    try:
        t1 = app.invoke("x", wait=True, timeout=5)
        assert app.wait_turn(t1, timeout=5)["status"] == "error"
        assert [e.kind for e in events] == ["error"]
    finally:
        app.shutdown()


def test_children_are_remounted_with_fresh_state():
    mounts = []

    @component
    def Counter():
        n, set_n = hooks.use_state(0)
        turn = use_turn()
        hooks.use_effect(lambda: mounts.append("mount"), [])
        if turn is not None and turn.text == "crash":
            raise RuntimeError("crash")
        hooks.use_effect(lambda: set_n(n + 1) if turn else None, [turn and turn.id])
        return [Note(key="n", text=f"n={n}")]

    @component
    def App():
        return [ErrorBoundary(key="eb", fallback=Note(text="down"), children=[Counter(key="c")])]

    app, events = _app(App)
    try:
        app.invoke("a", wait=True, timeout=5)
        app.invoke("crash", wait=True, timeout=5)
        app.invoke("b", wait=True, timeout=5)
        app.wait_idle(timeout=5)
    finally:
        app.shutdown()
    assert mounts == ["mount", "mount"]  # initial + after the reset


def test_effect_and_task_errors_are_caught():
    caught = []

    @component
    def SyncEffect():
        hooks.use_effect(lambda: 1 / 0, [])
        return []

    @component
    def AsyncEffect():
        async def fail():
            await asyncio.sleep(0)
            raise KeyError("async effect")

        hooks.use_effect(lambda: fail(), [])
        return []

    @component
    def Spawner():
        async def fail():
            raise LookupError("spawned task")

        hooks.use_effect(lambda: spawn(fail()) and None, [])
        return []

    @component
    def App():
        return [
            ErrorBoundary(key=name, fallback=None,
                          on_error=lambda e, info: caught.append((info["phase"], type(e).__name__)),
                          children=[C(key="c")])
            for name, C in (("a", SyncEffect), ("b", AsyncEffect), ("c", Spawner))
        ]

    app, events = _app(App)
    try:
        app.wait_idle(timeout=5)
    finally:
        app.shutdown()
    assert sorted(caught) == [("effect", "ZeroDivisionError"), ("task", "KeyError"),
                              ("task", "LookupError")]
    assert all(e.kind == "error.caught" for e in events)


def test_fallback_errors_and_the_boundary_own_errors_go_up():
    outer, inner = [], []

    @component
    def Bad():
        raise ValueError("child")

    @component
    def BadFallback(error):
        raise RuntimeError(f"fallback failed too ({error})")

    @component
    def App():
        return [ErrorBoundary(
            key="outer", fallback=Note(text="outer fallback", log=outer),
            on_error=lambda e, i: outer.append(str(e)),
            children=[ErrorBoundary(key="inner", fallback=BadFallback,
                                    on_error=lambda e, i: inner.append(str(e)),
                                    children=[Bad(key="bad")])])]

    app, _ = _app(App)
    try:
        app.wait_idle(timeout=5)
    finally:
        app.shutdown()
    assert inner == ["child"]
    assert outer == ["fallback failed too (child)", "outer fallback"]


def test_reset_on_props_callable_and_reset_function():
    seen = {}

    @component
    def Bad(fail):
        if fail:
            raise ValueError("bad")
        seen["ok"] = seen.get("ok", 0) + 1
        return []

    @component
    def Fallback(error, reset):
        seen["reset"] = reset
        seen["fallbacks"] = seen.get("fallbacks", 0) + 1
        return []

    @component
    def App():
        version, set_version = hooks.use_state(1)
        seen["set_version"] = set_version
        return [ErrorBoundary(key="eb", fallback=Fallback, reset_on="props",
                              reset_keys=[version],
                              children=[Bad(key="b", fail=version == 1)])]

    app, _ = _app(App)
    try:
        app.wait_idle(timeout=5)
        assert seen["fallbacks"] == 1 and "ok" not in seen
        app._submit(_call(seen["set_version"], 2)).result(5)
        app.wait_idle(timeout=5)
        assert seen["ok"] >= 1
    finally:
        app.shutdown()

    # callable reset_on + reset() from the fallback
    flags = {"fail": True, "resets": 0}

    @component
    def Flaky():
        if flags["fail"]:
            raise ValueError("x")
        return []

    @component
    def App2():
        return [ErrorBoundary(key="eb", fallback=Fallback,
                              reset_on=lambda e, info: flags["resets"] > 99,
                              children=[Flaky(key="f")])]

    seen.clear()
    app, events = _app(App2)
    try:
        app.wait_idle(timeout=5)
        assert seen["fallbacks"] == 1
        flags["fail"] = False
        app._submit(_call(seen["reset"])).result(5)
        app.wait_idle(timeout=5)
        assert seen["fallbacks"] == 1 and events == []  # re-mounted fine
    finally:
        app.shutdown()


async def _call(fn, *args):
    fn(*args)
