import asyncio
import time

import pytest

from pyreact.boot import bootstrap, current_session
from pyreact.core import EventBus, HookContext, component, hooks, runtime
from pyreact.core.runtime import run_renders
from pyreact.input.bus import InputBus
from pyreact.router import (
    Route,
    Router,
)


def _on_submit(fn):
    bus = hooks.get_service("input_bus", InputBus)
    hooks.use_effect(
        lambda: bus.subscribe(lambda e: e["type"] == "submit" and fn(e["value"])), []
    )


@pytest.fixture
def runners():
    made = []

    def _make(fn, **kw):
        r = bootstrap(fn, **kw)
        made.append(r)
        return r

    yield _make
    for r in made:
        r.shutdown()


# ---------------------------------------------------------------- services
def test_instance_services_are_per_session_root_and_class_services_are_global():
    a = HookContext("A", lambda **_: [], services={})
    b = HookContext("B", lambda **_: [], services={})
    child = HookContext("C", lambda **_: [], parent=a)
    bus_a = a.get_service("input_bus", InputBus)
    assert child.get_service("input_bus", InputBus) is bus_a
    assert b.get_service("input_bus", InputBus) is not bus_a
    assert HookContext.get_service("input_bus", InputBus) is not bus_a
    g = HookContext.get_service("thing", object)
    assert HookContext.get_service("thing", object) is g
    assert a.get_service("thing", object) is not g
    # Headless roots (no services dict) resolve the process-global services.
    headless = HookContext("H", lambda **_: [])
    assert headless.get_service("thing", object) is g


# ---------------------------------------------------------------- sessions
def test_two_sessions_do_not_share_state_or_inputs(runners):
    seen = []

    @component
    def App():
        items, set_items = hooks.use_state(())
        sess = hooks.get_service("session", lambda: None)
        _on_submit(lambda v: set_items(lambda xs: xs + (v,)))
        hooks.use_effect(lambda: seen.append((sess.id, items)), [items])
        return []

    app = runners(App, default_session=False)
    a, b = app.open_session("sess-a"), app.open_session("sess-b")
    assert a.input_bus is not b.input_bus
    assert a.nav_service is not b.nav_service
    assert isinstance(a.event_bus, EventBus) and a.event_bus is not b.event_bus
    assert a.services["event_bus"] is a.event_bus

    a.invoke("x", wait=True, timeout=5)
    a.invoke("y", wait=True, timeout=5)
    b.invoke("z", wait=True, timeout=5)
    assert ("sess-a", ("x", "y")) in seen
    assert ("sess-b", ("z",)) in seen
    assert not any(sid == "sess-b" and "x" in items for sid, items in seen)
    assert app.open_session("sess-a") is a  # resume
    assert {s.id for s in app.sessions()} == {"sess-a", "sess-b"}


def test_default_session_api_still_works(runners):
    got = []

    @component
    def App():
        _on_submit(got.append)
        return []

    app = runners(App)
    app.invoke("hi", wait=True, timeout=5)
    other = app.open_session("other-session")
    other.invoke("there", wait=True, timeout=5)
    assert got == ["hi", "there"]
    assert app.default_session.id == "default"
    assert app._root_ctx is app.default_session.root


def test_session_wait_ignores_other_sessions_work(runners):
    @component
    def App():
        v, setv = hooks.use_state(None)
        _on_submit(setv)

        async def work():
            if v == "slow":
                await asyncio.sleep(2.0)

        hooks.use_effect(work, [v])
        return []

    app = runners(App, default_session=False)
    a, b = app.open_session("slow-one"), app.open_session("fast-one")
    a.invoke("slow")
    time.sleep(0.05)
    t0 = time.time()
    b.invoke("fast", wait=True, timeout=5)
    assert time.time() - t0 < 1.0
    with pytest.raises(TimeoutError):
        a.wait_idle(timeout=0.2)


def test_close_session_unmounts_and_cancels_tasks(runners):
    cleaned, cancelled = [], []

    @component
    def App():
        sess = hooks.get_service("session", lambda: None)

        def eff():
            return lambda: cleaned.append(sess.id)

        async def forever():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.append(sess.id)
                raise

        hooks.use_effect(eff, [])
        hooks.use_effect(forever, [])
        return []

    app = runners(App, default_session=False)
    s = app.open_session("closing")
    keep = app.open_session("keeping")
    assert app.close_session("closing") is True
    app._submit(asyncio.sleep(0.05)).result()
    assert cleaned == ["closing"] and cancelled == ["closing"]
    assert s.closed and app.get_session("closing") is None
    assert app.close_session("closing") is False
    with pytest.raises(RuntimeError):
        s.invoke("x")
    assert not keep.closed


def test_close_idle_sessions(runners):
    @component
    def App():
        return []

    app = runners(App)
    s = app.open_session("old-session")
    app.open_session("new-session")
    s.last_active -= 100
    assert app.close_idle_sessions(50) == ["old-session"]
    assert app.get_session("new-session") is not None
    assert app.get_session("default") is not None


def test_spawned_tasks_and_current_session_are_attributed(runners):
    seen = []

    @component
    def App():
        def eff():
            async def inner():
                await asyncio.sleep(0)
                seen.append(current_session().id)

            runtime.spawn(inner())  # no explicit owner

        hooks.use_effect(eff, [])
        return []

    app = runners(App, default_session=False)
    s = app.open_session("attrib")
    s.wait_idle(timeout=5)
    assert seen == ["attrib"]


# ---------------------------------------------------------------- router
async def _flush():
    for _ in range(4):
        await asyncio.sleep(0)
        await run_renders()


def test_router_uses_session_nav_service(runners):
    paths = {}

    @component
    def Page(name):
        sess = hooks.get_service("session", lambda: None)
        paths[sess.id] = name
        return []

    @component
    def App():
        return [
            Router(
                children=[
                    Route(key="a", path="/a", children=[Page(key="pa", name="a")]),
                    Route(key="b", path="/b", children=[Page(key="pb", name="b")]),
                ]
            )
        ]

    app = runners(App, default_session=False)
    s1, s2 = app.open_session("router-1"), app.open_session("router-2")
    s1.nav("/a")
    s2.nav("/b")
    s1.wait_idle(timeout=5)
    s2.wait_idle(timeout=5)
    assert paths == {"router-1": "a", "router-2": "b"}
    assert s1.current_route()["path"] == "/a"
    assert s2.current_route()["path"] == "/b"


# ------------------------------------------------- turns / errors
def _turn_agent(log, delay):
    from pyreact.core import use_turn

    @component
    def Agent():
        sess = hooks.get_service("session", lambda: None)
        turn = use_turn()

        async def answer():
            if turn is None:
                return
            log.append(("start", sess.id, turn.id, runtime.current_turn_id()))
            await asyncio.sleep(delay)
            log.append(("answer", sess.id, turn.id, runtime.current_turn_id()))

        hooks.use_effect(answer, [turn])
        return []

    return Agent


def test_queue_in_one_session_does_not_block_another(runners):
    log = []
    app = runners(_turn_agent(log, 1.0), default_session=False)
    a, b = app.open_session("sess-a"), app.open_session("sess-b")
    ta1 = a.invoke("a1")
    ta2 = a.invoke("a2")  # queued behind a1 in session a
    time.sleep(0.1)
    t0 = time.time()
    tb = b.invoke("b1")
    # b's turn runs concurrently with a's (each session has its own queue);
    # it takes ~1s, not ~3s (a1 + a2 + b1).
    assert b.wait_turn(tb, timeout=5)["status"] == "done"
    assert time.time() - t0 < 1.6
    assert a.turns.result(ta2) is None  # a2 still waiting behind a1
    assert a.wait_turn(ta2, timeout=5)["status"] == "done"
    assert {ta1, ta2, tb} == {"sess-a/t1", "sess-a/t2", "sess-b/t1"}
    # Work is attributed to the right session's turn.
    for kind, sid, tid, current in log:
        assert tid.startswith(sid + "/") and current == tid, log


def test_interrupt_in_one_session_does_not_cancel_another(runners):
    log = []
    app = runners(_turn_agent(log, 0.5), default_session=False)
    a = app.open_session("int-a", turn_policy="interrupt")
    b = app.open_session("int-b")  # queue (runner default)
    ta1 = a.invoke("x")
    tb1 = b.invoke("y")
    time.sleep(0.1)
    ta2 = a.invoke("x2", wait=True, timeout=5)  # interrupts a's x only
    assert a.turns.result(ta1)["status"] == "interrupted"
    assert a.turns.result(ta2)["status"] == "done"
    assert b.wait_turn(tb1, timeout=5)["status"] == "done"
    answered = {(sid, tid) for k, sid, tid, _ in log if k == "answer"}
    assert ("int-b", tb1) in answered and ("int-a", ta1) not in answered


def test_hooks_usable_inside_effects_and_async_effects(runners):
    got = []

    @component
    def App():
        def eff():
            got.append(("sync", hooks.get_service("session", lambda: None).id))
            return lambda: got.append(
                ("cleanup", hooks.get_service("session", lambda: None).id)
            )

        async def aeff():
            await asyncio.sleep(0)
            got.append(("async", hooks.get_service("session", lambda: None).id))

        hooks.use_effect(eff, [])
        hooks.use_effect(aeff, [])
        return []

    app = runners(App, default_session=False)
    s = app.open_session("fx-sess")
    s.wait_idle(timeout=5)
    app.close_session("fx-sess")
    assert sorted(got) == [
        ("async", "fx-sess"),
        ("cleanup", "fx-sess"),
        ("sync", "fx-sess"),
    ]


def test_error_event_lands_on_the_owning_sessions_bus(runners, capsys):
    @component
    def App():
        v, setv = hooks.use_state(None)
        _on_submit(setv)

        def eff():
            if v == "boom":
                raise ValueError("kaboom")

        hooks.use_effect(eff, [v])
        return []

    app = runners(App, default_session=False)
    a, b = app.open_session("err-a"), app.open_session("err-b")
    errs = {"a": [], "b": []}
    a.event_bus.subscribe(errs["a"].append, ["error"])
    b.event_bus.subscribe(errs["b"].append, ["error"])
    ta = a.invoke("boom", wait=True, timeout=5)
    tb = b.invoke("fine", wait=True, timeout=5)
    assert len(errs["a"]) == 1 and errs["b"] == []
    ev = errs["a"][0]
    assert ev.turn_id == ta and "ValueError: kaboom" in ev.payload["error"]
    assert a.turns.result(ta)["status"] == "error"
    assert b.turns.result(tb)["status"] == "done"
    assert a.timeline.get(ta).errors
    assert "kaboom" in capsys.readouterr().err  # still printed
