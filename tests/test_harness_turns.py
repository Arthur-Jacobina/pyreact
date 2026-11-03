"""Turns flow through the harness (``"turn"`` boundaries)."""

import asyncio
import threading
import time

import pytest

from pyreact.boot import bootstrap
from pyreact.core import component, hooks, runtime, use_turn
from pyreact.core.harness import Blocked, install, middleware


@pytest.fixture
def run_app():
    apps, uninstalls = [], []

    def _start(fn, *mws, **kw):
        for mw in mws:
            uninstalls.append(install(mw))
        app = bootstrap(fn, **kw)
        apps.append(app)
        return app

    yield _start
    for app in apps:
        app.shutdown()
    for un in uninstalls:
        un()


def _agent(log, delay=0.05):
    @component
    def Agent():
        turn = use_turn()

        async def answer():
            if turn is None:
                return
            await asyncio.sleep(delay)
            log.append(("answer", turn.text, runtime.current_turn_id()))

        hooks.use_effect(answer, [turn])
        return []

    return Agent


def _events(app):
    evs, lock = [], threading.Lock()

    def _on(ev):
        with lock:
            evs.append(ev)

    app.event_bus.subscribe(_on, ["turn.*"])
    return evs


def test_turn_boundary_wraps_the_whole_lifecycle(run_app):
    log, seen = [], []

    @middleware(kinds=["turn"])
    async def spy(b, call_next):
        seen.append(("before", b.name, dict(b.inputs), dict(b.attrs), b.turn_id,
                     [x[0] for x in log]))
        out = await call_next(b)
        seen.append(("after", dict(b.attrs), [x[0] for x in log]))
        return out

    app = run_app(_agent(log), spy)
    evs = _events(app)
    tid = app.invoke("hello", wait=True, timeout=5)
    before, after = seen
    assert before[1:5] == ("turn", {"text": "hello"},
                           {"turn_id": tid, "index": 1, "source": "term"}, tid)
    assert before[5] == []  # nothing ran yet
    assert after[1]["status"] == "done" and after[1]["duration"] >= 0.05
    assert after[1]["turn_id"] == tid
    assert after[2] == ["answer"]  # settled inside the boundary
    assert [e.kind for e in evs] == ["turn.start", "turn.end"]


def test_turn_middleware_keeps_queue_order(run_app):
    log, spans, active = [], [], []

    async def timing(b, call_next):
        if b.kind != "turn":
            return await call_next(b)
        active.append(b.turn_id)
        assert len(active) == 1  # turns never overlap under "queue"
        try:
            return await call_next(b)
        finally:
            active.remove(b.turn_id)
            spans.append((b.inputs["text"], b.attrs["status"]))

    app = run_app(_agent(log), timing)
    t1, t2 = app.invoke("q1"), app.invoke("q2")
    assert app.wait_turn(t2, timeout=5)["status"] == "done"
    assert spans == [("q1", "done"), ("q2", "done")]
    assert [x[2] for x in log] == [t1, t2]


def test_interrupt_policy_still_works_through_the_harness(run_app):
    log, statuses = [], []

    async def passthrough(b, call_next):
        out = await call_next(b)
        if b.kind == "turn":
            statuses.append(b.attrs["status"])
        return out

    app = run_app(_agent(log, delay=0.5), passthrough, turn_policy="interrupt")
    t1 = app.invoke("q1")
    time.sleep(0.1)
    t2 = app.invoke("q2", wait=True, timeout=5)
    assert app.turns.result(t1)["status"] == "interrupted"
    assert app.turns.result(t2)["status"] == "done"
    assert statuses == ["interrupted", "done"]
    assert [x[1] for x in log] == ["q2"]


def test_blocked_turn_never_starts(run_app):
    log = []

    @middleware(kinds=["turn"], name="budget")
    async def budget(b, call_next):
        if "expensive" in b.inputs["text"]:
            raise Blocked("over budget")
        return await call_next(b)

    app = run_app(_agent(log), budget)
    evs = _events(app)
    t1 = app.invoke("expensive question", wait=True, timeout=5)
    t2 = app.invoke("cheap", wait=True, timeout=5)
    assert app.turns.result(t1)["status"] == "blocked"
    assert app.turns.result(t2)["status"] == "done"
    assert [x[1] for x in log] == ["cheap"]
    kinds = [(e.kind, e.turn_id) for e in evs]
    assert kinds == [("turn.blocked", t1), ("turn.end", t1),
                     ("turn.start", t2), ("turn.end", t2)]
    blocked = evs[0].payload
    assert blocked["by"] == "budget" and blocked["message"] == "turn 'turn' blocked by budget: over budget"
