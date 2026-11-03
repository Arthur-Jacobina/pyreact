"""Suspend / hydrate on the root API: on_active policies, carried turns,
lifecycle events, snapshot errors and the runner's idle_policy="suspend" /
auto_resume (``tests/test_roots.py`` covers the rest of the Root API)."""

import asyncio
import time

import pytest

from pyreact.boot import AppRunner, MemorySnapshots, create_root, hydrate_root
from pyreact.core import SnapshotError, component, hooks, use_dehydrate, use_turn

DELAYS = {"slow": 0.4, "long": 30.0}


def _agent(log):
    @component
    def Agent():
        use_dehydrate("v", dehydrate=lambda: 1)
        turn = use_turn()

        async def answer():
            if turn is None:
                return
            log.append(("start", turn.id, turn.text, turn.resumed))
            await asyncio.sleep(DELAYS.get(turn.text, 0.01))
            log.append(("done", turn.id, turn.text))

        hooks.use_effect(answer, [turn])
        return []

    return Agent


@pytest.fixture
def roots():
    made = []
    yield made
    for r in made:
        try:
            r.unmount()
        except Exception:
            pass


@pytest.fixture
def runner():
    made = []

    def _make(fn, **kw):
        kw.setdefault("default_session", False)
        r = AppRunner(fn, **kw)
        made.append(r)
        return r

    yield _make
    for r in made:
        r.shutdown()


def _kinds(sess, out):
    sess.event_bus.subscribe(lambda ev: out.append((ev.kind, ev.payload)), ["session.*"])
    return out


def test_suspend_publishes_lifecycle_events_and_unmounts(roots):
    store = MemorySnapshots()
    root = create_root(_agent([]), session_id="a1", store=store)
    roots.append(root)
    root.send("one", wait=True, timeout=5)
    root.send("two", wait=True, timeout=5)
    sess = root.session
    events = _kinds(sess, [])
    snap = root.suspend()
    assert root.closed and sess.close_reason == "suspend"
    assert snap["turns"]["next_index"] == 3 and snap["turns"]["carry"] == []
    assert store.load("a1")["session_id"] == "a1"
    assert [k for k, _ in events] == ["session.suspending", "session.suspended",
                                      "session.closed"]
    suspended = events[1][1]
    assert suspended["turn_seq"] == 2 and suspended["carried_turns"] == []
    assert suspended["ref"] == "memory:a1"  # where the store put it
    assert events[-1][1]["reason"] == "suspend"


def test_wait_policy_lets_the_active_turn_finish(roots):
    log = []
    root = create_root(_agent(log), session_id="w")
    roots.append(root)
    root.send("slow")
    time.sleep(0.1)
    snap = root.suspend(on_active="wait")
    assert ("done", "w/t1", "slow") in log
    assert snap["turns"]["carry"] == []


def test_wait_with_timeout_falls_back_to_interrupt(roots):
    log = []
    root = create_root(_agent(log), session_id="to")
    roots.append(root)
    root.send("long")
    time.sleep(0.1)
    t0 = time.time()
    snap = root.suspend(timeout=0.2)
    assert time.time() - t0 < 5
    assert [t["id"] for t in snap["turns"]["carry"]] == ["to/t1"]
    assert ("done", "to/t1", "long") not in log


def test_hydrated_root_reruns_carried_turns_and_publishes_resumed(roots):
    log = []
    store = MemorySnapshots()
    root = create_root(_agent(log), session_id="c", store=store)
    roots.append(root)
    root.send("long")
    time.sleep(0.1)
    root.suspend(on_active="interrupt")
    DELAYS["long"] = 0.01  # the re-run is quick
    try:
        root2 = hydrate_root(_agent(log), session_id="c", store=store, wait=True, timeout=10)
    finally:
        DELAYS["long"] = 30.0
    roots.append(root2)
    assert ("start", "c/t1", "long", True) in log and ("done", "c/t1", "long") in log
    assert root2.session.resumed_from["session_id"] == "c"
    resumed = root2.events.recent(None, ["session.resumed"])
    assert resumed and resumed[0].payload["turns"] == ["c/t1"]
    assert resumed[0].payload["slices"] == 1
    assert store.load("c") is None  # consumed


def test_snapshot_errors():
    root = create_root(_agent([]), session_id="e")
    snap = root.suspend()
    with pytest.raises(SnapshotError, match="is for session 'e'"):
        hydrate_root(_agent([]), session_id="other", snapshot=snap)
    with pytest.raises(SnapshotError):
        hydrate_root(_agent([]), session_id="e", snapshot={**snap, "format": "pyreact.snapshot/99"})
    runner = AppRunner(_agent([]), default_session=False)
    try:
        hydrate_root(_agent([]), session_id="e", snapshot=snap, runner=runner)
        with pytest.raises(RuntimeError, match="already open"):
            hydrate_root(_agent([]), session_id="e", snapshot=snap, runner=runner)
    finally:
        runner.shutdown()


def test_idle_policy_suspend_and_auto_resume(runner):
    log = []
    store = MemorySnapshots()
    app = runner(_agent(log), idle_policy="suspend", snapshot_store=store)
    assert app.auto_resume
    s = app.open_session("idle")
    events = _kinds(s, [])
    s.invoke("hi", wait=True, timeout=5)
    s.last_active -= 100
    assert app.close_idle_sessions(10) == ["idle"]
    assert s.close_reason == "suspend" and app.get_session("idle") is None
    assert [p["reason"] for k, p in events if k == "session.suspending"] == ["idle"]
    assert store.load("idle")["slices"] == {"Agent": {"v": 1}}
    r = app.open_session("idle")  # auto_resume: hydrated from the store
    assert r is not s and r.resumed_from["session_id"] == "idle"
    assert store.load("idle") is None  # consumed
    assert r.invoke("back", wait=True, timeout=5) == "idle/t2"  # numbering continues
    with pytest.raises(ValueError):
        AppRunner(_agent(log), idle_policy="nap", default_session=False)


def test_async_idle_suspend_and_auto_resume(runner):
    log = []
    app = runner(_agent(log), idle_policy="suspend")

    async def main():
        s = await app.aopen_session("aidle")
        await s.ainvoke("one", wait=True)
        s.last_active -= 100
        assert await app.aclose_idle_sessions(10) == ["aidle"]
        r = await app.aopen_session("aidle")
        return await r.ainvoke("two", wait=True)

    assert asyncio.run(main()) == "aidle/t2"


def test_close_policy_does_not_snapshot_and_open_does_not_hydrate(runner):
    store = MemorySnapshots()
    app = runner(_agent([]), snapshot_store=store)  # idle_policy="close"
    assert not app.auto_resume
    s = app.open_session("plain")
    s.last_active -= 100
    assert app.close_idle_sessions(10) == ["plain"]
    assert s.close_reason == "idle" and store.list() == []
    r = app.open_session("plain")
    assert r.resumed_from is None
