"""Session.suspend() / AppRunner.resume(): manifests, participants, turns, policies.

Deprecated wrappers (ADR 0001) over the root machinery
(``pyreact.boot.root.suspend_session`` / ``_hydrate_on_loop``), kept for one
release: their DeprecationWarnings are silenced here, except in the test
that checks them.
"""

import asyncio
import json
import time

import pytest

from pyreact.boot import (
    AppRunner,
    DirectoryManifestStore,
    ManifestError,
    MemoryManifestStore,
    SuspendError,
)
from pyreact.core import component, hooks, use_turn

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

DELAYS = {"slow": 0.4, "long": 30.0}


def _agent(log):
    @component
    def Agent():
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


class Notes:
    """A participant keeping a dict; resume seeds a service for the first render."""

    name = "notes"
    version = "7"

    def __init__(self, session):
        self.session = session
        self.data = {}
        self.calls = []

    async def suspend(self, session):
        self.calls.append("suspend")
        return {"data": dict(self.data)}

    async def resume(self, session, fragment):
        self.calls.append(("resume", session.root is None))
        self.data = dict(fragment["data"])
        session.services["notes"] = self.data

    async def resumed(self, session):
        self.calls.append(("resumed", session.root is not None))

    async def suspended(self, session):
        self.calls.append(("suspended", session.closed))


def test_suspend_writes_manifest_and_unmounts(runner):
    log = []
    store = MemoryManifestStore()
    app = runner(_agent(log), participants=[Notes], manifest_store=store)
    s = app.open_session("a1")
    s.participants.get("notes").data["k"] = 1
    s.invoke("one", wait=True, timeout=5)
    s.invoke("two", wait=True, timeout=5)
    events = _kinds(s, [])
    m = s.suspend()
    assert s.closed and s.close_reason == "suspend"
    assert app.get_session("a1") is None
    assert m["format"] == "pyreact.manifest/1" and m["session_id"] == "a1"
    assert m["turn_seq"] == 2 and m["reason"] == "manual"
    assert m["participants"]["turns"] == {"next_index": 3, "carry": []}
    assert m["participants"]["notes"] == {"data": {"k": 1}}
    assert m["versions"]["participants"] == {"turns": "1", "notes": "7"}
    assert m["ref"] == "memory:a1" and store.load("a1")["participants"]["notes"]
    json.dumps(m)
    assert [k for k, _ in events] == ["session.suspending", "session.suspended",
                                      "session.closed"]
    assert events[-1][1]["reason"] == "suspend"
    assert s.participants.get("notes").calls == ["suspend", ("suspended", True)]


def test_resume_hands_fragments_before_first_render_and_continues_turn_ids(runner):
    log, renders = [], []

    @component
    def App():
        renders.append(hooks.get_service("notes", lambda: None))
        return [_agent(log)(key="agent")]

    app = runner(App, participants=[Notes])
    s = app.open_session("a2")
    s.participants.get("notes").data["k"] = "v"
    s.invoke("one", wait=True, timeout=5)
    s.suspend()
    renders.clear()

    events = []
    app.on_session(open=lambda sess: _kinds(sess, events))
    r = app.resume("a2")
    assert renders[0] == {"k": "v"}  # restored before the first render
    p = r.participants.get("notes")
    assert p.calls == [("resume", True), ("resumed", True)]
    assert r.resumed_from["session_id"] == "a2" and r.degraded == []
    assert app.manifest_store.load("a2") is None  # consumed
    tid = r.invoke("two", wait=True, timeout=5)
    assert tid == "a2/t2"  # numbering continues
    kinds = [k for k, _ in events]
    assert kinds[:2] == ["session.opened", "session.resumed"]
    assert events[0][1]["resumed"] is True
    assert events[1][1]["participants"] == ["turns", "notes"]


def test_wait_policy_lets_the_active_turn_finish(runner):
    log = []
    app = runner(_agent(log))
    s = app.open_session("w")
    s.invoke("slow")
    time.sleep(0.1)
    m = s.suspend(on_active="wait")
    assert ("done", "w/t1", "slow") in log
    assert m["participants"]["turns"]["carry"] == []


def test_interrupt_policy_carries_the_cut_turn_and_queued_ones(runner):
    log = []
    app = runner(_agent(log))
    s = app.open_session("i")
    s.invoke("long")
    s.invoke("queued")
    time.sleep(0.1)
    t0 = time.time()
    m = s.suspend(on_active="interrupt")
    assert time.time() - t0 < 5
    carry = m["participants"]["turns"]["carry"]
    assert [(t["id"], t["text"]) for t in carry] == [("i/t1", "long"), ("i/t2", "queued")]
    assert ("done", "i/t1", "long") not in log

    DELAYS["long"] = 0.01  # the re-run is quick
    try:
        r = app.resume("i", m, wait=True, timeout=10)
    finally:
        DELAYS["long"] = 30.0
    assert ("start", "i/t1", "long", True) in log
    assert ("done", "i/t2", "queued") in log
    assert r.invoke("next", wait=True, timeout=5) == "i/t3"


def test_wait_with_timeout_falls_back_to_interrupt(runner):
    log = []
    app = runner(_agent(log))
    s = app.open_session("to")
    s.invoke("long")
    time.sleep(0.1)
    m = s.suspend(timeout=0.2)
    assert [t["id"] for t in m["participants"]["turns"]["carry"]] == ["to/t1"]


class Broken:
    name = "broken"

    def __init__(self):
        self.fail_suspend = True

    async def suspend(self, session):
        try:
            session.invoke("refused?")
        except RuntimeError as exc:
            self.refused = str(exc)
        if self.fail_suspend:
            raise OSError("disk full")
        return {"ok": True}

    async def resume(self, session, fragment):
        raise ValueError("cannot restore")


def test_failed_suspend_leaves_the_session_running(runner):
    log = []
    app = runner(_agent(log), participants=[Broken])
    s = app.open_session("f")
    events = _kinds(s, [])
    s.invoke("long")
    time.sleep(0.1)
    DELAYS["long"] = 0.01  # the running one keeps sleeping; its re-run will be quick
    try:
        with pytest.raises(SuspendError, match="disk full"):
            s.suspend(on_active="interrupt")
        assert not s.closed and not s.suspending
        assert "suspending" in s.participants.get("broken").refused
        assert [k for k, _ in events][-1] == "session.suspend.failed"
        assert app.manifest_store.load("f") is None
        # the cut turn is re-queued; new input works
        s.wait_turn("f/t1", timeout=5)
        tid = s.invoke("after", wait=True, timeout=5)
    finally:
        DELAYS["long"] = 30.0
    assert ("start", "f/t1", "long", True) in log and tid == "f/t2"


def test_resume_degrades_instead_of_failing(runner):
    log = []
    app = runner(_agent(log), participants=[Broken])
    s = app.open_session("d")
    s.participants.get("broken").fail_suspend = False
    m = s.suspend()
    m["participants"]["ghost"] = {"x": 1}  # nobody claims it
    events = []
    app.on_session(open=lambda sess: _kinds(sess, events))
    r = app.resume("d", m)
    assert {d["participant"] for d in r.degraded} == {"ghost", "broken"}
    deg = [p for k, p in events if k == "session.resume.degraded"]
    assert any(p["participant"] == "broken" and "cannot restore" in p["error"] for p in deg)
    resumed = [p for k, p in events if k == "session.resumed"][0]
    assert resumed["participants"] == ["turns"]
    assert sorted(resumed["degraded"]) == ["broken", "ghost"]
    assert r.invoke("x", wait=True, timeout=5) == "d/t1"


def test_directory_store_and_resume_by_path(runner, tmp_path):
    store = DirectoryManifestStore(tmp_path / "manifests")
    app = runner(_agent([]), manifest_store=store)
    s = app.open_session("user/42")
    s.invoke("hi", wait=True, timeout=5)
    m = s.suspend()
    path = tmp_path / "manifests" / "user%2F42.json"
    assert m["ref"] == str(path) and path.exists()
    assert store.list() == ["user/42"] and "user/42" in store
    r = app.resume("user/42", str(path), consume=False)
    assert path.exists()  # given explicitly: never consumed
    assert r.invoke("again", wait=True, timeout=5) == "user/42/t2"
    r.suspend()
    app.resume("user/42")  # from the store: consumed
    assert store.list() == []


def test_manifest_errors(runner):
    app = runner(_agent([]))
    s = app.open_session("e")
    m = s.suspend()
    with pytest.raises(ManifestError):
        app.resume("other", m)
    with pytest.raises(ManifestError):
        app.resume("e", {**m, "format": "pyreact.manifest/99"})
    with pytest.raises(KeyError):
        app.resume("never-suspended")
    app.resume("e", m)
    with pytest.raises(RuntimeError, match="already open"):
        app.resume("e", m)


def test_idle_policy_suspend_and_auto_resume(runner):
    log = []
    app = runner(_agent(log), idle_policy="suspend")
    s = app.open_session("idle")
    s.invoke("hi", wait=True, timeout=5)
    s.last_active -= 100
    assert app.close_idle_sessions(10) == ["idle"]
    assert s.close_reason == "suspend"
    assert app.manifest_store.load("idle")["reason"] == "idle"
    r = app.open_session("idle")  # auto_resume
    assert r is not s and r.resumed_from is not None
    assert r.invoke("back", wait=True, timeout=5) == "idle/t2"
    with pytest.raises(ValueError):
        AppRunner(_agent(log), idle_policy="nap", default_session=False)


def test_async_suspend_and_resume(runner):
    app = runner(_agent([]))

    async def main():
        s = await app.aopen_session("as")
        await s.ainvoke("one", wait=True)
        m = await s.asuspend()
        r = await app.aresume("as", m)
        return await r.ainvoke("two", wait=True)

    assert asyncio.run(main()) == "as/t2"


def test_legacy_suspend_and_resume_carry_use_dehydrate_slices(runner):
    from pyreact.core import use_dehydrate

    seen = []

    @component
    def Counter():
        start = use_dehydrate("n", dehydrate=lambda: n.current, initial=0)
        n = hooks.use_ref(start)
        turn = use_turn()
        hooks.use_effect(lambda: setattr(n, "current", n.current + (turn is not None)),
                         [turn and turn.id])
        seen.append(start)
        return []

    app = runner(Counter)
    s = app.open_session("legacy")
    s.invoke("a", wait=True, timeout=5)
    s.invoke("b", wait=True, timeout=5)
    m = s.suspend()
    assert m["slices"] == {"Counter": {"n": 2}}
    seen.clear()
    app.resume("legacy", m)
    assert seen[0] == 2


@pytest.mark.filterwarnings("default::DeprecationWarning")
def test_legacy_api_warns_and_runs_on_the_root_machinery(runner):
    from pyreact.core import use_dehydrate

    @component
    def Box():
        use_dehydrate("v", dehydrate=lambda: 7)
        return []

    with pytest.warns(DeprecationWarning, match="0001-look-like-react") as w:
        app = runner(Box, participants=[Notes])
    assert "participants=" in str(w[0].message)
    s = app.open_session("dep")
    events = _kinds(s, [])
    with pytest.warns(DeprecationWarning, match=r"Session.suspend\(\) is deprecated"):
        m = s.suspend()
    assert m["slices"] == {"Box": {"v": 7}}
    suspended = [p for k, p in events if k == "session.suspended"][0]
    # the root's payload plus the legacy participant list
    assert suspended["slices"] == 1 and suspended["participants"] == ["turns", "notes"]
    with pytest.warns(DeprecationWarning, match=r"AppRunner.resume\(\) is deprecated"):
        r = app.resume("dep", m)
    assert r.root.children == [] and r.resumed_from is m
    with pytest.warns(DeprecationWarning, match="add_participant"):
        app.add_participant(Notes)
