"""create_root / hydrate_root / Root: send, refs, dehydrate, suspend, stores."""

import asyncio
import json
import time

import pytest

from pyreact.boot import (
    AppRunner,
    DirectorySnapshots,
    MemorySnapshots,
    SnapshotStore,
    SuspendError,
    create_root,
    hydrate_root,
)
from pyreact.boot.root import acreate_root, ahydrate_root
from pyreact.core import (
    SnapshotError,
    component,
    hooks,
    use_cleanup_reason,
    use_dehydrate,
    use_imperative_handle,
    use_on_turn,
)

DELAYS = {"long": 30.0}


def make_app(log):
    @component
    def Agent():
        start = use_dehydrate("seen", dehydrate=lambda: seen.current, initial=[])
        seen = hooks.use_ref(list(start))

        async def on_turn(turn):
            log.append(("start", turn.id, turn.text, turn.resumed))
            await asyncio.sleep(DELAYS.get(turn.text, 0.01))
            seen.current.append(turn.text)
            log.append(("done", turn.id))

        use_on_turn(on_turn)
        use_imperative_handle("agent", lambda: {"seen": seen.current}, [])

        def fx():
            log.append(("effect", "mount"))
            return lambda: log.append(("cleanup", use_cleanup_reason()))

        hooks.use_effect(fx, [])
        return []

    @component
    def App():
        return [Agent(key="agent")]

    return App


@pytest.fixture
def roots():
    made = []
    yield made
    for r in made:
        try:
            r.unmount()
        except Exception:
            pass


def test_create_root_send_ref_and_dehydrate(roots):
    log = []
    root = create_root(make_app(log), session_id="r1")
    roots.append(root)
    assert root.send("hello", wait=True, timeout=5) == "r1/t1"
    assert root.ref("agent")["seen"] == ["hello"]
    with pytest.raises(KeyError, match="known: \\['agent'\\]"):
        root.ref("nope")
    snap = root.dehydrate()
    assert snap["format"] == "pyreact.snapshot/1" and snap["session_id"] == "r1"
    assert snap["slices"] == {"App/Agent[agent]": {"seen": ["hello"]}}
    assert snap["turns"]["next_index"] == 2 and snap["turns"]["carry"] == []
    assert not root.closed  # dehydrate does not unmount
    kinds = [e.kind for e in root.events.recent()]
    assert "turn.end" in kinds
    root.unmount()
    assert root.closed and not root.runner.is_running  # owned runner stopped
    assert ("cleanup", "unmount") in log


def test_suspend_to_directory_and_hydrate_in_a_new_runner(tmp_path, roots):
    log = []
    store = DirectorySnapshots(tmp_path / "state")
    assert isinstance(store, SnapshotStore)
    root = create_root(make_app(log), session_id="coder/1", store=store)
    root.send("one", wait=True, timeout=5)
    events = []
    root.event_bus.subscribe(lambda ev: events.append(ev.kind), ["session.*"])
    snap = root.suspend()
    assert root.closed and ("cleanup", "suspend") in log
    assert ("cleanup", "unmount") not in log  # cleanups ran once, as "suspend"
    assert events == ["session.suspending", "session.suspended", "session.closed"]
    path = tmp_path / "state" / "coder%2F1.json"
    assert json.loads(path.read_text()) == snap and store.list() == ["coder/1"]

    log.clear()
    root2 = hydrate_root(make_app(log), session_id="coder/1", store=store)
    roots.append(root2)
    assert root2.ref("agent")["seen"] == ["one"]
    assert root2.mismatches == []
    assert store.list() == []  # consumed
    assert root2.send("two", wait=True, timeout=5) == "coder/1/t2"  # numbering continues
    assert root2.ref("agent")["seen"] == ["one", "two"]
    root2.suspend()  # back into the same store
    assert store.list() == ["coder/1"]


def test_interrupt_carries_the_cut_and_queued_turns(roots):
    log = []
    store = MemorySnapshots()
    root = create_root(make_app(log), session_id="i", store=store)
    root.send("long")
    root.send("queued")
    time.sleep(0.15)
    t0 = time.time()
    snap = root.suspend(on_active="interrupt")
    assert time.time() - t0 < 5
    assert [(t["id"], t["text"]) for t in snap["turns"]["carry"]] == [
        ("i/t1", "long"), ("i/t2", "queued")]
    DELAYS["long"] = 0.01
    try:
        root2 = hydrate_root(make_app(log), session_id="i", store=store, wait=True, timeout=10)
    finally:
        DELAYS["long"] = 30.0
    roots.append(root2)
    assert ("start", "i/t1", "long", True) in log
    assert root2.ref("agent")["seen"] == ["long", "queued"]
    assert root2.send("next", wait=True, timeout=5) == "i/t3"


def test_suspend_cleanups_run_before_dehydrate_and_async_ones_are_awaited(roots):
    @component
    def Box():
        state = hooks.use_ref("running")
        use_dehydrate("box", dehydrate=lambda: state.current)

        def fx():
            async def cleanup():
                await asyncio.sleep(0.05)
                state.current = f"paused ({use_cleanup_reason()})"
            return cleanup

        hooks.use_effect(fx, [])
        return []

    root = create_root(Box)
    snap = root.suspend()
    assert snap["slices"] == {"Box": {"box": "paused (suspend)"}}


class FailingStore(MemorySnapshots):
    def save(self, snapshot):
        raise OSError("disk full")


def test_failed_suspend_keeps_the_root_running(roots):
    log = []
    root = create_root(make_app(log), session_id="f", store=FailingStore())
    roots.append(root)
    root.send("one", wait=True, timeout=5)
    with pytest.raises(SuspendError, match="disk full"):
        root.suspend()
    assert not root.closed and not root.session.suspending
    root.wait_idle(timeout=5)
    assert log.count(("effect", "mount")) == 2  # effects re-ran after the reveal
    assert root.send("two", wait=True, timeout=5) == "f/t2"


def test_many_roots_on_one_runner(roots):
    runner = AppRunner(make_app([]), default_session=False)
    try:
        a = create_root(make_app([]), session_id="a", runner=runner)
        b = create_root(make_app([]), session_id="b", runner=runner, turn_policy="interrupt")
        a.send("for a", wait=True, timeout=5)
        b.send("for b", wait=True, timeout=5)
        assert a.ref("agent")["seen"] == ["for a"] and b.ref("agent")["seen"] == ["for b"]
        with pytest.raises(RuntimeError, match="already open"):
            create_root(make_app([]), session_id="a", runner=runner)
        snap = a.suspend()
        assert runner.is_running  # not owned by the root
        a2 = hydrate_root(make_app([]), session_id="a", snapshot=snap, runner=runner)
        assert a2.ref("agent")["seen"] == ["for a"]
        b.unmount()
        assert runner.is_running
    finally:
        runner.shutdown()


def test_hydration_mismatches_are_reported_not_fatal(roots):
    log = []
    snap = {"format": "pyreact.snapshot/1", "session_id": "m", "created_at": 0.0,
            "slices": {"App/Agent[agent]": {"seen": ["x"]}, "App/Old[o]": {"k": 1}},
            "turns": {"next_index": 5, "carry": []}, "versions": {}}
    with pytest.warns(RuntimeWarning, match="hydration mismatch"):
        root = hydrate_root(make_app(log), session_id="m", snapshot=snap)
    roots.append(root)
    assert [(m["kind"], m["path"]) for m in root.mismatches] == [("unclaimed", "App/Old[o]")]
    kinds = [e.kind for e in root.events.recent()]
    assert "hydration.mismatch" in kinds and "session.resumed" in kinds
    assert root.ref("agent")["seen"] == ["x"]
    assert root.send("hi", wait=True, timeout=5) == "m/t5"


def test_hydrate_root_errors():
    store = MemorySnapshots()
    with pytest.raises(KeyError):
        hydrate_root(make_app([]), session_id="ghost", store=store, if_missing="raise")
    with pytest.raises(ValueError, match="if_missing"):
        hydrate_root(make_app([]), session_id="ghost", store=store, if_missing="nope")
    with pytest.raises(ValueError, match="snapshot= or store="):
        hydrate_root(make_app([]), session_id="x")
    snap = {"format": "pyreact.snapshot/1", "session_id": "a", "slices": {}, "turns": {}}
    with pytest.raises(SnapshotError, match="for session 'a'"):
        hydrate_root(make_app([]), session_id="b", snapshot=snap)
    with pytest.raises(SnapshotError):
        hydrate_root(make_app([]), session_id="a", snapshot={"format": "nope"})


def test_async_root_api():
    async def main():
        root = await acreate_root(make_app([]), session_id="as")
        await root.asend("one", wait=True)
        snap = await root.asuspend()
        root2 = await ahydrate_root(make_app([]), session_id="as", snapshot=snap)
        tid = await root2.asend("two", wait=True)
        await root2.aunmount()
        return tid, root2.closed

    assert asyncio.run(main()) == ("as/t2", True)


class PendingStore:
    """A TurnStore that saw ``pending`` start but never end (a crashed run)."""

    def __init__(self, pending=()):
        self._pending, self.ended = list(pending), []

    def on_turn_start(self, turn):
        pass

    def on_turn_end(self, turn, result):
        self.ended.append((turn.id, result["status"]))
        self._pending = [t for t in self._pending if t.id != turn.id]

    def pending(self):
        return list(self._pending)

    def get(self, turn_id):
        return next((t for t in self._pending if t.id == turn_id), None)


def test_root_resume_pending_reruns_unfinished_turns_from_the_turn_store(roots):
    from pyreact.core import Turn

    log = []
    store = PendingStore([Turn(id="p/t3", index=3, text="unfinished", source="term")])

    @component
    def App():
        def attach():  # a store attached on mount (like a Durable provider)
            from pyreact.boot.session import current_session

            current_session().turns.configure(store=store)

        hooks.use_effect(attach, [])
        return [make_app(log)(key="inner")]

    root = create_root(App, session_id="p")
    roots.append(root)
    assert root.resume_pending(wait=True, timeout=5) == ["p/t3"]
    assert ("start", "p/t3", "unfinished", True) in log
    assert store.ended == [("p/t3", "done")]
    assert root.resume_pending() == []
    assert root.send("next", wait=True, timeout=5) == "p/t4"


def test_render_to_renders_message_events_and_replays_recent_ones(roots, capsys):
    from pyreact.observability import ConsoleSink

    @component
    def Hello():
        return []

    root = create_root(Hello, session_id="rt")
    roots.append(root)
    bus = root.event_bus
    bus.publish("message", {"role": "assistant", "text": "before attach"})
    bus.publish("llm.end", {"model": "x"})

    got = []
    unsub = root.render_to(got.append)
    assert [e.payload["text"] for e in got] == ["before attach"]  # replayed
    bus.publish("message", {"role": "assistant", "text": "live"})
    bus.publish("tool.start", {"name": "t"})
    assert [e.kind for e in got] == ["message", "message"]
    unsub()
    bus.publish("message", {"role": "assistant", "text": "after"})
    assert len(got) == 2

    off = root.render_to("console", replay=False)
    bus.publish("message", {"role": "assistant", "text": "to the console"})
    off()
    assert "to the console" in capsys.readouterr().out

    class Collect(ConsoleSink):
        def __init__(self):
            super().__init__(kinds=["llm.*"])
            self.seen = []

        def handle(self, ev):
            self.seen.append(ev.kind)

    sink = Collect()
    root.render_to(sink)()  # a Sink keeps its own filter
    assert sink.seen == ["llm.end"]
    with pytest.raises(ValueError):
        root.render_to("html")
    with pytest.raises(TypeError):
        root.render_to(42)


def test_hydrate_root_with_no_stored_snapshot_mounts_a_fresh_tree(roots):
    log = []
    store = MemorySnapshots()
    with pytest.warns(RuntimeWarning, match="no snapshot for session 'new'"):
        root = hydrate_root(make_app(log), session_id="new", store=store)
    roots.append(root)
    assert ("effect", "mount") in log and root.mismatches == []
    assert root.session.degraded == [
        {"participant": "snapshot", "reason": "no snapshot", "store": repr(store)}]
    [ev] = root.events.recent(None, ["session.resume.degraded"])
    assert ev.payload["reason"] == "no snapshot" and ev.payload["session_id"] == "new"
    assert root.send("first", wait=True, timeout=5) == "new/t1"
    root.suspend()  # into the store: the next hydrate_root finds it
    root2 = hydrate_root(make_app(log), session_id="new", store=store)
    roots.append(root2)
    assert root2.ref("agent")["seen"] == ["first"] and root2.session.degraded == []
