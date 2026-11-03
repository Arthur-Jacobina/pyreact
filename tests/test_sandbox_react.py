"""SandboxProvider / use_sandbox: lifecycle, policies, harness boundary,
events and restart records (FakeDriver + LocalDriver, no Docker)."""

import asyncio

import pytest

from pyreact.boot import bootstrap
from pyreact.core import component, hooks, use_turn
from pyreact.core.harness import Blocked, HarnessProvider, install, middleware
from pyreact_sandbox import LocalDriver, SandboxSpec, Status
from pyreact_sandbox.react import SandboxProvider, use_sandbox, wait_closed
from pyreact_sandbox.store import FileSnapshotStore, MemorySnapshotStore
from pyreact_sandbox.testing import FakeDriver


@component
def Shell(sandbox_key="default", scope="session"):
    """Runs each turn's text as a command and replies with stdout."""
    sb = use_sandbox(key=sandbox_key, scope=scope)
    turn = use_turn()
    bus = hooks.get_service("event_bus", lambda: None)

    async def act():
        try:
            r = await sb.exec(turn.text)
            bus.publish("message", {"text": r.stdout.strip(), "sender": "sandbox"})
        except Blocked as exc:
            bus.publish("message", {"text": f"blocked: {exc.reason}", "sender": "sandbox"})

    hooks.use_effect(lambda: act() if turn else None, [turn and turn.id])
    return []


def boot(children_fn, **provider):
    @component
    def App():
        return [SandboxProvider(key="sbx", **provider, children=children_fn())]

    app = bootstrap(App, default_session=False)
    return app


def run_session(app, sid, lines, timeout=10):
    s = app.open_session(sid)
    events = []
    s.events.subscribe(events.append, ["sandbox.*", "message"], replay=True)
    for line in lines:
        s.invoke(line, wait=True, timeout=timeout)
    return s, events


def kinds(events):
    return [e.kind for e in events]


def texts(events):
    return [e.payload["text"] for e in events if e.kind == "message"]


def test_session_scope_exec_events_harness_and_destroy_on_close():
    drv = FakeDriver()
    seen = []

    async def spy(b, call_next):
        if b.kind == "sandbox.exec":
            seen.append((b.kind, b.name, b.session_id, b.turn_id, b.inputs["op"], b.component,
                         b.parent_id))
        return await call_next(b)

    uninstall = install(spy)
    app = boot(lambda: [Shell(key="a"), Shell(key="b", sandbox_key="other")], driver=drv)
    try:
        s, events = run_session(app, "s1", ["echo hello"])
        assert sorted(texts(events)) == ["hello", "hello"]
        assert len(drv.ops("create")) == 2  # one per key, created eagerly on mount
        ex = [e for e in events if e.kind == "sandbox.exec"]
        assert len(ex) == 2 and all(e.payload["exit_code"] == 0 for e in ex)
        assert {e.payload["key"] for e in ex} == {"default", "other"}
        assert all(e.turn_id and e.turn_id.endswith("t1") for e in ex)
        assert ex[0].payload["name"] == "echo" and ex[0].payload["stdout"] == "hello\n"
        sb_kinds = [k for k in seen if k[0] == "sandbox.exec"]
        assert len(sb_kinds) == 2
        kind, name, session, turn, op, comp, parent = sb_kinds[0]
        assert parent  # nested under the turn boundary
        assert (kind, name, session, op) == ("sandbox.exec", "echo", "s1", "exec")
        assert turn.endswith("t1") and comp.startswith("App/Shell[")
        app.close_session("s1")
        assert wait_closed(10)
    finally:
        app.shutdown()
        uninstall()
    assert len(drv.ops("destroy")) == 2 and drv.boxes == {}
    assert kinds(events).count("sandbox.created") == 2
    assert kinds(events).count("sandbox.destroyed") == 2


def test_two_components_share_a_session_sandbox_and_sessions_are_isolated():
    drv = FakeDriver()
    app = boot(lambda: [Shell(key="a"), Shell(key="b")], driver=drv)
    try:
        run_session(app, "s1", ["echo one"])
        run_session(app, "s2", ["echo two"])
        creates = drv.ops("create")
        assert len(creates) == 2  # one per session, shared by both components
        execs = drv.ops("exec")
        by_box = {}
        for _, box, cmd in execs:
            by_box.setdefault(box, []).append(cmd)
        assert sorted(by_box.values()) == [["echo one"] * 2, ["echo two"] * 2]
    finally:
        app.shutdown()
        wait_closed(10)
    assert drv.boxes == {}


def test_guard_middleware_blocks_and_rewrites_sandbox_exec():
    drv = FakeDriver()

    @middleware(kinds=["sandbox.*"])
    async def guard(b, call_next):
        if b.name == "rm":
            raise Blocked("no rm in the sandbox", boundary=b, by="guard")
        if b.name == "echo":
            b.inputs["cmd"] = b.inputs["cmd"].replace("secret", "*****")
        return await call_next(b)

    @component
    def App():
        return [HarnessProvider(key="h", middleware=[guard], children=[
            SandboxProvider(key="sbx", driver=drv, children=[Shell(key="sh")])])]

    app = bootstrap(App, default_session=False)
    try:
        s, events = run_session(app, "s", ["rm -rf /", "echo my secret"])
    finally:
        app.shutdown()
        wait_closed(10)
    assert texts(events) == ["blocked: no rm in the sandbox", "my *****"]
    assert [c[2] for c in drv.ops("exec")] == ["echo my *****"]  # rm never ran
    blocked = [e for e in events if e.kind == "sandbox.exec"][0]
    assert blocked.payload["error"].startswith("Blocked") and "exit_code" not in blocked.payload


def test_component_scope_closes_on_unmount():
    drv = FakeDriver()

    @component
    def Toggle():
        turn = use_turn()
        show = not (turn and turn.text == "hide")
        return [Shell(key="sh", scope="component")] if show else []

    app = boot(lambda: [Toggle(key="t")], driver=drv)
    try:
        s, events = run_session(app, "s", ["echo hi"])
        assert len(drv.boxes) == 1
        s.invoke("hide", wait=True, timeout=10)
        assert wait_closed(10)
        assert drv.boxes == {} and len(drv.ops("destroy")) == 1
    finally:
        app.shutdown()


def test_on_close_suspend_then_restart_reattaches():
    drv, store = FakeDriver(), MemorySnapshotStore()
    app = boot(lambda: [Shell(key="sh")], driver=drv, store=store, on_close="suspend")
    try:
        run_session(app, "s", ["echo first"])
    finally:
        app.shutdown()
        wait_closed(10)
    (sid,) = list(drv.boxes)
    assert drv.boxes[sid]["status"] == Status.SUSPENDED
    assert store.get("s/default")["sandbox_id"] == sid

    app = boot(lambda: [Shell(key="sh")], driver=drv, store=store, on_close="destroy")
    try:
        _, events = run_session(app, "s", ["echo again"])
    finally:
        app.shutdown()
        wait_closed(10)
    restored = [e for e in events if e.kind == "sandbox.restored"]
    assert restored and restored[0].payload["mode"] == "attach"
    assert restored[0].payload["sandbox_id"] == sid
    assert len(drv.ops("create")) == 1 and ("resume", sid) in drv.calls
    assert texts(events) == ["again"]
    assert drv.boxes == {} and store.get("s/default") is None  # destroyed, nothing to restore


@component
def Notes():
    """`write <text>` appends to notes.txt, `read` prints it."""
    sb = use_sandbox()
    turn = use_turn()
    bus = hooks.get_service("event_bus", lambda: None)

    async def act():
        cmd, _, arg = turn.text.partition(" ")
        if cmd == "write":
            try:
                old = await sb.read_text("notes.txt")
            except FileNotFoundError:
                old = ""
            await sb.write_file("notes.txt", old + arg + "\n")
        else:
            try:
                text = await sb.read_text("notes.txt")
            except FileNotFoundError:
                text = "(none)"
            bus.publish("message", {"text": text.strip(), "sender": "sandbox"})

    hooks.use_effect(lambda: act() if turn else None, [turn and turn.id])
    return []


@pytest.mark.parametrize("mode", ["snapshot", "autosnapshot"])
def test_snapshot_policies_survive_a_restart_with_local_driver(tmp_path, mode):
    drv = LocalDriver(root=tmp_path / "sbx")
    store = FileSnapshotStore(tmp_path / "sandboxes.json")
    opts = ({"on_close": "snapshot"} if mode == "snapshot"
            else {"on_close": "destroy", "autosnapshot": "turn"})
    app = boot(lambda: [Notes(key="n")], driver=drv, store=store, spec=SandboxSpec(), **opts)
    try:
        _, events1 = run_session(app, "s", ["write alpha", "write beta"])
    finally:
        app.shutdown()
        assert wait_closed(10)
    rec = store.get("s/default")
    assert rec["driver"] == "local" and rec["snapshot"] and rec["sandbox_id"] is None
    assert drv.list_sync() == []  # the sandbox itself is gone
    if mode == "autosnapshot":
        snaps = [e for e in events1 if e.kind == "sandbox.snapshot"]
        assert snaps and all(e.payload["auto"] for e in snaps)
        assert len(list((drv.root / "_snapshots").iterdir())) == 1  # keeps the latest only

    # "process restart": a new app, new driver object, same store file
    drv2 = LocalDriver(root=tmp_path / "sbx")
    app = boot(lambda: [Notes(key="n")], driver=drv2,
               store=FileSnapshotStore(tmp_path / "sandboxes.json"), **opts)
    try:
        _, events = run_session(app, "s", ["write gamma", "read"])
    finally:
        app.shutdown()
        wait_closed(10)
    assert texts(events) == ["alpha\nbeta\ngamma"]
    restored = [e for e in events if e.kind == "sandbox.restored"]
    assert restored[0].payload["mode"] == "snapshot"


def test_handle_direct_api_snapshot_restore_and_run_sync(tmp_path):
    drv = FakeDriver()
    box = {}

    @component
    def Holder():
        box["h"] = use_sandbox(eager=False)
        return []

    app = boot(lambda: [Holder(key="h")], driver=drv)
    try:
        s = app.open_session("s")
        s.wait_idle(timeout=5)
        h = box["h"]
        assert drv.ops("create") == []  # lazy: first use creates
        assert h.run_sync(lambda: h.status()) == Status.CREATING

        async def flow():
            await h.write_file("a.txt", "1")
            ref = await h.snapshot("one")
            await h.write_file("a.txt", "2")
            await h.restore(ref)
            return await h.read_text("a.txt"), ref

        text, ref = h.run_sync(flow)  # from a thread: runs on the app loop
        assert text == "1" and ref.label == "one"
        assert h.run_sync(lambda: h.exec("echo x")).stdout == "x\n"
        h.run_sync(lambda: h.suspend())
        assert h.run_sync(lambda: h.status()) == Status.SUSPENDED
        h.run_sync(lambda: h.resume())
        h.run_sync(lambda: h.destroy())
        assert h.closed and drv.boxes == {}
        with pytest.raises(Exception):
            h.run_sync(lambda: h.exec("echo nope"))
    finally:
        app.shutdown()
        wait_closed(10)


def test_close_waits_for_an_in_flight_create():
    drv = FakeDriver(create_delay=0.3)
    app = boot(lambda: [Shell(key="sh")], driver=drv)
    app.open_session("s")
    app.close_session("s")  # closes while the sandbox is still being created
    app.shutdown()
    assert wait_closed(10)
    assert len(drv.ops("create")) == 1 and drv.boxes == {}  # nothing leaked


def test_use_sandbox_validates_arguments():
    from pyreact_sandbox.react import SandboxHandle

    with pytest.raises(ValueError):
        SandboxHandle(key="k", store_key="k", driver=FakeDriver(), spec=SandboxSpec(),
                      store=None, on_close="explode", autosnapshot=None, bus=None)
    with pytest.raises(ValueError):
        SandboxHandle(key="k", store_key="k", driver=FakeDriver(), spec=SandboxSpec(),
                      store=None, on_close="destroy", autosnapshot="hourly", bus=None)
    asyncio.run(asyncio.sleep(0))


def test_on_close_policy_depends_on_the_close_reason():
    drv = FakeDriver()
    policy = {"shutdown": "suspend", "*": "destroy"}
    app = boot(lambda: [Shell(key="sh")], driver=drv, on_close=policy)
    try:
        run_session(app, "user-closed", ["echo a"])
        run_session(app, "still-open", ["echo b"])
        app.close_session("user-closed")  # reason "closed" -> destroy
        assert wait_closed(10)
        assert len(drv.boxes) == 1
    finally:
        app.shutdown()  # reason "shutdown" -> suspend (resume after a restart)
        assert wait_closed(10)
    (state,) = drv.boxes.values()
    assert state["status"] == Status.SUSPENDED
    assert len(drv.ops("destroy")) == 1 and len(drv.ops("suspend")) == 1

    from pyreact_sandbox.react import resolve_policy

    assert resolve_policy(policy, "idle") == "destroy"
    assert resolve_policy(lambda r: "keep" if r == "idle" else "destroy", "idle") == "keep"
    from pyreact_sandbox.react import SandboxHandle

    with pytest.raises(ValueError):
        SandboxHandle(key="k", store_key="k", driver=drv, spec=SandboxSpec(), store=None,
                      on_close={"*": "explode"}, autosnapshot=None, bus=None)


def test_autosnapshot_is_part_of_the_turn():
    drv, store = FakeDriver(), MemorySnapshotStore()
    app = boot(lambda: [Shell(key="sh")], driver=drv, store=store, autosnapshot="turn")
    try:
        s = app.open_session("s")
        order = []
        s.events.subscribe(lambda ev: order.append(ev.kind), ["sandbox.snapshot", "turn.end"])
        s.invoke("echo one", wait=True, timeout=10)
        # recorded before invoke(wait=True) returned, without wait_closed()
        assert store.get("s/default")["snapshot"]["label"] == "auto"
        assert order == ["sandbox.snapshot", "turn.end"]
        s.invoke("cat nothing-written", wait=True, timeout=10)  # exec -> snapshot again
        assert order[-2:] == ["sandbox.snapshot", "turn.end"]
    finally:
        app.shutdown()
        wait_closed(10)
