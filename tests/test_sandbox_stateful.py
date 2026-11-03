"""SandboxParticipant: sandboxes in suspend manifests (LocalDriver, no Docker).

The provider's SnapshotStore is in memory (lost with the runner), so the
only thing that brings a sandbox back is the manifest.
"""

import shutil

import pytest

from pyreact.boot import AppRunner, DirectoryManifestStore
from pyreact.core import component, hooks, use_turn
from pyreact_sandbox import LocalDriver, SandboxParticipant, SandboxSpec
from pyreact_sandbox.react import SandboxProvider, use_sandbox, wait_closed


@component
def Notes():
    """`write <text>` appends to notes.txt, `read` prints it."""
    sb = use_sandbox()
    turn = use_turn()
    bus = hooks.get_service("event_bus", lambda: None)

    async def act():
        cmd, _, arg = turn.text.partition(" ")
        try:
            old = await sb.read_text("notes.txt")
        except FileNotFoundError:
            old = ""
        if cmd == "write":
            await sb.write_file("notes.txt", old + arg + "\n")
        else:
            bus.publish("message", {"text": old.strip() or "(none)", "sender": "sandbox"})

    hooks.use_effect(lambda: act() if turn else None, [turn and turn.id])
    return []


@pytest.fixture
def make(tmp_path):
    made = []

    def _make(mode="snapshot"):
        drv = LocalDriver(root=tmp_path / "sbx")

        @component
        def App():
            return [SandboxProvider(key="sbx", driver=drv, spec=SandboxSpec(),
                                    on_close="destroy", children=[Notes(key="n")])]

        app = AppRunner(App, default_session=False,
                        participants=[lambda s: SandboxParticipant(s, mode=mode)],
                        manifest_store=DirectoryManifestStore(tmp_path / "manifests"))
        made.append(app)
        return app, drv

    yield _make
    for app in made:
        app.shutdown()
    wait_closed(10)


def _watch(app, kinds=("sandbox.*", "message", "session.resume.degraded")):
    events = []
    app.on_session(open=lambda s: s.event_bus.subscribe(events.append, list(kinds)))
    return events


def _texts(events):
    return [e.payload["text"] for e in events if e.kind == "message"]


@pytest.mark.parametrize("mode", ["snapshot", "suspend", "keep"])
def test_sandbox_comes_back_after_suspend_in_a_new_runner(make, mode):
    app, drv = make(mode)
    s = app.open_session("agent")
    s.invoke("write alpha", wait=True, timeout=10)
    m = s.suspend()
    [rec] = m["participants"]["sandbox"]["sandboxes"]
    assert m["participants"]["sandbox"]["mode"] == mode
    assert rec["store_key"] == "agent/default" and rec["driver"] == "local"
    live = drv.list_sync()
    if mode == "snapshot":
        assert rec["snapshot"] and rec["sandbox_id"] is None
        assert live == []  # destroyed by the time suspend() returned
    else:
        assert rec["sandbox_id"] in live
        status = drv.attach_sync(rec["sandbox_id"]).status_sync()
        assert status == ("suspended" if mode == "suspend" else "running")
    app.shutdown()
    wait_closed(10)

    app2, drv2 = make(mode)
    events = _watch(app2)
    r = app2.resume("agent")
    r.invoke("write beta", wait=True, timeout=10)
    r.invoke("read", wait=True, timeout=10)
    assert _texts(events) == ["alpha\nbeta"]
    restored = [e.payload for e in events if e.kind == "sandbox.restored"]
    assert restored and restored[0]["mode"] == ("snapshot" if mode == "snapshot" else "attach")
    assert r.degraded == []
    r.suspend()


def test_missing_snapshot_boots_a_fresh_sandbox_and_degrades(make, tmp_path):
    app, drv = make("snapshot")
    s = app.open_session("lost")
    s.invoke("write alpha", wait=True, timeout=10)
    m = s.suspend()
    app.shutdown()
    wait_closed(10)
    shutil.rmtree(tmp_path / "sbx" / "_snapshots")  # the snapshot is gone

    app2, _ = make("snapshot")
    events = _watch(app2)
    r = app2.resume("lost", m)
    r.invoke("read", wait=True, timeout=10)
    assert _texts(events) == ["(none)"]  # a fresh sandbox
    deg = [e.payload for e in events if e.kind == "session.resume.degraded"]
    assert [(d["participant"], d["reason"], d["key"]) for d in deg] == [
        ("sandbox", "snapshot missing", "default")]
    assert r.degraded[0]["reason"] == "snapshot missing"
    assert any(e.kind == "sandbox.created" for e in events)


def test_participant_validates_mode():
    with pytest.raises(ValueError):
        SandboxParticipant(mode="freeze")
