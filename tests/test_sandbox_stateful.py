"""Sandboxes dehydrate their snapshot ref (use_dehydrate): roots carry them.

LocalDriver, no Docker (``tests/test_sandbox_docker.py`` repeats the round
trip with Docker). The provider's SnapshotStore is in memory (lost with the
runner), so the only thing that brings a sandbox back is the snapshot.
"""

import shutil
import warnings

import pytest

from pyreact.boot import DirectorySnapshots
from pyreact.boot import create_root, hydrate_root
from pyreact.core import HydrationWarning, component, hooks, use_turn
from pyreact_sandbox import LocalDriver, SandboxSpec
from pyreact_sandbox.react import SandboxProvider, resolve_policy, use_sandbox, wait_closed

SLICE = "App/SandboxProvider[sbx]"


def _notes(**sandbox_kw):
    @component
    def Notes():
        """`write <text>` appends to notes.txt, `read` prints it."""
        sb = use_sandbox(**sandbox_kw)
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

    return Notes


def make_app(drv, on_close="destroy"):
    Notes = _notes()

    @component
    def App():
        return [SandboxProvider(key="sbx", driver=drv, spec=SandboxSpec(),
                                on_close=on_close, children=[Notes(key="n")])]

    return App


@pytest.fixture
def env(tmp_path):
    roots = []
    yield tmp_path, roots
    for r in roots:
        try:
            r.unmount()
        except Exception:
            pass
    wait_closed(10)


def _texts(root):
    return [e.payload["text"] for e in root.events.recent(None, ["message"])]


def _kinds(root, kind):
    return [e.payload for e in root.events.recent(None, [kind])]


def test_suspend_policy_never_destroys_implicitly():
    assert resolve_policy("destroy", "suspend") == "snapshot"
    assert resolve_policy({"shutdown": "suspend", "*": "destroy"}, "suspend") == "snapshot"
    assert resolve_policy({"suspend": "destroy"}, "suspend") == "destroy"  # explicit
    assert resolve_policy("suspend", "suspend") == "suspend"
    assert resolve_policy(lambda r: "keep" if r == "suspend" else "destroy", "suspend") == "keep"
    assert resolve_policy("destroy", "closed") == "destroy"


@pytest.mark.parametrize("on_close,mode", [
    ("destroy", "snapshot"),                              # the default: snapshot, not destroy
    ({"suspend": "suspend", "*": "destroy"}, "suspend"),  # pause, reattach
    ({"suspend": "keep"}, "keep"),
])
def test_sandbox_comes_back_after_suspend_in_a_new_runner(env, on_close, mode):
    tmp_path, roots = env
    drv = LocalDriver(root=tmp_path / "sbx")
    store = DirectorySnapshots(tmp_path / "state")
    root = create_root(make_app(drv, on_close), session_id="agent", store=store)
    root.send("write alpha", wait=True, timeout=10)
    snap = root.suspend()
    rec = snap["slices"][SLICE]["sandboxes"]["default"]
    assert rec["driver"] == "local" and rec["spec"]
    live = drv.list_sync()
    if mode == "snapshot":
        assert rec["snapshot"] and rec["sandbox_id"] is None
        assert live == []  # snapshotted then destroyed by the time suspend() returned
    else:
        assert rec["sandbox_id"] in live
        status = drv.attach_sync(rec["sandbox_id"]).status_sync()
        assert status == ("suspended" if mode == "suspend" else "running")
    wait_closed(10)

    r = hydrate_root(make_app(LocalDriver(root=tmp_path / "sbx"), on_close),
                     session_id="agent", store=store)
    roots.append(r)
    r.send("write beta", wait=True, timeout=10)
    r.send("read", wait=True, timeout=10)
    assert _texts(r) == ["alpha\nbeta"]
    restored = _kinds(r, "sandbox.restored")
    assert restored and restored[0]["mode"] == ("snapshot" if mode == "snapshot" else "attach")
    assert r.mismatches == [] and r.session.degraded == []
    snap2 = r.suspend()  # and again
    assert snap2["slices"][SLICE]["sandboxes"]["default"]


def test_missing_snapshot_boots_a_fresh_sandbox_and_is_a_mismatch(env):
    tmp_path, roots = env
    drv = LocalDriver(root=tmp_path / "sbx")
    root = create_root(make_app(drv), session_id="lost")
    root.send("write alpha", wait=True, timeout=10)
    snap = root.suspend()
    wait_closed(10)
    shutil.rmtree(tmp_path / "sbx" / "_snapshots")  # the snapshot is gone

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HydrationWarning)
        r = hydrate_root(make_app(drv), session_id="lost", snapshot=snap)
        roots.append(r)
        r.send("read", wait=True, timeout=10)
    assert _texts(r) == ["(none)"]  # a fresh sandbox
    [m] = r.mismatches
    assert (m["kind"], m["path"], m["key"], m["reason"]) == (
        "error", SLICE, "sandboxes", "snapshot missing")
    assert [(d["participant"], d["reason"], d["key"]) for d in r.session.degraded] == [
        ("sandbox", "snapshot missing", "default")]
    assert _kinds(r, "hydration.mismatch") and _kinds(r, "sandbox.created")


def test_explicit_destroy_on_suspend_carries_nothing(env):
    tmp_path, roots = env
    drv = LocalDriver(root=tmp_path / "sbx")
    root = create_root(make_app(drv, {"suspend": "destroy"}), session_id="d")
    root.send("write alpha", wait=True, timeout=10)
    snap = root.suspend()
    assert snap["slices"][SLICE]["sandboxes"] == {}
    assert drv.list_sync() == []


def test_component_scoped_sandbox_dehydrates_in_its_component(env):
    tmp_path, roots = env
    drv = LocalDriver(root=tmp_path / "sbx")
    Notes = _notes(scope="component", driver=drv)

    @component
    def App():
        return [Notes(key="n")]  # no provider

    root = create_root(App, session_id="c")
    root.send("write alpha", wait=True, timeout=10)
    snap = root.suspend()
    rec = snap["slices"]["App/Notes[n]"]["sandbox"]
    assert rec["snapshot"] and rec["sandbox_id"] is None
    wait_closed(10)
    r = hydrate_root(App, session_id="c", snapshot=snap)
    roots.append(r)
    r.send("read", wait=True, timeout=10)
    assert _texts(r) == ["alpha"] and r.mismatches == []


def test_live_dehydrate_keeps_the_sandbox_running(env):
    tmp_path, roots = env
    drv = LocalDriver(root=tmp_path / "sbx")
    root = create_root(make_app(drv), session_id="live")
    roots.append(root)
    root.send("write alpha", wait=True, timeout=10)
    rec = root.dehydrate()["slices"][SLICE]["sandboxes"]["default"]
    assert rec["sandbox_id"] in drv.list_sync()
    root.send("read", wait=True, timeout=10)
    assert _texts(root) == ["alpha"]
