"""pyreact_sandbox.contrib.durable: restart records in a pyreact_durable log."""

import subprocess
import sys
from pathlib import Path

from pyreact.boot import bootstrap
from pyreact.core import component, hooks, use_turn
from pyreact_durable import Durable, SessionLog
from pyreact_sandbox import LocalDriver, SandboxSpec
from pyreact_sandbox.contrib.durable import DurableSnapshotStore, use_durable_snapshot_store
from pyreact_sandbox.react import SandboxProvider, use_sandbox, wait_closed

ROOT = Path(__file__).resolve().parents[1]


@component
def Notes():
    store = use_durable_snapshot_store()
    sb = use_sandbox(store=store, autosnapshot="turn")
    turn = use_turn()
    bus = hooks.get_service("event_bus", lambda: None)

    async def act():
        cmd, _, arg = turn.text.partition(" ")
        if cmd == "write":
            await sb.exec(f"echo {arg} >> notes.txt")
        else:
            r = await sb.exec("cat notes.txt")
            bus.publish("message", {"text": r.stdout.strip(), "sender": "sandbox"})

    hooks.use_effect(lambda: act() if turn else None, [turn and turn.id])
    return []


def start(log, root):
    @component
    def App():
        return [Durable(key="durable", log=log, children=[
            SandboxProvider(key="sbx", driver=LocalDriver(root=root), spec=SandboxSpec(),
                            children=[Notes(key="notes")])])]

    return bootstrap(App)


def say(app, lines):
    out = []
    unsub = app.event_bus.subscribe(lambda ev: out.append(ev.payload["text"]), ["message"])
    for line in lines:
        app.invoke(line, wait=True, timeout=10)
    wait_closed(10)  # pending autosnapshots
    unsub()
    return out


def test_restart_restores_from_the_log_and_branches_follow_checkout(tmp_path):
    log_path, root = tmp_path / "agent.jsonl", tmp_path / "sbx"
    app = start(str(log_path), root)
    try:
        say(app, ["write one"])
    finally:
        app.shutdown()
        wait_closed(10)

    with SessionLog.open(log_path) as log:
        rec = log.get_state("pyreact_sandbox", "default/default")
        assert rec["driver"] == "local" and rec["snapshot"]["label"] == "auto"
        log.label("after-one")

    app = start(str(log_path), root)  # restart: restored from the recorded snapshot
    try:
        assert say(app, ["write two", "read"]) == ["one\ntwo"]
    finally:
        app.shutdown()
        wait_closed(10)

    with SessionLog.open(log_path) as log:  # time travel: back to "after-one"
        log.checkout("after-one")
    app = start(str(log_path), root)
    try:
        assert say(app, ["read"]) == ["one"]
    finally:
        app.shutdown()
        wait_closed(10)


def test_store_api_and_closed_log(tmp_path):
    log = SessionLog.open(tmp_path / "x.jsonl")
    store = DurableSnapshotStore(log)
    assert store.get("k") is None
    store.set("k", {"driver": "fake", "sandbox_id": "a"})
    assert store.get("k") == {"driver": "fake", "sandbox_id": "a"}
    store.delete("k")
    assert store.get("k") is None
    log.close()
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        store.set("k", {"x": 1})
    assert w and "log closed" in str(w[0].message)


def test_pyreact_sandbox_core_does_not_import_pyreact_durable():
    code = ("import sys, pyreact_sandbox, pyreact_sandbox.react, pyreact_sandbox.contrib\n"
            "assert not any(m.startswith('pyreact_durable') for m in sys.modules)\n")
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0
