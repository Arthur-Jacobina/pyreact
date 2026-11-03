"""Durable dehydrates its log reference (use_dehydrate): roots carry it.

No participants: ``root.suspend()`` puts ``{path, log_id, leaf, ...}`` in the
snapshot (the Durable provider's slice) and ``hydrate_root`` reopens the
log at that leaf before the children's first render.
"""

import asyncio
import time
import warnings

import pytest

from pyreact.boot import DirectorySnapshots, hydrate_root
from pyreact.boot import create_root
from pyreact.core import HydrationWarning, component, hooks, use_turn
from pyreact_durable import (
    Durable,
    SessionLog,
    use_conversation,
    use_persistent_state,
)

SLOW = {"long": 30.0}
SLICE = "App/Durable[d]"


@component
def Chat():
    turn = use_turn()
    conv = use_conversation()
    count, set_count = use_persistent_state(0, key="count")
    hooks.get_service("seen", list).append((count, len(conv)))

    async def answer():
        if turn is None:
            return
        await asyncio.sleep(SLOW.get(turn.text, 0.01))
        conv.extend([{"role": "user", "content": turn.text},
                     {"role": "assistant", "content": turn.text.upper()}])
        set_count(lambda c: c + 1)

    hooks.use_effect(answer, [turn and turn.id])
    return []


def make_app(tmp_path):
    @component
    def App():
        return [Durable(key="d", log_factory=lambda sid: tmp_path / f"{sid}.jsonl",
                        children=[Chat(key="chat")])]

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


def _seen(root):
    return root.session.services["seen"]


def _store(tmp_path):
    return DirectorySnapshots(tmp_path / "state")


def test_snapshot_carries_the_log_and_hydrate_restores_state(roots, tmp_path):
    root = create_root(make_app(tmp_path), session_id="agent", store=_store(tmp_path))
    root.send("hello", wait=True, timeout=5)
    root.send("world", wait=True, timeout=5)
    snap = root.suspend()
    rec = snap["slices"][SLICE]["log"]
    assert rec["path"] == str((tmp_path / "agent.jsonl").resolve())
    assert rec["leaf"] and rec["log_id"] and rec["pending_turns"] == []
    assert snap["turns"]["next_index"] == 3

    # "new process": hydrate_root makes a new runner; same directory
    r = hydrate_root(make_app(tmp_path), session_id="agent", store=_store(tmp_path))
    roots.append(r)
    assert _seen(r)[0] == (2, 4)  # first render already has the state
    assert r.mismatches == [] and r.session.degraded == []
    assert r.send("again", wait=True, timeout=5) == "agent/t3"
    r.unmount()
    log = SessionLog.open(tmp_path / "agent.jsonl", readonly=True)
    try:
        conv = log.get_state("App/Chat[chat]", "history")
        assert [c["content"] for c in conv][-2:] == ["again", "AGAIN"]
        starts = [e["turn_id"] for e in log.events("turn.start")]
        assert starts == ["agent/t1", "agent/t2", "agent/t3"]
    finally:
        log.close()


def test_hydrate_puts_the_leaf_back(roots, tmp_path):
    root = create_root(make_app(tmp_path), session_id="leafy")
    root.send("one", wait=True, timeout=5)
    snap = root.suspend()
    # Someone writes to the log after the suspend (e.g. a CLI): the leaf moves.
    log = SessionLog.open(tmp_path / "leafy.jsonl")
    log.set_state("App/Chat[chat]", "count", 99)
    log.close()

    r = hydrate_root(make_app(tmp_path), session_id="leafy", snapshot=snap)
    roots.append(r)
    assert _seen(r)[0] == (1, 2)  # the suspended state, not 99


def test_live_dehydrate_reads_the_open_log(roots, tmp_path):
    root = create_root(make_app(tmp_path), session_id="live")
    roots.append(root)
    root.send("x", wait=True, timeout=5)
    rec = root.dehydrate()["slices"][SLICE]["log"]
    log = SessionLog.open(tmp_path / "live.jsonl", readonly=True)
    try:
        assert rec["leaf"] == log.leaf and rec["entries"] == len(log.entries)
    finally:
        log.close()


def test_carried_turn_reruns_after_hydration(roots, tmp_path):
    root = create_root(make_app(tmp_path), session_id="cut")
    root.send("long")
    time.sleep(0.2)
    snap = root.suspend(on_active="interrupt")
    assert [t["id"] for t in snap["turns"]["carry"]] == ["cut/t1"]

    SLOW["long"] = 0.01
    try:
        r = hydrate_root(make_app(tmp_path), session_id="cut", snapshot=snap,
                         wait=True, timeout=10)
    finally:
        SLOW["long"] = 30.0
    roots.append(r)
    starts = [(e.turn_id, e.payload.get("resumed")) for e in r.events.recent()
              if e.kind == "turn.start"]
    assert starts == [("cut/t1", True)]
    assert r.session.turns.pending_turns() == []
    assert _seen(r)[-1] == (1, 2)
    assert r.send("next", wait=True, timeout=5) == "cut/t2"


def _hydrate_quiet(*a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HydrationWarning)
        return hydrate_root(*a, **kw)


def test_missing_log_is_a_mismatch_and_starts_fresh(roots, tmp_path):
    root = create_root(make_app(tmp_path), session_id="gone")
    root.send("x", wait=True, timeout=5)
    snap = root.suspend()
    (tmp_path / "gone.jsonl").unlink()

    r = _hydrate_quiet(make_app(tmp_path), session_id="gone", snapshot=snap)
    roots.append(r)
    [m] = r.mismatches
    assert (m["kind"], m["path"], m["key"]) == ("error", SLICE, "log")
    assert "log missing" in m["message"]
    assert [(d["participant"], d["reason"]) for d in r.session.degraded] == [
        ("durable", "log missing")]
    assert _seen(r)[0] == (0, 0)


def test_replaced_log_is_a_mismatch(roots, tmp_path):
    root = create_root(make_app(tmp_path), session_id="swap")
    root.send("x", wait=True, timeout=5)
    snap = root.suspend()
    (tmp_path / "swap.jsonl").unlink()
    SessionLog.open(tmp_path / "swap.jsonl").close()  # a different log, same path

    r = _hydrate_quiet(make_app(tmp_path), session_id="swap", snapshot=snap)
    roots.append(r)
    assert [m["kind"] for m in r.mismatches] == ["error"]
    assert [d["reason"] for d in r.session.degraded] == ["log replaced"]


def test_hidden_activity_keeps_the_log_open(roots, tmp_path):
    from pyreact.core import Activity, use_imperative_handle
    from pyreact_durable import use_durable

    @component
    def Peek():
        log = use_durable()
        use_imperative_handle("log", lambda: log, [id(log)])
        return []

    @component
    def App():
        mode, set_mode = hooks.use_state("visible")
        use_imperative_handle("app", lambda: set_mode, [])
        return [Activity(mode=mode, children=[
            Durable(key="d", log=tmp_path / "hidden.jsonl", children=[Peek(key="p")])])]

    root = create_root(App, session_id="h")
    roots.append(root)
    log = root.ref("log")
    root.ref("app")("hidden")
    root.wait_idle(timeout=5)
    assert not log.closed  # state is kept while hidden
    root.ref("app")("visible")
    root.wait_idle(timeout=5)
    root.unmount()
    assert log.closed
