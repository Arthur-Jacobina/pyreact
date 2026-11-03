"""DurableParticipant: durable logs in suspend manifests, restored on resume."""

import asyncio
import time

import pytest

from pyreact.boot import AppRunner, DirectoryManifestStore
from pyreact.core import EventBus, component, hooks, use_turn
from pyreact_durable import (
    Durable,
    DurableParticipant,
    SessionLog,
    use_conversation,
    use_persistent_state,
)

SLOW = {"long": 30.0}


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
def runners():
    made = []

    def _make(tmp_path, **kw):
        r = AppRunner(make_app(tmp_path), default_session=False,
                      participants=[DurableParticipant],
                      manifest_store=DirectoryManifestStore(tmp_path / "manifests"), **kw)
        made.append(r)
        return r

    yield _make
    for r in made:
        r.shutdown()


def _seen(sess):
    return sess.services["seen"]


def test_manifest_records_log_and_resume_restores_state(runners, tmp_path):
    app = runners(tmp_path)
    s = app.open_session("agent")
    s.invoke("hello", wait=True, timeout=5)
    s.invoke("world", wait=True, timeout=5)
    m = s.suspend()
    frag = m["participants"]["durable"]
    [rec] = frag["logs"]
    assert rec["path"] == str((tmp_path / "agent.jsonl").resolve())
    assert rec["leaf"] and rec["log_id"] and rec["pending_turns"] == []
    app.shutdown()

    # "new process": a new runner, same directory
    app2 = runners(tmp_path)
    r = app2.resume("agent")
    assert _seen(r)[0] == (2, 4)  # first render already has the state
    assert r.degraded == []
    r.invoke("again", wait=True, timeout=5)
    log = SessionLog.open(tmp_path / "agent.jsonl", readonly=True)
    try:
        conv = log.get_state("App/Chat[chat]", "history")
        assert [c["content"] for c in conv][-2:] == ["again", "AGAIN"]
        starts = [e["turn_id"] for e in log.events("turn.start")]
        assert starts == ["agent/t1", "agent/t2", "agent/t3"]
    finally:
        log.close()


def test_resume_puts_the_leaf_back(runners, tmp_path):
    app = runners(tmp_path)
    s = app.open_session("leafy")
    s.invoke("one", wait=True, timeout=5)
    m = s.suspend()
    app.shutdown()
    # Someone writes to the log after the suspend (e.g. a CLI): the leaf moves.
    log = SessionLog.open(tmp_path / "leafy.jsonl")
    log.set_state("App/Chat[chat]", "count", 99)
    log.close()

    app2 = runners(tmp_path)
    r = app2.resume("leafy", m)
    assert _seen(r)[0] == (1, 2)  # the suspended state, not 99


def test_carried_turn_reruns_after_a_resume(runners, tmp_path):
    app = runners(tmp_path)
    s = app.open_session("cut")
    s.invoke("long")
    time.sleep(0.2)
    m = s.suspend(on_active="interrupt")
    assert [t["id"] for t in m["participants"]["turns"]["carry"]] == ["cut/t1"]
    app.shutdown()

    SLOW["long"] = 0.01
    try:
        app2 = runners(tmp_path)
        events = []
        app2.on_session(open=lambda sess: sess.event_bus.subscribe(
            lambda ev: events.append((ev.kind, ev.turn_id, ev.payload)), ["turn.*"]))
        r = app2.resume("cut", wait=True, timeout=10)
    finally:
        SLOW["long"] = 30.0
    starts = [(t, p.get("resumed")) for k, t, p in events if k == "turn.start"]
    assert starts == [("cut/t1", True)]
    assert r.turns.pending_turns() == []
    assert r.invoke("next", wait=True, timeout=5) == "cut/t2"


def test_missing_log_degrades_to_a_fresh_one(runners, tmp_path):
    app = runners(tmp_path)
    s = app.open_session("gone")
    s.invoke("x", wait=True, timeout=5)
    m = s.suspend()
    app.shutdown()
    (tmp_path / "gone.jsonl").unlink()

    app2 = runners(tmp_path)
    deg = []
    app2.on_session(open=lambda sess: sess.event_bus.subscribe(
        lambda ev: deg.append(ev.payload), ["session.resume.degraded"]))
    r = app2.resume("gone", m)
    assert [(d["participant"], d["reason"]) for d in deg] == [("durable", "log missing")]
    assert _seen(r)[0] == (0, 0)
    assert isinstance(r.event_bus, EventBus)


def test_replaced_log_degrades(runners, tmp_path):
    app = runners(tmp_path)
    s = app.open_session("swap")
    s.invoke("x", wait=True, timeout=5)
    m = s.suspend()
    app.shutdown()
    (tmp_path / "swap.jsonl").unlink()
    SessionLog.open(tmp_path / "swap.jsonl").close()  # a different log, same path

    app2 = runners(tmp_path)
    r = app2.resume("swap", m)
    assert [d["reason"] for d in r.degraded] == ["log replaced"]
