"""The whole agent in one snapshot, no participants (ADR 0001).

Durable conversation + sandbox with a file + learned program + a turn cut
short: ``create_root`` -> ``root.suspend(on_active="interrupt")`` -> a new
runner -> ``hydrate_root`` -> the carried turn re-runs *after* hydration
completed (it sees every library's state) with its LM call replayed by
``durable_lm``, and numbering continues. LocalDriver, DummyLM (offline);
``scripts/e2e_roots.py`` does the same with Docker and a real LM.
"""

import asyncio
import time

import dspy
import pytest
from dspy.utils import DummyLM

from integrations import DSPyProvider, use_module
from integrations.learn import LearnedProgram, OptimizationResult, use_learned
from pyreact.boot import DirectorySnapshots, create_root, hydrate_root
from pyreact.core import component, hooks, use_on_turn
from pyreact_durable import Durable, use_conversation, use_durable_lm
from pyreact_sandbox import LocalDriver, SandboxProvider, SandboxSpec, use_sandbox, wait_closed

SLOW = {"long": 30.0}
LMS: dict = {}


class QA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _result(tmp_path, name, text):
    p = dspy.Predict(QA)
    p.signature = p.signature.with_instructions(text)
    path = str(tmp_path / f"{name}.json")
    OptimizationResult(program=p, optimizer="gepa", arch="predict").save(path)
    return path


@component
def Coder():
    conv = use_conversation()
    sb = use_sandbox()
    qa = use_module(QA, name="qa", program=use_learned("qa"))
    seen = hooks.get_service("seen", list)

    async def on_turn(turn):
        try:
            notes = await sb.read_text("notes.txt")
        except FileNotFoundError:
            notes = ""
        seen.append({"turn": turn.id, "resumed": turn.resumed, "messages": len(conv),
                     "notes": notes, "instructions": qa.signature.instructions})
        lm = LMS["durable"]

        def ask():
            with dspy.context(lm=lm):
                return qa(question=turn.text).answer

        answer = await asyncio.to_thread(ask)
        await asyncio.sleep(SLOW.get(turn.text.split()[0], 0.01))
        await sb.write_file("notes.txt", notes + answer + "\n")
        conv.extend([{"role": "user", "content": turn.text},
                     {"role": "assistant", "content": answer}])

    use_on_turn(on_turn)
    return []


@component(transparent=True)
def LMLayer(lm=None, children=None):
    LMS["durable"] = use_durable_lm(lm)
    return [DSPyProvider(key="dspy", lm=LMS["durable"], children=children or [])]


def make_app(tmp_path, lm, learned):
    drv = LocalDriver(root=tmp_path / "sbx")

    @component
    def App():
        return [Durable(key="d", log_factory=lambda sid: tmp_path / f"{sid}.jsonl", children=[
            LearnedProgram(key="qa", src=learned, children=[
                LMLayer(key="lm", lm=lm, children=[
                    SandboxProvider(key="sbx", driver=drv, spec=SandboxSpec(),
                                    children=[Coder(key="coder")])])])])]

    return App


@pytest.fixture
def cleanup():
    roots = []
    yield roots
    for r in roots:
        try:
            r.unmount()
        except Exception:
            pass
    wait_closed(10)


def test_whole_agent_round_trips_with_a_carried_turn(tmp_path, cleanup):
    one, two = _result(tmp_path, "one", "ONE"), _result(tmp_path, "two", "TWO")
    store = DirectorySnapshots(tmp_path / "state")
    lm1 = DummyLM([{"answer": "A1"}, {"answer": "A2"}, {"answer": "A3"}])
    root = create_root(make_app(tmp_path, lm1, one), session_id="agent", store=store)
    assert root.send("first", wait=True, timeout=10) == "agent/t1"
    # learn: swap the program (the LearnedProgram's state changes)
    root.session.services["learn.programs"]["App/LearnedProgram[qa]"].set("qa", two)
    root.wait_idle(timeout=5)
    assert root.send("second", wait=True, timeout=10) == "agent/t2"
    cut = root.send("long third")
    while not any(s["turn"] == cut for s in root.session.services["seen"]):
        time.sleep(0.02)
    time.sleep(0.3)  # its LM call answered; now it sleeps
    snap = root.suspend(on_active="interrupt")
    wait_closed(10)

    # one snapshot, every library's slice
    slices = snap["slices"]
    assert set(slices) == {"App/Durable[d]", "App/LearnedProgram[qa]",
                           "App/DSPyProvider[dspy]", "App/SandboxProvider[sbx]"}
    assert slices["App/Durable[d]"]["log"]["leaf"]
    assert slices["App/LearnedProgram[qa]"] == {"programs": {"qa": two}}
    assert slices["App/DSPyProvider[dspy]"] == {"compiled": []}
    assert slices["App/SandboxProvider[sbx]"]["sandboxes"]["default"]["snapshot"]
    assert snap["turns"]["next_index"] == 4
    assert [t["id"] for t in snap["turns"]["carry"]] == ["agent/t3"]

    SLOW["long"] = 0.01
    try:
        lm2 = DummyLM([{"answer": "A4"}])  # the re-run's call replays: not this LM
        r = hydrate_root(make_app(tmp_path, lm2, one), session_id="agent", store=store,
                         wait=True, timeout=20)
    finally:
        SLOW["long"] = 30.0
    cleanup.append(r)
    assert r.mismatches == [] and r.session.degraded == []
    rerun = r.session.services["seen"][0]
    # the carried turn ran after hydration completed: it sees the whole agent
    assert rerun == {"turn": "agent/t3", "resumed": True, "messages": 4,
                     "notes": "A1\nA2\n", "instructions": "TWO"}
    assert LMS["durable"].stats["replayed"] == 1 and LMS["durable"].stats["inner_calls"] == 0
    assert r.send("fourth", wait=True, timeout=10) == "agent/t4"  # numbering continues
    last = r.session.services["seen"][-1]
    assert last["notes"] == "A1\nA2\nA3\n" and last["messages"] == 6


def test_resume_turns_false_keeps_numbering_but_does_not_rerun(tmp_path, cleanup):
    one = _result(tmp_path, "one", "ONE")
    lm = DummyLM([{"answer": "A1"}, {"answer": "A2"}])
    root = create_root(make_app(tmp_path, lm, one), session_id="nr")
    root.send("first", wait=True, timeout=10)
    root.send("long second")
    time.sleep(0.3)
    snap = root.suspend(on_active="interrupt")
    wait_closed(10)
    r = hydrate_root(make_app(tmp_path, DummyLM([{"answer": "B"}]), one), session_id="nr",
                     snapshot=snap, resume_turns=False)
    cleanup.append(r)
    assert r.session.services.get("seen", []) == []
    assert r.send("third", wait=True, timeout=10) == "nr/t3"
