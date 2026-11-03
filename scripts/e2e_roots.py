"""End-to-end: suspend a whole coding agent with a root, hydrate it in a new process.

The agent is its tree's state, and every library dehydrates its own slice
(ADR 0001) -- no participants, no runner configuration:

- ``Durable``: the conversation / persistent state (log + leaf), every LM
  call journaled by ``durable_lm``;
- ``SandboxProvider``: a Docker sandbox with the files the agent wrote
  (snapshotted on suspend, restored on hydrate);
- ``LearnedProgram``: a learned prompt swapped in at runtime;
- the turn manager: a turn cut short by the suspend, re-run after hydration.

Process 1 (``--phase work``): ``create_root`` -> seed ``TASK.md`` -> Q1 (the
agent writes and runs ``fib.py``) -> "learn" (swap a better prompt into the
``LearnedProgram``, no LLM call) -> Q2, and once it made a tool call,
``root.suspend(on_active="interrupt")``; the snapshot is printed.

Process 2 (``--phase resume``): ``hydrate_root`` -> the carried turn re-runs
by itself (its answered LM calls replayed from the log) using the files of
process 1 and the learned prompt; then the root is unmounted.

    PYREACT_EXAMPLE_LM=claude uv run python scripts/e2e_roots.py            # both phases
    uv run python scripts/e2e_roots.py --cleanup

LM calls: ``MAX_ITERS=4`` ReAct steps + 1 extract per question (<= 10 real
calls in all; replayed calls are free). Containers and snapshot images are
labelled and removed with ``pyreact_sandbox.gc`` at the end.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dspy  # noqa: E402

import pyreact_sandbox as ps  # noqa: E402
from examples._lm import example_lm  # noqa: E402
from integrations import DSPyProvider, use_agent, use_dspy_env  # noqa: E402
from integrations.learn import (  # noqa: E402
    LearnedProgram,
    OptimizationResult,
    use_learned,
    use_learned_registry,
)
from pyreact.boot import DirectorySnapshots, create_root, hydrate_root  # noqa: E402
from pyreact.core import component, hooks, use_emit, use_imperative_handle, use_turn  # noqa: E402
from pyreact_durable import Durable, SessionLog, use_conversation, use_durable_lm  # noqa: E402
from pyreact_sandbox import SandboxProvider, SandboxSpec, sandbox_tools, use_sandbox  # noqa: E402

SESSION = "coder-1"
LABELS = {"pyreact.e2e": "roots"}
SPEC = SandboxSpec(image="python:3.12-slim", cpu=1, memory="512m", network=False,
                   labels=LABELS)
MAX_ITERS = 4

TASK = """# Task
Maintain fib.py: a script that prints Fibonacci numbers (fib(1) = fib(2) = 1).
Print each result labelled, e.g. `fib(30) = 832040`.
"""
Q1 = ("Read TASK.md, then write fib.py that prints the 30th Fibonacci number, run it, "
      "and tell me the result.")
Q2 = ("Extend the fib.py you wrote earlier so it also prints the 40th Fibonacci number, "
      "run it, and tell me both numbers.")
LESSON = (" Lesson learned: before changing a file from an earlier turn, read it with "
          "read_file first; keep scripts short.")


class Coder(dspy.Signature):
    """You are a careful engineer with a private Linux sandbox (Python 3, no network).
    Answer by writing a small Python script with write_file, running it with run_shell
    (e.g. `python3 fib.py`), and reporting what it printed. Files from earlier turns are
    still in the sandbox; use the conversation history."""

    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField(desc="the final answer, one or two sentences")


LMS: dict = {}


@component(transparent=True)
def LMLayer(lm=None, children=None):
    dlm = use_durable_lm(lm)  # every LM call journaled; a re-run replays them
    LMS["durable"] = dlm
    return [DSPyProvider(key="dspy", lm=dlm, children=children or [])]


@component
def CodingAgent():
    conv = use_conversation()
    sb = use_sandbox(ref="sandbox")  # root.ref("sandbox")
    tools = hooks.use_memo(lambda: sandbox_tools(sb, timeout=30), [id(sb)])
    agent = use_agent(Coder, tools=tools, max_iters=MAX_ITERS, name="coder",
                      program=use_learned("coder"))
    env = use_dspy_env()
    registry = use_learned_registry()
    use_imperative_handle("coder", lambda: {
        "module": env.caches.get(("ReAct", "Coder", "coder")), "learned": registry,
        "messages": len(conv)}, [len(conv)])
    turn = use_turn()
    emit = use_emit()

    def mounted():
        emit("message", {"text": f"(mounted with {len(conv)} messages in the conversation)",
                         "sender": "system"})

    hooks.use_effect(mounted, [])

    def ask():
        if turn is not None:
            agent.run(question=turn.text, history=conv.as_dspy_history())

    hooks.use_effect(ask, [turn and turn.id])

    def answered():
        if not agent.ver or turn is None:
            return
        if agent.error is not None:
            emit("message", {"text": f"error: {agent.error!r}", "sender": "system"})
            return
        conv.extend([{"role": "user", "content": turn.text},
                     {"role": "assistant", "content": agent.result.answer}])
        emit("message", {"text": agent.result.answer, "sender": "assistant"})

    hooks.use_effect(answered, [agent.ver])
    return []


def make_app(state: Path, driver, lm):
    logs = state / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    @component
    def App():
        return [Durable(key="durable", log_factory=lambda sid: logs / f"{sid}.jsonl", children=[
            LearnedProgram(key="coder", src=None, children=[
                LMLayer(key="lm", lm=lm, children=[
                    SandboxProvider(key="sandbox", driver=driver, spec=SPEC, on_close="destroy",
                                    children=[CodingAgent(key="coder")]),
                ]),
            ]),
        ])]

    return App


# ---------------------------------------------------------------------------
# Output (an event-bus subscription: devtools, not rendering)
# ---------------------------------------------------------------------------
def show(ev) -> None:
    p = ev.payload
    if ev.kind == "message":
        print(f"\n{p.get('sender')}> {p.get('text')}", flush=True)
    elif ev.kind == "tool.start":
        print(f"  [tool] {p.get('tool')}({str(p.get('args'))[:110]})", flush=True)
    elif ev.kind == "sandbox.exec":
        out = (p.get("stdout") or p.get("stderr") or p.get("error") or "").strip()
        code = p.get("exit_code")
        print(f"  [sandbox.exec {p.get('op')}] {p.get('name')} -> "
              f"{'' if code is None else f'exit {code} '}{out[:120]!r}", flush=True)
    elif ev.kind in ("sandbox.created", "sandbox.restored", "sandbox.snapshot",
                     "sandbox.destroyed", "sandbox.suspended", "sandbox.error"):
        keep = {k: v for k, v in p.items() if k in ("sandbox_id", "mode", "snapshot", "error",
                                                   "policy", "label")}
        print(f"  [{ev.kind}] {keep}", flush=True)
    elif ev.kind == "turn.start":
        print(f"  [turn.start] {ev.turn_id}{' (resumed)' if p.get('resumed') else ''}",
              flush=True)
    elif ev.kind == "turn.end":
        print(f"  [turn.end] {ev.turn_id} {p.get('status')}", flush=True)
    elif ev.kind.startswith("session.") or ev.kind == "hydration.mismatch":
        keep = {k: v for k, v in p.items() if k != "session_id"}
        print(f"  [{ev.kind}] {keep}", flush=True)


KINDS = ["message", "tool.start", "sandbox.*", "turn.start", "turn.end", "session.*",
         "hydration.mismatch"]


def _short(obj, n=120):
    if isinstance(obj, dict):
        return {k: _short(v, n) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_short(v, n) for v in obj]
    if isinstance(obj, str) and len(obj) > n:
        return obj[:n] + "…"
    return obj


def lm_stats() -> str:
    st = getattr(LMS.get("durable"), "stats", None) or {}
    return (f"LM calls this process: {st.get('inner_calls', 0)} real, "
            f"{st.get('replayed', 0)} replayed from the log")


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def learn(root, state: Path) -> None:
    """Stand-in for an optimizer: a better prompt saved as a learn result and
    swapped into the LearnedProgram (its state: a re-render applies it)."""
    coder = root.ref("coder")
    learned = copy.deepcopy(coder["module"])
    for name, pred in learned.named_predictors():
        if name.startswith("react"):
            pred.signature = pred.signature.with_instructions(
                pred.signature.instructions + LESSON)
    path = OptimizationResult(program=learned, optimizer="manual", arch="react").save(
        str(state / "learned" / "coder.json"))
    coder["learned"].set("coder", path)
    root.wait_idle(timeout=10)
    print(f"\n[learn] swapped a learned prompt into LearnedProgram 'coder' ({path})",
          flush=True)


def phase_work(state: Path, driver, lm) -> None:
    store = DirectorySnapshots(state / "snapshots")
    root = create_root(make_app(state, driver, lm), session_id=SESSION, store=store)
    root.event_bus.subscribe(show, KINDS)
    try:
        sb = root.ref("sandbox")
        sb.run_sync(lambda: sb.write_file("TASK.md", TASK), timeout=300)
        print(f"\nuser> {Q1}", flush=True)
        root.send(Q1, wait=True, timeout=600)

        learn(root, state)

        tool_done = threading.Event()
        unsub = root.event_bus.subscribe(lambda ev: tool_done.set(), ["tool.end"])
        print(f"\nuser> {Q2}", flush=True)
        tid = root.send(Q2)
        while not tool_done.wait(0.05) and root.session.turns.result(tid) is None:
            pass
        unsub()
        print(f"\n--- root.suspend(on_active='interrupt') while {tid} is running ---",
              flush=True)
        snap = root.suspend(on_active="interrupt")
        print(f"\nsnapshot {store.path_for(SESSION)}:", flush=True)
        print(json.dumps(_short(snap), indent=2), flush=True)
    finally:
        if not root.closed:
            root.unmount()
        ps.wait_closed(120)
    print(f"\n  ({lm_stats()})", flush=True)


def phase_resume(state: Path, driver, lm) -> None:
    store = DirectorySnapshots(state / "snapshots")
    print(f"\n--- new process: hydrate_root({SESSION!r}) ---", flush=True)
    t0 = time.time()
    root = hydrate_root(make_app(state, driver, lm), session_id=SESSION, store=store)
    root.event_bus.subscribe(show, KINDS)
    try:
        resumed = [e.payload for e in root.events.recent(None, ["session.resumed"])][-1]
        print(f"  [session.resumed] {_short({k: v for k, v in resumed.items() if k != 'session_id'})}"
              f" ({time.time() - t0:.1f}s)", flush=True)
        print(f"  mismatches: {root.mismatches}", flush=True)
        coder = root.ref("coder")
        instr = coder["module"].react.signature.instructions
        print(f"[learn] learned prompt hydrated: {LESSON.strip() in instr}", flush=True)
        print(f"[durable] conversation hydrated: {coder['messages']} messages", flush=True)
        for tid in resumed.get("turns") or []:  # the carried turn re-runs by itself
            root.wait_turn(tid, timeout=600)
        sb = root.ref("sandbox")
        ls = sb.run_sync(lambda: sb.exec("ls -1"), timeout=300)
        out = sb.run_sync(lambda: sb.exec("python3 fib.py"), timeout=300)
        print(f"\n[sandbox] files: {ls.stdout.split()}", flush=True)
        print(f"[sandbox] python3 fib.py -> {out.stdout.strip()!r}", flush=True)
    finally:
        root.unmount()  # on_close="destroy": the container goes away
        ps.wait_closed(120)
    log = SessionLog.open(state / "logs" / f"{SESSION}.jsonl", readonly=True)
    try:
        conv = log.get_state("App/CodingAgent[coder]", "history", []) or []
        journaled = sum(1 for e in log.entries if e.get("type") == "llm")
    finally:
        log.close()
    print(f"\nconversation in the log: {len(conv)} messages "
          f"({journaled} LM calls journaled across both processes)", flush=True)
    for m in conv:
        print(f"  {m['role']:>9}: {str(m['content'])[:110]}", flush=True)
    print(f"\n  ({lm_stats()})", flush=True)


def cleanup(state: Path) -> None:
    drv = ps.get_driver("docker")
    if drv.available():
        report = ps.gc([drv], all=True, snapshots=True, labels=LABELS)
        if any(report.values()):
            print(f"gc: {report}", flush=True)
    shutil.rmtree(state, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=["work", "resume"])
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--state", default="runs/e2e_roots")
    args = ap.parse_args()
    state = Path(args.state).resolve()
    if args.cleanup:
        cleanup(state)
        return 0
    drv = ps.get_driver("docker")
    if not drv.available():
        print("Docker is not available", file=sys.stderr)
        return 1
    if args.phase is None:  # both, each in its own process
        cleanup(state)
        base = [sys.executable, str(Path(__file__).resolve()), "--state", str(state)]
        try:
            for phase in ("work", "resume"):
                print(f"\n========== process: --phase {phase} ==========", flush=True)
                t0 = time.time()
                rc = subprocess.run([*base, "--phase", phase]).returncode
                print(f"========== exit {rc} ({time.time() - t0:.1f}s) ==========", flush=True)
                if rc:
                    return rc
        finally:
            cleanup(state)
        return 0
    lm = example_lm()
    (state / "learned").mkdir(parents=True, exist_ok=True)
    if args.phase == "work":
        phase_work(state, drv, lm)
    else:
        phase_resume(state, drv, lm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
