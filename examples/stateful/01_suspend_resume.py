"""01 - Suspend a whole coding agent, hydrate it in another process.

An *agent* here is its tree's state: its durable conversation
(``pyreact_durable``), its sandbox with the files it wrote
(``pyreact_sandbox``), its learned prompt (``integrations.learn``) and the
turn it was in the middle of. The tree declares all of it; outside code only
talks to the **root** (ADR 0001)::

    root = create_root(App, session_id="coder-1", store=DirectorySnapshots(...))
    root.send(Q1, wait=True)
    root.ref("coder").learned.set("coder", path)   # LearnedProgram state: a re-render
    root.suspend(on_active="interrupt") # hide -> dehydrate -> snapshot -> unmount
    ...
    root = hydrate_root(App, session_id="coder-1", store=...)   # next process

Process 1 (``--phase work``):
  1. the agent mounts with a brief: an effect writes ``TASK.md`` into its
     sandbox; it is asked a question and writes and runs ``fib.py``;
  2. "learn": a better prompt is saved as an ``integrations.learn`` result
     and becomes the program of ``LearnedProgram(key="coder")`` -- the
     provider's state, set through its registry (no optimizer run, no LLM
     call);
  3. a follow-up is asked and, while the agent is working on it, the root
     is **suspended** (``on_active="interrupt"``): the turn is cut short and
     carried in the snapshot, which is printed.

Process 2 (``--phase resume``): ``hydrate_root`` mounts the same tree with the
snapshot. Every library hands its own slice back during the first render
(``use_dehydrate``): ``Durable`` its log and leaf (the conversation),
``SandboxProvider`` / ``use_sandbox`` the snapshot taken on suspend (the
files), ``LearnedProgram`` the learned prompt. Then the cut turn re-runs by
itself -- its LLM calls that had answered are **replayed** from the log
(``durable_lm``) -- and the agent finishes the task using the files from
process 1.

    uv run python -m examples.stateful.01_suspend_resume --demo            # both, 2 processes
    uv run python -m examples.stateful.01_suspend_resume --phase work
    uv run python -m examples.stateful.01_suspend_resume --phase resume
    uv run python -m examples.stateful.01_suspend_resume --cleanup         # sandboxes, snapshots, state

``--driver local`` uses the LocalDriver (NOT isolated; host temp dirs) -- what
the offline smoke test does. State goes to ``--state`` (default
``runs/stateful``). LM: ``examples._lm`` (``PYREACT_EXAMPLE_LM``).
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
from types import SimpleNamespace

import dspy

import pyreact_sandbox as ps
from examples._devtools import trace
from examples._lm import example_lm
from integrations import DSPyProvider, use_agent, use_dspy_env
from integrations.learn import (
    LearnedProgram,
    OptimizationResult,
    use_learned,
    use_learned_registry,
)
from pyreact.boot import DirectorySnapshots, create_root, hydrate_root
from pyreact.components.message import Message
from pyreact.core import component, hooks, use_event, use_imperative_handle, use_turn
from pyreact_durable import Durable, SessionLog, use_conversation, use_durable_lm
from pyreact_sandbox import SandboxProvider, SandboxSpec, sandbox_tools, use_sandbox

# Offline smoke run (tests/test_examples_smoke.py): local driver, state in a temp dir.
SMOKE = {"args": ["--demo", "--driver", "local", "--state", "{tmp}/state"], "timeout": 240}

SESSION = "coder-1"
LABELS = {"pyreact.example": "stateful"}
SPEC = SandboxSpec(image="python:3.12-slim", cpu=1, memory="512m", network=False,
                   labels=LABELS)
MAX_ITERS = 5  # ReAct steps per question (+1 extract call)
SUSPEND_AFTER_TOOLS = 1  # process 1 suspends once the follow-up made this many tool calls

TASK = """# Task
Maintain fib.py: a script that prints Fibonacci numbers (fib(1) = fib(2) = 1).
Print each result labelled, e.g. `fib(30) = 832040`.
"""
Q1 = ("Read TASK.md, then write fib.py that computes and prints the 30th Fibonacci "
      "number, run it, and tell me the result.")
Q2 = ("Extend the fib.py you wrote earlier so it also prints the 40th Fibonacci number, "
      "run it, and tell me both numbers.")
Q3 = "Run fib.py once more and tell me what it prints."
LESSON = (" Lesson learned: before changing a file from an earlier turn, read it with "
          "read_file first; keep scripts short.")


class Coder(dspy.Signature):
    """You are a careful engineer with a private Linux sandbox (Python 3, no network).
    Answer by writing a small Python script to a file with write_file, running it with
    run_shell (e.g. `python3 script.py`), and reporting what it printed. Files you wrote
    in earlier turns are still in the sandbox; use the conversation history."""

    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField(desc="the final answer, one or two sentences")


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------
@component(transparent=True)  # no path segment: state lives at App/CodingAgent[coder]
def LMLayer(lm=None, children=None):
    dlm = use_durable_lm(lm)  # every LM call journaled; re-runs replay them
    use_imperative_handle("lm", lambda: dlm, [id(dlm)])  # root.ref("lm").stats
    return [DSPyProvider(key="dspy", lm=dlm, children=children or [])]


@component
def CodingAgent(brief: str = "", on_tool_end=None):
    conv = use_conversation()                     # persisted Q/A pairs
    sb = use_sandbox(ref="sandbox")               # the session's sandbox; root.ref("sandbox")
    tools = hooks.use_memo(lambda: sandbox_tools(sb, timeout=30), [id(sb)])
    program = use_learned("coder")                # the LearnedProgram above (None: baseline)
    agent = use_agent(Coder, tools=tools, max_iters=MAX_ITERS, name="coder", program=program)
    turn = use_turn()
    restored = hooks.use_ref(len(conv))           # what the first render saw

    # The brief is part of the workspace: an effect writes it into the sandbox
    # (idempotent, so a hydrated agent may write it again).
    hooks.use_effect(lambda: sb.write_file("TASK.md", brief) if brief else None,
                     [id(sb), brief])
    use_event("tool.end", lambda ev: on_tool_end(ev) if on_tool_end else None)
    env = use_dspy_env()
    registry = use_learned_registry()            # the LearnedProgram provider's handle
    # imperative access for outside code: the live module, the learned registry
    use_imperative_handle("coder", lambda: SimpleNamespace(
        module=env.caches.get(("ReAct", "Coder", "coder")), learned=registry,
        messages=len(conv)), [len(conv), id(program)])

    def ask():
        if turn is not None:
            agent.run(question=turn.text, history=conv.as_dspy_history())

    hooks.use_effect(ask, [turn and turn.id])

    def remember():
        if agent.ver and turn is not None and agent.error is None:
            conv.extend([{"role": "user", "content": turn.text},
                         {"role": "assistant", "content": agent.result.answer}])

    hooks.use_effect(remember, [agent.ver])

    out = [Message(key="mounted", sender="system", trigger="mount",
                   text=f"(mounted with {restored.current} messages in the conversation)")]
    if agent.ver and agent.error is not None:
        out.append(Message(key=f"error-{agent.ver}", text=f"error: {agent.error!r}",
                           sender="system", message_type="error"))
    elif agent.ver and agent.result is not None:
        out.append(Message(key=f"answer-{agent.ver}", text=agent.result.answer,
                           sender="assistant"))
    return out


def make_app(state: Path, driver, lm, on_tool_end=None):
    logs = state / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    @component
    def App():
        # Every provider dehydrates its own state on root.suspend() and gets
        # it back on hydrate_root(): the log ref, the sandbox snapshot ref
        # (taken on suspend, destroyed otherwise), the learned program.
        return [Durable(key="durable", log_factory=lambda sid: logs / f"{sid}.jsonl", children=[
            LearnedProgram(key="coder", src=None, children=[
                LMLayer(key="lm", lm=lm, children=[
                    SandboxProvider(key="sandbox", driver=driver, spec=SPEC, on_close="destroy",
                                    children=[
                        CodingAgent(key="coder", brief=TASK, on_tool_end=on_tool_end),
                    ]),
                ]),
            ]),
        ])]

    return App


# ---------------------------------------------------------------------------
# Devtools output
# ---------------------------------------------------------------------------
def show(ev):
    p = ev.payload
    if ev.kind == "message":
        return f"\n{p.get('sender')}> {p.get('text')}"
    if ev.kind == "tool.start":
        return f"  [tool] {p.get('tool')}({str(p.get('args'))[:120]})"
    if ev.kind == "sandbox.exec":
        out = (p.get("stdout") or p.get("stderr") or p.get("error") or "").strip()
        code = p.get("exit_code")
        return (f"  [sandbox.exec {p.get('op')}] {p.get('name')} -> "
                f"{'' if code is None else f'exit {code} '}{out[:160]!r}")
    if ev.kind in ("sandbox.created", "sandbox.restored", "sandbox.snapshot",
                   "sandbox.destroyed", "sandbox.error"):
        keep = {k: v for k, v in p.items() if k in ("sandbox_id", "mode", "snapshot", "error",
                                                   "policy")}
        return f"  [{ev.kind}] {keep}"
    if ev.kind == "turn.start":
        return f"  [turn.start] {ev.turn_id}{' (resumed)' if p.get('resumed') else ''}"
    if ev.kind == "turn.end":
        return f"  [turn.end] {ev.turn_id} {p.get('status')}"
    if ev.kind.startswith("session.") or ev.kind == "hydration.mismatch":
        keep = {k: v for k, v in p.items() if k not in ("session_id",)}
        return f"  [{ev.kind}] {keep}"
    return None


KINDS = ["message", "tool.start", "sandbox.*", "turn.start", "turn.end", "session.*",
         "hydration.mismatch"]


def _short(obj, n=160):
    if isinstance(obj, dict):
        return {k: _short(v, n) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_short(v, n) for v in obj]
    if isinstance(obj, str) and len(obj) > n:
        return obj[:n] + "…"
    return obj


def lm_stats(dlm) -> str:
    stats = getattr(dlm, "stats", None) or {}
    return (f"model calls this process: {stats.get('inner_calls', 0)}, replayed from the "
            f"log: {stats.get('replayed', 0)}")


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def learn(root, state: Path) -> str:
    """Stand-in for an optimizer run: the agent's program with a better
    prompt, saved as a learn result and set as the LearnedProgram's state."""
    coder = root.ref("coder")
    learned = copy.deepcopy(coder.module)
    for name, pred in learned.named_predictors():
        if name.startswith("react"):
            pred.signature = pred.signature.with_instructions(pred.signature.instructions
                                                              + LESSON)
    path = OptimizationResult(program=learned, optimizer="manual", arch="react").save(
        str(state / "learned" / "coder.json"))
    coder.learned.set("coder", path)  # the provider's state: a re-render applies it
    root.wait_idle(timeout=10)
    print(f"\n[learn] LearnedProgram 'coder' <- {path}: the agent re-rendered with the "
          "learned prompt", flush=True)
    return path


def phase_work(state: Path, driver, lm) -> None:
    tools_done = threading.Event()
    seen = {"n": 0}

    def on_tool_end(_ev):  # a callback prop (onToolEnd)
        seen["n"] += 1
        if seen["n"] >= SUSPEND_AFTER_TOOLS:
            tools_done.set()

    store = DirectorySnapshots(state / "snapshots")
    root = create_root(make_app(state, driver, lm, on_tool_end), session_id=SESSION,
                       store=store)
    dlm = root.ref("lm")
    try:
        trace(root, KINDS, show)
        root.wait_idle(timeout=300)  # the brief is in the sandbox
        print(f"\nuser> {Q1}", flush=True)
        root.send(Q1, wait=True, timeout=600)

        learn(root, state)

        # The follow-up: suspend while the agent is working on it.
        tools_done.clear()
        seen["n"] = 0
        print(f"\nuser> {Q2}", flush=True)
        tid = root.send(Q2)
        while not tools_done.is_set():
            try:
                root.wait_turn(tid, timeout=0.05)
                break  # it finished first (e.g. the offline LM makes no tool calls)
            except TimeoutError:
                pass
        print(f"\n--- suspending {SESSION} (on_active='interrupt') ---", flush=True)
        snap = root.suspend(on_active="interrupt")
        print(f"\nsnapshot {store.path_for(SESSION)}:", flush=True)
        print(json.dumps(_short(snap), indent=2), flush=True)
    finally:
        if not root.closed:
            root.unmount()
        ps.wait_closed(120)
    print(f"\n  ({lm_stats(dlm)})", flush=True)


def phase_resume(state: Path, driver, lm) -> None:
    store = DirectorySnapshots(state / "snapshots")
    snap = store.load(SESSION)
    if snap is None:
        raise SystemExit(f"no snapshot for {SESSION!r} in {state / 'snapshots'}; "
                         "run --phase work first")
    carried = [t["id"] for t in (snap.get("turns") or {}).get("carry", [])]
    print(f"\n--- new process: hydrating {SESSION} (carried turns: {carried}) ---", flush=True)
    root = hydrate_root(make_app(state, driver, lm), session_id=SESSION, store=store)
    dlm = root.ref("lm")
    try:
        trace(root, KINDS, show)  # replays what hydration published
        sb = root.ref("sandbox")
        listing = sb.run_sync(lambda: sb.exec("ls -1"), timeout=300)
        print(f"\n[sandbox] files after hydrate: {listing.stdout.split()}", flush=True)
        coder = root.ref("coder")
        instr = coder.module.react.signature.instructions
        print(f"[learn] learned prompt hydrated: {LESSON.strip() in instr}", flush=True)
        print(f"[durable] conversation hydrated: {coder.messages} messages", flush=True)
        print(f"  mismatches: {root.mismatches}", flush=True)

        for tid in carried:  # the cut turn re-runs on its own
            root.wait_turn(tid, timeout=600)
        if not carried:  # nothing was cut short (e.g. the offline LM): continue anyway
            print(f"\nuser> {Q3}", flush=True)
            root.send(Q3, wait=True, timeout=600)

        print(f"\n--- suspending {SESSION} again ---", flush=True)
        snap = root.suspend()
        print(f"snapshot: turns next_index={snap['turns'].get('next_index')}, "
              f"slices: {sorted(snap['slices'])}", flush=True)
    finally:
        if not root.closed:
            root.unmount()
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
    print(f"\n  ({lm_stats(dlm)})", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def driver_for(name: str, state: Path):
    if name == "local":
        return ps.LocalDriver(root=state / "local")  # NOT isolated: dev/tests only
    return ps.get_driver(name)


def cleanup(driver_name: str, state: Path) -> None:
    drv = driver_for(driver_name, state)
    if drv.available():
        report = ps.gc([drv], all=True, snapshots=True, labels=LABELS)
        if any(report.values()):
            print(f"gc: {report}", flush=True)
    for sub in ("logs", "snapshots", "learned", "local"):
        shutil.rmtree(state / sub, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=["work", "resume"])
    ap.add_argument("--demo", action="store_true", help="work, then resume in a new process")
    ap.add_argument("--cleanup", action="store_true", help="remove sandboxes, snapshots, state")
    ap.add_argument("--driver", default="docker")
    ap.add_argument("--state", default="runs/stateful")
    args = ap.parse_args()
    state = Path(args.state)

    if args.cleanup:
        cleanup(args.driver, state)
        return 0
    drv = driver_for(args.driver, state)
    if not drv.available():
        print(f"driver {args.driver!r} is not available (is Docker running?)", file=sys.stderr)
        return 1

    if args.demo:
        cleanup(args.driver, state)
        base = [sys.executable, "-m", __spec__.name, "--driver", args.driver,
                "--state", str(state)]
        try:
            for phase in ("work", "resume"):
                print(f"\n========== process: --phase {phase} ==========", flush=True)
                t0 = time.time()
                rc = subprocess.run([*base, "--phase", phase]).returncode
                print(f"========== exit {rc} ({time.time() - t0:.1f}s) ==========", flush=True)
                if rc:
                    return rc
        finally:
            cleanup(args.driver, state)
        return 0

    lm = example_lm()
    if args.phase == "work":
        cleanup(args.driver, state)  # a fresh agent
        phase_work(state, drv, lm)
    elif args.phase == "resume":
        phase_resume(state, drv, lm)
    else:
        ap.error("pick --phase work|resume, --demo or --cleanup")
    return 0


if __name__ == "__main__":
    sys.exit(main())
