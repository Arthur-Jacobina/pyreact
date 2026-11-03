"""01 - Suspend a whole coding agent, resume it in another process.

An *agent* here is a session's whole state: its durable conversation and
component state (``pyreact_durable``), its Docker sandbox with the files it
wrote (``pyreact_sandbox``), its learned prompt (``integrations.learn``) and
the turn it was in the middle of. ``session.suspend()`` turns all of that
into one manifest; ``app.resume(session_id)`` brings it back.

Process 1 (``--phase work``):
  1. seed the sandbox with a brief (``TASK.md``) and ask the coding agent a
     question; it writes and runs ``fib.py`` in the sandbox;
  2. "learn": save a better prompt as an ``integrations.learn`` result and
     hot-swap it into the agent (no optimizer run, no LLM call);
  3. ask a follow-up and, while the agent is working on it, **suspend**
     (``on_active="interrupt"``): the turn is cut short and carried in the
     manifest, the sandbox is snapshotted then destroyed, the manifest is
     printed.

Process 2 (``--phase resume``): a new runner **resumes** the session. Before
the first render the conversation, the log leaf, the sandbox snapshot and
the learned prompt are handed back; then the cut turn re-runs by itself --
its LLM calls that had answered are **replayed** from the log (``durable_lm``)
-- and the agent finishes the task using the files from process 1.

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

import dspy

import pyreact_sandbox as ps
from integrations import DSPyProvider, use_agent
from integrations.learn import OptimizationResult
from integrations.learn.hmr import resolve_module
from integrations.stateful import CompiledModulesParticipant
from pyreact.boot import AppRunner, DirectoryManifestStore
from pyreact.core import component, hooks, use_turn
from pyreact_durable import (
    Durable,
    DurableParticipant,
    SessionLog,
    use_conversation,
    use_durable_lm,
)
from pyreact_sandbox import SandboxParticipant, SandboxProvider, SandboxSpec, sandbox_tools
from pyreact_sandbox import use_sandbox
from pyreact_sandbox.react import HANDLES_SERVICE

from examples._lm import example_lm

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


LMS: dict = {}  # this process's durable LM wrapper (for its replay stats)


@component(transparent=True)  # no path segment: state lives at App/CodingAgent[coder]
def LMLayer(lm=None, children=None):
    dlm = use_durable_lm(lm)  # every LM call journaled; re-runs replay them
    LMS["durable"] = dlm
    return [DSPyProvider(key="dspy", lm=dlm, children=children or [])]


@component
def CodingAgent():
    conv = use_conversation()                     # persisted Q/A pairs
    sb = use_sandbox()                            # the session's sandbox
    tools = hooks.use_memo(lambda: sandbox_tools(sb, timeout=30), [id(sb)])
    agent = use_agent(Coder, tools=tools, max_iters=MAX_ITERS, name="coder")
    turn = use_turn()
    bus = hooks.get_service("event_bus", lambda: None)

    def mounted():  # what the first render saw (restored before it, on resume)
        bus.publish("message", {"text": f"(mounted with {len(conv)} messages in the "
                                        "conversation)", "sender": "system"})

    hooks.use_effect(mounted, [])

    def ask():
        if turn is not None:
            agent.run(question=turn.text, history=conv.as_dspy_history())

    hooks.use_effect(ask, [turn and turn.id])

    def answered():
        if not agent.ver or turn is None:
            return
        if agent.error is not None:
            bus.publish("message", {"text": f"error: {agent.error!r}", "sender": "system"})
            return
        answer = agent.result.answer
        conv.extend([{"role": "user", "content": turn.text},
                     {"role": "assistant", "content": answer}])
        bus.publish("message", {"text": answer, "sender": "assistant"})

    hooks.use_effect(answered, [agent.ver])
    return []


def make_app(state: Path, driver, lm) -> AppRunner:
    logs = state / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    @component
    def App():
        return [Durable(key="durable", log_factory=lambda sid: logs / f"{sid}.jsonl", children=[
            LMLayer(key="lm", lm=lm, children=[
                SandboxProvider(key="sandbox", driver=driver, spec=SPEC, on_close="destroy",
                                children=[CodingAgent(key="coder")]),
            ]),
        ])]

    return AppRunner(
        App, default_session=False,
        participants=[
            DurableParticipant,                                        # log + leaf
            lambda s: SandboxParticipant(s, mode="snapshot"),          # snapshot -> restore
            lambda s: CompiledModulesParticipant(s, dir=state / "learned"),  # learned prompts
        ],
        manifest_store=DirectoryManifestStore(state / "manifests"),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def show(ev) -> None:
    p = ev.payload
    if ev.kind == "message":
        print(f"\n{p.get('sender')}> {p.get('text')}", flush=True)
    elif ev.kind == "tool.start":
        print(f"  [tool] {p.get('tool')}({str(p.get('args'))[:120]})", flush=True)
    elif ev.kind == "sandbox.exec":
        out = (p.get("stdout") or p.get("stderr") or p.get("error") or "").strip()
        code = p.get("exit_code")
        print(f"  [sandbox.exec {p.get('op')}] {p.get('name')} -> "
              f"{'' if code is None else f'exit {code} '}{out[:160]!r}", flush=True)
    elif ev.kind in ("sandbox.created", "sandbox.restored", "sandbox.snapshot",
                     "sandbox.destroyed", "sandbox.error"):
        keep = {k: v for k, v in p.items() if k in ("sandbox_id", "mode", "snapshot", "error",
                                                   "policy")}
        print(f"  [{ev.kind}] {keep}", flush=True)
    elif ev.kind == "turn.start":
        print(f"  [turn.start] {ev.turn_id}{' (resumed)' if p.get('resumed') else ''}",
              flush=True)
    elif ev.kind == "turn.end":
        print(f"  [turn.end] {ev.turn_id} {p.get('status')}", flush=True)
    elif ev.kind.startswith("session."):
        keep = {k: v for k, v in p.items() if k not in ("session_id",)}
        print(f"  [{ev.kind}] {keep}", flush=True)


KINDS = ["message", "tool.start", "sandbox.*", "turn.start", "turn.end", "session.*"]


def _short(obj, n=160):
    if isinstance(obj, dict):
        return {k: _short(v, n) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_short(v, n) for v in obj]
    if isinstance(obj, str) and len(obj) > n:
        return obj[:n] + "…"
    return obj


def lm_stats() -> str:
    stats = getattr(LMS.get("durable"), "stats", None) or {}
    return (f"model calls this process: {stats.get('inner_calls', 0)}, replayed from the "
            f"log: {stats.get('replayed', 0)}")


def sandbox_handle(session):
    return session.services[HANDLES_SERVICE][f"{session.id}/default"]


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def learn(session, state: Path) -> None:
    """Stand-in for an optimizer run: a better prompt, saved as a learn result
    and hot-swapped in. CompiledModulesParticipant carries it across the restart."""
    _env, key, mod = resolve_module(session, "coder")
    learned = copy.deepcopy(mod)
    for name, pred in learned.named_predictors():
        if name.startswith("react"):
            pred.signature = pred.signature.with_instructions(pred.signature.instructions
                                                              + LESSON)
    res = OptimizationResult(program=learned, optimizer="manual", module_key=key)
    path = res.save(str(state / "learned" / "coder.json"))
    res.hot_swap(session, key)
    session.wait_idle(timeout=10)
    print(f"\n[learn] hot-swapped a learned prompt into {key[0]}:{key[2]} (saved: {path})",
          flush=True)


def phase_work(state: Path, driver, lm) -> None:
    app = make_app(state, driver, lm)
    app.on_session(open=lambda s: s.event_bus.subscribe(show, KINDS))
    try:
        s = app.open_session(SESSION)
        h = sandbox_handle(s)
        h.run_sync(lambda: h.write_file("TASK.md", TASK), timeout=300)
        print(f"\nuser> {Q1}", flush=True)
        s.invoke(Q1, wait=True, timeout=600)

        learn(s, state)

        # The follow-up: suspend while the agent is working on it.
        tools_done = threading.Event()
        counter = {"n": 0}

        def on_tool_end(ev):
            counter["n"] += 1
            if counter["n"] >= SUSPEND_AFTER_TOOLS:
                tools_done.set()

        unsub = s.event_bus.subscribe(on_tool_end, ["tool.end"])
        print(f"\nuser> {Q2}", flush=True)
        tid = s.invoke(Q2)
        while not tools_done.wait(0.05) and s.turns.result(tid) is None:
            pass
        unsub()
        print(f"\n--- suspending {SESSION} (on_active='interrupt') ---", flush=True)
        manifest = s.suspend(on_active="interrupt")
        print("\nmanifest " + manifest["ref"] + ":", flush=True)
        print(json.dumps(_short(manifest), indent=2), flush=True)
    finally:
        app.shutdown()
        ps.wait_closed(120)
    print(f"\n  ({lm_stats()})", flush=True)


def phase_resume(state: Path, driver, lm) -> None:
    app = make_app(state, driver, lm)
    app.on_session(open=lambda s: s.event_bus.subscribe(show, KINDS))
    resumed: dict = {}
    app.on_session(open=lambda s: s.event_bus.subscribe(
        lambda ev: resumed.update(ev.payload), ["session.resumed"]))
    try:
        print(f"\n--- new process: resuming {SESSION} from its manifest ---", flush=True)
        s = app.resume(SESSION)

        h = sandbox_handle(s)
        listing = h.run_sync(lambda: h.exec("ls -1"), timeout=300)
        print(f"\n[sandbox] files after resume: {listing.stdout.split()}", flush=True)
        _env, _key, mod = resolve_module(s, "coder")
        instr = mod.react.signature.instructions
        print(f"[learn] learned prompt restored: {LESSON.strip()[:40]!r} in instructions = "
              f"{LESSON.strip() in instr}", flush=True)

        carried = resumed.get("turns") or []
        for tid in carried:  # the cut turn re-runs on its own
            s.wait_turn(tid, timeout=600)
        if not carried:  # nothing was cut short (e.g. the offline LM): continue anyway
            print(f"\nuser> {Q3}", flush=True)
            s.invoke(Q3, wait=True, timeout=600)

        print(f"\n--- suspending {SESSION} again ---", flush=True)
        manifest = s.suspend()
        frag = manifest["participants"]
        print(f"manifest {manifest['ref']}: turn_seq={manifest['turn_seq']}, "
              f"sandbox snapshot={frag['sandbox']['sandboxes'][0]['snapshot']['id']}",
              flush=True)
    finally:
        app.shutdown()
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
    for sub in ("logs", "manifests", "learned", "local"):
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
