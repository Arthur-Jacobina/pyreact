"""01 - A coding agent that works in a Docker sandbox and survives restarts.

A ReAct agent (``use_agent``) gets four tools over its sandbox
(``sandbox_tools``: ``run_shell``, ``run_python``, ``read_file``,
``write_file``). It answers by *writing and running* a small Python script in
a container (python:3.12-slim, no network, 512 MB, 1 CPU).

State that survives the process:

- the conversation (``use_conversation``) and the sandbox's restart record
  (``use_durable_snapshot_store``) live in a ``pyreact_durable`` log;
- after every turn that touched the sandbox it is snapshotted
  (``autosnapshot="turn"`` -> ``docker commit``), and on exit the container is
  destroyed (``on_close="destroy"``) — the snapshot is the state.

So run it, let it exit, then ``--resume``: a new runner restores the sandbox
from the snapshot (``sandbox.restored``) and the agent continues where it
was — its files are still there.

    uv run python -m examples.sandbox.01_coding_agent \\
        "What is the 30th Fibonacci number? Write fib.py that computes it."
    uv run python -m examples.sandbox.01_coding_agent --resume \\
        "Extend the script you wrote to also print the 40th; run it."
    uv run python -m examples.sandbox.01_coding_agent --cleanup   # remove snapshots
    uv run python -m examples.sandbox.01_coding_agent --demo      # both runs in one go

``--driver local`` uses the LocalDriver (NOT isolated: host subprocesses in a
temp dir) — what the offline smoke test does. State goes to ``--state``
(default ``runs/sandbox``). LM: ``examples._lm`` (``PYREACT_EXAMPLE_LM``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import dspy

import pyreact_sandbox as ps
from integrations import DSPyProvider, use_agent
from pyreact.boot import bootstrap
from pyreact.core import component, hooks, use_turn
from pyreact_durable import Durable, use_conversation
from pyreact_sandbox import SandboxProvider, SandboxSpec, sandbox_tools, use_sandbox
from pyreact_sandbox.contrib.durable import use_durable_snapshot_store

from examples._lm import example_lm

# Offline smoke run (tests/test_examples_smoke.py): local driver, state in a temp dir.
SMOKE = {"args": ["--demo", "--driver", "local", "--state", "{tmp}/state"], "timeout": 120}

LABELS = {"pyreact.example": "coding_agent"}
SPEC = SandboxSpec(image="python:3.12-slim", cpu=1, memory="512m", network=False,
                   labels=LABELS)
MAX_ITERS = 5  # ReAct steps per question (+1 extract call): <= 6 LM calls per run


class Coder(dspy.Signature):
    """You are a careful engineer with a private Linux sandbox (Python 3, no network).
    Answer by writing a small Python script to a file with write_file, running it with
    run_shell (e.g. `python3 script.py`), and reporting what it printed. Files you wrote
    in earlier turns are still in the sandbox; use the conversation history.
    Keep scripts short."""

    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField(desc="the final answer, one or two sentences")


@component
def CodingAgent():
    conv = use_conversation()                     # persisted Q/A pairs
    store = use_durable_snapshot_store()          # sandbox restart record, in the same log
    sb = use_sandbox(store=store, autosnapshot="turn")
    tools = hooks.use_memo(lambda: sandbox_tools(sb, timeout=30), [id(sb)])
    agent = use_agent(Coder, tools=tools, max_iters=MAX_ITERS)
    turn = use_turn()
    bus = hooks.get_service("event_bus", lambda: None)

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


def make_app(log_path: str, driver, lm):
    @component
    def App():
        return [Durable(key="durable", log=log_path, children=[
            DSPyProvider(key="dspy", lm=lm, children=[
                SandboxProvider(key="sandbox", driver=driver, spec=SPEC, on_close="destroy",
                                children=[CodingAgent(key="coder")]),
            ]),
        ])]

    return bootstrap(App)


def show(ev) -> None:
    p = ev.payload
    if ev.kind == "message":
        print(f"\n{p.get('sender')}> {p.get('text')}", flush=True)
    elif ev.kind == "tool.start":
        print(f"  [tool] {p.get('tool')}({p.get('args')})", flush=True)
    elif ev.kind == "sandbox.exec":
        out = (p.get("stdout") or p.get("stderr") or p.get("error") or "").strip()
        code = p.get("exit_code")
        print(f"  [sandbox.exec {p.get('op')}] {p.get('name')} -> "
              f"{'' if code is None else f'exit {code} '}{out[:200]!r}", flush=True)
    elif ev.kind in ("sandbox.created", "sandbox.restored", "sandbox.snapshot",
                     "sandbox.destroyed", "sandbox.error"):
        keep = {k: v for k, v in p.items() if k in ("sandbox_id", "mode", "snapshot", "duration",
                                                   "error", "policy", "image")}
        if "duration" in keep:
            keep["duration"] = round(keep["duration"], 2)
        print(f"  [{ev.kind}] {keep}", flush=True)


def driver_for(name: str, state: Path):
    if name == "local":
        return ps.LocalDriver(root=state / "local")  # NOT isolated: dev/tests only
    return ps.get_driver(name)


def cleanup(driver_name: str, state: Path) -> None:
    drv = driver_for(driver_name, state)
    if drv.available():
        report = ps.gc([drv], all=True, snapshots=True, labels=LABELS)
        print(f"gc: {report}")
    log = state / "agent.jsonl"
    if log.exists():
        log.unlink()


def run_once(question: str, *, state: Path, driver_name: str, lm) -> None:
    state.mkdir(parents=True, exist_ok=True)
    app = make_app(str(state / "agent.jsonl"), driver_for(driver_name, state), lm)
    app.event_bus.subscribe(show, ["message", "tool.start", "sandbox.*"])
    print(f"\nuser> {question}", flush=True)
    calls0 = len(getattr(lm, "history", []) or [])
    try:
        app.invoke(question, wait=True, timeout=600)
    finally:
        app.shutdown()
        ps.wait_closed(120)  # autosnapshot + destroy finish before we exit
    calls = len(getattr(lm, "history", []) or []) - calls0
    print(f"  (LM calls this run: {calls})", flush=True)


DEFAULT_Q1 = ("What is the 30th Fibonacci number (fib(1) = fib(2) = 1)? "
              "Write fib.py that computes and prints it, run it, and answer.")
DEFAULT_Q2 = ("Extend the fib.py you wrote earlier so it also prints the 40th Fibonacci "
              "number, run it, and tell me both numbers.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--resume", action="store_true", help="continue the previous session")
    ap.add_argument("--demo", action="store_true", help="fresh run, restart, resumed run")
    ap.add_argument("--cleanup", action="store_true", help="remove sandboxes, snapshots, log")
    ap.add_argument("--driver", default="docker")
    ap.add_argument("--state", default="runs/sandbox")
    args = ap.parse_args()
    state = Path(args.state)

    if args.cleanup:
        cleanup(args.driver, state)
        return 0
    drv = driver_for(args.driver, state)
    if not drv.available():
        print(f"driver {args.driver!r} is not available (is Docker running?)", file=sys.stderr)
        return 1
    lm = example_lm()

    if args.demo:
        cleanup(args.driver, state)
        run_once(args.question or DEFAULT_Q1, state=state, driver_name=args.driver, lm=lm)
        print("\n--- process 'restarts': new runner, same log ---")
        run_once(DEFAULT_Q2, state=state, driver_name=args.driver, lm=lm)
        cleanup(args.driver, state)
        if args.driver == "local":
            shutil.rmtree(state / "local", ignore_errors=True)
        return 0
    if not args.resume:
        cleanup(args.driver, state)  # a fresh session: drop the previous one
    question = args.question or (DEFAULT_Q2 if args.resume else DEFAULT_Q1)
    run_once(question, state=state, driver_name=args.driver, lm=lm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
