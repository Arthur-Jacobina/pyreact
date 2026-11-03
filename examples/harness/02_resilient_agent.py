"""02 - A resilient agent: retry, fallback model, circuit breaker, ErrorBoundary, resume.

A weather assistant whose world is unreliable, simulated deterministically:

- the ``weather`` tool drops the first two connections;
- the primary model (``"default"``) is a *flaky provider*: overloaded once
  (with a ``Retry-After`` hint), then it hangs, then it recovers;
- the chart widget crashes when asked for a chart.

The recovery is declared once, as harness middleware and one component::

    HarnessProvider(middleware=[
        fallback_model(chain=["default", "fast"]),     # degrade to the fast model
        retry(max_attempts=2, backoff=exponential(...)),# llm: retry once
        circuit_breaker(failure_threshold=2, ...),      # stop calling a dead model
        timeout(...),                                   # per attempt
        retry(max_attempts=4, kinds=["tool"]),          # tools: retry harder
    ])
    ErrorBoundary(fallback=..., children=[ChartWidget()])

Turns (watch the ``recovery.*`` lines):

1. tool retried twice; primary overloaded -> retried after its Retry-After hint
2. primary hangs twice -> both attempts time out, the circuit opens, the
   call falls back to ``fast``
3. circuit open -> primary not even called (fail fast) -> ``fast``
4. the chart widget crashes -> ``ErrorBoundary`` renders a fallback message;
   the answer still arrives
5. 31 s later (simulated clock) -> half-open trial on primary succeeds, the
   circuit closes; the boundary has re-mounted the widget (renders fine)

Then **crash & resume**: a child process answers a question with a durable
session log (``pyreact_durable``), and is ``kill -9``'d while saving a
note, after the model answered but before the turn ended. A new process
mounts the same root (same ``session_id``) and resumes the unfinished turn
(``resume_pending()``): the journaled LLM call is replayed (0 model calls)
and the answer is the same.

Run from the repo root::

    uv run python -m examples.harness.02_resilient_agent [--log-dir DIR]

LM: ``examples._lm`` (``PYREACT_EXAMPLE_LM=claude|openai|offline``); 6 real
model calls with a real LM.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict

import dspy

from integrations import DSPyProvider, call_tool, use_call, use_module
from integrations.recovery import (
    circuit_breaker,
    exponential,
    fallback_model,
    retry,
    timeout,
)
from pyreact.boot import create_root
from pyreact.components.message import Message
from pyreact.core import ErrorBoundary, component, current_boundary, hooks, use_turn
from pyreact.core.harness import HarnessProvider
from pyreact.tools import make_tool

from examples._devtools import trace
from examples._lm import OFFLINE, example_lm

# Offline smoke run (tests/test_examples_smoke.py): logs go to the temp dir.
SMOKE = {"args": ["--log-dir", "{tmp}"], "timeout": 180}

# A real model answers in seconds; the simulated hang must outlast the limit.
LIMIT = 0.5 if OFFLINE else 20.0
CITIES = ("Lisbon", "Porto", "Faro")


# ---------------------------------------------------------------------------
# The unreliable world (deterministic)
# ---------------------------------------------------------------------------
class Overloaded(Exception):
    """What a provider's 529 looks like: a status and a Retry-After hint."""

    status_code = 529

    def __init__(self, retry_after: float):
        super().__init__(f"model overloaded, retry after {retry_after}s")
        self.retry_after = retry_after


class FlakyLM(dspy.BaseLM):
    """A flaky provider in front of a real LM. ``plan(turn_index, attempt)``
    says what happens to each attempt of each call: ``"ok"``,
    ``"overloaded"`` or ``"hang"``. Decided per boundary attempt (DSPy's
    adapter fallback may call the LM twice per attempt)."""

    def __init__(self, inner: Any, name: str, plan) -> None:
        super().__init__(model=name, cache=False)
        self.inner, self.plan = inner, plan
        self.real_calls = 0

    def _behavior(self) -> str:
        b = current_boundary()
        index = int(str(b.turn_id).rsplit("t", 1)[-1]) if b and b.turn_id else 0
        return self.plan(index, b.attempt if b else 1)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        what = self._behavior()
        if what == "overloaded":
            raise Overloaded(retry_after=0.3)
        if what == "hang":
            await asyncio.sleep(LIMIT * 10)  # cancelled by timeout()
        self.real_calls += 1
        if type(self.inner).aforward is not dspy.BaseLM.aforward:
            return await self.inner.aforward(prompt=prompt, messages=messages, **kwargs)
        return await asyncio.to_thread(self.inner.forward, prompt=prompt, messages=messages,
                                       **kwargs)

    def forward(self, prompt=None, messages=None, **kwargs):  # pragma: no cover
        raise RuntimeError("FlakyLM is async-only")


def primary_plan(turn_index: int, attempt: int) -> str:
    if turn_index == 1:
        return "overloaded" if attempt == 1 else "ok"
    if turn_index == 2:
        return "hang"
    return "ok"


def weather_tool(flaky: bool):
    drops = {"left": 2 if flaky else 0}
    forecast = {"Lisbon": "sunny, 24°C", "Porto": "light rain, 17°C", "Faro": "clear, 27°C"}

    async def weather(city: str) -> str:
        """Current weather for a city."""
        await asyncio.sleep(0.01)
        if drops["left"]:
            drops["left"] -= 1
            raise ConnectionError("weather service: connection reset")
        return forecast.get(city, "unknown")

    return make_tool(weather)


def note_tool(hang: bool):
    async def save_note(text: str) -> str:
        """Save the answer to the user's notes."""
        if hang:
            print("  [save_note] saving... (hangs: kill me now)", flush=True)
            await asyncio.sleep(120)
        return "saved"

    return make_tool(save_note)


class DemoClock:
    """The circuit breaker's clock, advanced by the script (deterministic)."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------
class Answer(dspy.Signature):
    """Answer the weather question in one short sentence using the observation."""

    question: str = dspy.InputField()
    observation: str = dspy.InputField()
    answer: str = dspy.OutputField()


def city_of(text: str) -> str:
    return next((c for c in CITIES if c.lower() in text.lower()), "Lisbon")


@component
def Assistant(tools: Dict[str, Any]):
    turn = use_turn()
    mod = use_module(Answer, name="answer")
    call = use_call(mod)

    async def research():
        city = city_of(turn.text)
        try:
            obs = await call_tool(tools["weather"], kwargs={"city": city})
        except Exception as exc:
            obs = f"weather unavailable ({exc})"
        call.run(question=turn.text, observation=f"{city}: {obs}", tag=turn.id)

    hooks.use_effect(lambda: research() if turn else None, [turn and turn.id])

    settled = call.ver and not call.loading

    async def save():  # a side effect of the answer: part of the turn
        await call_tool(tools["save_note"], kwargs={"text": call.result.answer})

    hooks.use_effect(lambda: save() if settled and call.error is None else None, [call.ver])

    if not settled:
        return []
    if call.error is not None:
        return [Message(key=f"error-{call.ver}", sender="assistant", message_type="error",
                        text=f"sorry, no model could answer ({call.error})")]
    return [Message(key=f"answer-{call.ver}", text=call.result.answer, sender="assistant")]


@component
def ChartWidget():
    turn = use_turn()
    if turn is not None and "chart" in turn.text.lower():
        raise ValueError("chart renderer not installed")
    return []


@component
def App(models, middleware, tools, durable_log=None):
    caught = hooks.use_memo(lambda: [], [])
    body = [
        Assistant(key="assistant", tools=tools),
        ErrorBoundary(
            key="widgets",
            fallback=lambda error: Message(
                text=f"(the chart could not be drawn: {error}; answering in text)",
                sender="system", message_type="warning"),
            on_error=lambda error, info: caught.append(info["component"]),
            reset_on="turn",
            children=[ChartWidget(key="chart")],
        ),
    ]
    tree = DSPyProvider(key="dspy", models=models, children=[
        HarnessProvider(key="harness", middleware=middleware, children=body)])
    if durable_log is not None:
        from pyreact_durable import Durable

        return [Durable(key="durable", log=durable_log, children=[tree])]
    return [tree]


def middleware_stack(clock=None):
    breaker = circuit_breaker(failure_threshold=2, reset_timeout=30,
                              clock=clock or time.monotonic, kinds=["llm"])
    return breaker, [
        fallback_model(chain=["default", "fast"]),
        retry(max_attempts=2, backoff=exponential(0.1, 1.0, jitter=False), kinds=["llm"]),
        breaker,
        timeout(LIMIT, kinds=["llm"]),
        retry(max_attempts=4, backoff=exponential(0.05, 0.5, jitter=False), kinds=["tool"]),
    ]


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------
def printer(prefix: str = ""):
    """Devtools: one line per event of KINDS (the transcript of the demo)."""

    def show(ev):
        p = ev.payload
        if ev.kind == "turn.start":
            tag = " (resumed)" if p.get("resumed") else ""
            return f"{prefix}\n{prefix}> {p['text']}   [{ev.turn_id}{tag}]"
        if ev.kind == "turn.end":
            return f"{prefix}  turn {ev.turn_id}: {p['status']} in {p['duration']:.2f}s"
        if ev.kind == "message":
            return f"{prefix}  {p.get('sender')}: {p.get('text')}"
        if ev.kind == "llm.end":
            key = f" ({p['model_key']})" if p.get("model_key") else ""
            return f"{prefix}  · llm answered by {p.get('model')}{key}"
        if ev.kind == "error.caught":
            return f"{prefix}  · ErrorBoundary caught {p['error']} in {p['component']}"
        if ev.kind == "recovery.retry":
            hint = f", Retry-After {p['hint']}s" if p.get("hint") else ""
            return (f"{prefix}  · retry {p['kind']} {p['name']!r} attempt {p['next_attempt']} "
                    f"in {p['delay']:.2f}s{hint} <- {p['error']}")
        if ev.kind == "recovery.timeout":
            return f"{prefix}  · timeout {p['kind']} {p['name']!r} after {p['seconds']}s"
        if ev.kind == "recovery.circuit":
            return f"{prefix}  · circuit {p['key']}: {p['prev']} -> {p['state']}"
        if ev.kind == "recovery.fallback":
            return f"{prefix}  · fallback {p.get('from')} -> {p['to']} <- {p['error']}"
        return None

    return show


KINDS = ["turn.*", "message", "llm.end", "error.caught", "recovery.*"]


# ---------------------------------------------------------------------------
# Part 1: retry / fallback / circuit breaker / ErrorBoundary
# ---------------------------------------------------------------------------
def resilience_demo() -> FlakyLM:
    real = example_lm()
    primary = FlakyLM(real, f"primary/{real.model}", primary_plan)
    fast = FlakyLM(real, f"fast/{real.model}", lambda turn, attempt: "ok")
    models = {"default": primary, "fast": fast}
    clock = DemoClock()
    breaker, stack = middleware_stack(clock)
    tools = {"weather": weather_tool(flaky=True), "save_note": note_tool(hang=False)}

    @component
    def Root():
        return [App(key="app", models=models, middleware=stack, tools=tools)]

    questions = [
        "What's the weather in Lisbon?",
        "And in Porto?",
        "And in Faro?",
        "Show me a chart for Lisbon",
        "Thanks! And Lisbon tomorrow?",
    ]
    with create_root(Root, session_id="weather") as root:
        trace(root, KINDS, printer())
        for i, q in enumerate(questions):
            if i == 4:
                clock.t += 31
                print("\n  ... 31 seconds later (simulated clock) ...")
            root.send(q, wait=True, timeout=600)
    print(f"\n  circuit {primary.model}: {breaker.state(primary.model)}; real model "
          f"calls: primary {primary.real_calls}, fast {fast.real_calls}")
    return primary


# ---------------------------------------------------------------------------
# Part 2: kill -9 mid-turn, then resume
# ---------------------------------------------------------------------------
def child(mode: str, log_path: str) -> None:
    from pyreact_durable import SessionLog, durable_lm

    slog = SessionLog.open(log_path)
    dlm = durable_lm(example_lm(), slog)
    _, stack = middleware_stack()
    tools = {"weather": weather_tool(flaky=False), "save_note": note_tool(hang=mode == "crash")}

    @component
    def Root():
        return [App(key="app", models={"default": dlm}, middleware=stack, tools=tools,
                    durable_log=slog)]

    # the same session id in both processes: the log's turns are this root's
    root = create_root(Root, session_id="weather")
    trace(root, KINDS, printer("    |"))
    if mode == "crash":
        root.send("What's the weather in Porto?")
        time.sleep(300)  # never returns: the parent kills us mid-turn
        return
    # Re-run the turn the log saw start but never end. TODO(api): there is no
    # Root.resume_pending() yet; the session is the root's low-level side.
    resumed = root.session.resume_pending(wait=True, timeout=300)
    print(f"    | resumed: {resumed}; durable_lm: {dlm.stats}", flush=True)
    root.unmount()
    slog.close()


def crash_and_resume(log_dir: str) -> None:
    log_path = os.path.join(log_dir, "resilient-agent.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    cmd = [sys.executable, "-m", "examples.harness.02_resilient_agent", "--log-dir", log_dir]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    print(f"\n=== crash & resume (log: {log_path}) ===")
    print("  process 1: answers, then hangs while saving the note")
    proc = subprocess.Popen([*cmd, "--child", "crash"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env)
    answer, deadline = None, time.time() + 300
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if line.startswith("    |  assistant:"):
                answer = line.split("assistant:", 1)[1].strip()
            if "kill me now" in line or time.time() > deadline:
                break
        os.kill(proc.pid, signal.SIGKILL)  # kill -9, mid-turn
        proc.wait(10)
    finally:
        if proc.poll() is None:
            proc.kill()
    print(f"  process 1: killed with SIGKILL (exit {proc.returncode}) before the turn ended")

    print("  process 2: same root, resume_pending()")
    out = subprocess.run([*cmd, "--child", "resume"], capture_output=True, text=True, env=env,
                         timeout=600)
    print(out.stdout, end="")
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
        raise SystemExit(out.returncode)
    again = [ln.split("assistant:", 1)[1].strip() for ln in out.stdout.splitlines()
             if ln.startswith("    |  assistant:")]
    print(f"  same answer after resume: {bool(again) and again[-1] == answer}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=None, help="where the durable log goes (default: temp)")
    ap.add_argument("--child", choices=["crash", "resume"], help=argparse.SUPPRESS)
    args = ap.parse_args()
    try:
        import dotenv

        dotenv.load_dotenv()
    except ImportError:  # pragma: no cover
        pass
    log_dir = args.log_dir or tempfile.mkdtemp(prefix="pyreact-resilient-")
    if args.child:
        child(args.child, os.path.join(log_dir, "resilient-agent.jsonl"))
        return
    print("=== resilience: retry, fallback model, circuit breaker, ErrorBoundary ===")
    resilience_demo()
    crash_and_resume(log_dir)


if __name__ == "__main__":
    main()
