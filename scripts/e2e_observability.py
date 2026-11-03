"""End-to-end observability check with a real LLM (``claude -p`` CLI).

Runs the tool-calling store app (examples/tools.py) for two questions with a
JSONLSink + TurnTimeline + UsageAggregator attached and prints the timeline.

    uv run python -m scripts.e2e_observability [--jsonl PATH] [--max-calls 15]

Notes
- ``ClaudeCLILM`` reports APPROXIMATE char-based usage (``~est``). Pass
  ``--no-usage`` to make it report none and see the aggregator count calls
  "without usage" instead.
- Until the dspy layer publishes ``llm.*`` events itself, this script wraps the
  LM to publish ``llm.start/llm.end/llm.error`` (source ``e2e-lm``). Pass
  ``--no-lm-events`` once the integration emits them, to avoid double counting.
- Likewise, ``turn.start/turn.end`` are published around each question unless
  ``pyreact.core.turns`` exists (then the runtime is expected to emit them).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

from integrations.claude_cli_lm import CALLS, ClaudeCLILM
from pyreact.boot import bootstrap
from pyreact.core.core import component
from pyreact.observability import ConsoleSink, JSONLSink, TurnTimeline, UsageAggregator

QUESTIONS = [
    "Quanto custa a camisa vermelha (SHIRT-RED)?",
    "Quanto custa o frete do SHOE-RUN-42 para o BR?",
]


class InstrumentedLM(ClaudeCLILM):
    """ClaudeCLILM that publishes llm.* events and enforces a call budget."""

    def __init__(self, *a, role: str = "default", max_calls: int = 15, emit: bool = True, **kw):
        super().__init__(*a, **kw)
        self.role = role
        self.max_calls = max_calls
        self.emit = emit
        self.bus = None  # set after bootstrap

    def _pub(self, kind, payload):
        if self.emit and self.bus is not None:
            self.bus.publish(kind, {"name": f"lm:{self.role}", "model": self.model, **payload}, source="e2e-lm")

    async def aforward(self, prompt=None, messages=None, **kwargs):
        if len(CALLS) >= self.max_calls:
            err = RuntimeError(f"e2e LLM call budget ({self.max_calls}) exhausted")
            self._pub("llm.error", {"error": str(err)})
            raise err
        self._pub("llm.start", {})
        t0 = time.time()
        try:
            resp = await super().aforward(prompt=prompt, messages=messages, **kwargs)
        except BaseException as exc:
            self._pub("llm.error", {"error": f"{type(exc).__name__}: {exc}", "latency": time.time() - t0})
            raise
        self._pub("llm.end", {"usage": dict(resp.usage), "cost": None, "latency": time.time() - t0})
        return resp

    def forward(self, prompt=None, messages=None, **kwargs):
        if len(CALLS) >= self.max_calls:
            raise RuntimeError(f"e2e LLM call budget ({self.max_calls}) exhausted")
        self._pub("llm.start", {})
        t0 = time.time()
        try:
            resp = super().forward(prompt=prompt, messages=messages, **kwargs)
        except BaseException as exc:
            self._pub("llm.error", {"error": f"{type(exc).__name__}: {exc}", "latency": time.time() - t0})
            raise
        self._pub("llm.end", {"usage": dict(resp.usage), "cost": None, "latency": time.time() - t0})
        return resp


def _has_runtime_turns() -> bool:
    try:
        import pyreact.core.turns  # noqa: F401
    except Exception:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.path.join(tempfile.gettempdir(), "pyreact_e2e_events.jsonl"))
    ap.add_argument("--max-calls", type=int, default=15)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--no-usage", action="store_true", help="LM reports empty usage")
    ap.add_argument("--no-lm-events", action="store_true", help="don't publish llm.* from the LM wrapper")
    ap.add_argument("--timeout", type=float, default=240.0)
    args = ap.parse_args()

    from examples.tools import Root

    lms = {
        k: InstrumentedLM(
            args.model,
            role=k,
            max_calls=args.max_calls,
            emit=not args.no_lm_events,
            approx_usage=not args.no_usage,
        )
        for k in ["default", "fast", "reasoning"]
    }

    @component
    def Boot():
        return [Root(key="root", models=lms)]

    if os.path.exists(args.jsonl):
        os.remove(args.jsonl)

    app = bootstrap(Boot)
    jsonl = JSONLSink(args.jsonl)
    timeline = TurnTimeline()
    usage = UsageAggregator()
    console = ConsoleSink(kinds=("message", "error", "llm.end", "llm.error", "tool.*", "turn.*"))
    for sink in (jsonl, timeline, usage, console):
        sink.attach(app.events, replay=True)  # replay: includes the welcome message
    for lm in lms.values():
        lm.bus = app.event_bus

    manual_turns = not _has_runtime_turns()
    t_start = time.time()
    try:
        for i, q in enumerate(QUESTIONS, 1):
            tid = f"q{i}"
            print(f"\n>>> {q}")
            if manual_turns:
                app.event_bus.publish("turn.start", {"input": q}, turn_id=tid, source="e2e")
            try:
                app.invoke(q, wait=True, timeout=args.timeout)
            except TimeoutError as exc:
                print(f"[e2e] {exc}", file=sys.stderr)
            if manual_turns:
                app.event_bus.publish("turn.end", {}, turn_id=tid, source="e2e")
    finally:
        app.shutdown()
        jsonl.close()

    color = sys.stdout.isatty()
    print("\n=================== TURN TIMELINE ===================")
    print(timeline.render(color=color))
    print("\n======================= USAGE =======================")
    print(usage.render())
    with open(args.jsonl, encoding="utf-8") as fh:
        n = sum(1 for _ in fh)
    print(f"\nLLM subprocess calls: {len(CALLS)}  wall: {time.time() - t_start:.1f}s")
    print(f"JSONL: {args.jsonl} ({n} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
