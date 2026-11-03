"""01 - Architectures are props: same signature, three ways of thinking.

One component, one signature (``context, question -> answer``). The ``arch``
prop decides *how* it thinks:

    predict  one LM call, answer straight away
    cot      reason step by step, then answer
    rlm      write Python in a sandboxed REPL to explore the context (Deno),
             asking a sub-LM about pieces when useful

Everything else is identical: the same ``Call`` (run/result/loading/error),
the same ``llm.*`` events (that's where latency and LM-call counts below come
from), the same cancellation. Each user input is a turn naming the arch to use.

    uv run python -m examples.learn.01_architectures            # predict, cot, rlm
    uv run python -m examples.learn.01_architectures cot rlm    # pick some

Uses the ``claude -p`` dev LM (no API key). ~8 LM calls.
"""

from __future__ import annotations

import random
import sys
import time

from integrations import DSPyProvider, use_agent_call
from examples._lm import OFFLINE, example_lm
from integrations.learn import explain
from pyreact.boot import bootstrap
from pyreact.core import component, hooks, use_turn

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"args": []}

SIG = "context: str, question: str -> answer: str"

# Per-architecture options (kwargs of use_module / use_agent_call).
ARCH_OPTIONS = {
    "predict": {},
    "cot": {},
    "rlm": {"max_iterations": 6, "max_llm_calls": 4, "sub_model": "fast"},
}


def make_orders(n: int = 300, seed: int = 7) -> tuple[str, int]:
    """A log too tedious to count by eye; returns (text, true answer)."""
    rng = random.Random(seed)
    lines, truth = [], 0
    for i in range(1, n + 1):
        state = rng.choice(["SP", "RJ", "MG", "RS"])
        status = rng.choice(["delivered", "in_transit", "cancelled"])
        qty = rng.randint(1, 5)
        truth += qty if (state == "RJ" and status == "delivered") else 0
        lines.append(f"order #{i:03d} | dest={state} | status={status} | qty={qty}")
    return "\n".join(lines), truth


CONTEXT, TRUTH = make_orders()
QUESTION = ("What is the total qty of all orders with dest=RJ and status=delivered? "
            "Answer with just the number.")


@component
def Solver(arch: str = "predict", turn: int = 0):
    # The one line that changes between architectures: the `arch` prop.
    call = use_agent_call(SIG, arch=arch, **ARCH_OPTIONS.get(arch, {}))

    def ask():
        if turn:
            call.run(context=CONTEXT, question=QUESTION)

    hooks.use_effect(ask, [turn])
    return []


@component
def App():
    turn = use_turn()  # each input line is a turn: "predict", "cot", "rlm"
    arch = turn.text.strip() if turn is not None else "predict"
    return [Solver(key="solver", arch=arch, turn=turn.index if turn else 0)]


def main(archs: list[str]) -> None:
    lm = example_lm()
    models = {"default": lm, "fast": example_lm()}

    @component
    def Root():
        return [DSPyProvider(key="dspy", models=models, children=[App(key="app")])]

    app = bootstrap(Root)
    ends, iters = {}, {}

    # The trace you debug with: llm.end carries latency, usage and outputs.
    def on_event(ev):
        p = ev.payload
        if ev.kind in ("llm.end", "llm.error"):
            ends[p["call_id"]] = p
        elif ev.kind == "rlm.iteration":
            print(f"   rlm step {p['iteration']}/{p['max_iterations']} "
                  f"(sub-LM calls so far: {p['sub_llm_calls']})", flush=True)
            iters[p["call_id"]] = p["iteration"]

    app.event_bus.subscribe(on_event, ["llm.*", "rlm.*"])
    print(f"context: {len(CONTEXT.splitlines())} order lines, {len(CONTEXT)} chars; "
          f"true answer = {TRUTH}\n")
    rows = []
    try:
        for arch in archs:
            print(f"-> arch={arch}", flush=True)
            n = len(ends)
            t0 = time.time()
            app.invoke(arch, wait=True, timeout=900)
            new = list(ends.values())[n:]
            p = new[-1] if new else {}
            answer = (p.get("outputs") or {}).get("answer", p.get("error", "?"))
            rows.append((arch, str(answer).strip(), str(answer).strip() == str(TRUTH),
                         time.time() - t0, (p.get("usage") or {}).get("lm_calls"),
                         p.get("iterations"), p.get("call_id")))
    finally:
        app.shutdown()

    print(f"\n{'arch':<8} {'answer':<10} {'ok':<4} {'latency':>8} {'LM calls':>9} {'REPL steps':>11}")
    for arch, ans, ok, lat, calls, steps, _cid in rows:
        print(f"{arch:<8} {ans[:10]:<10} {'yes' if ok else 'no':<4} {lat:>7.1f}s "
              f"{calls if calls is not None else '?':>9} {steps if steps else '-':>11}")

    # Debugging: what exactly did the LM see for the first call?
    if rows and rows[0][-1]:
        print("\nexplain(first call), truncated:\n")
        print(str(explain(rows[0][-1], max_chars=400))[:1500])


if __name__ == "__main__":
    # rlm needs a real model writing code in the sandbox; skipped offline
    main(sys.argv[1:] or (["predict", "cot"] if OFFLINE else ["predict", "cot", "rlm"]))
