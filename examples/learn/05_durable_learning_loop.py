"""05 - The whole loop across restarts: serve -> journal -> learn -> serve better.

Two decoupled libraries meet only at the border of the flow:

- ``pyreact_durable`` journals what happened (EventBus events) into an
  append-only session log, the same way it persists state. The app does not
  know it is being recorded.
- ``integrations.learn`` reads that journal later, offline, as a dataset;
  GEPA turns the user's feedback into a better prompt; the optimized program is
  saved as an artifact and hot-swapped into the next run.

Phases (each one is a separate process in real life)::

    # 1. serve: a simulated customer chats with the store router and rates
    #    each answer (feedback events); everything lands in the journal
    uv run python -m examples.learn.05_durable_learning_loop serve

    # 2. learn: journal -> dataset -> GEPA -> artifacts/learn/router_from_journal.json
    uv run python -m examples.learn.05_durable_learning_loop learn

    # 3. serve again with the optimized program (loaded + hot-swapped at boot)
    uv run python -m examples.learn.05_durable_learning_loop serve --program artifacts/learn/router_from_journal.json

    # or all three in a row, then compare accuracy
    uv run python -m examples.learn.05_durable_learning_loop all

    # inspect the journal at any time (works while the app is writing it)
    uv run python -m pyreact_durable show runs/store_router.jsonl
    uv run python -m pyreact_durable events runs/store_router.jsonl --kind feedback

LM: ``examples._lm`` (``claude -p`` by default; ``PYREACT_EXAMPLE_LM=openai|offline``).
``all`` makes roughly 50-70 LM calls.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional, Tuple

from examples.optimize_router import TRAIN, VAL, normalize, route_metric, to_examples
from examples.tools import SPECIALISTS, RouteSpecialistSig
from integrations import DSPyProvider, use_call, use_module
from examples._lm import example_lm
from integrations.feedback import send_feedback
from integrations.learn import build_module, dataset_from_events, load, optimize
from pyreact.boot import bootstrap
from pyreact.core import component, hooks, use_turn
from pyreact_durable import Durable, SessionLog

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"args": ["all", "--log", "{tmp}/store.jsonl", "--out", "{tmp}/router.json", "--budget", "16"]}

LOG = "runs/store_router.jsonl"
ARTIFACT = "artifacts/learn/router_from_journal.json"  # default --out
KEY = "supervisor"  # module name -> the handle hot_swap / datasets use


# ---------------------------------------------------------------- the app
@component
def StoreRouter(results):
    """Routes each turn's message to a specialist. Knows nothing about
    journaling, datasets or optimizers."""
    turn = use_turn()
    router = use_module(RouteSpecialistSig, arch="cot", name=KEY)
    call = use_call(router)

    def route():
        if turn is not None and turn.text.strip():
            call.run(message=turn.text, specialists=SPECIALISTS)

    hooks.use_effect(route, [turn and turn.id])

    def record():
        if turn is not None and call.result is not None:
            results[turn.id] = normalize(getattr(call.result, "name", ""))

    hooks.use_effect(record, [call.ver])
    return []


def make_app(lm, log_path: str, results: dict):
    @component
    def App():
        # The ONLY durable-specific line: wrap the tree, pick what to journal.
        return [
            Durable(
                key="durable",
                log=log_path,
                journal=("turn.*", "llm.*", "feedback"),
                children=[
                    DSPyProvider(
                        key="dspy",
                        lm=lm,
                        children=[StoreRouter(key="router", results=results)],
                    )
                ],
            )
        ]

    return App


# ---------------------------------------------------------------- phases
def serve(
    rows: List[Tuple[str, str]],
    *,
    log_path: str,
    program: Optional[str] = None,
    rate: bool = True,
    lm=None,
) -> float:
    """Chat through ``rows`` (message, gold route); optionally rate answers.

    Returns accuracy. With ``program``, the saved optimization result is
    loaded and hot-swapped into the running app before the first message.
    """
    lm = lm or example_lm()
    results: dict = {}
    app = bootstrap(make_app(lm, log_path, results))
    try:
        if program:
            swapped = load(program).hot_swap(app, KEY)
            print(f"  hot-swapped {program} into {len(swapped)} module(s)")
        correct = 0
        for message, gold in rows:
            tid = app.invoke(message, wait=True, timeout=180)
            got = results.get(tid, "?")
            ok = got == gold
            correct += ok
            print(f"  {'✓' if ok else '✗'} {got:<9} gold={gold:<9} {message}")
            if rate:
                # Feedback is just another event; the journal keeps it.
                send_feedback(
                    tid,
                    1.0 if ok else 0.0,
                    "correct" if ok else f"wrong: {message!r} is a {gold} question",
                    expected=None if ok else {"name": gold},
                    bus=app.event_bus,
                )
        app.wait_idle(timeout=30)
        return correct / max(1, len(rows))
    finally:
        app.shutdown()


def learn(*, log_path: str, out: str, budget: int = 24, lm=None):
    """Journal -> dataset -> GEPA -> artifact. Runs offline (no app)."""
    lm = lm or example_lm()

    # 1. The journal is the dataset. Only *rated* calls count (unrated ones,
    #    e.g. the held-out evaluation, would leak and carry unchecked labels);
    #    feedback corrections (expected={"name": ...}) become the labels.
    records = dataset_from_events(log_path, KEY, require_feedback=True)
    print(f"  {len(records)} rated calls found in {log_path}")

    # 2. ``specialists`` is a constant list (and long: llm.start only keeps a
    #    preview of each input), so rebuild examples from the message + label.
    rows = [(ex.message, normalize(ex.name)) for ex in records]
    trainset = to_examples(rows)

    # 3. One call for any optimizer; the metric is the same one used offline.
    program = build_module(RouteSpecialistSig, "cot")
    result = optimize(
        program,
        trainset=trainset,
        valset=to_examples(VAL),
        metric=route_metric,
        optimizer="gepa",
        budget=budget,
        lm=lm,
        reflection_lm=lm,
    )
    print(result.report())
    print(result.diff())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    result.save(out)
    print(f"  saved {out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", choices=["serve", "learn", "all"])
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--program", default=None, help="optimized artifact to hot-swap")
    ap.add_argument("--out", default=ARTIFACT)
    ap.add_argument("--budget", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    if args.phase == "serve":
        rows = VAL if args.program else TRAIN
        acc = serve(rows, log_path=args.log, program=args.program)
        print(f"accuracy: {acc:.2f}")
        return
    if args.phase == "learn":
        learn(log_path=args.log, out=args.out, budget=args.budget)
        return

    # all: fresh journal, then the three phases
    if os.path.exists(args.log):
        os.replace(args.log, args.log + f".{int(time.time())}.bak")
    print("== 1. serve (baseline) on the training conversations, with feedback")
    serve(TRAIN, log_path=args.log)
    print("== baseline on held-out messages (no feedback)")
    before = serve(VAL, log_path=args.log, rate=False)
    print("== 2. learn from the journal")
    learn(log_path=args.log, out=args.out, budget=args.budget)
    print("== 3. serve again with the optimized program")
    after = serve(VAL, log_path=args.log, program=args.out, rate=False)
    log = SessionLog.open(args.log, readonly=True)
    print(f"\nheld-out accuracy: {before:.2f} -> {after:.2f}")
    print(f"journal: {args.log} ({log.stats()})")


if __name__ == "__main__":
    main()
