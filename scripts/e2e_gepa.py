"""End-to-end: events -> dataset -> GEPA -> saved program -> live hot-swap.

Uses the ``claude -p`` CLI (``ClaudeCLILM("haiku")``) as student and
reflection LM. Small budget on purpose (~70-110 LLM calls, several minutes).

    uv run python scripts/e2e_gepa.py [--budget 40] [--skip-demo]

Writes ``artifacts/router_gepa.json`` (compiled program),
``artifacts/router_events.jsonl`` (recorded events) and
``artifacts/gepa_report.json`` (numbers).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dspy  # noqa: E402

from examples.optimize_router import (  # noqa: E402
    ROUTER_KEY,
    TRAIN,
    VAL,
    accuracy,
    demo_hot_swap,
    make_router,
    predict_all,
    record_runs,
    route_metric,
    to_examples,
)
from integrations import claude_cli_lm  # noqa: E402
from integrations.claude_cli_lm import ClaudeCLILM  # noqa: E402
from integrations.feedback import TraceRecorder  # noqa: E402
from integrations.optimize import (  # noqa: E402
    instructions_of,
    optimize_with_gepa,
    save_compiled,
)
from pyreact.core.events import EventBus  # noqa: E402


def log(*a):
    print(f"[e2e_gepa {time.strftime('%H:%M:%S')}]", *a, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40, help="GEPA max_metric_calls")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--skip-demo", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.artifacts, exist_ok=True)

    student_lm = ClaudeCLILM(args.model)
    reflection_lm = ClaudeCLILM(args.model)
    dspy.configure(lm=student_lm)
    t0 = time.time()
    report = {"budget": args.budget, "model": args.model}

    def calls():
        return len(claude_cli_lm.CALLS)

    # 1. baseline on val
    val = to_examples(VAL)
    base_rows = predict_all(make_router(), val, student_lm, threads=args.threads)
    report["baseline_val_acc"] = accuracy(base_rows)
    report["baseline_val"] = base_rows
    log("baseline val acc", report["baseline_val_acc"], "calls", calls())

    # 2. runs -> events -> dataset (train)
    events_path = os.path.join(args.artifacts, "router_events.jsonl")
    if os.path.exists(events_path):
        os.remove(events_path)
    bus = EventBus()
    recorder = TraceRecorder(bus, path=events_path)
    train_rows = record_runs(make_router(), to_examples(TRAIN), student_lm, recorder,
                             threads=args.threads)
    report["baseline_train_acc"] = accuracy(train_rows)
    trainset = recorder.dataset(ROUTER_KEY, input_keys=["message", "specialists"],
                                require_feedback=True)
    recorder.close()
    log("baseline train acc", report["baseline_train_acc"], "dataset", len(trainset),
        "calls", calls())

    # 3. GEPA (fresh log_dir: GEPA resumes from saved state in an existing one)
    gepa_logs = os.path.join(args.artifacts, "gepa_logs")
    shutil.rmtree(gepa_logs, ignore_errors=True)
    calls_before = calls()
    compiled = optimize_with_gepa(
        make_router(), trainset, route_metric,
        reflection_lm=reflection_lm, valset=val, budget=args.budget,
        lm=student_lm, num_threads=args.threads,
        log_dir=gepa_logs,
    )
    report["gepa_llm_calls"] = calls() - calls_before
    report["instructions_before"] = instructions_of(make_router())
    report["instructions_after"] = instructions_of(compiled)
    dr = getattr(compiled, "detailed_results", None)
    if dr is not None:
        report["gepa_val_scores"] = list(getattr(dr, "val_aggregate_scores", []) or [])
        report["gepa_best_idx"] = getattr(dr, "best_idx", None)
    out = save_compiled(compiled, os.path.join(args.artifacts, "router_gepa.json"))
    log("compiled saved to", out, "gepa calls", report["gepa_llm_calls"])

    # 4. optimized on val (fresh evaluation, same examples as baseline)
    opt_rows = predict_all(compiled, val, student_lm, threads=args.threads)
    report["optimized_val_acc"] = accuracy(opt_rows)
    report["optimized_val"] = opt_rows
    log("optimized val acc", report["optimized_val_acc"], "calls", calls())

    # 5. live hot-swap into a running app
    if not args.skip_demo:
        wrong = [r["message"] for r in base_rows if not r["ok"]]
        msg = wrong[0] if wrong else VAL[3][0]
        report["hot_swap"] = demo_hot_swap(student_lm, compiled, msg)
        log("hot swap", report["hot_swap"])

    report["total_llm_calls"] = calls()
    report["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.artifacts, "gepa_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log("DONE baseline", report["baseline_val_acc"], "-> optimized",
        report["optimized_val_acc"], "total calls", report["total_llm_calls"],
        f"{report['seconds']}s")


if __name__ == "__main__":
    main()
