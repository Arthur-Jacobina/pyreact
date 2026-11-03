"""``python -m integrations.learn`` -- optimize / inspect / diff from the shell.

    python -m integrations.learn optimize examples.optimize_router:make_router \\
        --data artifacts/router_events.jsonl --module-key supervisor \\
        --optimizer gepa --budget 40 --lm claude:haiku --out artifacts/router.json
    python -m integrations.learn optimize "question -> answer" --arch cot --data run.jsonl ...
    python -m integrations.learn inspect artifacts/router.json
    python -m integrations.learn diff artifacts/a.json artifacts/b.json
    python -m integrations.learn list

``<program>`` is ``module:attr`` (a ``dspy.Module``, a zero-arg factory or a
Signature class) or a signature string. ``--metric module:attr`` defaults to
``exact_match()`` over the label fields. LMs: ``claude:<model>`` uses the
``claude -p`` dev LM; anything else is a ``dspy.LM`` model string.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any, List, Optional


def _import(spec: str) -> Any:
    mod_name, sep, attr = spec.partition(":")
    if not sep:
        raise SystemExit(f"expected module:attr, got {spec!r}")
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def make_lm(spec: Optional[str]) -> Any:
    if not spec:
        return None
    import dspy

    if spec.startswith("claude:"):
        from integrations.claude_cli_lm import ClaudeCLILM

        return ClaudeCLILM(spec.split(":", 1)[1] or "haiku")
    return dspy.LM(spec)


def load_program(spec: str, arch: Optional[str] = None) -> Any:
    import dspy

    from integrations.architectures import build_module

    if "->" in spec:
        return build_module(spec, arch or "predict")
    obj = _import(spec)
    if isinstance(obj, dspy.Module):
        return obj.deepcopy()
    if isinstance(obj, type) and issubclass(obj, dspy.Signature):
        return build_module(obj, arch or "predict")
    if callable(obj):
        out = obj()
        if not isinstance(out, dspy.Module):
            raise SystemExit(f"{spec} returned {type(out).__name__}, not a dspy.Module")
        return out
    raise SystemExit(f"{spec} is not a dspy.Module, a factory, a Signature or a signature string")


def _budget(v: str) -> Any:
    return int(v) if v.isdigit() else v


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m integrations.learn",
                                 description="Optimize, inspect and diff DSPy programs.")
    sub = ap.add_subparsers(dest="command", required=True)

    o = sub.add_parser("optimize", help="optimize a program from recorded events")
    o.add_argument("program", help="module:attr (Module / factory / Signature) or 'a -> b'")
    o.add_argument("--data", action="append", required=True,
                   help="events JSONL (repeatable): TraceRecorder / JSONLSink / session log")
    o.add_argument("--module-key", required=True, help="module key, label or component key")
    o.add_argument("--arch", default=None, help="architecture when building from a signature")
    o.add_argument("--optimizer", default="gepa")
    o.add_argument("--budget", type=_budget, default="light", help="int or light|medium|heavy")
    o.add_argument("--metric", default=None, help="module:attr (default: exact_match())")
    o.add_argument("--input-keys", default=None, help="comma-separated input fields")
    o.add_argument("--min-score", type=float, default=None,
                   help="keep calls with feedback >= this (corrections always kept)")
    o.add_argument("--require-feedback", action="store_true")
    o.add_argument("--val", type=float, default=0.3, help="validation fraction (0 = none)")
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--lm", default="claude:haiku", help="student LM (claude:<m> or dspy model)")
    o.add_argument("--reflection-lm", default=None, help="default: same as --lm")
    o.add_argument("--threads", type=int, default=4)
    o.add_argument("--log-dir", default=None)
    o.add_argument("--resume", action="store_true", help="resume an existing --log-dir run")
    o.add_argument("--out", default=None, help="save the result JSON here")

    i = sub.add_parser("inspect", help="show a saved result (report + program)")
    i.add_argument("path")

    d = sub.add_parser("diff", help="diff the programs of two saved results")
    d.add_argument("a")
    d.add_argument("b")

    sub.add_parser("list", help="registered architectures and optimizers")
    return ap


def cmd_optimize(args) -> int:
    from . import dataset_from_events, exact_match, optimize, split

    program = load_program(args.program, args.arch)
    input_keys = args.input_keys.split(",") if args.input_keys else None
    data: List[Any] = []
    for path in args.data:
        data += dataset_from_events(path, args.module_key, input_keys=input_keys,
                                    min_score=args.min_score,
                                    require_feedback=args.require_feedback)
    if not data:
        print(f"no examples for {args.module_key!r} in {args.data}", file=sys.stderr)
        return 2
    train, val = split(data, val=args.val, seed=args.seed) if args.val else (data, [])
    metric = _import(args.metric) if args.metric else exact_match()
    lm = make_lm(args.lm)
    reflection_lm = make_lm(args.reflection_lm) if args.reflection_lm else lm
    print(f"[learn] {len(train)} train / {len(val)} val examples for {args.module_key!r}",
          flush=True)
    res = optimize(program, trainset=train, valset=val or None, metric=metric,
                   optimizer=args.optimizer, budget=args.budget, lm=lm,
                   reflection_lm=reflection_lm, log_dir=args.log_dir, resume=args.resume,
                   num_threads=args.threads, seed=args.seed, module_key=args.module_key)
    if args.out:
        res.save(args.out)
    print(res.report())
    print()
    print(res.diff())
    return 0


def cmd_inspect(args) -> int:
    from . import load

    res = load(args.path)
    print(res.report())
    print()
    print(res.diff())
    return 0


def cmd_diff(args) -> int:
    from .result import diff_states

    with open(args.a, encoding="utf-8") as f:
        a = json.load(f)
    with open(args.b, encoding="utf-8") as f:
        b = json.load(f)
    sa, sb = a.get("state", a), b.get("state", b)
    for label, d in (("a", a), ("b", b)):
        if "score" in d:
            print(f"{label}: {d.get('optimizer')} score={d.get('score')} "
                  f"(baseline {d.get('baseline')}, {d.get('calls')} calls)")
    print(diff_states(sa, sb, labels=(args.a, args.b)))
    return 0


def cmd_list(_args) -> int:
    from . import list_architectures, list_optimizers
    from .optimizers import get_optimizer

    print("architectures:")
    for n, doc in list_architectures().items():
        print(f"  {n:<12} {doc}")
    print("optimizers (budget int means):")
    for n, doc in list_optimizers().items():
        print(f"  {n:<12} {get_optimizer(n).budget_doc:<24} {doc}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return {"optimize": cmd_optimize, "inspect": cmd_inspect, "diff": cmd_diff,
            "list": cmd_list}[args.command](args)
