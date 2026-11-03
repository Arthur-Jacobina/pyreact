"""02 - Optimizers are build steps: same data, same metric, three optimizers.

A support-ticket router with house rules the model can't guess (refunds go
to *account*, anything about leaving goes to *retention*, 2FA is *tech*).
We run ``bootstrap`` (few-shot demos), ``mipro`` (instructions + demos) and
``gepa`` (reflective instruction rewriting from metric feedback) with tiny
budgets, print a comparison table and the winning program's diff.

The metric is written once, the simple way -- ``(example, pred) -> (score,
feedback)`` -- and ``optimize`` adapts it to each optimizer (GEPA reads the
feedback text; the others use the score).

    uv run python -m examples.learn.02_optimizers                 # all three
    uv run python -m examples.learn.02_optimizers gepa            # just one

LM: ``examples._lm`` (``claude -p`` by default; ``PYREACT_EXAMPLE_LM=openai|offline``).
~60 LM calls with the budgets below.
Results are saved to ``artifacts/learn/02_<optimizer>.json``; compare them
with ``python -m integrations.learn diff a.json b.json``.
"""

from __future__ import annotations

import sys

import dspy

from integrations.architectures import build_module
from examples._lm import example_lm, out_path
from integrations.learn import evaluate, optimize

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"args": ["bootstrap", "gepa"]}


class TicketSig(dspy.Signature):
    """Route the support ticket to one team: billing, tech, account or retention."""

    ticket: str = dspy.InputField()
    team: str = dspy.OutputField(desc="one of: billing, tech, account, retention")


TEAMS = ("billing", "tech", "account", "retention")
RULES = {
    "account": "Refunds and charge disputes go to account (not billing).",
    "retention": "Any intent to cancel, leave or switch provider goes to retention, "
                 "whatever the reason.",
    "tech": "Login, password and 2FA problems go to tech, like bugs and slowness.",
    "billing": "Invoices, receipts and payment-method changes go to billing.",
}

TRAIN = [
    ("I was charged twice this month, please refund one of them", "account"),
    ("Can you send me a copy of my March invoice?", "billing"),
    ("I want to cancel, your app keeps crashing", "retention"),
    ("I can't log in, the password reset email never arrives", "tech"),
    ("How do I update the credit card on file?", "billing"),
    ("The dashboard takes forever to load", "tech"),
]
VAL = [
    ("I dispute this charge, I never ordered it", "account"),
    ("Thinking about switching to another provider because of the price", "retention"),
    ("My 2FA codes are not accepted", "tech"),
    ("Need a receipt for my last payment", "billing"),
]


def examples(rows):
    return [dspy.Example(ticket=t, team=team).with_inputs("ticket") for t, team in rows]


def normalize(team) -> str:
    s = str(team or "").strip().strip("'\"`.").lower()
    hits = [t for t in TEAMS if t in s]
    return hits[0] if len(hits) == 1 else s


def route_metric(example, pred):
    """One metric for every optimizer: a score plus *why* (GEPA learns from it)."""
    got = normalize(getattr(pred, "team", None))
    if got == example.team:
        return 1.0, f"Correct: {example.team}. {RULES[example.team]}"
    return 0.0, (f"'{example.ticket}' was routed to '{got}' but belongs to "
                 f"'{example.team}'. Rule: {RULES[example.team]}")


# optimizer -> (budget, extra options). Budgets are tiny on purpose.
PLAN = {
    "bootstrap": (2, {}),  # up to 2 bootstrapped demos
    "mipro": (3, dict(max_bootstrapped_demos=1, max_labeled_demos=1,  # 3 trials
                      data_aware_proposer=False)),
    "gepa": (14, {}),  # 14 metric calls (4 to score the seed on val)
}


def main(names: list[str]) -> None:
    lm = example_lm()
    train, val = examples(TRAIN), examples(VAL)
    program = build_module(TicketSig, "predict")

    # Baseline once (each optimize() would otherwise re-measure it).
    baseline, rows = evaluate(program, val, route_metric, lm=lm)
    print(f"baseline on val: {baseline:.2f}  "
          f"(wrong: {[r['inputs']['ticket'][:30] for r in rows if r['score'] < 1]})\n")

    results = {}
    for name in names:
        budget, opts = PLAN[name]
        print(f"=== {name} (budget={budget}) ===", flush=True)
        res = optimize(program, trainset=train, valset=val, metric=route_metric,
                       optimizer=name, budget=budget, lm=lm, reflection_lm=lm,
                       baseline=baseline, num_threads=4,
                       on_progress=lambda phase, info: print(f"  [{phase}] {info}", flush=True),
                       **opts)
        res.save(out_path(f"02_{name}.json"))
        results[name] = res

    print(f"\n{'optimizer':<10} {'budget':>6} {'val before':>10} {'val after':>9} "
          f"{'LM calls':>9} {'seconds':>8}")
    for name, r in results.items():
        print(f"{name:<10} {str(r.budget):>6} {r.baseline:>10.2f} {r.score:>9.2f} "
              f"{r.calls:>9} {r.seconds:>8.0f}")

    best = max(results.values(), key=lambda r: (r.score, -r.calls))
    print(f"\nbest: {best.optimizer}\n{best.report()}\n\n--- diff ---\n{best.diff()}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(PLAN))
