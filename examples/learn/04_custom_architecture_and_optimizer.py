"""04 - Bring your own architecture and optimizer.

1. ``register_architecture("majority", ...)``: self-consistency -- sample a
   base program N times concurrently and return the majority answer. From
   then on it is a value of the ``arch`` prop like any built-in:
   ``use_agent_call(sig, arch="majority", n=3)`` (same Call, same events).

2. ``register_optimizer("instruction_menu", ...)``: no LLM proposals -- try
   a menu of hand-written instructions on the validation set and keep the
   best. ``optimize(..., optimizer="instruction_menu", candidates=[...])``
   gets baseline/score/calls/diff/save/hot_swap for free.

    uv run python -m examples.learn.04_custom_architecture_and_optimizer

LM: ``examples._lm`` (``claude -p`` by default; ``PYREACT_EXAMPLE_LM=openai|offline``). ~17 LM calls.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import dspy

from examples._devtools import collect, show_messages
from examples._lm import example_lm, out_path
from integrations import DSPyProvider, use_agent_call
from integrations.architectures import build_module
from integrations.learn import evaluate, optimize, register_architecture, register_optimizer
from pyreact.boot import create_root
from pyreact.components.message import Message
from pyreact.core import component, hooks, use_turn

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"args": []}


# ---------------------------------------------------------------------------
# 1. A custom architecture
# ---------------------------------------------------------------------------
class MajorityVote(dspy.Module):
    """Self-consistency: sample a base program N times, return the majority answer."""

    def __init__(self, signature, n: int = 3, base: str = "predict"):
        super().__init__()
        self.voter = build_module(signature, base)  # any registered arch
        self.n = n
        self.field = list(dspy.ensure_signature(signature).output_fields)[-1]

    def _vote(self, preds):
        votes = Counter(str(getattr(p, self.field)).strip().lower() for p in preds)
        winner, count = votes.most_common(1)[0]
        best = next(p for p in preds if str(getattr(p, self.field)).strip().lower() == winner)
        return dspy.Prediction(**{**best, "votes": dict(votes), "agreement": count / len(preds)})

    def forward(self, **inputs):
        return self._vote([self.voter(**inputs, config={"rollout_id": i, "temperature": 1.0})
                           for i in range(self.n)])

    async def aforward(self, **inputs):  # use_call awaits this: samples run concurrently
        preds = await asyncio.gather(*[
            self.voter.acall(**inputs, config={"rollout_id": i, "temperature": 1.0})
            for i in range(self.n)])
        return self._vote(preds)


register_architecture("majority", MajorityVote,
                      doc="Self-consistency: N samples of a base arch, majority vote.")

SIG = "riddle: str -> answer: str"


@component
def Riddler():
    turn = use_turn()
    call = use_agent_call(SIG, arch="majority", n=3)  # <- the custom arch, as a prop
    hooks.use_effect(lambda: turn and call.run(riddle=turn.text), [turn and turn.id])
    if call.error is not None:
        return [Message(key=f"error-{call.ver}", text=f"error: {call.error}", sender="system",
                        message_type="error")]
    if call.result is not None:
        r = call.result  # the majority answer, plus how the samples voted
        return [Message(key=f"answer-{call.ver}", sender="assistant",
                        text=f"{r.answer}  (votes={r.votes}, agreement={r.agreement:.0%})")]
    return []


# ---------------------------------------------------------------------------
# 2. A custom optimizer
# ---------------------------------------------------------------------------
def instruction_menu(program, *, trainset, valset, metric, lm, candidates=(), **_):
    """Try hand-written instructions on the valset; keep the best."""
    scored = []
    for text in [None, *candidates]:  # None = keep the current instructions
        p = program.deepcopy()
        if text is not None:
            for _name, pred in p.named_predictors():
                pred.signature = pred.signature.with_instructions(text)
        score, _rows = evaluate(p, valset or trainset, metric, lm=lm)
        print(f"   menu: {score:.2f}  {text or '(current)'}"[:110], flush=True)
        scored.append((score, p))
    best_score, best = max(scored, key=lambda s: s[0])
    return best, {"menu_scores": [s for s, _ in scored]}


register_optimizer("instruction_menu", instruction_menu, budget_doc="(unused)")


class TicketSig(dspy.Signature):
    """Route the support ticket to one team: billing, tech, account or retention."""

    ticket: str = dspy.InputField()
    team: str = dspy.OutputField(desc="one of: billing, tech, account, retention")


VAL = [
    dspy.Example(ticket="I dispute this charge, I never ordered it", team="account"),
    dspy.Example(ticket="My 2FA codes are not accepted", team="tech"),
    dspy.Example(ticket="Need a receipt for my last payment", team="billing"),
]
VAL = [e.with_inputs("ticket") for e in VAL]
HOUSE_RULES = ("Route the support ticket to one team: billing, tech, account or retention. "
               "House rules: refunds and charge disputes -> account; login, password and 2FA "
               "-> tech; invoices, receipts, payment methods -> billing; any wish to cancel "
               "or switch provider -> retention.")


def team_match(example, pred):
    return str(getattr(pred, "team", "")).strip().strip(".").lower() == example.team


def main() -> None:
    lm = example_lm()

    print("1) custom architecture 'majority' inside a component")

    @component
    def Root():
        return [DSPyProvider(key="dspy", lm=lm, children=[Riddler(key="riddler")])]

    with create_root(Root, session_id="riddler") as root:
        show_messages(root)
        ends = collect(root, ["llm.end"])  # devtools: latency + LM calls of the call
        root.send("I speak without a mouth and hear without ears. What am I? One word.",
                  wait=True, timeout=300)
    end = ends[-1].payload
    print(f"   arch={end.get('module')} LM calls={end['usage']['lm_calls']} "
          f"latency={end['latency']:.1f}s")

    print("\n2) custom optimizer 'instruction_menu'")
    res = optimize(build_module(TicketSig, "predict"), trainset=VAL, valset=VAL,
                   metric=team_match, optimizer="instruction_menu", lm=lm,
                   candidates=[HOUSE_RULES])
    res.save(out_path("04_instruction_menu.json"))
    print(res.report())
    print(res.diff())


if __name__ == "__main__":
    main()
