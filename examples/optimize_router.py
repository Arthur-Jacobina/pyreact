"""Learning from events: GEPA-optimize the store's supervisor and hot-swap it.

Pieces (the full, costed pipeline is ``scripts/e2e_gepa.py``):

- ``TRAIN`` / ``VAL``: labeled Portuguese messages for ``RouteSpecialistSig``
  (catalog / shipping / order), including tricky ones such as
  "comprar e saber o frete" (shipping) and "pronta entrega" (stock -> order).
- ``route_metric``: GEPA feedback metric that explains *why* a route is wrong.
- ``record_runs``: runs the router through a recorder-aware ``TracedModule``,
  then publishes ``feedback`` events; ``TraceRecorder.dataset`` turns them
  into the GEPA trainset (runs -> events -> dataset).
- ``RouterDemo``: a running app (a root) with the real ``SupervisorAgent``
  from ``examples/tools.py`` under ``LearnedProgram(key="supervisor",
  src=...)``; the hot swap is a new ``src`` (state of the app, set through
  its ``"learned"`` ref) -- a re-render, not a mutation of the module.

Run the live hot-swap demo with a previously saved program::

    uv run python -m examples.optimize_router --compiled artifacts/router_gepa.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import dspy

from examples.tools import SPECIALISTS, RouteSpecialistSig, SupervisorAgent
from integrations.dspy_integration import DSPyProvider
from integrations.feedback import TracedModule, TraceRecorder, send_feedback
from integrations.learn import LearnedProgram, use_learned
from integrations.optimize import instructions_of, load_compiled, module_key_for
from pyreact.boot import create_root
from pyreact.core import hooks, use_imperative_handle, use_turn
from pyreact.core.core import component

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"args": []}

ROUTER_KEY = module_key_for(dspy.ChainOfThought, RouteSpecialistSig, "supervisor")
CLASSES = ("catalog", "shipping", "order")

RULES = {
    "shipping": (
        "Anything about frete/entrega (cost, deadline, free shipping) goes to "
        "shipping, even when combined with purchase intent ('comprar e saber o "
        "frete') or phrased as 'quanto custa/sai' -- the customer needs the "
        "delivery answer before ordering."
    ),
    "order": (
        "Availability of a specific item ('ainda tem', 'em estoque', 'pronta "
        "entrega' = in stock now, NOT shipping) and placing/closing an order "
        "without a shipping question go to order."
    ),
    "catalog": (
        "What the store sells ('vocês vendem/têm <tipo de produto>?' asks about "
        "the catalog, not stock), listing products and product prices go to catalog."
    ),
}

# (message, gold)
TRAIN: List[Tuple[str, str]] = [
    ("quais produtos vocês têm?", "catalog"),
    ("quanto custa a camisa vermelha?", "catalog"),
    ("vocês vendem boné?", "catalog"),
    ("qual o produto mais barato?", "catalog"),
    ("quero comprar a camisa azul e ver quanto fica o frete", "shipping"),
    ("fechar pedido de 2 tênis e calcular a entrega pra SP", "shipping"),
    ("comprar chapéu + frete pro RJ", "shipping"),
    ("quero 3 camisas, quanto dá com o frete?", "shipping"),
    ("pedido acima de 100 reais tem frete grátis?", "shipping"),
    ("o tênis 42 tá disponível pra pronta entrega?", "order"),
    ("ainda tem camisa azul?", "order"),
    ("quero comprar 2 chapéus", "order"),
]
VAL: List[Tuple[str, str]] = [
    ("comprar e saber o frete", "shipping"),
    ("quanto sai a entrega do chapéu?", "shipping"),
    ("pedir duas camisas e calcular o frete", "shipping"),
    ("finalizar a compra do tênis e ver o prazo de entrega", "shipping"),
    ("qual o valor do tênis?", "catalog"),
    ("vocês têm camisa?", "catalog"),
    ("tem chapéu de lã pra pronta entrega?", "order"),
    ("quero fazer um pedido de 3 camisas vermelhas", "order"),
]


def to_examples(rows: List[Tuple[str, str]]) -> List[dspy.Example]:
    return [
        dspy.Example(message=m, specialists=SPECIALISTS, name=g).with_inputs(
            "message", "specialists"
        )
        for m, g in rows
    ]


def normalize(name: Any) -> str:
    s = str(name or "").strip().strip("'\"`.").lower()
    for c in CLASSES:
        if s == c:
            return c
    hits = [c for c in CLASSES if c in s]
    return hits[0] if len(hits) == 1 else s


def explain(message: str, predicted: str, gold: str) -> str:
    if predicted == gold:
        return f"Correct: '{message}' -> {gold}. {RULES[gold]}"
    return (
        f"Misrouted: '{message}' was sent to '{predicted or '<empty>'}' but the "
        f"correct specialist is '{gold}'. Rule: {RULES[gold]}"
        + (f" (Why not {predicted}: {RULES[predicted]})" if predicted in RULES else "")
    )


def route_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA feedback metric: 1/0 accuracy + a textual explanation."""
    predicted = normalize(getattr(pred, "name", None))
    score = float(predicted == gold.name)
    # Explain the *current* prediction (recorded feedback_text describes the
    # run that produced the example, which may have routed differently).
    feedback = explain(gold.message, predicted, gold.name)
    if pred_name is None:  # plain evaluation (dspy.Evaluate etc.)
        return score
    return dspy.Prediction(score=score, feedback=feedback)


def make_router() -> dspy.Module:
    return dspy.ChainOfThought(RouteSpecialistSig)


def predict_all(
    module: dspy.Module, examples: List[dspy.Example], lm: Any, *, threads: int = 4
) -> List[Dict[str, Any]]:
    """Run ``module`` over examples (threaded); returns per-example rows."""

    def one(ex):
        try:
            with dspy.context(lm=lm):
                pred = module(**ex.inputs())
            return pred, None
        except Exception as exc:
            return None, exc

    with cf.ThreadPoolExecutor(max_workers=threads) as pool:
        results = list(pool.map(one, examples))
    rows = []
    for ex, (pred, err) in zip(examples, results):
        p = normalize(getattr(pred, "name", None)) if pred is not None else None
        rows.append(
            {"message": ex.message, "gold": ex.name, "pred": p, "ok": p == ex.name,
             "error": None if err is None else str(err)[:200]}
        )
    return rows


def accuracy(rows: List[Dict[str, Any]]) -> float:
    return sum(r["ok"] for r in rows) / max(1, len(rows))


def record_runs(
    module: dspy.Module,
    examples: List[dspy.Example],
    lm: Any,
    recorder: TraceRecorder,
    *,
    threads: int = 4,
) -> List[Dict[str, Any]]:
    """Runs -> events: call through a TracedModule, then publish feedback.

    In production the feedback comes from users / a judge / downstream
    signals (e.g. the specialist reported "not my job"); here a labeler with
    the gold route plays that role.
    """
    traced = TracedModule(module, ROUTER_KEY, bus=recorder.bus, source="router")

    def one(ex):
        try:
            with dspy.context(lm=lm):
                pred = traced(**ex.inputs())
            # the TracedModule's last_call_id is racy across threads; read the
            # id back from the recorder by matching the message instead.
            return pred
        except Exception:
            return None

    with cf.ThreadPoolExecutor(max_workers=threads) as pool:
        preds = list(pool.map(one, examples))

    by_msg = {r["inputs"].get("message"): r for r in recorder.records(ROUTER_KEY)}
    rows = []
    for ex, pred in zip(examples, preds):
        rec = by_msg.get(ex.message)
        p = normalize(getattr(pred, "name", None)) if pred is not None else None
        if rec is not None:
            send_feedback(
                rec["call_id"],
                score=float(p == ex.name),
                text=explain(ex.message, p, ex.name),
                expected={"name": ex.name},
                bus=recorder.bus,
            )
        rows.append({"message": ex.message, "gold": ex.name, "pred": p, "ok": p == ex.name})
    return rows


# ---------------------------------------------------------------------------
# A running app with the real SupervisorAgent, for the hot-swap demo
# ---------------------------------------------------------------------------
def make_router_app(models: Dict[str, Any], routes: List[Tuple[str, str, int]]):
    """The store's supervisor under a ``LearnedProgram``. Refs: ``"learned"``
    (``swap(program)``: a new ``src``) and ``"router"`` (``program``: the
    learned program the supervisor reads)."""

    @component
    def RouterHome():
        turn = use_turn()  # a new turn per input, even when the text repeats
        program = use_learned("supervisor")  # what SupervisorAgent reads too
        use_imperative_handle("router", lambda: SimpleNamespace(program=program),
                              [id(program)])

        def on_route(name, message, ver):  # a callback prop (onRoute)
            routes.append((message, name, ver))

        return [
            SupervisorAgent(key="supervisor", message=turn.text if turn else "",
                            turn=turn.index if turn else 0, on_route=on_route),
        ]

    @component
    def RouterDemo():
        program, set_program = hooks.use_state(None)  # None: the baseline prompt
        # set_program(lambda _: p): a program is callable, so never pass it bare
        use_imperative_handle("learned", lambda: SimpleNamespace(
            swap=lambda p: set_program(lambda _old: p)), [])
        return [DSPyProvider(key="dspy", models=models, children=[
            LearnedProgram(key="supervisor", src=program, children=[RouterHome(key="home")]),
        ])]

    return RouterDemo


def ask(root, routes: List[Tuple[str, str, int]], message: str,
        timeout: float = 180) -> Optional[str]:
    n = len(routes)
    root.send(message, wait=True, timeout=timeout)
    return routes[-1][1] if len(routes) > n else None


def demo_hot_swap(lm: Any, compiled: dspy.Module, message: str) -> Dict[str, Any]:
    """Mount the app, route ``message``, swap ``compiled`` in as the learned
    program (a re-render), route again."""
    routes: List[Tuple[str, str, int]] = []
    models = {"default": lm, "fast": lm, "reasoning": lm}
    with create_root(make_router_app(models, routes), session_id="router-demo") as root:
        before = ask(root, routes, message)
        root.ref("learned").swap(compiled)  # LearnedProgram(src=compiled): a re-render
        root.wait_idle(timeout=30)
        after = ask(root, routes, message)
        in_tree = root.ref("router").program is compiled
    # Did the post-swap LM call actually carry the optimized instructions?
    new_instr = next(iter(instructions_of(compiled).values()), "")
    probe = new_instr.strip().splitlines()[0][:60] if new_instr.strip() else ""
    last = (getattr(lm, "history", None) or [{}])[-1]
    msgs = last.get("messages") or [{"content": last.get("prompt") or ""}]
    prompt_text = "\n".join(str(m.get("content", "")) for m in msgs)
    return {
        "message": message,
        "before": before,
        "after": after,
        "learned_in_tree": in_tree,
        "prompt_has_new_instructions": bool(probe) and probe in prompt_text,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compiled", default="artifacts/router_gepa.json")
    ap.add_argument("--message", default="tem chapéu de lã pra pronta entrega?")
    ap.add_argument("--model", default="haiku")
    args = ap.parse_args(argv)

    from examples._lm import example_lm

    lm = example_lm(args.model)
    compiled = load_compiled(dspy.ChainOfThought, RouteSpecialistSig, args.compiled)
    print(demo_hot_swap(lm, compiled, args.message))


if __name__ == "__main__":
    main()
