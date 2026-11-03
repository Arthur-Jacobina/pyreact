"""03 - Live feedback loop: chat, rate answers, optimize, hot-swap -- no restart.

The store's router (``RouteSpecialistSig`` from ``examples/tools.py``) picks a
specialist (catalog / shipping / order) for each customer message. While the
app runs you rate its answers; ``:optimize`` turns those ratings (events) into
a dataset, runs GEPA in a worker thread and hot-swaps the better program into
the running component (HMR). The next message already uses the new prompt.

The learned program is a prop: ``LearnedProgram(key="supervisor", src=...)``
wraps the router and the router reads it with ``use_learned("supervisor")``.
``:optimize`` saves the result and makes it that provider's new ``src`` -- a
re-render, not a mutation of the module cache. Start from a saved program
with ``--program artifacts/learn/<file>.json``.

Interactive::

    uv run python -m examples.learn.03_live_feedback_loop
    >> quero comprar a camisa azul e ver quanto fica o frete
    [router] -> order
    >> :feedback - frete junto com compra é shipping name=shipping
    >> :feedback +              (on a good answer)
    >> :optimize supervisor gepa 14
    >> :modules   :inspect supervisor   :explain   :events optimize   :help

Scripted (a simulated user rates each answer against a gold route)::

    uv run python -m examples.learn.03_live_feedback_loop --demo

LM: ``examples._lm`` (``claude -p`` by default; ``PYREACT_EXAMPLE_LM=openai|offline``).
``--demo`` makes ~25 LM calls.
"""

from __future__ import annotations

import argparse
import asyncio

from examples.optimize_router import TRAIN, VAL, normalize, route_metric, to_examples
from examples.tools import SPECIALISTS, RouteSpecialistSig
from integrations import DSPyProvider, use_call, use_module
from examples._lm import example_lm, out_path
from integrations.learn import (
    LearnedProgram,
    configure_learning,
    find_registries,
    use_learned,
    wait_optimizations,
)
from pyreact.boot import bootstrap, handle_command, read_terminal_and_invoke
from pyreact.components.message import Message
from pyreact.core import component, hooks, use_turn

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"args": ["--demo"]}


@component
def StoreRouter():
    turn = use_turn()
    # name="supervisor" -> module key ("ChainOfThought", "RouteSpecialistSig", "supervisor"):
    # the handle :optimize / :inspect use. program=: the learned program the
    # LearnedProgram above provides (None until :optimize swaps one in).
    router = use_module(RouteSpecialistSig, arch="cot", name="supervisor",
                        program=use_learned("supervisor"))
    call = use_call(router)

    def route():
        if turn is not None and turn.text.strip():
            call.run(message=turn.text, specialists=SPECIALISTS)

    hooks.use_effect(route, [turn and turn.id])

    if call.loading:
        return [Message(key="status", text="roteando…", sender="system", message_type="info")]
    if call.error is not None:
        return [Message(key="err", text=f"erro: {call.error}", sender="system",
                        message_type="error")]
    if call.result is not None:
        return [Message(key=f"route-{call.ver}", text=f"[router] -> {normalize(call.result.name)}",
                        sender="assistant")]
    return []


def make_root(lm, program=None):
    @component
    def Root():
        return [DSPyProvider(key="dspy", lm=lm, children=[
            # src: a saved result (or None: the baseline). :optimize updates it.
            LearnedProgram(key="supervisor", src=program, children=[StoreRouter(key="router")]),
        ])]

    return Root


def configure(budget: int = 14) -> None:
    """What :optimize uses: GEPA, the router's feedback metric, a few extra
    labeled examples and a fixed validation set (so scores are comparable)."""
    configure_learning(
        metric=route_metric,  # a GEPA-style 5-arg metric works as-is
        optimizer="gepa",
        budget=budget,
        trainset=to_examples(TRAIN[4:7]),  # seed data on top of your ratings
        valset=to_examples(VAL[:4]),
        min_examples=3,
        save_dir=out_path(),
    )


DEMO = [
    ("quero comprar a camisa azul e ver quanto fica o frete", "shipping"),
    ("quais produtos vocês têm?", "catalog"),
    ("o tênis 42 tá disponível pra pronta entrega?", "order"),
]
AFTER = ("comprar e saber o frete", "shipping")


def demo(lm) -> None:
    app = bootstrap(make_root(lm))
    routes = []
    app.event_bus.subscribe(
        lambda ev: routes.append(normalize((ev.payload.get("outputs") or {}).get("name"))),
        ["llm.end"])

    def say(text):
        print(f"\n>> {text}", flush=True)
        n = len(routes)
        app.invoke(text, wait=True, timeout=300)
        got = routes[-1] if len(routes) > n else None
        print(f"[router] -> {got}")
        return got

    def cmd(line):
        print(f">> {line}", flush=True)
        name, _, args = line[1:].partition(" ")
        handle_command(app, name, args)

    try:
        configure()
        for message, gold in DEMO:
            got = say(message)
            # the simulated user: approve, or correct with the right route
            cmd(":feedback +" if got == gold else f":feedback - {RULE_HINTS[gold]} name={gold}")
        cmd(":modules")
        cmd(":optimize supervisor")
        wait_optimizations(timeout=900)
        app.wait_idle(timeout=30)
        for path, reg in find_registries(app):  # the swap is the provider's new src
            print(f"[learn] {path}: {reg.programs}")
        say(AFTER[0])
        cmd(":inspect supervisor")
        cmd(":events optimize.* 10")
    finally:
        app.shutdown()


RULE_HINTS = {
    "shipping": "frete/entrega vai para shipping mesmo com intenção de compra",
    "order": "pronta entrega = estoque agora, vai para order",
    "catalog": "o que a loja vende e preços vão para catalog",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", action="store_true", help="scripted run")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--program", default=None, help="saved result to start from")
    args = ap.parse_args()
    lm = example_lm(args.model)
    if args.demo:
        demo(lm)
        return
    configure()
    app = bootstrap(make_root(lm, args.program))
    print("Loja (router). Rate answers with :feedback +/- [name=<route>], then "
          ":optimize supervisor. :help lists all commands.")
    try:
        asyncio.run(read_terminal_and_invoke(app, prompt=">> "))
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
