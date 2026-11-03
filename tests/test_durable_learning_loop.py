"""A run journaled by pyreact_durable is a dataset for integrations.learn.

The two libraries never import each other; they meet at the border (events
in a session log). No network: DummyLM.
"""

import warnings

from dspy.utils import DummyLM

from integrations import DSPyProvider, use_call, use_module
from integrations.feedback import send_feedback
from integrations.learn import dataset_from_events
from pyreact.boot import bootstrap
from pyreact.core import component, hooks, use_turn
from pyreact_durable import Durable, SessionLog


def _serve(path, lm, messages, ratings):
    @component
    def Router():
        turn = use_turn()
        call = use_call(use_module("message -> name", name="router"))
        hooks.use_effect(lambda: turn and call.run(message=turn.text), [turn and turn.id])
        return []

    @component
    def App():
        return [
            Durable(key="d", log=path, journal=("turn.*", "llm.*", "feedback"), children=[
                DSPyProvider(key="p", lm=lm, children=[Router(key="r")])
            ])
        ]

    app = bootstrap(App)
    try:
        for msg in messages:
            tid = app.invoke(msg, wait=True, timeout=10)
            if msg in ratings:
                score, expected = ratings[msg]
                send_feedback(tid, score, "rated", expected=expected, bus=app.event_bus)
        app.wait_idle(timeout=5)
    finally:
        app.shutdown()


def test_journal_becomes_dataset_with_corrections_and_no_unrated_leak(tmp_path):
    warnings.simplefilter("ignore")
    path = str(tmp_path / "run.sqlite")  # works the same with .jsonl
    lm = DummyLM([{"name": "order"}] * 10)
    _serve(
        path,
        lm,
        ["comprar e saber o frete", "tem em estoque?", "held-out question"],
        {
            "comprar e saber o frete": (0.0, {"name": "shipping"}),  # correction
            "tem em estoque?": (1.0, None),  # confirmed
        },
    )

    kinds = {e["kind"] for e in SessionLog.open(path, readonly=True).events()}
    assert {"turn.start", "llm.start", "llm.end", "feedback"} <= kinds

    rated = dataset_from_events(path, "router", require_feedback=True)
    assert sorted((ex.message, ex.name) for ex in rated) == [
        ("comprar e saber o frete", "shipping"),  # label from the correction
        ("tem em estoque?", "order"),  # label = confirmed output
    ]
    # Without require_feedback the unrated call would leak in with an
    # unchecked label.
    assert len(dataset_from_events(path, "router")) == 3
