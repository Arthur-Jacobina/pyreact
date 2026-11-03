"""Verifier failures -> feedback events -> dataset -> GEPA, end to end. No network."""

import asyncio
import re

import dspy

from integrations import DSPyProvider, use_call, use_module
from integrations.claude_cli_lm import _wrap
from integrations.harness import check_metric, use_verify, verify
from integrations.learn import dataset_from_events, optimize
from pyreact.boot import bootstrap
from pyreact.core import EventBus, HookContext, component, hooks, use_turn
from pyreact.core.harness import Boundary, Harness


class RuleLM(dspy.BaseLM):
    """Sentiment 'classifier': pos iff 'good' in text (or 'great' once instructed)."""

    def __init__(self, reflect=False):
        super().__init__(model="fake/rule", cache=False)
        self.reflect = reflect

    def forward(self, prompt=None, messages=None, **kw):
        if self.reflect:  # GEPA reflection: propose a new instruction
            return _wrap("```\nClassify sentiment. Words like 'great' are pos.\n```", self.model)
        system, user = messages[0]["content"], messages[-1]["content"]
        found = re.findall(r"\[\[ ## text ## \]\]\n(.*)", user)
        text = found[-1] if found else ""
        pos = "good" in text or ("great" in text and "great" in system)
        return _wrap(f"[[ ## label ## ]]\n{'pos' if pos else 'neg'}\n\n[[ ## completed ## ]]",
                     self.model)

    async def aforward(self, prompt=None, messages=None, **kw):
        return self.forward(prompt, messages, **kw)


def sentiment_ok(b, pred):
    if "great" in b.inputs["text"] and pred.label != "pos":
        return "A text that says 'great' is positive: label it pos."


ROWS = ["good day", "bad day", "great food", "awful", "great job"]


def test_flagged_failures_become_a_gepa_dataset_and_gepa_fixes_them():
    events = []
    HookContext.get_service("event_bus", EventBus).subscribe(events.append)

    @component
    def Classifier():
        turn = use_turn()
        call = use_call(use_module("text -> label", name="clf"))
        use_verify(sentiment_ok, on_fail="flag", to_feedback=True)
        hooks.use_effect(lambda: call.run(text=turn.text) if turn else None, [turn and turn.id])
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=RuleLM(), children=[Classifier(key="c")])]

    app = bootstrap(Root)
    try:
        for row in ROWS:
            app.invoke(row, wait=True, timeout=5)
    finally:
        app.shutdown()

    fb = [e for e in events if e.kind == "feedback"]
    assert len(fb) == 2 and all(e.payload["score"] == 0.0 for e in fb)
    assert fb[0].payload["by"] == "sentiment_ok" and fb[0].payload["target"].startswith("llm-")
    assert fb[0].payload["text"] == ("[sentiment_ok] A text that says 'great' is positive: "
                                     "label it pos.")

    data = dataset_from_events(events, "clf", require_feedback=True)
    assert sorted(ex.text for ex in data) == ["great food", "great job"]
    assert all(ex.feedback_score == 0.0 and "positive" in ex.feedback_text for ex in data)
    assert all(ex.recorded_outputs["label"] == "neg" for ex in data)  # what it said

    res = optimize(dspy.Predict("text -> label"), trainset=data, valset=data,
                   metric=check_metric(sentiment_ok), optimizer="gepa", budget=8,
                   lm=RuleLM(), reflection_lm=RuleLM(reflect=True), num_threads=1)
    assert res.baseline == 0.0 and res.score == 1.0 and res.improved
    assert "great" in res.diff()


def test_no_feedback_when_a_retry_fixed_the_call():
    events = []
    HookContext.get_service("event_bus", EventBus).subscribe(events.append, ["feedback"])

    async def owner(b):
        return "fixed" if b.attempt > 1 else "bad"

    v = verify(lambda b, r: r == "fixed" or "wrong", on_fail="retry", to_feedback=True)
    out = asyncio.run(Harness([v]).run(Boundary(kind="llm", name="x"), owner))
    assert out == "fixed" and events == []

    v = verify(lambda b, r: "wrong", on_fail="retry", on_exhausted="flag", to_feedback=True)
    b = Boundary(kind="llm", name="x", attrs={"call_id": "llm-9"}, turn_id="t")
    asyncio.run(Harness([v]).run(b, owner))
    (e,) = events
    assert e.payload["target"] == "llm-9" and e.turn_id == "t"


def test_check_metric_runs_sync_and_async_checks():
    ex = dspy.Example(text="great view", label="pos").with_inputs("text")
    m = check_metric(sentiment_ok)
    assert m(ex, dspy.Prediction(label="neg"))[0] == 0.0
    assert m(ex, dspy.Prediction(label="pos")) == (1.0, "Passes the check.")

    async def acheck(b, pred):
        return (0.5, "half")

    assert check_metric(acheck)(ex, dspy.Prediction()) == (0.5, "half")
