"""optimize(): metric adapter, optimizers, OptimizationResult. No network."""

import json
import re
import warnings

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from integrations.claude_cli_lm import _wrap
from integrations.dspy_integration import DSPyEnv
from integrations.learn import (
    BudgetError,
    Metric,
    MetricError,
    OptimizationResult,
    OptimizeError,
    as_metric,
    contains,
    count_lm_calls,
    exact_match,
    list_optimizers,
    llm_judge,
    load,
    optimize,
    register_optimizer,
    unregister_optimizer,
)
from integrations.learn.metrics import normalize_output


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


ROWS = [("good day", "pos"), ("bad day", "neg"), ("great food", "pos"), ("awful", "neg"),
        ("good stuff", "pos"), ("great job", "pos")]


def _data():
    ex = [dspy.Example(text=t, label=lab).with_inputs("text") for t, lab in ROWS]
    return ex[:4], ex[4:]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_metric_outputs_are_normalized_and_wrong_types_are_clear():
    assert normalize_output(True) == (1.0, None)
    assert normalize_output(0.25) == (0.25, None)
    assert normalize_output((0, "nope")) == (0.0, "nope")
    assert normalize_output(dspy.Prediction(score=0.5, feedback="half")) == (0.5, "half")
    assert normalize_output({"score": 1, "feedback": "ok"}) == (1.0, "ok")
    with pytest.raises(MetricError, match="float, a bool, \\(score, feedback_text\\)"):
        normalize_output("yes")
    with pytest.raises(MetricError):
        normalize_output(None)


def test_one_metric_adapts_to_gepa_and_standard_signatures():
    gold = dspy.Example(text="good", label="pos", feedback_text="be positive").with_inputs("text")
    pred = dspy.Prediction(label="neg")

    m2 = as_metric(lambda ex, p: p.label == ex.label)  # 2-arg, bool
    assert m2(gold, pred) == 0.0 and m2(gold, pred, None) == 0.0
    out = m2.for_gepa()(gold, pred, None, "self", None)  # GEPA's 5-arg call
    assert isinstance(out, dspy.Prediction) and out.score == 0.0
    # auto feedback: expected vs got + recorded human feedback
    assert "'pos'" in out.feedback and "'neg'" in out.feedback and "be positive" in out.feedback

    seen = []

    def m5(gold, pred, trace=None, pred_name=None, pred_trace=None):
        seen.append(pred_name)
        return dspy.Prediction(score=1.0, feedback=f"fine for {pred_name}")

    gm = as_metric(m5).for_gepa()
    assert gm(gold, pred, None, "predict", []).feedback == "fine for predict"
    assert as_metric(m5)(gold, pred) == 1.0 and seen == ["predict", None]
    assert as_metric(as_metric(m5)).fn is m5  # idempotent


def test_metric_helpers():
    gold = dspy.Example(q="x", label="Pos", reasoning="ignored").with_inputs("q")
    assert exact_match("label")(gold, dspy.Prediction(label=" pos.")) == 1.0
    s, fb = exact_match()(gold, dspy.Prediction(label="neg")), None
    assert s == 0.0  # reasoning is not a label field
    s, fb = exact_match("label").feedback(gold, dspy.Prediction(label="neg"))
    assert s == 0.0 and "should be 'Pos'" in fb
    assert contains("label")(gold, dspy.Prediction(label="it is POS")) == 1.0
    judge_lm = DummyLM([{"score": "0.8", "feedback": "mostly right"}])
    s, fb = llm_judge("label must be the sentiment", lm=judge_lm).feedback(
        gold, dspy.Prediction(label="pos"))
    assert s == 0.8 and fb == "mostly right"


# ---------------------------------------------------------------------------
# optimize(): clear errors
# ---------------------------------------------------------------------------
def test_optimize_input_errors_are_clear(tmp_path):
    train, val = _data()
    prog = dspy.Predict("text -> label")
    lm = RuleLM()
    with pytest.raises(OptimizeError, match="trainset is empty"):
        optimize(prog, trainset=[], metric=exact_match(), optimizer="bootstrap", lm=lm)
    with pytest.raises(OptimizeError, match="needs reflection_lm"):
        optimize(prog, trainset=train, metric=exact_match(), optimizer="gepa", lm=lm)
    with pytest.raises(BudgetError, match="too small"):
        optimize(prog, trainset=train, valset=val, metric=exact_match(), optimizer="gepa",
                 budget=3, lm=lm, reflection_lm=lm)
    with pytest.raises(BudgetError):
        optimize(prog, trainset=train, metric=exact_match(), budget="huge", lm=lm,
                 optimizer="bootstrap")
    with pytest.raises(MetricError, match="metric 'bad'"):
        def bad(ex, pred):
            return "great"
        optimize(prog, trainset=train, metric=bad, optimizer="bootstrap", lm=lm)
    with pytest.raises(KeyError, match="register_optimizer"):
        optimize(prog, trainset=train, metric=exact_match(), optimizer="nope", lm=lm)
    with pytest.raises(OptimizeError, match="No LM"):
        optimize(prog, trainset=train, metric=exact_match(), optimizer="bootstrap")


def test_dict_examples_get_inputs_from_signature():
    res = optimize(dspy.Predict("text -> label"), trainset=[{"text": "good", "label": "pos"}],
                   metric=exact_match("label"), optimizer="bootstrap", budget=1, lm=RuleLM(),
                   num_threads=1)
    assert res.score == 1.0 and res.train_size == 1


# ---------------------------------------------------------------------------
# optimize(): optimizers, result, diff, save/load, hot swap
# ---------------------------------------------------------------------------
def test_bootstrap_result_diff_report_save_load(tmp_path):
    train, val = _data()
    prog = dspy.Predict("text -> label")
    phases = []
    res = optimize(prog, trainset=train, valset=val, metric=exact_match("label"),
                   optimizer="bootstrap", budget=2, lm=RuleLM(), num_threads=1,
                   on_progress=lambda p, info: phases.append(p))
    assert isinstance(res, OptimizationResult)
    assert phases == ["baseline", "compile", "evaluate", "done"]
    assert res.baseline == 0.5 and res.score == 0.5 and res.calls >= 4
    assert res.details["calls_by_phase"]["baseline"] == 2
    assert prog.demos == []  # the input program is untouched
    d = res.diff()
    assert "demos: 0 -> 2" in d and "+ text='good day', label='pos'" in d
    rep = res.report()
    assert "bootstrap" in rep and "0.500 -> 0.500" in rep and "LM calls" in rep

    path = res.save(str(tmp_path / "r.json"))
    data = json.loads(open(path).read())
    assert data["format"] == "pyreact.learn/1" and data["arch"] == "predict"
    back = load(path)  # rebuilt from the saved signature + arch
    assert len(back.program.demos) == 2 and back.score == 0.5
    assert back.diff() == d
    assert "demos: 2 -> 0" in back.diff(dspy.Predict("text -> label"))

    # hot swap into an env by key
    env = DSPyEnv(models={"default": RuleLM()})
    key = ("Predict", "text -> label", "App/Clf#clf")
    env.caches[key] = prog
    res.hot_swap(env, key)
    assert env.compiled[key] is res.program and env.caches[key] is res.program


def test_gepa_uses_feedback_and_improves():
    train, val = _data()
    res = optimize(dspy.Predict("text -> label"), trainset=train, valset=val,
                   metric=exact_match("label"), optimizer="gepa", budget=9, lm=RuleLM(),
                   reflection_lm=RuleLM(reflect=True), num_threads=1)
    assert res.baseline == 0.5 and res.score == 1.0 and res.improved
    assert "+ Classify sentiment" in res.diff()
    assert res.details["val_aggregate_scores"] == [0.5, 1.0]


def test_existing_log_dir_is_not_resumed_silently(tmp_path):
    train, val = _data()
    (tmp_path / "gepa_state.bin").write_bytes(b"old")
    with pytest.warns(UserWarning, match="RESUME"):
        res = optimize(dspy.Predict("text -> label"), trainset=train, valset=val,
                       metric=exact_match("label"), optimizer="bootstrap", budget=1,
                       lm=RuleLM(), log_dir=str(tmp_path), num_threads=1)
    assert res.details["log_dir"].startswith(str(tmp_path / "run-"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = optimize(dspy.Predict("text -> label"), trainset=train, valset=val,
                       metric=exact_match("label"), optimizer="bootstrap", budget=1,
                       lm=RuleLM(), log_dir=str(tmp_path), resume=True, num_threads=1)
    assert res.details["log_dir"] == str(tmp_path)


def test_register_custom_optimizer_and_count_calls():
    seen = {}

    def best_instruction(program, *, trainset, valset, metric, budget, lm, candidates=(),
                         **_):
        """Pick the best of a few handwritten instructions."""
        from integrations.learn import evaluate

        best = (None, -1.0)
        for text in candidates:
            p = program.deepcopy()
            p.signature = p.signature.with_instructions(text)
            score, _ = evaluate(p, valset or trainset, metric, lm=lm, num_threads=1)
            best = max(best, (p, score), key=lambda x: x[1])
        seen["metric"] = metric
        return best[0], {"best_score": best[1]}

    register_optimizer("best_instruction", best_instruction)
    try:
        assert "best_instruction" in list_optimizers()
        train, val = _data()
        res = optimize(dspy.Predict("text -> label"), trainset=train, valset=val,
                       metric=exact_match("label"), optimizer="best_instruction", budget=1,
                       lm=RuleLM(), num_threads=1,
                       candidates=["Label it.", "Words like 'great' are pos."])
        assert isinstance(seen["metric"], Metric)
        assert res.score == 1.0 and res.details["best_score"] == 1.0
        assert res.details["calls_by_phase"]["compile"] == 4  # 2 candidates x 2 val
    finally:
        unregister_optimizer("best_instruction")

    lm = RuleLM()
    with count_lm_calls() as c:
        with dspy.context(lm=lm):
            dspy.Predict("text -> label")(text="good")
    assert c.n == 1
