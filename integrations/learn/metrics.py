"""One metric, every optimizer.

Write the metric the simple way::

    def metric(example, pred):            # (example, pred[, trace])
        return pred.label == example.label             # bool | float
        # or (score, "why")  |  dspy.Prediction(score=..., feedback=...)

:func:`as_metric` wraps it in a :class:`Metric` that

- validates the return value (clear :class:`MetricError` on a wrong type),
- is a standard DSPy metric (``metric(example, pred, trace=None) -> float``)
  for Evaluate / BootstrapFewShot / MIPROv2 / SIMBA / COPRO,
- adapts to GEPA's feedback signature via :meth:`Metric.for_gepa`
  (``(gold, pred, trace, pred_name, pred_trace) -> Prediction(score, feedback)``);
  when your metric returns no feedback text, one is generated from the
  example's labels (expected vs got) plus any recorded human feedback
  (``feedback_text`` on examples built by ``dataset_from_events``).

Helpers: :func:`exact_match`, :func:`contains`, :func:`llm_judge`.
"""

from __future__ import annotations

import inspect
import numbers
from typing import Any, Callable, Iterable, List, Optional, Tuple

import dspy

__all__ = [
    "Metric",
    "MetricError",
    "as_metric",
    "exact_match",
    "contains",
    "llm_judge",
    "label_fields",
    "META_FIELDS",
]

# Fields dataset_from_events attaches to examples that are *not* labels.
META_FIELDS = frozenset({
    "call_id", "recorded_outputs", "feedback_score", "feedback_text",
    "reasoning", "rationale", "trajectory",
})


class MetricError(TypeError):
    """The metric returned something that is not a score."""


def label_fields(example: Any) -> List[str]:
    """Label (non-input, non-meta) field names of an example."""
    try:
        keys = list(example.labels().keys())
    except Exception:
        keys = [k for k in getattr(example, "keys", lambda: [])()]
    return [k for k in keys if k not in META_FIELDS]


def _norm(v: Any) -> str:
    return " ".join(str(v if v is not None else "").strip().strip("'\"`.").lower().split())


def normalize_output(out: Any, *, metric_name: str = "metric") -> Tuple[float, Optional[str]]:
    """Metric return value -> ``(score, feedback or None)`` or :class:`MetricError`."""
    if isinstance(out, bool):
        return float(out), None
    if isinstance(out, numbers.Real):
        return float(out), None
    if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[0], (bool, numbers.Real)):
        fb = out[1]
        return float(out[0]), None if fb is None else str(fb)
    score = None
    fb = None
    if isinstance(out, dict) or isinstance(out, dspy.Prediction):
        try:
            score = out["score"]
            fb = out.get("feedback") if hasattr(out, "get") else None
        except Exception:
            score = None
    elif hasattr(out, "score"):
        score, fb = getattr(out, "score"), getattr(out, "feedback", None)
    if isinstance(score, (bool, numbers.Real)):
        return float(score), None if fb is None else str(fb)
    raise MetricError(
        f"{metric_name} returned {type(out).__name__} {out!r:.120}; a metric must return "
        "a float, a bool, (score, feedback_text), or dspy.Prediction(score=..., feedback=...)."
    )


def _arity(fn: Callable) -> int:
    """How many positional args ``fn`` accepts (5 for *args)."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return 2
    n = 0
    for p in params:
        if p.kind == p.VAR_POSITIONAL:
            return 5
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
    return max(2, min(n, 5))


def auto_feedback(example: Any, pred: Any, score: float) -> str:
    """Feedback text from labels (expected vs got) + recorded human feedback."""
    parts = []
    for f in label_fields(example):
        gold = example.get(f) if hasattr(example, "get") else getattr(example, f, None)
        got = getattr(pred, f, None) if pred is not None else None
        if got is None and isinstance(pred, dict):
            got = pred.get(f)
        if _norm(got) == _norm(gold):
            parts.append(f"`{f}` is correct ({str(gold)[:200]!r}).")
        else:
            parts.append(f"`{f}` should be {str(gold)[:200]!r} but the program said "
                         f"{str(got)[:200]!r}.")
    human = example.get("feedback_text") if hasattr(example, "get") else None
    if human:
        parts.append(f"User feedback on an earlier answer to this input: {human}")
    head = f"Score {score:g}."
    return " ".join([head, *parts]) if parts else head + " (no labels to compare against)"


class Metric:
    """A user metric normalized for every optimizer (see module docstring)."""

    def __init__(self, fn: Callable, name: Optional[str] = None) -> None:
        if isinstance(fn, Metric):
            fn = fn.fn
        if not callable(fn):
            raise TypeError(f"metric must be callable, got {fn!r}")
        self.fn = fn
        self.name = name or getattr(fn, "__name__", type(fn).__name__)
        self.arity = _arity(fn)
        self.calls = 0

    def score(self, example, pred, trace=None, pred_name=None, pred_trace=None
              ) -> Tuple[float, Optional[str]]:
        self.calls += 1
        args = (example, pred, trace, pred_name, pred_trace)[: self.arity]
        return normalize_output(self.fn(*args), metric_name=f"metric {self.name!r}")

    def __call__(self, example, pred, trace=None) -> float:
        """Standard DSPy metric: a float score."""
        return self.score(example, pred, trace)[0]

    def feedback(self, example, pred, trace=None, pred_name=None, pred_trace=None
                 ) -> Tuple[float, str]:
        score, fb = self.score(example, pred, trace, pred_name, pred_trace)
        return score, fb if fb else auto_feedback(example, pred, score)

    def for_gepa(self) -> Callable:
        """GEPA's 5-arg feedback metric returning ``dspy.Prediction(score, feedback)``."""

        def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            score, fb = self.feedback(gold, pred, trace, pred_name, pred_trace)
            return dspy.Prediction(score=score, feedback=fb)

        gepa_metric.__name__ = f"gepa_{self.name}"
        return gepa_metric

    def __repr__(self) -> str:
        return f"Metric({self.name})"


def as_metric(fn: Callable) -> Metric:
    return fn if isinstance(fn, Metric) else Metric(fn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fields(example, field: Optional[Iterable[str] | str]) -> List[str]:
    if field is None:
        return label_fields(example)
    return [field] if isinstance(field, str) else list(field)


def exact_match(field: Optional[Iterable[str] | str] = None, *, normalize: bool = True) -> Metric:
    """Fraction of ``field`` (default: every label field) the prediction gets
    exactly right (case/whitespace/quote-insensitive unless ``normalize=False``),
    with feedback naming each wrong field."""

    def exact_match_metric(example, pred):
        fields = _fields(example, field)
        if not fields:
            return 0.0, "No label fields on the example to compare."
        ok, notes = 0, []
        for f in fields:
            gold, got = example.get(f), getattr(pred, f, None)
            same = _norm(gold) == _norm(got) if normalize else gold == got
            ok += same
            if not same:
                notes.append(f"`{f}` should be {str(gold)[:200]!r}, got {str(got)[:200]!r}.")
        score = ok / len(fields)
        human = example.get("feedback_text")
        if human and score < 1:
            notes.append(f"User feedback: {human}")
        return score, " ".join(notes) or "Correct."

    exact_match_metric.__name__ = f"exact_match({field or '*'})"
    return Metric(exact_match_metric)


def contains(field: Optional[Iterable[str] | str] = None, *, case: bool = False) -> Metric:
    """1.0 when the expected value appears inside the predicted text (per field, averaged)."""

    def contains_metric(example, pred):
        fields = _fields(example, field)
        if not fields:
            return 0.0, "No label fields on the example to compare."
        ok, notes = 0, []
        for f in fields:
            gold, got = str(example.get(f) or ""), str(getattr(pred, f, "") or "")
            hit = gold in got if case else gold.lower() in got.lower()
            ok += hit
            if not hit:
                notes.append(f"`{f}` must mention {gold[:200]!r}; it said {got[:200]!r}.")
        return ok / len(fields), " ".join(notes) or "Correct."

    contains_metric.__name__ = f"contains({field or '*'})"
    return Metric(contains_metric)


class _JudgeSig(dspy.Signature):
    """Grade a program's answer against the criteria (and the expected answer, if any).
    Be strict and specific: the feedback is used to improve the program's instructions."""

    criteria: str = dspy.InputField()
    task_inputs: str = dspy.InputField()
    expected: str = dspy.InputField(desc="reference answer or '(none)'")
    answer: str = dspy.InputField()
    score: float = dspy.OutputField(desc="0.0 (fails) to 1.0 (fully meets the criteria)")
    feedback: str = dspy.OutputField(desc="one or two sentences: what is wrong / right and why")


def llm_judge(criteria: str, lm: Any = None, *, fields: Optional[Iterable[str]] = None) -> Metric:
    """LM-graded metric: ``score`` in [0, 1] plus feedback (great for GEPA).

    Costs one ``lm`` call per evaluation; ``lm`` defaults to the configured LM.
    """
    judge = dspy.Predict(_JudgeSig)

    def llm_judge_metric(example, pred):
        outs = list(fields) if fields else [k for k in getattr(pred, "keys", lambda: [])()
                                             if k not in META_FIELDS]
        answer = "\n".join(f"{k}: {getattr(pred, k, None)}" for k in outs)
        inputs = "\n".join(f"{k}: {v}" for k, v in example.inputs().items())
        labels = label_fields(example)
        expected = "\n".join(f"{k}: {example.get(k)}" for k in labels) or "(none)"
        if lm is not None:
            with dspy.context(lm=lm):
                g = judge(criteria=criteria, task_inputs=inputs, expected=expected, answer=answer)
        else:
            g = judge(criteria=criteria, task_inputs=inputs, expected=expected, answer=answer)
        try:
            score = max(0.0, min(1.0, float(g.score)))
        except Exception:
            score = 0.0
        return score, str(g.feedback)

    llm_judge_metric.__name__ = "llm_judge"
    return Metric(llm_judge_metric)
