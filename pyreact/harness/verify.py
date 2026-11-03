"""Verification: check a boundary call's *result*, then raise, retry or flag.

``verify(check, ...)`` is harness middleware that runs the call, then
``check(b, result)``::

    None / True            -> pass (score 1.0)
    False / "feedback"     -> fail (score 0.0), with that feedback
    score (float)          -> pass iff score >= threshold
    (score, "feedback")    -> same, with feedback (also any object with
                              .score / .feedback, e.g. a dspy.Prediction)

and on failure, by ``on_fail``:

``"raise"``                :class:`VerificationFailed` (the call fails)
``"retry"``                run the call again (up to ``max_attempts`` in all)
``"retry_with_feedback"``  run it again *conditioned on the feedback*
``"flag"``                 keep the result, record the failed verdict

Every verdict is appended to ``b.attrs["verdicts"]`` (a list of dicts) and
published as ``verify.passed`` / ``verify.failed``::

    {"verifier", "score", "threshold", "feedback", "action", "kind", "name",
     "boundary_id", "call_id", "component", "attempt"}

How a re-run is conditioned (the retry contract): before calling
``call_next`` again the verifier bumps ``b.attempt`` and sets two attrs the
*owner* of the boundary honors:

- ``b.attrs["rollout_id"]`` = the attempt number: re-sample instead of
  replaying a cached answer;
- ``b.attrs["feedback"]`` (``retry_with_feedback`` only) = the failed
  verdict's feedback. ``llm`` boundaries run by ``integrations`` add it to
  every predictor call of the re-run as an extra input field ``feedback``
  ("feedback on a previous attempt: fix these problems"), like
  ``dspy.Refine``'s hint -- the module's signature and the caller's inputs
  are untouched. Pass ``inject=fn(b, feedback)`` to condition differently
  (e.g. rewrite ``b.inputs`` for your own boundary kind).

Retries exhausted: ``on_exhausted="raise"`` (default) raises
:class:`VerificationFailed`, ``"flag"`` returns the last result.

Built-in checks: :func:`schema`, :func:`assert_that`, :func:`grounded_in`
(``integrations.harness.llm_judge`` for an LM judge). Pass a list to run
several as one verifier (all must pass; feedback is joined).
"""

from __future__ import annotations

import numbers
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from pyreact.core.harness import Boundary, Next

from ._base import HarnessMiddleware, Kinds
from ._base import resolve as _resolve
from ._events import emit

__all__ = [
    "Verdict",
    "VerificationFailed",
    "as_mapping",
    "assert_that",
    "grounded_in",
    "schema",
    "verify",
]

ON_FAIL = ("raise", "retry", "retry_with_feedback", "flag")
Check = Callable[[Boundary, Any], Any]


@dataclass
class Verdict:
    """One check of one attempt."""

    verifier: str
    passed: bool
    score: float
    threshold: float
    feedback: Optional[str]
    attempt: int
    action: str = "pass"  # pass | raise | retry | retry_with_feedback | flag | exhausted

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VerificationFailed(Exception):
    """A call's result failed verification (``on_fail="raise"`` or retries
    exhausted). ``verdict`` holds the score/feedback, ``boundary`` the call."""

    def __init__(self, verdict: Verdict, boundary: Boundary) -> None:
        super().__init__(verdict.feedback or "verification failed")
        self.verdict, self.boundary = verdict, boundary

    def __str__(self) -> str:
        b, v = self.boundary, self.verdict
        tries = f" after {v.attempt} attempt(s)" if v.attempt > 1 else ""
        return (f"{b.kind} {b.name!r} failed verification by {v.verifier}{tries}: "
                f"{v.feedback or 'score ' + format(v.score, 'g')}")


def _check_name(check: Any) -> str:
    return (getattr(check, "check_name", None) or getattr(check, "__name__", None)
            or type(check).__name__)


def _normalize(out: Any, name: str) -> Tuple[Optional[float], Optional[str]]:
    """Check output -> ``(score or None for a plain pass/fail, feedback)``."""
    if out is None or out is True:
        return 1.0, None
    if out is False:
        return 0.0, None
    if isinstance(out, str):
        return 0.0, out or None
    if isinstance(out, numbers.Real):
        return float(out), None
    if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[0], (bool, numbers.Real)):
        return float(out[0]), None if out[1] is None else str(out[1])
    score = getattr(out, "score", None)
    if score is None and isinstance(out, Mapping):
        score = out.get("score")
    if isinstance(score, (bool, numbers.Real)):
        fb = getattr(out, "feedback", None)
        if fb is None and isinstance(out, Mapping):
            fb = out.get("feedback")
        return float(score), None if fb is None else str(fb)
    raise TypeError(
        f"check {name!r} returned {type(out).__name__}; a check returns None/True (pass), "
        "False or a feedback string (fail), a score, or (score, feedback)"
    )


class verify(HarnessMiddleware):  # noqa: N801 - reads as a function
    """Verification middleware (see the module docstring).

    ``check``: ``check(b, result)`` (sync or async) or a list of checks.
    ``kinds``: default ``("llm",)``. ``threshold``: minimum passing score
    (default: the check's own ``threshold`` attribute, else 1.0).
    ``max_attempts``: total runs for the retry modes (default 2).
    """

    default_kinds = ("llm",)

    def __init__(self, check: Union[Check, Sequence[Check]], *, kinds: Kinds = None,
                 on_fail: str = "raise", max_attempts: int = 2,
                 threshold: Optional[float] = None, on_exhausted: str = "raise",
                 inject: Optional[Callable[[Boundary, str], None]] = None,
                 name: Optional[str] = None) -> None:
        checks = list(check) if isinstance(check, (list, tuple)) else [check]
        if not checks or not all(callable(c) for c in checks):
            raise TypeError("verify needs a check callable (or a list of them)")
        if on_fail not in ON_FAIL:
            raise ValueError(f"on_fail must be one of {ON_FAIL}, got {on_fail!r}")
        if on_exhausted not in ("raise", "flag"):
            raise ValueError("on_exhausted must be 'raise' or 'flag'")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        super().__init__(kinds=kinds, name=name or "+".join(_check_name(c) for c in checks))
        self.checks = checks
        self.on_fail, self.max_attempts, self.on_exhausted = on_fail, max_attempts, on_exhausted
        self.threshold = threshold
        self.inject = inject

    # -- one verdict ------------------------------------------------------------
    async def evaluate(self, b: Boundary, result: Any) -> Verdict:
        """Run every check on ``result``; all must pass (min score, joined feedback)."""
        scores, notes, passed, thresholds = [], [], True, []
        for c in self.checks:
            name = _check_name(c)
            score, fb = _normalize(await _resolve(c(b, result)), name)
            thr = self.threshold if self.threshold is not None else getattr(c, "threshold", 1.0)
            ok = score >= thr
            if not ok:
                passed = False
                notes.append(fb or f"{name}: score {score:g} is below {thr:g}")
            scores.append(score)
            thresholds.append(thr)
        return Verdict(verifier=self.harness_name, passed=passed, score=min(scores),
                       threshold=max(thresholds), feedback="\n".join(notes) or None,
                       attempt=b.attempt)

    def _record(self, b: Boundary, v: Verdict) -> None:
        b.attrs.setdefault("verdicts", []).append(v.as_dict())
        emit("verify.passed" if v.passed else "verify.failed", b,
             **{k: val for k, val in v.as_dict().items() if k not in ("passed", "attempt")})

    def _condition(self, b: Boundary, v: Verdict) -> None:
        b.attempt += 1
        b.attrs["rollout_id"] = b.attempt
        if self.on_fail == "retry_with_feedback":
            fb = v.feedback or f"the previous answer failed {self.harness_name}"
            if self.inject is not None:
                self.inject(b, fb)
            else:
                b.attrs["feedback"] = fb

    async def __call__(self, b: Boundary, call_next: Next) -> Any:
        tries = 0
        while True:
            result = await call_next(b)
            tries += 1
            v = await self.evaluate(b, result)
            if v.passed:
                self._record(b, v)
                return result
            retrying = self.on_fail in ("retry", "retry_with_feedback")
            if retrying and tries < self.max_attempts:
                v.action = self.on_fail
            elif retrying:
                v.action = "exhausted" if self.on_exhausted == "raise" else "flag"
            else:
                v.action = self.on_fail
            self._record(b, v)
            if v.action == "flag":
                return result
            if v.action in ("raise", "exhausted"):
                raise VerificationFailed(v, b)
            self._condition(b, v)


# ---------------------------------------------------------------------------
# Built-in checks
# ---------------------------------------------------------------------------
def as_mapping(result: Any) -> Dict[str, Any]:
    """Fields of a result: a ``dspy.Prediction`` / dict / pydantic model /
    object; a plain value (e.g. a tool's string) is ``{"result": value}``."""
    if result is None:
        return {}
    if isinstance(result, Mapping):
        return dict(result)
    for attr in ("toDict", "model_dump"):
        fn = getattr(result, attr, None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:
                pass
    if isinstance(result, (str, bytes, numbers.Number, list, tuple)):
        return {"result": result}
    data = getattr(result, "__dict__", None)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return {"result": result}


def _named(fn: Callable, name: str, threshold: Optional[float] = None) -> Callable:
    fn.check_name = name  # type: ignore[attr-defined]
    if threshold is not None:
        fn.threshold = threshold  # type: ignore[attr-defined]
    return fn


def schema(model_or_fields: Any, *, name: str = "schema") -> Check:
    """The result's fields must validate against a schema.

    ``model_or_fields``: a pydantic model class (its fields are read from the
    result: extra fields such as ``reasoning`` are ignored), a dict
    ``{field: type}`` (``Annotated[str, Field(max_length=200)]``,
    ``Literal[...]``, ...), or a list of field names that must be non-empty.
    Feedback names each invalid field, so ``retry_with_feedback`` can fix it.
    """
    from pydantic import BaseModel, ValidationError, create_model

    if isinstance(model_or_fields, type) and issubclass(model_or_fields, BaseModel):
        model = model_or_fields
        required: List[str] = []
    elif isinstance(model_or_fields, Mapping):
        model = create_model("Schema", **{k: (t, ...) for k, t in model_or_fields.items()})
        required = []
    elif isinstance(model_or_fields, Iterable) and not isinstance(model_or_fields, str):
        required = list(model_or_fields)
        model = create_model("Schema", **{k: (Any, ...) for k in required})
    else:
        raise TypeError("schema() takes a pydantic model, {field: type} or a list of fields")
    fields = list(model.model_fields)

    def schema_check(b: Boundary, result: Any):
        data = as_mapping(result)
        problems = []
        try:
            model.model_validate({k: data[k] for k in fields if k in data})
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
                got = f" (got {err['input']!r:.80})" if "input" in err and err["type"] != "missing" else ""
                problems.append(f"`{loc}`: {err.get('msg')}{got}")
        for k in required:
            v = data.get(k)
            if k in data and (v is None or (isinstance(v, (str, list, dict)) and not v)):
                problems.append(f"`{k}`: must not be empty")
        if problems:
            return "The output does not match the expected schema: " + "; ".join(problems) + "."
        return None

    return _named(schema_check, name)


def assert_that(predicate: Callable[[Any], Any], message: str, *,
                name: Optional[str] = None) -> Check:
    """``predicate(result)`` must be truthy; otherwise fail with ``message``
    (the feedback the model gets on ``retry_with_feedback``). A predicate
    that raises (e.g. a missing field) fails too."""

    def assert_check(b: Boundary, result: Any):
        try:
            ok = predicate(result)
        except Exception as exc:
            return f"{message} (check raised {type(exc).__name__}: {exc})"
        return None if ok else message

    return _named(assert_check, name or getattr(predicate, "__name__", "assert_that")
                  .replace("<lambda>", "assert_that"))


_STOP = frozenset(
    "the and for are but not you your with that this from was were has have had its it's "
    "they them their there what which who whom will would can could should into onto about "
    "than then also only just very more most some such any all each our out over under".split()
)


def _words(text: Any) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if len(w) > 2 and w not in _STOP]


def grounded_in(field: str, source_field: str, *, min_overlap: float = 0.6,
                name: Optional[str] = None) -> Check:
    """A cheap groundedness check: the share of content words of the
    result's ``field`` that also appear in ``source_field`` (looked up in the
    call's inputs, then in the result) must be >= ``min_overlap``. Score =
    that share; the feedback lists unsupported words."""

    def grounded_check(b: Boundary, result: Any):
        data = as_mapping(result)
        source = b.inputs.get(source_field, data.get(source_field))
        if source is None:
            return 0.0, f"no `{source_field}` to check `{field}` against"
        words = _words(data.get(field))
        if not words:
            return 1.0, None
        known = set(_words(source))
        missing = [w for w in dict.fromkeys(words) if w not in known]
        score = 1 - sum(w not in known for w in words) / len(words)
        if score >= min_overlap:
            return score, None
        return score, (f"`{field}` is not grounded in `{source_field}`: only {score:.0%} of its "
                       f"words appear there (unsupported: {', '.join(missing[:12])}). "
                       f"Answer only from `{source_field}`.")

    return _named(grounded_check, name or f"grounded_in({field}, {source_field})", min_overlap)
