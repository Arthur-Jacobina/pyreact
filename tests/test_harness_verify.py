"""Verification primitives (pyreact.harness.verify), generic layer. No network."""

import asyncio
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field

from pyreact.core import EventBus, HookContext
from pyreact.core.harness import Boundary, Harness
from pyreact.harness import (
    VerificationFailed,
    as_mapping,
    assert_that,
    grounded_in,
    schema,
    verify,
)


def _events():
    got = []
    HookContext.get_service("event_bus", EventBus).subscribe(got.append, ["verify.*"])
    return got


def _run(h, answers, inputs=None, kind="llm"):
    """Owner returns answers[i] on attempt i; records what each attempt saw."""
    seen = []

    async def owner(b):
        seen.append({"attempt": b.attempt, "feedback": b.attrs.get("feedback"),
                     "rollout_id": b.attrs.get("rollout_id")})
        return answers[min(len(seen), len(answers)) - 1]

    b = Boundary(kind=kind, name="qa", inputs=dict(inputs or {}), turn_id="t1",
                 attrs={"call_id": "llm-1"})
    out = asyncio.run(Harness([*h]).run(b, owner))
    return out, b, seen


def test_check_return_values_are_normalized():
    def ok(b, r):
        return None

    v = verify([ok, lambda b, r: True, lambda b, r: 0.9, lambda b, r: (1, "fine")],
               threshold=0.5)
    out, b, _ = _run([v], ["x"])
    assert out == "x" and b.attrs["verdicts"][0]["passed"] and b.attrs["verdicts"][0]["score"] == 0.9
    with pytest.raises(TypeError, match="a check returns None/True"):
        _run([verify(lambda b, r: object())], ["x"])
    with pytest.raises(ValueError):
        verify(ok, on_fail="explode")


def test_raise_publishes_verify_failed_and_names_the_verifier():
    ev = _events()
    v = verify(assert_that(lambda r: r["answer"].endswith("."), "End with a period.",
                           name="period"))
    with pytest.raises(VerificationFailed) as exc:
        _run([v], [{"answer": "no period"}])
    assert str(exc.value) == "llm 'qa' failed verification by period: End with a period."
    assert exc.value.verdict.score == 0.0
    (e,) = ev
    assert e.kind == "verify.failed" and e.turn_id == "t1"
    p = e.payload
    assert p["verifier"] == "period" and p["feedback"] == "End with a period."
    assert p["action"] == "raise" and p["call_id"] == "llm-1" and p["name"] == "qa"


def test_retry_with_feedback_conditions_the_rerun_then_passes():
    ev = _events()
    v = verify(assert_that(lambda r: len(r) <= 5, "Answer in at most 5 characters."),
               on_fail="retry_with_feedback", max_attempts=3)
    out, b, seen = _run([v], ["too long", "still long", "ok"])
    assert out == "ok"
    assert seen == [
        {"attempt": 1, "feedback": None, "rollout_id": None},
        {"attempt": 2, "feedback": "Answer in at most 5 characters.", "rollout_id": 2},
        {"attempt": 3, "feedback": "Answer in at most 5 characters.", "rollout_id": 3},
    ]
    assert [e.kind for e in ev] == ["verify.failed", "verify.failed", "verify.passed"]
    assert [d["action"] for d in b.attrs["verdicts"]] == ["retry_with_feedback"] * 2 + ["pass"]


def test_plain_retry_resamples_without_feedback_and_exhaustion():
    v = verify(lambda b, r: r == "good" or "bad answer", on_fail="retry", max_attempts=2)
    out, b, seen = _run([v], ["bad", "good"])
    assert out == "good" and [s["feedback"] for s in seen] == [None, None]
    assert seen[1]["rollout_id"] == 2
    with pytest.raises(VerificationFailed, match="after 2 attempt\\(s\\): bad answer"):
        _run([v], ["bad"])
    soft = verify(lambda b, r: "bad answer", on_fail="retry", on_exhausted="flag")
    out, b, seen = _run([soft], ["bad1", "bad2", "bad3"])
    assert out == "bad2" and len(seen) == 2
    assert [d["action"] for d in b.attrs["verdicts"]] == ["retry", "flag"]


def test_flag_keeps_the_result_and_custom_inject():
    out, b, _ = _run([verify(lambda b, r: (0.2, "weak"), on_fail="flag")], ["meh"])
    assert out == "meh"
    (d,) = b.attrs["verdicts"]
    assert d["passed"] is False and d["action"] == "flag" and d["feedback"] == "weak"

    def inject(b, fb):
        b.inputs = {**b.inputs, "hint": fb}

    seen_inputs = []

    async def owner(b):
        seen_inputs.append(dict(b.inputs))
        return "fixed" if "hint" in b.inputs else "wrong"

    v = verify(lambda b, r: r == "fixed" or "use the hint", on_fail="retry_with_feedback",
               inject=inject)
    b = Boundary(kind="llm", name="x", inputs={"q": 1})
    assert asyncio.run(Harness([v]).run(b, owner)) == "fixed"
    assert seen_inputs == [{"q": 1}, {"q": 1, "hint": "use the hint"}]
    assert "feedback" not in b.attrs


def test_kinds_filter_and_tool_results():
    v = verify(assert_that(lambda r: "error" not in r, "tool failed"), kinds="tool")
    out, _, _ = _run([v], ["fine"], kind="tool")
    assert out == "fine"
    with pytest.raises(VerificationFailed):
        _run([v], ["an error"], kind="tool")
    out, b, _ = _run([v], ["an error"], kind="llm")  # not verified
    assert "verdicts" not in b.attrs


class Answer(BaseModel):
    answer: Annotated[str, Field(min_length=1, max_length=20)]
    confidence: Literal["low", "high"]


def test_schema_with_a_model_a_field_map_and_required_fields():
    v = verify(schema(Answer))
    out, _, _ = _run([v], [{"answer": "yes", "confidence": "high", "reasoning": "ignored"}])
    assert out["answer"] == "yes"
    with pytest.raises(VerificationFailed) as exc:
        _run([v], [{"answer": "x" * 30, "confidence": "maybe"}])
    msg = exc.value.verdict.feedback
    assert msg.startswith("The output does not match the expected schema:")
    assert "`answer`: String should have at most 20 characters" in msg
    assert "`confidence`: Input should be 'low' or 'high' (got 'maybe')" in msg
    with pytest.raises(VerificationFailed, match="`confidence`: Field required"):
        _run([v], [{"answer": "ok"}])
    m = verify(schema({"n": int}))
    _run([m], [{"n": "3"}])  # lax: "3" validates as int
    with pytest.raises(VerificationFailed, match="`n`: Input should be a valid integer"):
        _run([m], [{"n": "three"}])
    r = verify(schema(["answer"]))
    with pytest.raises(VerificationFailed, match="`answer`: must not be empty"):
        _run([r], [{"answer": ""}])


def test_assert_that_failing_predicate_and_grounded_in():
    v = verify(assert_that(lambda r: r.missing, "need `missing`"))
    with pytest.raises(VerificationFailed, match="need `missing` \\(check raised AttributeError"):
        _run([v], [object()])

    ctx = "The Eiffel Tower is in Paris and was completed in 1889."
    g = verify(grounded_in("answer", "context"))
    out, b, _ = _run([g], [{"answer": "It was completed in 1889 in Paris."}],
                     inputs={"context": ctx})
    assert b.attrs["verdicts"][0]["score"] == 1.0
    with pytest.raises(VerificationFailed) as exc:
        _run([g], [{"answer": "It opened in 1925 for the Berlin Olympics."}],
             inputs={"context": ctx})
    assert "unsupported: opened, 1925, berlin, olympics" in exc.value.verdict.feedback
    assert exc.value.verdict.threshold == 0.6
    with pytest.raises(VerificationFailed, match="no `context`"):
        _run([g], [{"answer": "x"}])


def test_as_mapping_shapes():
    import dspy

    assert as_mapping(dspy.Prediction(answer="a")) == {"answer": "a"}
    assert as_mapping(Answer(answer="a", confidence="low")) == {"answer": "a", "confidence": "low"}
    assert as_mapping("text") == {"result": "text"}
    assert as_mapping(None) == {}
