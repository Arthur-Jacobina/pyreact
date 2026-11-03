"""Verification on real llm boundaries: retry-with-feedback reaches the
prompt, rollout ids bypass the cache, llm_judge. No network."""

import asyncio
from types import SimpleNamespace

import dspy
from dspy.utils.dummies import DummyLM

from integrations import DSPyProvider, use_call, use_module
from integrations.harness import (
    FEEDBACK_FIELD,
    VerificationFailed,
    assert_that,
    attempt_adapter,
    llm_judge,
    schema,
    verify,
)
from pyreact.core import EventBus, HookContext, component, hooks, runtime
from pyreact.core.harness import HarnessProvider, install
from pyreact.core.runtime import schedule_rerender


class FeedbackLM(dspy.BaseLM):
    """Answers lowercase without a period, unless the prompt carries feedback."""

    def __init__(self):
        super().__init__(model="fake/feedback", cache=False)
        self.prompts, self.kwargs_seen = [], []

    def forward(self, prompt=None, messages=None, **kw):
        user = messages[-1]["content"]
        self.prompts.append(messages)
        self.kwargs_seen.append(kw)
        text = "Paris." if "[[ ## feedback ## ]]" in user else "paris"
        msg = SimpleNamespace(content=f"[[ ## answer ## ]]\n{text}\n\n[[ ## completed ## ]]",
                              tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            model=self.model, _hidden_params={},
        )

    async def aforward(self, prompt=None, messages=None, **kw):
        return self.forward(prompt, messages, **kw)


def _run_app(mws, lm, question="capital of France?"):
    seen = {}

    @component
    def Agent():
        mod = use_module("question -> answer", name="qa")
        call = use_call(mod)
        seen["call"] = call
        hooks.use_effect(lambda: call.run(question=question), [])
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=lm, children=[
            HarnessProvider(key="h", middleware=mws, children=[Agent(key="a")])
        ])]

    async def main():
        root = HookContext("Root", Root)
        schedule_rerender(root)
        await runtime.run_until_idle()
        await runtime.run_until_idle()

    asyncio.run(main())
    return seen["call"]


PERIOD = assert_that(lambda p: p.answer[:1].isupper() and p.answer.endswith("."),
                     "Start with a capital letter and end with a period.", name="style")


def test_retry_with_feedback_reaches_the_prompt_as_an_input_field():
    ev = []
    HookContext.get_service("event_bus", EventBus).subscribe(ev.append, ["verify.*", "llm.*"])
    lm = FeedbackLM()
    call = _run_app([verify(PERIOD, on_fail="retry_with_feedback")], lm)
    assert call.error is None and call.result.answer == "Paris."
    assert len(lm.prompts) == 2
    first, second = lm.prompts
    assert "feedback" not in first[0]["content"]  # untouched first attempt
    assert f"`{FEEDBACK_FIELD}`" in second[0]["content"]  # field declared in the system prompt
    assert "Start with a capital letter and end with a period." in second[-1]["content"]
    assert lm.kwargs_seen[0].get("rollout_id") is None and lm.kwargs_seen[1]["rollout_id"] == 2
    kinds = [e.kind for e in ev]
    assert kinds.count("llm.start") == 1 and kinds.count("llm.end") == 1  # one call, 2 attempts
    assert [e.kind for e in ev if e.kind.startswith("verify")] == ["verify.failed",
                                                                    "verify.passed"]
    end = [e for e in ev if e.kind == "llm.end"][0].payload
    assert end["usage"]["total_tokens"] == 30 and end["outputs"]["answer"] == "Paris."


def test_raise_fails_the_call_with_verification_failed():
    lm = FeedbackLM()
    call = _run_app([verify(PERIOD)], lm)
    assert isinstance(call.error, VerificationFailed) and call.result is None
    assert "failed verification by style" in str(call.error)


def test_schema_on_a_prediction_and_process_wide_install():
    lm = FeedbackLM()
    un = install(verify(schema({"answer": str}), name="has-answer"))
    try:
        call = _run_app([], lm)
    finally:
        un()
    assert call.error is None and len(lm.prompts) == 1


def test_attempt_adapter_keeps_class_name_and_renames_on_collision():
    a = attempt_adapter(dspy.JSONAdapter(), feedback="x")
    assert type(a).__name__ == "JSONAdapter" and isinstance(a, dspy.JSONAdapter)
    lm = FeedbackLM()
    sig = dspy.Signature("question, feedback -> answer")
    wrapped = attempt_adapter(dspy.ChatAdapter(), feedback="be brief", rollout_id=3)
    with dspy.context(lm=lm, adapter=wrapped):
        dspy.Predict(sig)(question="q", feedback="user feedback")
    user = lm.prompts[0][-1]["content"]
    assert "[[ ## feedback_ ## ]]\nbe brief" in user and "user feedback" in user
    assert lm.kwargs_seen[0]["rollout_id"] == 3


def test_llm_judge_is_a_nested_llm_judge_boundary():
    spans = []

    async def recorder(b, call_next):
        try:
            return await call_next(b)
        finally:
            spans.append((b.kind, b.name, b.id, b.parent_id))

    judge_lm = DummyLM([{"score": 0.2, "feedback": "too vague"},
                        {"score": 0.9, "feedback": "good"}])
    lm = FeedbackLM()
    v = verify(llm_judge("Names the city precisely.", lm=judge_lm),
               on_fail="retry_with_feedback")
    un = install(recorder)
    try:
        call = _run_app([v], lm)
    finally:
        un()
    assert call.error is None and call.result.answer == "Paris."
    llm = [s for s in spans if s[0] == "llm"][0]
    judges = [s for s in spans if s[0] == "llm.judge"]
    assert len(judges) == 2 and all(j[3] == llm[2] for j in judges)
    assert "too vague" in lm.prompts[1][-1]["content"]


def test_llm_judge_needs_an_lm():
    call = _run_app([verify(llm_judge("anything"))], FeedbackLM())
    assert isinstance(call.error, RuntimeError) and "llm_judge needs lm" in str(call.error)


def test_plain_retry_passes_a_new_rollout_id():
    lm = FeedbackLM()
    call = _run_app([verify(PERIOD, on_fail="retry", on_exhausted="flag")], lm)
    assert call.error is None and call.result.answer == "paris"  # flagged, kept
    assert [k.get("rollout_id") for k in lm.kwargs_seen] == [None, 2]
    assert "feedback" not in lm.prompts[1][-1]["content"]
