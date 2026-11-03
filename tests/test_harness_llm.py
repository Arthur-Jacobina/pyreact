"""LLM calls flow through the harness (``"llm"`` boundaries). No network."""

import asyncio
from types import SimpleNamespace

import dspy
import pytest

from integrations import telemetry as tm
from integrations import DSPyProvider, invoke_module, use_call, use_module
from pyreact.core import EventBus, HookContext, component, hooks, runtime
from pyreact.core.harness import Blocked, Harness, HarnessProvider, install, middleware
from pyreact.core.runtime import schedule_rerender


class CountingLM(dspy.BaseLM):
    """Answers ``answer`` fields; counts calls; reports token usage."""

    def __init__(self, text="hi"):
        super().__init__(model="fake/counting", cache=False)
        self.text, self.calls = text, 0

    def forward(self, prompt=None, messages=None, **kw):
        self.calls += 1
        msg = SimpleNamespace(content=f"[[ ## answer ## ]]\n{self.text}\n\n[[ ## completed ## ]]",
                              tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            model=self.model, _hidden_params={"response_cost": 0.001},
        )

    async def aforward(self, prompt=None, messages=None, **kw):
        return self.forward(prompt, messages, **kw)


def _events(kinds=("llm.*",)):
    got = []
    HookContext.get_service("event_bus", EventBus).subscribe(got.append, kinds)
    return got


def _run_app(mws, lm, question="q?"):
    """Mount Root -> DSPyProvider -> HarnessProvider(mws) -> Agent; run one call."""
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


def test_llm_boundary_attrs_before_and_after_the_call():
    seen = {}

    async def spy(b, call_next):
        seen["before"] = dict(b.attrs)
        seen["kind"], seen["name"], seen["inputs"] = b.kind, b.name, dict(b.inputs)
        out = await call_next(b)
        seen["after"] = dict(b.attrs)
        return out

    lm = CountingLM("hello")
    call = _run_app([spy], lm)
    assert call.result.answer == "hello" and call.error is None
    assert seen["kind"] == "llm" and seen["name"] == "qa"
    assert seen["inputs"] == {"question": "q?"}
    before = seen["before"]
    assert set(before) >= {"call_id", "module_key", "arch", "model"}
    assert before["arch"] == "predict" and before["model"] == "fake/counting"
    assert before["call_id"] == call.call_id
    after = seen["after"]
    assert after["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert after["cost"] == pytest.approx(0.001)
    assert after["latency"] >= 0
    assert after["outputs"] == {"answer": "hello"}
    assert len(after["lm_calls"]) == 1 and after["lm_calls"][0]["completion"]


def test_retry_reruns_the_module_and_transformed_inputs_are_used():
    lm = CountingLM("x")
    attempts = []

    async def retry(b, call_next):
        await call_next(b)  # first answer discarded
        b.attempt += 1
        b.inputs = {**b.inputs, "question": "rewritten"}
        out = await call_next(b)
        attempts.append((b.attempt, b.attrs["usage"]["total_tokens"]))
        return out

    call = _run_app([retry], lm)
    assert lm.calls == 2
    assert attempts == [(2, 30)]  # usage accumulates across attempts
    assert call.usage["lm_calls"] == 2
    rec = tm.get_call(call.call_id)  # the transformed inputs reached the module
    assert "rewritten" in str(rec["lm_calls"][-1]["messages"])


def test_short_circuit_supplies_the_prediction_without_an_lm_call():
    lm = CountingLM("never")

    @middleware(kinds=["llm"])
    async def cache(b, call_next):
        return dspy.Prediction(answer="cached")

    call = _run_app([cache], lm)
    assert lm.calls == 0
    assert call.result.answer == "cached" and call.error is None


def test_blocked_becomes_call_error_with_events():
    lm = CountingLM()
    got = _events()

    @middleware(kinds=["llm"], name="no-qa")
    async def guard(b, call_next):
        raise Blocked("questions are closed today")

    call = _run_app([guard], lm)
    assert lm.calls == 0
    assert isinstance(call.error, Blocked)
    assert str(call.error) == "llm 'qa' blocked by no-qa: questions are closed today"
    kinds = [e.kind for e in got]
    assert kinds[0] == "llm.start" and "llm.blocked" in kinds and kinds[-1] == "llm.error"
    blocked = next(e for e in got if e.kind == "llm.blocked").payload
    assert blocked["reason"] == "questions are closed today" and blocked["by"] == "no-qa"
    assert next(e for e in got if e.kind == "llm.error").payload["blocked"]["by"] == "no-qa"


def test_events_unchanged_with_empty_pipeline():
    got = _events()
    call = _run_app([], CountingLM("ok"))
    assert call.result.answer == "ok"
    assert [e.kind for e in got] == ["llm.start", "llm.end"]
    end = got[-1].payload
    assert end["usage"]["total_tokens"] == 15 and end["outputs"] == {"answer": "ok"}


def test_invoke_module_outside_components_uses_process_wide_middleware():
    names = []

    async def spy(b, call_next):
        names.append((b.kind, b.name, b.component))
        return await call_next(b)

    async def main():
        mod = dspy.Predict("question -> answer")
        return await invoke_module(mod, {"question": "q"}, lm=CountingLM("y"), name="script")

    un = install(spy)
    try:
        result, usage = asyncio.run(main())
    finally:
        un()
    assert result.answer == "y" and usage["total_tokens"] == 15
    assert names == [("llm", "script", None)]

    # explicit harness wins
    h = Harness([spy])
    names.clear()

    async def main2():
        mod = dspy.Predict("question -> answer")
        return await invoke_module(mod, {"question": "q"}, lm=CountingLM("z"), harness=h)

    assert asyncio.run(main2())[0].answer == "z" and len(names) == 1


def test_process_wide_middleware_installed_after_mount_applies():
    """Scoped middleware is captured at render; process-wide is read at call time."""
    names = []

    async def spy(b, call_next):
        names.append(b.kind)
        return await call_next(b)

    hs = {}

    @component
    def Agent():
        call = use_call(use_module("question -> answer"))
        hs["call"] = call
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=CountingLM(), children=[Agent(key="a")])]

    async def main():
        root = HookContext("Root", Root)
        schedule_rerender(root)
        await runtime.run_until_idle()
        un = install(spy)
        try:
            hs["call"].run(question="late")
            await runtime.run_until_idle()
        finally:
            un()

    asyncio.run(main())
    assert names == ["llm"]
