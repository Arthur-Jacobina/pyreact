"""fallback(fn) (generic) and fallback_model(chain) (llm boundaries)."""

import asyncio
from types import SimpleNamespace

import dspy
import pytest

from integrations import DSPyProvider, use_call, use_module
from integrations.recovery import fallback_model
from pyreact.core import EventBus, HookContext, component, hooks, runtime
from pyreact.core.harness import Blocked, Harness, HarnessProvider, through
from pyreact.core.recovery import fallback, retry
from pyreact.core.runtime import schedule_rerender


def _events(kinds=("recovery.*",)):
    got = []
    HookContext.get_service("event_bus", EventBus).subscribe(got.append, kinds)
    return got


def test_fallback_returns_substitute():
    events = _events()

    async def broken(b):
        raise ConnectionError("down")

    async def cached(b, exc):
        return f"cached {b.inputs['city']} ({type(exc).__name__})"

    h = Harness([fallback(cached, kinds=["tool"])])
    assert asyncio.run(through("tool", "weather", {"city": "Rome"}, broken, harness=h)) \
        == "cached Rome (ConnectionError)"
    assert events[-1].kind == "recovery.fallback"
    assert events[-1].payload["to"] == "cached"

    h = Harness([fallback(lambda b, e: "sync ok")])
    assert asyncio.run(through("tool", "t", {}, broken, harness=h)) == "sync ok"

    async def refused(b):
        raise Blocked("no")

    with pytest.raises(Blocked):
        asyncio.run(through("tool", "t", {}, refused, harness=h))
    h = Harness([fallback(lambda b, e: "x", on=TimeoutError)])
    with pytest.raises(ConnectionError):
        asyncio.run(through("tool", "t", {}, broken, harness=h))


class LM(dspy.BaseLM):
    def __init__(self, model, text="hi", fail=False):
        super().__init__(model=model, cache=False)
        self.text, self.fail, self.calls = text, fail, 0

    def forward(self, prompt=None, messages=None, **kw):
        self.calls += 1
        if self.fail:
            raise ConnectionError(f"{self.model} overloaded")
        msg = SimpleNamespace(content=f"[[ ## answer ## ]]\n{self.text}\n\n[[ ## completed ## ]]",
                              tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage={}, model=self.model)

    async def aforward(self, prompt=None, messages=None, **kw):
        return self.forward(prompt, messages, **kw)


def _run(models, mws, model=None):
    seen = {}

    @component
    def Agent():
        mod = use_module("question -> answer", name="qa")
        call = use_call(mod, model=model)
        seen["call"] = call
        hooks.use_effect(lambda: call.run(question="q"), [])
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", models=models, children=[
            HarnessProvider(key="h", middleware=mws, children=[Agent(key="a")])])]

    async def main():
        root = HookContext("Root", Root)
        schedule_rerender(root)
        await runtime.run_until_idle()
        await runtime.run_until_idle()

    asyncio.run(main())
    return seen["call"]


def test_fallback_model_switches_to_the_next_registry_model():
    events = _events(("recovery.*", "llm.end"))
    big, small = LM("big/one", fail=True), LM("small/one", text="from small")
    seen = {}

    async def spy(b, call_next):
        seen["before"] = (b.attrs.get("model_key"), b.attrs.get("model_names"))
        try:
            return await call_next(b)
        finally:
            seen["after"] = (b.attrs.get("model_key"), b.attrs.get("model"), b.attempt)

    call = _run({"default": big, "fast": small},
                [spy, fallback_model(chain=["default", "fast"])])
    assert call.error is None and call.result.answer == "from small"
    assert big.calls >= 1 and small.calls == 1
    assert seen["before"] == ("default", {"default": "big/one", "fast": "small/one"})
    assert seen["after"] == ("fast", "small/one", 2)
    fb = [e for e in events if e.kind == "recovery.fallback"]
    assert fb[0].payload["from"] == "default" and fb[0].payload["to"] == "fast"
    end = [e for e in events if e.kind == "llm.end"][-1]
    assert end.payload["model"] == "small/one" and end.payload["model_key"] == "fast"


def test_fallback_model_composes_with_retry_and_raises_when_all_fail():
    big, small = LM("big/one", fail=True), LM("small/one", fail=True)
    call = _run({"default": big, "fast": small},
                [fallback_model(chain=["default", "fast"]),
                 retry(max_attempts=2, backoff=lambda n: 0)])
    assert isinstance(call.error, ConnectionError) and "small/one" in str(call.error)
    # 2 retries per model; each attempt = ChatAdapter call + JSONAdapter fallback
    assert big.calls == 4 and small.calls == 4


def test_fallback_model_degrades_only_along_the_chain():
    big, small, other = LM("big/one"), LM("small/one", fail=True), LM("other/x", text="other")
    # the call uses "fast", which is last in the chain: nothing to fall back to
    call = _run({"default": big, "fast": small, "other": other},
                [fallback_model(chain=["default", "fast"])], model="fast")
    assert isinstance(call.error, ConnectionError) and big.calls == 0
    # a call outside the chain tries its own model, then the chain
    call = _run({"default": big, "fast": small, "other": LM("other/x", fail=True)},
                [fallback_model(chain=["fast", "default"])], model="other")
    assert call.result.answer == "hi"
