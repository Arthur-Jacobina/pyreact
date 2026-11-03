"""Architectures: one switch, same Call shape. No network (DummyLM / fakes)."""

import asyncio
import time

import dspy
import pytest
from dspy.primitives.code_interpreter import FinalOutput
from dspy.utils.dummies import DummyLM

from integrations import DSPyProvider, use_agent_call, use_call, use_module
from integrations.architectures import (
    arch_of,
    build_module,
    get_architecture,
    list_architectures,
    register_architecture,
    unregister_architecture,
)
from integrations.claude_cli_lm import _wrap
from pyreact.core import EventBus, HookContext, component, hooks, runtime
from pyreact.core.runtime import schedule_rerender


def _events(kinds=("llm.*", "rlm.*")):
    got = []
    HookContext.get_service("event_bus", EventBus).subscribe(got.append, list(kinds))
    return got


async def _settle():
    await runtime.run_until_idle()


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_builtin_registry_and_build_module():
    names = list_architectures()
    for n in ("predict", "cot", "react", "rlm", "pot", "refine", "best_of_n"):
        assert n in names and names[n]
    assert isinstance(build_module("q -> a", "predict"), dspy.Predict)
    cot = build_module("q -> a", "cot")
    assert isinstance(cot, dspy.ChainOfThought) and arch_of(cot) == "cot"
    assert arch_of(dspy.ChainOfThought("q -> a")) == "cot"  # inferred from class
    assert isinstance(build_module("q -> a", "best_of_n", N=2), dspy.BestOfN)
    assert isinstance(build_module("q -> a", "refine", N=2, base="predict").module, dspy.Predict)
    assert get_architecture("ChainOfThought").name == "cot"  # alias
    with pytest.raises(KeyError, match="register_architecture"):
        get_architecture("nope")


def test_register_custom_architecture():
    class Echo(dspy.Module):
        def __init__(self, signature, prefix=">"):
            super().__init__()
            self.prefix = prefix

        def forward(self, **kw):
            return dspy.Prediction(answer=f"{self.prefix}{kw['question']}")

    register_architecture("echo", Echo, doc="Echo the question back.")
    try:
        assert list_architectures()["echo"] == "Echo the question back."
        mod = build_module("question -> answer", "echo", prefix="!")
        assert arch_of(mod) == "echo" and mod(question="hi").answer == "!hi"
        with pytest.raises(ValueError):
            register_architecture("echo", Echo, replace=False)
    finally:
        unregister_architecture("echo")
    assert "echo" not in list_architectures()


# ---------------------------------------------------------------------------
# switching arch like a prop keeps the Call shape
# ---------------------------------------------------------------------------
def test_switching_arch_prop_rebuilds_module_and_keeps_call_shape():
    lm = DummyLM([{"answer": "p1"}, {"reasoning": "r", "answer": "c1"}])
    events = _events(["llm.start", "llm.end"])
    hs, mods = {}, []

    @component
    def QA(arch):
        mod = use_module("question -> answer", arch=arch)
        call = use_call(mod)
        mods.append(mod)
        hs["call"] = call
        return []

    @component
    def Root():
        arch, set_arch = hooks.use_state("predict")
        hs["set_arch"] = set_arch
        return [DSPyProvider(key="d", lm=lm, children=[QA(key="qa", arch=arch)])]

    async def main():
        schedule_rerender(HookContext("Root", Root))
        await _settle()
        hs["call"].run(question="q1")
        await _settle()
        first = hs["call"]
        hs["set_arch"]("cot")
        await _settle()
        hs["call"].run(question="q2")
        await _settle()
        return first

    first = asyncio.run(main())
    second = hs["call"]
    assert type(first) is type(second)
    assert first.result.answer == "p1" and second.result.answer == "c1"
    assert isinstance(mods[0], dspy.Predict) and isinstance(mods[-1], dspy.ChainOfThought)
    starts = [e.payload for e in events if e.kind == "llm.start"]
    assert [s["arch"] for s in starts] == ["predict", "cot"]
    ends = [e.payload for e in events if e.kind == "llm.end"]
    assert ends[0]["module_key"].startswith("Predict:question -> answer:")
    assert ends[1]["module_key"].startswith("ChainOfThought:question -> answer:")
    # call ids flow to the Call snapshot (for explain / feedback)
    assert second.call_id == ends[1]["call_id"]


def test_use_module_cache_key_matches_legacy_keys():
    seen = {}

    @component
    def A():
        seen["cot"] = use_module("q -> a", arch="cot", name="shared")
        seen["legacy"] = use_module("q -> a", dspy.ChainOfThought, name="shared")
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=DummyLM([]), children=[A(key="a")])]

    async def main():
        schedule_rerender(HookContext("Root", Root))
        await _settle()

    asyncio.run(main())
    assert seen["cot"] is seen["legacy"]  # same ("ChainOfThought", sig, "shared") key


# ---------------------------------------------------------------------------
# RLM through the same Call: worker thread, events, cancellation
# ---------------------------------------------------------------------------
class RLMFakeLM(dspy.BaseLM):
    def __init__(self):
        super().__init__(model="fake/rlm", cache=False)

    def forward(self, prompt=None, messages=None, **kw):
        return _wrap("[[ ## reasoning ## ]]\nr\n\n[[ ## code ## ]]\nstep()\n\n"
                     "[[ ## completed ## ]]", self.model)


class FakeInterpreter:
    def __init__(self, steps=1, delay=0.0):
        self.tools, self.output_fields = {}, None
        self.steps, self.delay, self.executed, self.shut = steps, delay, 0, False

    def start(self):
        pass

    def execute(self, code, variables=None):
        if self.delay:
            time.sleep(self.delay)
        self.executed += 1
        if self.executed > self.steps:
            return FinalOutput({"answer": f"chars={len(variables['context'])}"})
        return "ok"

    def shutdown(self):
        self.shut = True


def test_use_agent_call_rlm_has_call_shape_events_and_cancel():
    events = _events()
    hs, interps, delay = {}, [], {"v": 0.0}

    def make():
        interps.append(FakeInterpreter(steps=2, delay=delay["v"]))
        return interps[-1]

    @component
    def Agent():
        hs["call"] = use_agent_call("context, question -> answer", arch="rlm",
                                    max_iterations=6, interpreter=make)
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=RLMFakeLM(), children=[Agent(key="a")])]

    async def main():
        schedule_rerender(HookContext("Root", Root))
        await _settle()
        hs["call"].run(context="x" * 42, question="?")
        await _settle()
        ok = hs["call"]
        delay["v"] = 0.05  # next call is slow: cancel it from the Call snapshot
        hs["call"].run(context="y", question="?")
        await asyncio.sleep(0.03)
        hs["call"].cancel()
        await _settle()
        await asyncio.sleep(0.2)  # let the worker thread unwind
        return ok

    ok = asyncio.run(main())
    assert ok.result.answer == "chars=42" and ok.ver == 1 and ok.error is None
    assert ok.usage["lm_calls"] == 3  # 3 REPL steps, recorded for this call only
    kinds = [e.kind for e in events]
    assert kinds[:5] == ["llm.start", "rlm.iteration", "rlm.iteration", "rlm.iteration", "llm.end"]
    assert events[0].payload["arch"] == "rlm"
    end = events[4].payload
    assert end["iterations"] == 3 and end["outputs"]["answer"] == "chars=42"
    assert kinds[5:6] == ["llm.start"] and "llm.cancel" in kinds[6:]
    assert "llm.end" not in kinds[5:]
    assert interps[-1].shut and interps[-1].executed < 3  # stopped in the worker
    assert hs["call"].loading is False
