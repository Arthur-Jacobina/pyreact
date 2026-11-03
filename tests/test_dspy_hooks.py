"""DSPy hooks layer (integrations/hooks.py). No network: DummyLM / fake LMs."""

import asyncio
import time
from types import SimpleNamespace

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from integrations import (
    DSPyProvider,
    use_agent,
    use_call,
    use_chain_of_thought,
    use_dspy_agent,
    use_dspy_call,
    use_dspy_module,
    use_module,
    use_predict,
    use_stream,
)
from pyreact.core import EventBus, HookContext, component, hooks, runtime
from pyreact.core.runtime import schedule_rerender
from pyreact.tools import ToolProvider


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _resp(text, usage=None, model="fake"):
    msg = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=usage or {},
        model=model,
        _hidden_params={"response_cost": 0.001},
    )


class FakeLM(dspy.BaseLM):
    """Answers 'answer' fields; optionally streams chunks like dspy.LM does."""

    def __init__(self, text="hello streaming world", stream=False, delay=0.0):
        super().__init__(model="fake/lm", cache=False)
        self.text, self.stream, self.delay = text, stream, delay

    def _full(self):
        return f"[[ ## answer ## ]]\n{self.text}\n\n[[ ## completed ## ]]"

    def forward(self, prompt=None, messages=None, **kw):
        return _resp(self._full(), {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

    async def aforward(self, prompt=None, messages=None, **kw):
        from litellm import ModelResponseStream
        from litellm.types.utils import Delta, StreamingChoices

        if self.delay:
            await asyncio.sleep(self.delay)
        send = dspy.settings.send_stream
        if self.stream and send is not None:
            pid = id(dspy.settings.caller_predict)
            full = self._full()
            for i in range(0, len(full), 4):
                chunk = ModelResponseStream(
                    choices=[StreamingChoices(delta=Delta(content=full[i : i + 4]))]
                )
                chunk.predict_id = pid
                await send.send(chunk)
                await asyncio.sleep(0.005)
        return self.forward(prompt, messages, **kw)


def _events(kinds=("llm.*", "tool.*")):
    got = []
    HookContext.get_service("event_bus", EventBus).subscribe(got.append, kinds)
    return got


async def _pump(cond, timeout=5.0):
    """Drive renders while background tasks run (run_until_idle would wait them out)."""
    t0 = time.perf_counter()
    while not cond() and time.perf_counter() - t0 < timeout:
        await asyncio.sleep(0.005)
        await runtime.run_renders()


def _mount(Root):
    async def main(driver=None):
        root = HookContext("Root", Root)
        schedule_rerender(root)
        await runtime.run_until_idle()
        if driver is not None:
            await driver()
        await runtime.run_until_idle()
        return root

    return main


# --------------------------------------------------------------------------
# DSPyProvider
# --------------------------------------------------------------------------
def test_provider_memoizes_env_and_never_configures_globally():
    lm = DummyLM([{"answer": "x"}])
    envs, hs = [], {}
    from integrations import use_dspy_env

    @component
    def Child():
        envs.append(use_dspy_env())
        return []

    @component
    def Root():
        n, set_n = hooks.use_state(0)
        hs["bump"] = lambda: set_n(lambda v: v + 1)
        # inline dict rebuilt each render: still the same env
        return [DSPyProvider(key="d", models={"default": lm}, children=[Child(key="c")])]

    before = dspy.settings.lm

    async def drive():
        hs["bump"]()
        await runtime.run_until_idle()

    asyncio.run(_mount(Root)(drive))
    assert len(envs) == 2 and envs[0] is envs[1]
    assert dspy.settings.lm is before


# --------------------------------------------------------------------------
# use_module
# --------------------------------------------------------------------------
def test_use_module_is_memoized_per_component_or_shared_by_name():
    mods = {}

    @component
    def A(tag):
        mods[(tag, "own")] = use_predict("question -> answer")
        mods[(tag, "shared")] = use_chain_of_thought("question -> answer", name="shared")
        mods[(tag, "legacy")] = use_dspy_module("question -> answer", dspy.Predict)
        return []

    @component
    def Root():
        n, set_n = hooks.use_state(0)
        mods.setdefault("bumps", []).append(set_n)
        return [DSPyProvider(key="d", lm=DummyLM([]), children=[A(key="a", tag="a"), A(key="b", tag="b")])]

    seen = []

    async def drive():
        seen.append(dict(mods))
        mods["bumps"][0](1)
        await runtime.run_until_idle()

    asyncio.run(_mount(Root)(drive))
    first = seen[0]
    assert first[("a", "own")] is not first[("b", "own")]
    assert first[("a", "shared")] is first[("b", "shared")]
    assert first[("a", "legacy")] is first[("b", "legacy")]
    assert isinstance(first[("a", "shared")], dspy.ChainOfThought)
    # stable across rerenders
    assert mods[("a", "own")] is first[("a", "own")]
    # module instances pass through untouched
    m = dspy.Predict("q -> a")

    @component
    def P():
        assert use_module(m) is m
        return []

    asyncio.run(_mount(P)())


def test_use_module_prefers_compiled_program():
    compiled = dspy.Predict("question -> answer")
    got = []

    class Opt:
        def compile(self, mod):
            return compiled

    @component
    def A():
        got.append(use_dspy_module("question -> answer", dspy.Predict, compile_with_optimizer=True))
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=DummyLM([]), optimizer=Opt(), children=[A(key="a")])]

    asyncio.run(_mount(Root)())
    assert got[0] is not compiled and got[-1] is compiled


# --------------------------------------------------------------------------
# use_call
# --------------------------------------------------------------------------
def test_use_call_result_ver_usage_and_events():
    fast = FakeLM("fast answer", delay=0.02)
    calls, hs = [], {}
    events = _events()

    @component
    def Agent():
        qa = use_predict("question -> answer")
        call = use_call(qa, model="fast")
        hs["call"] = call
        calls.append((call.loading, call.ver, call.result and call.result.answer))
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", models={"default": FakeLM("default"), "fast": fast},
                             children=[Agent(key="agent")])]

    async def drive():
        hs["call"].run(question="hi")
        await runtime.run_until_idle()

    asyncio.run(_mount(Root)(drive))
    c = hs["call"]
    assert c.result.answer == "fast answer" and c.ver == 1 and not c.loading and c.error is None
    assert c.usage["total_tokens"] == 15 and c.usage["lm_calls"] == 1
    assert c.usage["cost"] == pytest.approx(0.001)
    assert (True, 0, None) in calls
    kinds = [e.kind for e in events]
    assert kinds == ["llm.start", "llm.end"]
    end = events[1]
    assert end.payload["model"] == "fake/lm" and end.payload["usage"]["total_tokens"] == 15
    assert end.payload["component"].endswith("Agent#agent")
    assert end.payload["outputs"] == {"answer": "fast answer"}
    assert end.payload["latency"] >= 0
    # legacy unpacking still works
    run, (res, ver), loading, err = c
    assert res is c.result and ver == 1


def test_use_call_error_bumps_ver_and_publishes_llm_error():
    hs = {}
    events = _events()

    class Boom:
        async def acall(self, **kw):
            raise ValueError("nope")

    mod = Boom()

    @component
    def Agent():
        hs["call"] = use_call(mod)
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=object(), children=[Agent(key="a")])]

    async def drive():
        hs["call"].run(x=1)
        await runtime.run_until_idle()

    asyncio.run(_mount(Root)(drive))
    c = hs["call"]
    assert isinstance(c.error, ValueError) and c.ver == 1 and c.result is None
    assert [e.kind for e in events] == ["llm.start", "llm.error"]
    assert "nope" in events[1].payload["error"]


@pytest.mark.parametrize("policy,expect_calls,expect_ver", [
    ("replace", ["a", "b"], 1),
    ("parallel", ["a", "b"], 2),
    ("ignore", ["a"], 1),
])
def test_call_policies(policy, expect_calls, expect_ver):
    started, finished, hs = [], [], {}
    events = _events(["llm.cancel"])

    class Slow:
        async def acall(self, q):
            started.append(q)
            await asyncio.sleep(0.05)
            finished.append(q)
            return q

    mod = Slow()

    @component
    def Agent():
        hs["call"] = use_call(mod, lm=object(), policy=policy)
        return []

    async def drive():
        hs["call"].run(q="a")
        await asyncio.sleep(0.01)
        hs["call"].run(q="b")
        await runtime.run_until_idle()

    asyncio.run(_mount(Agent)(drive))
    assert started == expect_calls
    assert hs["call"].ver == expect_ver and not hs["call"].loading
    if policy == "replace":
        assert finished == ["b"] and hs["call"].result == "b"
        assert [e.kind for e in events] == ["llm.cancel"]


def test_legacy_use_dspy_call_tuple_shape():
    hs = {}

    @component
    def Agent():
        hs["t"] = use_dspy_call(dspy.Predict("question -> answer"), lm=FakeLM("ok"))
        return []

    async def drive():
        hs["t"][0](question="q")
        await runtime.run_until_idle()

    asyncio.run(_mount(Agent)(drive))
    run, (res, ver), loading, err = hs["t"]
    assert res.answer == "ok" and ver == 1 and loading is False and err is None


def test_turn_id_is_attached_when_available(monkeypatch):
    monkeypatch.setattr(runtime, "current_turn_id", lambda *a, **k: "turn-7")
    events = _events()
    hs = {}

    @component
    def Agent():
        hs["call"] = use_call(dspy.Predict("question -> answer"), lm=FakeLM())
        return []

    async def drive():
        hs["call"].run(question="q")
        await runtime.run_until_idle()

    asyncio.run(_mount(Agent)(drive))
    assert {e.turn_id for e in events} == {"turn-7"}


# --------------------------------------------------------------------------
# use_stream
# --------------------------------------------------------------------------
def _stream_app(lm, renders, hs):
    @component
    def Chat():
        mod = use_predict("question -> answer")
        s = use_stream(mod, listen=["answer"], throttle=0.0)
        hs["s"] = s
        renders.append((s.loading, s.text))
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=lm, children=[Chat(key="chat")])]

    return Root


def test_use_stream_rerenders_with_partial_text_and_publishes_chunks():
    renders, hs = [], {}
    events = _events(["llm.chunk", "llm.end"])

    async def drive():
        hs["s"].run(question="q")
        await _pump(lambda: hs["s"].ver > 0)

    asyncio.run(_mount(_stream_app(FakeLM("hello streaming world", stream=True), renders, hs))(drive))
    s = hs["s"]
    assert s.text == "hello streaming world" and s.result.answer == "hello streaming world"
    partials = [t for loading, t in renders if loading and t]
    assert len(partials) >= 2, renders  # several intermediate rerenders
    assert all("hello streaming world".startswith(p.strip()) for p in partials)
    chunks = [e for e in events if e.kind == "llm.chunk"]
    assert chunks and "".join(e.payload["chunk"] for e in chunks).strip() == "hello streaming world"
    assert events[-1].kind == "llm.end"


def test_use_stream_falls_back_when_lm_does_not_stream():
    renders, hs = [], {}

    async def drive():
        hs["s"].run(question="q")
        await runtime.run_until_idle()

    asyncio.run(_mount(_stream_app(DummyLM([{"answer": "whole"}]), renders, hs))(drive))
    assert hs["s"].text == "whole" and hs["s"].ver == 1


# --------------------------------------------------------------------------
# use_agent
# --------------------------------------------------------------------------
def test_use_agent_runs_sync_tools_off_the_loop_and_emits_tool_events():
    def slow_lookup(sku: str) -> str:
        """Look up a SKU slowly.
        sku: the sku
        """
        time.sleep(0.3)
        return f"price({sku})=10"

    lm = DummyLM([
        {"next_thought": "look it up", "next_tool_name": "slow_lookup", "next_tool_args": {"sku": "A1"}},
        {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
        {"reasoning": "found", "answer": "10"},
    ])
    hs, ticks = {}, []
    events = _events()

    @component
    def Agent():
        hs["call"] = use_agent("question -> answer", tools=["slow_lookup"], max_iters=3)
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=lm, children=[
            ToolProvider(key="t", tools=[slow_lookup], children=[Agent(key="agent")])
        ])]

    async def drive():
        hs["call"].run(question="price of A1?")

        def done():
            ticks.append(time.perf_counter())
            return hs["call"].ver > 0

        await _pump(done)

    asyncio.run(_mount(Root)(drive))
    c = hs["call"]
    assert c.error is None and c.result.answer == "10"
    assert c.usage["lm_calls"] == 3
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    # The property: the loop is never blocked by the 0.3s tool (it runs in a
    # worker thread). A blocked loop shows a gap >= 0.3s; under load
    # (parallel test runs, Docker) scheduler jitter alone can exceed 0.1s,
    # so the bound is 0.25s -- still well under the sleep -- and many ticks
    # during the call show the loop kept turning.
    assert len(ticks) > 20 and max(gaps) < 0.25, gaps
    kinds = [e.kind for e in events]
    assert kinds == ["llm.start", "tool.start", "tool.end", "tool.start", "tool.end", "llm.end"] or (
        kinds[0] == "llm.start" and kinds[-1] == "llm.end" and "tool.end" in kinds
    )
    tool_end = next(e for e in events if e.kind == "tool.end" and e.payload["tool"] == "slow_lookup")
    assert tool_end.payload["duration"] >= 0.3 and tool_end.payload["error"] is None
    assert tool_end.payload["args"] == {"sku": "A1"}
    assert tool_end.payload["call_id"] == events[0].payload["call_id"]


def test_legacy_use_dspy_agent_tuple_and_tool_errors():
    def broken(x: int) -> int:
        """Always fails."""
        raise RuntimeError("bad tool")

    lm = DummyLM([
        {"next_thought": "t", "next_tool_name": "broken", "next_tool_args": {"x": 1}},
        {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
        {"reasoning": "r", "answer": "gave up"},
    ])
    hs = {}
    events = _events(["tool.end"])

    @component
    def Agent():
        hs["t"] = use_dspy_agent("question -> answer", tools=[broken], lm=lm, max_iters=3)
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=lm, children=[Agent(key="a")])]

    async def drive():
        hs["t"][0](question="q")
        await runtime.run_until_idle()

    asyncio.run(_mount(Root)(drive))
    run, (res, ver), loading, err = hs["t"]
    assert res.answer == "gave up" and ver == 1
    assert "bad tool" in events[0].payload["error"]
