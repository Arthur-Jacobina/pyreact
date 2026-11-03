"""RLM hook plumbing + learning-loop plumbing (recorder, hot swap). No network."""

import asyncio
import json
import time

import dspy
from dspy.primitives.code_interpreter import FinalOutput

from integrations.claude_cli_lm import _wrap
from integrations.dspy_integration import DSPyProvider, use_dspy_env, use_dspy_module
from integrations.feedback import (
    TracedModule,
    TraceRecorder,
    key_str,
    send_feedback,
    use_traced,
)
from integrations.optimize import (
    ModuleVersions,
    hot_swap,
    instructions_of,
    load_compiled,
    module_key_for,
    save_compiled,
    use_module_version,
)
from integrations.use_dspy import use_dspy_call
from pyreact.core import EventBus, HookContext, component, hooks, runtime
from pyreact.core.runtime import schedule_rerender


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class FakeLM(dspy.BaseLM):
    """Always proposes the same REPL step; the fake interpreter decides when to stop."""

    def __init__(self, code="step()"):
        super().__init__(model="fake/lm", cache=False)
        self.code = code
        self.n = 0

    def forward(self, prompt=None, messages=None, **kw):
        self.n += 1
        text = f"[[ ## reasoning ## ]]\nthink\n\n[[ ## code ## ]]\n{self.code}\n\n[[ ## completed ## ]]"
        return _wrap(text, self.model)


class FakeInterpreter:
    """CodeInterpreter double: N plain steps, then SUBMIT; optionally slow."""

    instances = []

    def __init__(self, steps=2, delay=0.0, use_sub_lm=False):
        self.tools = {}
        self.output_fields = None
        self.steps, self.delay, self.use_sub_lm = steps, delay, use_sub_lm
        self.executed = 0
        self.shut = False
        FakeInterpreter.instances.append(self)

    def start(self):
        pass

    def execute(self, code, variables=None):
        if self.delay:
            time.sleep(self.delay)
        self.executed += 1
        if self.use_sub_lm and self.executed == 1:
            return self.tools["llm_query"]("summarize this")
        if self.executed > self.steps:
            return FinalOutput({"answer": f"len={len(variables['context'])}"})
        return f"step {self.executed}"

    def shutdown(self):
        self.shut = True


async def _settle():
    await runtime.run_until_idle()


def _collect(kinds=("llm.*", "rlm.*")):
    bus = HookContext.get_service("event_bus", EventBus)
    events = []
    bus.subscribe(events.append, kinds=list(kinds))
    return events


# ---------------------------------------------------------------------------
# feedback: runs -> events -> dataset
# ---------------------------------------------------------------------------
class Router:
    def __call__(self, message):
        return dspy.Prediction(name="order" if "comprar" in message else "catalog")


def test_recorder_builds_dataset_from_calls_and_feedback(tmp_path):
    bus = EventBus()
    path = tmp_path / "events.jsonl"
    rec = TraceRecorder(bus, path=str(path))
    key = ("ChainOfThought", "RouteSpecialistSig", "supervisor")
    traced = TracedModule(Router(), key, bus=bus)

    traced(message="quais produtos")
    ok_id = traced.last_call_id
    traced(message="comprar e saber o frete")
    bad_id = traced.last_call_id
    send_feedback(ok_id, 1.0, "right", bus=bus)
    send_feedback(
        bad_id, 0.0, "frete -> shipping", expected={"name": "shipping"}, bus=bus
    )
    # turn-targeted feedback that arrives before its call is attached later
    send_feedback("turn-9", 0.5, "meh", bus=bus)
    TracedModule(Router(), key, bus=bus, turn_id="turn-9")(message="vocês têm boné?")
    TracedModule(Router(), "other", bus=bus)(message="ignored")

    ds = rec.dataset(key)
    assert [e.message for e in ds] == [
        "quais produtos",
        "comprar e saber o frete",
        "vocês têm boné?",
    ]
    assert [e.name for e in ds] == [
        "catalog",
        "shipping",
        "catalog",
    ]  # label = correction
    assert ds[1].recorded_outputs == {"name": "order"}
    assert ds[1].feedback_text == "frete -> shipping" and ds[2].feedback_score == 0.5
    assert set(ds[0].inputs().keys()) == {"message"}
    # only good runs or corrected ones
    assert [e.message for e in rec.dataset(key, min_score=1.0)] == [
        "quais produtos",
        "comprar e saber o frete",
    ]
    assert (
        len(rec.dataset("other")) == 1
        and rec.dataset("other", require_feedback=True) == []
    )

    # persisted JSONL replays into the same dataset
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert {x["kind"] for x in lines} == {"module.call", "feedback"}
    rec.close()
    replay = TraceRecorder(EventBus()).load(str(path))
    assert [(e.message, e.name) for e in replay.dataset(key_str(key))] == [
        (e.message, e.name) for e in ds
    ]


def test_traced_module_works_with_use_dspy_call():
    rec = TraceRecorder(HookContext.get_service("event_bus", EventBus))

    class AsyncRouter:
        async def acall(self, message):
            return dspy.Prediction(name=f"r:{message}")

    mod, hs, got = AsyncRouter(), {}, []

    @component
    def Agent():
        traced = use_traced(mod, "router")
        run, result, _l, _e = use_dspy_call(traced, lm=object())
        hs["run"], hs["traced"] = run, traced
        hooks.use_effect(
            lambda: result[0] is not None and got.append(result[0].name), [result]
        )
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=object(), children=[Agent(key="a")])]

    async def main():
        schedule_rerender(HookContext("Root", Root))
        await _settle()
        hs["run"](message="hi")
        await _settle()

    asyncio.run(main())
    assert got == ["r:hi"]
    (r,) = rec.records("router")
    assert r["inputs"] == {"message": "hi"} and r["outputs"] == {"name": "r:hi"}
    assert r["call_id"] == hs["traced"].last_call_id


# ---------------------------------------------------------------------------
# optimize: save/load + hot swap
# ---------------------------------------------------------------------------
def test_save_and_load_compiled_roundtrip(tmp_path):
    mod = dspy.ChainOfThought("question -> answer")
    mod.predict.signature = mod.predict.signature.with_instructions("Be terse. v2")
    p = save_compiled(mod, str(tmp_path / "a" / "prog.json"))
    loaded = load_compiled(dspy.ChainOfThought, "question -> answer", p)
    assert instructions_of(loaded) == {"predict": "Be terse. v2"}


class FakeMod:
    def __init__(self, signature, tag="base"):
        self.signature, self.tag = signature, tag


def test_hot_swap_rerenders_users_and_next_call_uses_compiled():
    ModuleVersions._ctx.value = {}
    hs, seen, other_renders = {}, [], []

    @component
    def Agent():
        hs["env"] = use_dspy_env()
        mod = use_dspy_module("q -> a", FakeMod, name="router")
        version = use_module_version(module_key_for(FakeMod, "q -> a", "router"))
        seen.append((mod.tag, version))
        return []

    @component
    def Bystander():
        use_dspy_env()
        other_renders.append(1)
        return []

    @component
    def Root():
        return [
            DSPyProvider(
                key="d", lm=object(), children=[Agent(key="a"), Bystander(key="b")]
            )
        ]

    key = module_key_for(FakeMod, "q -> a", "router")
    compiled = FakeMod("q -> a", tag="optimized")

    async def main():
        schedule_rerender(HookContext("Root", Root))
        await _settle()
        scheduled = hot_swap(hs["env"], key, compiled)
        await _settle()
        return scheduled

    scheduled = asyncio.run(main())
    assert [c.name for c in scheduled] == ["Agent"]
    assert seen[0] == ("base", 0)
    assert seen[-1] == ("optimized", 1)
    assert len(other_renders) == 1  # unrelated DSPy users are not rerendered
    assert hs["env"].compiled[key] is compiled and hs["env"].caches[key] is compiled
    ModuleVersions._ctx.value = {}
