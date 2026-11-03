"""RLM hook plumbing + learning-loop plumbing (recorder, hot swap). No network."""

import asyncio
import time

import dspy
import pytest
from dspy.primitives.code_interpreter import FinalOutput

from integrations import rlm as rlm_mod
from integrations.claude_cli_lm import _wrap
from integrations.dspy_integration import DSPyProvider
from integrations.rlm import LenientRLMAdapter, RLMSandboxError, use_rlm
from pyreact.core import EventBus, HookContext, component, hooks, runtime
from pyreact.core.harness import HarnessProvider
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
# use_rlm
# ---------------------------------------------------------------------------
def test_use_rlm_runs_in_worker_and_publishes_events():
    events = _collect()
    hs, seen = {}, []
    lm, sub = FakeLM(), FakeLM()

    @component
    def Agent():
        run, (result, ver), loading, error = use_rlm(
            "context, question -> answer",
            sub_model="fast",
            max_iterations=5,
            interpreter=lambda: FakeInterpreter(steps=2, use_sub_lm=True),
        )
        hs["run"] = run
        seen.append((loading, ver, None if result is None else result.answer, error))
        return []

    @component
    def Root():
        return [
            DSPyProvider(
                key="d", models={"default": lm, "fast": sub}, children=[Agent(key="a")]
            )
        ]

    async def main():
        schedule_rerender(HookContext("Root", Root))
        await _settle()
        hs["run"](context="x" * 5000, question="how long?")
        await _settle()

    asyncio.run(main())
    loading, ver, answer, error = seen[-1]
    assert error is None and not loading and ver == 1
    assert answer == "len=5000"
    assert any(s[0] for s in seen)  # was loading at some point
    kinds = [e.kind for e in events]
    assert kinds[0] == "llm.start" and kinds[-1] == "llm.end"
    assert kinds.count("rlm.iteration") == 3
    end = events[-1].payload
    assert end["iterations"] == 3 and end["sub_llm_calls"] == 1
    assert end["trajectory_len"] == 3
    assert events[0].payload["input_chars"] == 5000 + len("how long?")
    assert sub.n == 1  # sub_model picked the sub-LM by name
    assert FakeInterpreter.instances[-1].shut  # per-call interpreter shut down


def test_use_rlm_superseded_call_is_cancelled_in_the_worker():
    events = _collect()
    hs, results = {}, []

    @component
    def Agent():
        run, (result, ver), _l, error = use_rlm(
            "context, question -> answer",
            lm=FakeLM(),
            max_iterations=10,
            interpreter=lambda: FakeInterpreter(steps=3, delay=0.05),
        )
        hs["run"] = run
        hooks.use_effect(lambda: ver and results.append((result.answer, error)), [ver])
        return []

    async def main():
        schedule_rerender(HookContext("Agent", Agent))
        await _settle()
        hs["run"](context="old", question="q")
        await asyncio.sleep(0.08)
        hs["run"](context="newer", question="q")
        await _settle()
        await asyncio.sleep(0.3)  # let the cancelled worker thread unwind

    asyncio.run(main())
    assert results == [("len=5", None)]
    # Unified Call path: a superseded call publishes llm.cancel (like every
    # other architecture) and its worker thread stops at the next checkpoint.
    assert [e for e in events if e.kind == "llm.error"] == []
    cancels = [e for e in events if e.kind == "llm.cancel"]
    assert len(cancels) == 1
    ends = [e for e in events if e.kind == "llm.end"]
    assert len(ends) == 1
    assert cancels[0].payload["call_id"] != ends[0].payload["call_id"]
    old = FakeInterpreter.instances[-2]
    assert old.shut and old.executed < 4  # stopped before SUBMIT


def test_use_rlm_missing_sandbox_surfaces_clear_error(monkeypatch):
    monkeypatch.setattr(rlm_mod.shutil, "which", lambda _name: None)
    events = _collect()
    hs, errors = {}, []

    @component
    def Agent():
        run, (_r, ver), _l, error = use_rlm("context, question -> answer", lm=FakeLM())
        hs["run"] = run
        hooks.use_effect(lambda: ver and errors.append(error), [ver])
        return []

    async def main():
        schedule_rerender(HookContext("Agent", Agent))
        await _settle()
        hs["run"](context="c", question="q")
        await _settle()

    asyncio.run(main())
    assert len(errors) == 1 and isinstance(errors[0], RLMSandboxError)
    assert "Deno" in str(errors[0]) and "interpreter=" in str(errors[0])
    assert [e.kind for e in events] == ["llm.start", "llm.error"]


def test_lenient_adapter_accepts_prose_plus_code_block():
    sig = dspy.Signature("question -> reasoning, code")
    out = LenientRLMAdapter().parse(
        sig, "I'll look first.\n\n```python\nprint(1)\n```\n"
    )
    assert out == {"reasoning": "I'll look first.", "code": "print(1)"}
    with pytest.raises(Exception):
        LenientRLMAdapter().parse(dspy.Signature("q -> answer"), "no markers")


def test_rlm_run_is_an_llm_boundary():
    seen = []

    async def spy(b, call_next):
        out = await call_next(b)
        seen.append((b.kind, b.attrs["arch"], b.attrs["call_id"], b.attrs["outputs"]["answer"]))
        return out

    hs = {}

    @component
    def Agent():
        run, (result, ver), _l, error = use_rlm(
            "context, question -> answer", lm=FakeLM(), max_iterations=5,
            interpreter=lambda: FakeInterpreter(steps=1),
        )
        hs["run"], hs["result"] = run, result
        return []

    @component
    def Root():
        return [HarnessProvider(key="h", middleware=[spy], children=[Agent(key="a")])]

    async def main():
        schedule_rerender(HookContext("Root", Root))
        await _settle()
        hs["cid"] = hs["run"](context="abc", question="q")
        await _settle()

    asyncio.run(main())
    assert seen == [("llm", "rlm", hs["cid"], "len=3")]


class ToolInterpreter(FakeInterpreter):
    """Calls the ``shout`` tool from the REPL (worker thread), then SUBMITs."""

    def execute(self, code, variables=None):
        self.executed += 1
        if self.executed == 1:
            return self.tools["shout"](text="hi")
        return FinalOutput({"answer": "done"})


def test_rlm_tools_run_as_tool_boundaries_from_the_worker_thread():
    import threading

    seen, threads = [], []

    def shout(text: str) -> str:
        """Shout the text.
        text: what to shout
        """
        threads.append(threading.current_thread().name)
        return text.upper()

    async def spy(b, call_next):
        out = await call_next(b)
        seen.append((b.kind, b.name, dict(b.inputs), b.attrs.get("result"),
                     b.attrs.get("call_id")))
        return out

    hs = {}

    @component
    def Agent():
        run, (result, ver), _l, error = use_rlm(
            "context, question -> answer", lm=FakeLM(), tools=[shout], max_iterations=5,
            interpreter=lambda: ToolInterpreter(),
        )
        hs["run"] = run
        return []

    @component
    def Root():
        return [HarnessProvider(key="h", middleware=[spy], children=[Agent(key="a")])]

    async def main():
        schedule_rerender(HookContext("Root", Root))
        await _settle()
        hs["cid"] = hs["run"](context="abc", question="q")
        await _settle()

    asyncio.run(main())
    tool = [s for s in seen if s[0] == "tool"]
    assert tool == [("tool", "shout", {"text": "hi"}, "HI", hs["cid"])]
    assert threads and threads[0] != "MainThread"  # sync tool still off the loop
    assert [s[0] for s in seen] == ["tool", "llm"]
