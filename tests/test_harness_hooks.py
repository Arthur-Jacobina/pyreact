"""use_guard / use_verify / use_middleware: component-scoped harness. No network."""

import asyncio
from types import SimpleNamespace

import dspy
import pytest

from integrations import DSPyProvider, use_call, use_module
from pyreact.core import HookContext, component, hooks, runtime
from pyreact.core.harness import Blocked, Boundary, HarnessProvider, resolve_harness
from pyreact.core.runtime import schedule_rerender
from pyreact.harness import (
    assert_that,
    budget,
    scoped_middleware,
    use_guard,
    use_middleware,
    use_verify,
)
from pyreact.harness import hooks as harness_hooks
from pyreact.harness.hooks import _Entry


class EchoLM(dspy.BaseLM):
    """answer = 'ok' (or 'OK.' when the prompt carries verifier feedback)."""

    def __init__(self):
        super().__init__(model="fake/echo", cache=False)
        self.calls = 0

    def forward(self, prompt=None, messages=None, **kw):
        self.calls += 1
        text = "OK." if "[[ ## feedback ## ]]" in messages[-1]["content"] else "ok"
        msg = SimpleNamespace(content=f"[[ ## answer ## ]]\n{text}\n\n[[ ## completed ## ]]",
                              tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage={"prompt_tokens": 1, "completion_tokens": 1,
                                      "total_tokens": 2},
                               model=self.model, _hidden_params={})

    async def aforward(self, prompt=None, messages=None, **kw):
        return self.forward(prompt, messages, **kw)


calls = {}


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    for e in list(harness_hooks._registry):  # roots dropped without unmounting
        harness_hooks._unregister(e)
    assert resolve_harness().middleware == []


@component
def Asker(name: str, question: str = "q", children=None):
    mod = use_module("question -> answer", name=name)
    call = use_call(mod)
    calls[name] = call
    hooks.use_effect(lambda: call.run(question=question), [question])
    return children or []


def _mount(Root, steps=2):
    async def main():
        root = HookContext("Root", Root)
        schedule_rerender(root)
        for _ in range(steps):
            await runtime.run_until_idle()
        return root

    return asyncio.run(main())


def test_use_guard_scopes_to_the_component_and_its_subtree_only():
    calls.clear()

    @component
    def Guarded():
        use_guard(lambda b: f"{b.name} is guarded", kinds="llm", name="lock")
        return [Asker(key="mine", name="mine", children=[Asker(key="child", name="child")])]

    @component
    def Root():
        return [DSPyProvider(key="d", lm=EchoLM(), children=[
            Guarded(key="g"), Asker(key="free", name="free"),
        ])]

    _mount(Root)
    assert isinstance(calls["mine"].error, Blocked)
    assert str(calls["mine"].error) == "llm 'mine' blocked by lock: mine is guarded"
    assert isinstance(calls["child"].error, Blocked)  # subtree
    assert calls["free"].error is None and calls["free"].result.answer == "ok"  # sibling


def test_unmount_unregisters_and_the_pipeline_is_empty_again():
    calls.clear()
    ui = {}

    @component
    def Guarded():
        use_guard(lambda b: False, name="never")
        return []

    @component
    def Root():
        show, set_show = hooks.use_state(True)
        ui["hide"] = lambda: set_show(False)
        return [Guarded(key="g")] if show else []

    async def main():
        root = HookContext("Root", Root)
        schedule_rerender(root)
        await runtime.run_until_idle()
        mounted = len(scoped_middleware()), len(resolve_harness().middleware)
        ui["hide"]()
        await runtime.run_until_idle()
        return mounted

    assert asyncio.run(main()) == (1, 1)
    assert scoped_middleware() == [] and resolve_harness().middleware == []


def test_predicates_see_latest_props_and_stateful_guards_are_kept():
    calls.clear()
    ui, seen = {}, {}

    @component
    def Guarded(limit: int):
        use_guard(lambda b: len(b.inputs["question"]) <= limit or f"over {limit}",
                  kinds="llm", name="len")
        seen.setdefault("budgets", set()).add(id(use_guard(budget(max_tokens_per_turn=99))))
        return [Asker(key="a", name="a", question="x" * 5)]

    @component
    def Root():
        limit, set_limit = hooks.use_state(10)
        ui["set"] = set_limit
        return [DSPyProvider(key="d", lm=EchoLM(), children=[Guarded(key="g", limit=limit)])]

    async def main():
        root = HookContext("Root", Root)
        schedule_rerender(root)
        await runtime.run_until_idle()
        first = calls["a"].error
        ui["set"](3)
        await runtime.run_until_idle()
        calls["a"].run(question="x" * 5)
        await runtime.run_until_idle()
        return first, calls["a"].error

    first, second = asyncio.run(main())
    assert first is None
    assert isinstance(second, Blocked) and "over 3" in str(second)
    assert len(seen["budgets"]) == 1  # same budget instance across renders


def test_order_providers_then_ancestor_hooks_then_descendant_hooks():
    calls.clear()
    log = []

    def tag(name):
        async def mw(b, call_next):
            log.append(name)
            return await call_next(b)

        mw.kinds = ("llm",)
        mw.harness_name = name
        return mw

    @component
    def Inner():
        use_middleware(tag("inner-hook"))
        return [Asker(key="a", name="a")]

    @component
    def Outer():
        use_middleware(tag("outer-hook-1"))
        use_middleware(tag("outer-hook-2"))
        return [HarnessProvider(key="h", middleware=[tag("provider")], children=[
            Inner(key="i")])]

    @component
    def Root():
        return [DSPyProvider(key="d", lm=EchoLM(), children=[Outer(key="o")])]

    _mount(Root)
    assert calls["a"].error is None
    assert log == ["provider", "outer-hook-1", "outer-hook-2", "inner-hook"]


def test_use_verify_retries_with_feedback_for_calls_below():
    calls.clear()
    lm = EchoLM()

    @component
    def Strict():
        use_verify(assert_that(lambda p: p.answer.endswith("."), "End with a period."),
                   on_fail="retry_with_feedback")
        return [Asker(key="a", name="a")]

    @component
    def Root():
        return [DSPyProvider(key="d", lm=lm, children=[Strict(key="s"),
                                                       Asker(key="b", name="b")])]

    _mount(Root)
    assert calls["a"].result.answer == "OK." and calls["b"].result.answer == "ok"
    assert lm.calls == 3  # a: 2 attempts, b: 1


def test_entries_match_on_session_and_path_prefix():
    e = _Entry(mw=None, path="Root/Shop[s]", session_id="s1", seq=1)
    assert e.covers(Boundary(kind="llm", name="x", component="Root/Shop[s]", session_id="s1"))
    assert e.covers(Boundary(kind="llm", name="x", component="Root/Shop[s]/A", session_id="s1"))
    assert not e.covers(Boundary(kind="llm", name="x", component="Root/Shop[s]2",
                                 session_id="s1"))
    assert not e.covers(Boundary(kind="llm", name="x", component="Root/Shop[s]",
                                 session_id="s2"))
    assert not e.covers(Boundary(kind="llm", name="x", component=None, session_id="s1"))
    root, other = object(), object()
    e = _Entry(mw=None, path="Root", session_id=None, seq=1, root=lambda: root)
    b = Boundary(kind="llm", name="x", component="Root/A")
    assert e.covers(b, root) and not e.covers(b, other)  # same path, another tree
