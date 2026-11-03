"""Learned programs are props: LearnedProgram + use_learned + program= (ADR 0001)."""

import asyncio
import copy
import os
import time

import dspy
import pytest
from dspy.utils import DummyLM

from integrations import DSPyProvider, use_agent, use_call, use_dspy_env, use_module
from integrations.learn import LearnedProgram, OptimizationResult, SavedProgram, use_learned
from integrations.optimize import ModuleVersions
from pyreact.core import component, hooks, runtime
from pyreact.core.hook import HookContext
from pyreact.core.runtime import schedule_rerender


class QA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def learned(text, sig=QA):
    p = dspy.Predict(sig)
    p.signature = p.signature.with_instructions(text)
    return p


def learned_like(program, text):
    p = copy.deepcopy(program)
    for _name, pred in p.named_predictors():
        pred.signature = pred.signature.with_instructions(text)
    return p


def run(Root, *steps):
    """Render ``Root`` headless; each step runs between idle points."""

    async def main():
        schedule_rerender(HookContext("Root", Root))
        await runtime.run_until_idle()
        for step in steps:
            step()
            await runtime.run_until_idle()

    asyncio.run(main())


def make_tree(Leaf, initial=None):
    """Root holds ``src`` as state (a prop of LearnedProgram); returns (Root, set_src)."""
    box = {}

    @component
    def Root():
        src, set_src = hooks.use_state(initial)
        box["set"] = set_src
        return [DSPyProvider(key="dspy", lm=DummyLM([]), children=[
            LearnedProgram(key="qa", src=src, children=[Leaf(key="leaf")])])]

    # (a lambda: use_state setters call callables, and a dspy program is one)
    return Root, lambda v: box["set"](lambda _old: v)


def test_changing_src_rerenders_users_with_the_new_program():
    seen = []

    @component
    def Leaf():
        qa = use_module(QA, name="qa", program=use_learned("qa"))
        env = use_dspy_env()
        seen.append((qa.signature.instructions, qa is env.caches[("Predict", "QA", "qa")],
                     ("Predict", "QA", "qa") in env.compiled))
        return []

    Root, set_src = make_tree(Leaf)
    p1, p2 = learned("V1"), learned("V2")
    run(Root, lambda: set_src(p1), lambda: set_src(p2), lambda: set_src(None))
    insts = [s[0] for s in seen]
    assert insts[0] == "Answer the question."
    assert "V1" in insts and "V2" in insts
    assert insts.index("V1") < insts.index("V2")
    assert insts[-1] == "Answer the question."  # back to None: the baseline again
    assert all(s[1] for s in seen)  # always the component's own (live) module
    assert seen[-1][2] is False and any(s[2] for s in seen if s[0] == "V2")


def test_use_learned_outside_a_provider_is_none_and_nearest_provider_wins():
    got = {}

    @component
    def Leaf():
        got[hooks.path] = (use_learned("qa"), use_learned("other"))
        return []

    outer, inner = learned("OUTER"), learned("INNER")

    @component
    def Root():
        return [Leaf(key="bare"), LearnedProgram(key="qa", src=outer, children=[
            LearnedProgram(key="other", src=None, children=[Leaf(key="a")]),
            LearnedProgram(key="qa", src=inner, children=[Leaf(key="b")]),
        ])]

    run(Root)
    assert got["Root/Leaf[bare]"] == (None, None)
    assert got["Root/Leaf[a]"] == (outer, None)  # transparent: no provider segments
    assert got["Root/Leaf[b]"] == (inner, None)


def test_learned_program_needs_a_key():
    errors = []

    @component
    def Root():
        return [LearnedProgram(src=None, children=[])]

    prev = runtime._error_handler
    runtime.set_error_handler(lambda exc, ctx, phase: errors.append(exc))
    try:
        run(Root)
    finally:
        runtime.set_error_handler(prev)
    assert errors and isinstance(errors[0], TypeError) and "key=" in str(errors[0])


def test_file_source_and_bad_file_degrades(tmp_path):
    seen = []

    @component
    def Leaf():
        qa = use_module(QA, name="qa", program=use_learned("qa"))
        seen.append(qa.signature.instructions)
        return []

    path = str(tmp_path / "qa.json")
    OptimizationResult(program=learned("FROM FILE"), optimizer="gepa", arch="predict").save(path)
    Root, set_src = make_tree(Leaf, initial=path)
    run(Root, lambda: set_src(str(tmp_path / "missing.json")))
    assert seen[0] == "FROM FILE"
    assert seen[-1] == "Answer the question."  # unreadable source -> baseline


def test_watch_reloads_a_changed_file(tmp_path):
    seen = []

    @component
    def Leaf():
        qa = use_module(QA, name="qa", program=use_learned("qa"))
        seen.append(qa.signature.instructions)
        return []

    path = str(tmp_path / "qa.json")
    OptimizationResult(program=learned("ONE"), optimizer="gepa", arch="predict").save(path)

    @component
    def Root():
        return [DSPyProvider(key="dspy", lm=DummyLM([]), children=[
            LearnedProgram(key="qa", src=path, watch=True, interval=0.02,
                           children=[Leaf(key="leaf")])])]

    async def main():
        schedule_rerender(HookContext("Root", Root))
        await runtime.run_until_idle()  # the watch timer does not hold idle
        assert seen[-1] == "ONE"
        OptimizationResult(program=learned("TWO"), optimizer="gepa", arch="predict").save(path)
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        for _ in range(100):
            await asyncio.sleep(0.02)
            await runtime.run_until_idle()
            if seen[-1] == "TWO":
                break
        assert seen[-1] == "TWO"

    t0 = time.time()
    asyncio.run(main())
    assert time.time() - t0 < 5


def test_use_agent_program_keeps_tools_and_counts_swaps():
    seen = []

    @component
    def Leaf():
        use_agent(QA, tools=[add], max_iters=2, name="coder", program=use_learned("qa"))
        react = use_dspy_env().caches[("ReAct", "QA", "coder")]
        seen.append((react.react.signature.instructions, sorted(react.tools)))
        return []

    Root, set_src = make_tree(Leaf)
    base = dspy.ReAct(QA, tools=[add], max_iters=2)
    before = (ModuleVersions._ctx.value or {}).get(("ReAct", "QA", "coder"), 0)
    run(Root, lambda: set_src(learned_like(base, "AGENT LESSON")))
    assert seen[-1][0] == "AGENT LESSON"
    assert "add" in seen[-1][1]  # the live tool binding stays
    assert (ModuleVersions._ctx.value or {})[("ReAct", "QA", "coder")] == before + 1


def test_use_call_program_and_saved_program_state():
    seen = []

    @component
    def Leaf():
        qa = use_module(QA, name="plain")
        use_call(qa, program=use_learned("qa"))
        seen.append(qa.signature.instructions)
        return []

    Root, set_src = make_tree(Leaf)
    saved = SavedProgram(learned("FROM STATE").dump_state(), path="x.json")
    run(Root, lambda: set_src(saved), lambda: set_src(None))
    assert "FROM STATE" in seen and seen[-1] == "Answer the question."


def test_program_wins_over_an_imperative_compiled_entry():
    seen = []

    @component
    def Leaf():
        env = use_dspy_env()
        env.compiled.setdefault(("Predict", "QA", "qa"), learned("OLD SWAP"))
        qa = use_module(QA, name="qa", program=use_learned("qa"))
        seen.append(qa.signature.instructions)
        return []

    Root, set_src = make_tree(Leaf, initial=learned("PROP"))
    run(Root)
    assert seen[-1] == "PROP"


def test_bad_src_type_is_a_type_error():
    with pytest.raises(TypeError):
        from integrations.learn.programs import _resolve

        _resolve("x", 42, {}, False)
