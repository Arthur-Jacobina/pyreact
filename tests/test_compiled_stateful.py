"""Learned programs dehydrate themselves (use_dehydrate): roots carry them.

- ``DSPyProvider``: modules an optimizer compiled / a hot swap by key put in
  ``env.compiled`` (slice ``"compiled"``);
- ``LearnedProgram`` / ``LearnedPrograms``: the provider's ``{name: src}``
  state, a swap included (slice ``"programs"``).
"""

import copy
import warnings

import dspy
import pytest
from dspy.utils import DummyLM

from integrations import DSPyProvider, use_agent, use_dspy_env, use_module
from integrations.learn import (
    LearnedProgram,
    OptimizationResult,
    find_registries,
    use_learned,
    use_learned_registry,
)
from integrations.learn.hmr import hot_swap_into, resolve_module
from integrations.stateful import CompiledModulesParticipant
from pyreact.boot import AppRunner, DirectoryManifestStore, create_root, hydrate_root
from pyreact.core import HydrationWarning, component, hooks, use_imperative_handle

DSPY = "App/DSPyProvider[dspy]"


class QA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@component
def Agent():
    qa = use_module(QA, name="qa")
    use_agent(QA, tools=[add], max_iters=2, name="coder")
    react = use_dspy_env().caches[("ReAct", "QA", "coder")]
    seen = hooks.get_service("seen", list)
    seen.append((qa.signature.instructions, react.react.signature.instructions))
    return []


@component
def App():
    return [DSPyProvider(key="dspy", lm=DummyLM([]), children=[Agent(key="agent")])]


@pytest.fixture
def roots():
    made = []
    yield made
    for r in made:
        try:
            r.unmount()
        except Exception:
            pass


def _learned(program, text):
    p = copy.deepcopy(program)
    for _name, pred in p.named_predictors():
        pred.signature = pred.signature.with_instructions(text)
    return p


def _swap(sess, key, text):
    _env, k, mod = resolve_module(sess, key)
    hot_swap_into(sess, k, _learned(mod, text))
    sess.wait_idle(timeout=5)


def _seen(root):
    return root.session.services["seen"]


def test_compiled_modules_come_back_before_the_first_render(roots):
    root = create_root(App, session_id="a")
    _swap(root, "qa", "LEARNED QA")  # a Root is resolved through its session
    _swap(root, "coder", "LEARNED AGENT")
    assert _seen(root)[-1] == ("LEARNED QA", "LEARNED AGENT")
    snap = root.suspend()
    mods = {e["key_str"]: e for e in snap["slices"][DSPY]["compiled"]}
    assert set(mods) == {"Predict:QA:qa", "ReAct:QA:coder"}
    assert all("state" in e for e in mods.values())  # no result file: inlined

    r = hydrate_root(App, session_id="a", snapshot=snap)  # a new runner
    roots.append(r)
    assert _seen(r)[0] == ("LEARNED QA", "LEARNED AGENT")  # first render
    assert r.mismatches == []
    # still compiled: a second suspend keeps them
    assert len(r.suspend()["slices"][DSPY]["compiled"]) == 2


def test_result_path_is_reused_and_missing_file_is_a_mismatch(roots, tmp_path):
    root = create_root(App, session_id="p")
    _env, key, mod = resolve_module(root, "qa")
    res = OptimizationResult(program=_learned(mod, "FROM FILE"), optimizer="gepa",
                             module_key=key)
    res.save(str(tmp_path / "qa.json"))
    res.hot_swap(root, key)
    root.wait_idle(timeout=5)
    snap = root.suspend()
    [e] = snap["slices"][DSPY]["compiled"]
    assert e["path"] == str(tmp_path / "qa.json") and "state" not in e

    r = hydrate_root(App, session_id="p", snapshot=snap)
    assert _seen(r)[0][0] == "FROM FILE"
    r.unmount()

    (tmp_path / "qa.json").unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HydrationWarning)
        r2 = hydrate_root(App, session_id="p", snapshot=snap)
    roots.append(r2)
    assert [(m["path"], m["key"], m["reason"]) for m in r2.mismatches] == [
        (DSPY, "compiled", "result missing")]
    assert [(d["participant"], d["reason"]) for d in r2.session.degraded] == [
        ("dspy.compiled", "result missing")]
    assert _seen(r2)[0][0] == "Answer the question."


# ---------------------------------------------------------------------------
# LearnedProgram: the provider's state is the learned program reference
# ---------------------------------------------------------------------------
def _result(tmp_path, name, text):
    p = dspy.Predict(QA)
    p.signature = p.signature.with_instructions(text)
    path = str(tmp_path / f"{name}.json")
    OptimizationResult(program=p, optimizer="gepa", arch="predict").save(path)
    return path


def learned_app(src):
    @component
    def Leaf():
        qa = use_module(QA, name="qa", program=use_learned("qa"))
        use_imperative_handle("learned", lambda: use_learned_registry(), [])
        hooks.get_service("seen", list).append(qa.signature.instructions)
        return []

    @component
    def App():
        return [DSPyProvider(key="dspy", lm=DummyLM([]), children=[
            LearnedProgram(key="qa", src=src, children=[Leaf(key="leaf")])])]

    return App


def test_learned_program_swap_survives_suspend(roots, tmp_path):
    one, two = _result(tmp_path, "one", "ONE"), _result(tmp_path, "two", "TWO")
    root = create_root(learned_app(one), session_id="lp")
    assert _seen(root)[-1] == "ONE"
    [(path, reg)] = find_registries(root)  # a Root, resolved through its session
    assert path == "App/LearnedProgram[qa]"
    reg.set("qa", two)  # a hot swap: the provider's state changes
    root.wait_idle(timeout=5)
    assert _seen(root)[-1] == "TWO"
    res = OptimizationResult(program=_learned(dspy.Predict(QA), "THREE"), optimizer="gepa")
    assert res.hot_swap(root, "qa") == ["App/Leaf[leaf]"]
    root.wait_idle(timeout=5)
    assert _seen(root)[-1] == "THREE"
    reg.set("qa", two)
    root.wait_idle(timeout=5)
    snap = root.suspend()
    assert snap["slices"]["App/LearnedProgram[qa]"] == {"programs": {"qa": two}}
    # the DSPy env does not carry it twice (the LearnedProgram owns it)
    assert snap["slices"][DSPY] == {"compiled": []}

    r = hydrate_root(learned_app(one), session_id="lp", snapshot=snap)
    roots.append(r)
    assert _seen(r)[0] == "TWO"  # the swapped program, on the first render
    assert r.mismatches == []


def test_learned_program_with_a_lost_file_falls_back_to_the_prop(roots, tmp_path):
    one, two = _result(tmp_path, "one", "ONE"), _result(tmp_path, "two", "TWO")
    root = create_root(learned_app(one), session_id="lost")
    root.session.services["learn.programs"]["App/LearnedProgram[qa]"].set("qa", two)
    root.wait_idle(timeout=5)
    snap = root.suspend()
    (tmp_path / "two.json").unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HydrationWarning)
        r = hydrate_root(learned_app(one), session_id="lost", snapshot=snap)
    roots.append(r)
    assert _seen(r)[0] == "ONE"
    [m] = r.mismatches
    assert (m["path"], m["key"], m["reason"]) == (
        "App/LearnedProgram[qa]", "programs", "result missing")


# ---------------------------------------------------------------------------
# Deprecated: CompiledModulesParticipant keeps working for one release
# ---------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_deprecated_participant_still_round_trips(tmp_path):
    with pytest.warns(DeprecationWarning, match="CompiledModulesParticipant is deprecated"):
        CompiledModulesParticipant()

    def make():
        return AppRunner(App, default_session=False,
                         participants=[lambda s: CompiledModulesParticipant(s)],
                         manifest_store=DirectoryManifestStore(tmp_path / "m"))

    app = make()
    try:
        s = app.open_session("old")
        _swap(s, "qa", "LEARNED QA")
        m = s.suspend()
        assert m["participants"]["dspy.compiled"] == {}
        assert [e["key_str"] for e in m["slices"][DSPY]["compiled"]] == ["Predict:QA:qa"]
    finally:
        app.shutdown()
    app2 = make()
    try:
        r = app2.resume("old")
        assert r.services["seen"][0][0] == "LEARNED QA" and r.degraded == []
    finally:
        app2.shutdown()
