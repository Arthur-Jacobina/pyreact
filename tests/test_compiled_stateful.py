"""CompiledModulesParticipant: learned programs survive suspend / resume."""

import copy

import dspy
import pytest
from dspy.utils import DummyLM

from integrations import DSPyProvider, use_agent, use_dspy_env, use_module
from integrations.learn import OptimizationResult
from integrations.learn.hmr import hot_swap_into, resolve_module
from integrations.stateful import CompiledModulesParticipant
from pyreact.boot import AppRunner, DirectoryManifestStore
from pyreact.core import component, hooks


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
def make(tmp_path):
    made = []

    def _make(**kw):
        app = AppRunner(App, default_session=False,
                        participants=[lambda s: CompiledModulesParticipant(s, **kw)],
                        manifest_store=DirectoryManifestStore(tmp_path / "m"))
        made.append(app)
        return app

    yield _make
    for app in made:
        app.shutdown()


def _learned(program, text):
    p = copy.deepcopy(program)
    for _name, pred in p.named_predictors():
        pred.signature = pred.signature.with_instructions(text)
    return p


def _swap(sess, key, text):
    _env, k, mod = resolve_module(sess, key)
    hot_swap_into(sess, k, _learned(mod, text))
    sess.wait_idle(timeout=5)


@pytest.mark.parametrize("where", ["inline", "dir"])
def test_learned_programs_come_back_before_the_first_render(make, tmp_path, where):
    app = make(dir=tmp_path / "learned") if where == "dir" else make()
    s = app.open_session("a")
    _swap(s, "qa", "LEARNED QA")
    _swap(s, "coder", "LEARNED AGENT")
    assert s.services["seen"][-1] == ("LEARNED QA", "LEARNED AGENT")
    m = s.suspend()
    mods = {e["key_str"]: e for e in m["participants"]["dspy.compiled"]["modules"]}
    assert set(mods) == {"Predict:QA:qa", "ReAct:QA:coder"}
    for e in mods.values():
        assert ("path" in e) == (where == "dir") and ("state" in e) == (where == "inline")
    app.shutdown()

    app2 = make()
    r = app2.resume("a")
    assert r.services["seen"][0] == ("LEARNED QA", "LEARNED AGENT")  # first render
    # still compiled: a second suspend keeps them
    m2 = r.suspend()
    assert len(m2["participants"]["dspy.compiled"]["modules"]) == 2


def test_result_path_is_reused_and_missing_file_degrades(make, tmp_path):
    app = make()
    s = app.open_session("p")
    _env, key, mod = resolve_module(s, "qa")
    res = OptimizationResult(program=_learned(mod, "FROM FILE"), optimizer="gepa",
                             module_key=key)
    res.save(str(tmp_path / "qa.json"))
    res.hot_swap(s, key)
    s.wait_idle(timeout=5)
    m = s.suspend()
    [e] = m["participants"]["dspy.compiled"]["modules"]
    assert e["path"] == str(tmp_path / "qa.json")

    r = app.resume("p", m, consume=False)
    assert r.services["seen"][0][0] == "FROM FILE"
    r.suspend()

    (tmp_path / "qa.json").unlink()
    r2 = app.resume("p", m)
    assert [(d["participant"], d["reason"]) for d in r2.degraded] == [
        ("dspy.compiled", "result missing")]
    assert r2.services["seen"][0][0] == "Answer the question."
