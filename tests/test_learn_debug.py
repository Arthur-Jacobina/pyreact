"""explain / inspect_program / terminal dev commands (incl. a live app loop). No network."""

import asyncio
import re

import dspy
import pytest

from integrations import DSPyProvider, telemetry, use_agent_call, use_call, use_predict
from integrations.claude_cli_lm import _wrap
from integrations.learn import (
    configure_learning,
    explain,
    inspect_program,
    list_modules,
    run_optimization,
    set_debug,
)
from integrations.learn.devtools import _CONFIG, LearnConfig, _parse_feedback
from pyreact.boot import bootstrap, handle_command, register_command
from pyreact.boot.terminal import help_text, unregister_command
from pyreact.core import EventBus, HookContext, component, hooks, runtime, use_turn
from pyreact.core.runtime import schedule_rerender


class RuleLM(dspy.BaseLM):
    """label = pos iff 'good' in text, or 'great' once the instructions mention it."""

    def __init__(self, reflect=False):
        super().__init__(model="fake/rule", cache=False)
        self.reflect = reflect

    def forward(self, prompt=None, messages=None, **kw):
        if self.reflect:
            return _wrap("```\nClassify sentiment. Words like 'great' are pos.\n```", self.model)
        system, user = messages[0]["content"], messages[-1]["content"]
        found = re.findall(r"\[\[ ## text ## \]\]\n(.*)", user)
        text = found[-1] if found else ""
        pos = "good" in text or ("great" in text and "great" in system)
        return _wrap(f"[[ ## label ## ]]\n{'pos' if pos else 'neg'}\n\n[[ ## completed ## ]]",
                     self.model)

    async def aforward(self, prompt=None, messages=None, **kw):
        return self.forward(prompt, messages, **kw)


def _run_headless(Root, drive):
    async def main():
        schedule_rerender(HookContext("Root", Root))
        await runtime.run_until_idle()
        await drive()
        await runtime.run_until_idle()

    asyncio.run(main())


# ---------------------------------------------------------------------------
# explain / inspect
# ---------------------------------------------------------------------------
def test_explain_shows_exact_prompt_and_completion(monkeypatch):
    hs = {}
    events = []
    HookContext.get_service("event_bus", EventBus).subscribe(events.append, ["llm.end"])

    @component
    def Clf():
        hs["call"] = use_call(use_predict("text -> label"))
        return []

    @component
    def Root():
        return [DSPyProvider(key="d", lm=RuleLM(), children=[Clf(key="clf")])]

    async def drive():
        hs["call"].run(text="good day")
        await runtime.run_until_idle()
        set_debug(True)
        hs["call"].run(text="bad day")
        await runtime.run_until_idle()
        set_debug(False)

    _run_headless(Root, drive)
    first, second = events
    assert "lm_calls" not in first.payload  # small payloads by default
    assert second.payload["lm_calls"][0]["completion"].startswith("[[ ## label ## ]]")

    e = explain(first.payload["call_id"])
    assert e["arch"] == "predict" and e["outputs"] == {"label": "pos"}
    text = str(e)
    assert ">>> SYSTEM" in text and ">>> USER" in text and "good day" in text
    assert "<<< COMPLETION" in text and "pos" in text
    assert explain(last=True)["call_id"] == second.payload["call_id"]

    # from a trace in another process: buffer empty, prompts on the llm.end event
    monkeypatch.setattr(telemetry, "_calls", type(telemetry._calls)())
    with pytest.raises(LookupError, match="set_debug"):
        explain(first.payload["call_id"])
    e2 = explain(second.payload["call_id"], events=events)
    assert "bad day" in str(e2)


def test_inspect_program_shows_arch_instructions_demos():
    cot = dspy.ChainOfThought("question -> answer")
    cot.predict.signature = cot.predict.signature.with_instructions("Be terse.")
    cot.predict.demos = [dspy.Example(question="1+1?", answer="2")]
    info = inspect_program(cot)
    assert info["arch"] == "cot" and info["predictors"]["predict"]["instructions"] == "Be terse."
    s = str(info)
    assert "Be terse." in s and "demos: 1" in s and "question='1+1?'" in s


# ---------------------------------------------------------------------------
# terminal extension point
# ---------------------------------------------------------------------------
def test_register_command_dispatch_help_and_errors(capsys):
    got = []
    register_command("hello", lambda app, args: got.append((app, args)), "say hi")
    register_command("boom", lambda app, args: 1 / 0, "explodes")
    try:
        assert handle_command("APP", "hello", "a b") is True and got == [("APP", "a b")]
        assert ":hello" in help_text() and "say hi" in help_text()
        assert handle_command("APP", "boom", "") is True  # errors are printed, not raised
        assert "ZeroDivisionError" in capsys.readouterr().out
        assert handle_command("APP", "nope", "") is False
        # learn commands are registered on import of integrations.learn
        assert all(f":{c}" in help_text() for c in
                   ("modules", "inspect", "explain", "feedback", "optimize"))
    finally:
        unregister_command("hello")
        unregister_command("boom")


def test_parse_feedback():
    assert _parse_feedback("+") == (1.0, None, "", {})
    assert _parse_feedback('- @call wrong one label=pos "name=a b"') == (
        0.0, "call", "wrong one", {"label": "pos", "name": "a b"})
    assert _parse_feedback("0.5 meh")[0] == 0.5
    with pytest.raises(ValueError):
        _parse_feedback("maybe")


# ---------------------------------------------------------------------------
# live app: chat -> :feedback -> :optimize -> hot swap
# ---------------------------------------------------------------------------
@pytest.fixture
def learn_cfg():
    saved = dict(_CONFIG)
    yield
    _CONFIG.clear()
    _CONFIG.update(saved)
    _CONFIG.setdefault(None, LearnConfig())


def test_live_feedback_optimize_hot_swap(capsys, learn_cfg, tmp_path):
    answers = []

    @component
    def Clf():
        turn = use_turn()
        call = use_agent_call("text -> label", arch="predict")
        hooks.use_effect(lambda: turn and call.run(text=turn.text), [turn and turn.id])
        hooks.use_effect(lambda: call.ver and answers.append(call.result.label), [call.ver])
        return []

    @component
    def App():
        return [DSPyProvider(key="dspy", lm=RuleLM(), children=[Clf(key="clf")])]

    app = bootstrap(App)
    try:
        def say(text):
            app.invoke(text, wait=True, timeout=10)
            return answers[-1]

        assert say("great food") == "neg"  # wrong: 'great' is positive
        handle_command(app, "feedback", "- 'great' is positive label=pos")
        assert say("good day") == "pos"
        handle_command(app, "feedback", "+")
        assert say("bad day") == "neg"
        handle_command(app, "feedback", "+")
        out = capsys.readouterr().out
        assert "feedback 0 on turn" in out and "correction {'label': 'pos'}" in out

        # llm.* events land in the session log (what :events / datasets read)
        assert len(app.events.recent(None, ["llm.end"])) == 3
        handle_command(app, "modules", "")
        handle_command(app, "explain", "")
        out = capsys.readouterr().out
        assert "predict" in out and "Predict:text -> label:" in out
        assert ">>> USER" in out and "bad day" in out

        val = [dspy.Example(text="great stuff", label="pos").with_inputs("text")]
        configure_learning(optimizer="gepa", budget=10, min_examples=3, valset=val,
                           reflection_lm=RuleLM(reflect=True), save_dir=str(tmp_path))
        t = run_optimization(app, "clf", block=True)
        assert t is not None
        app.wait_idle(timeout=10)
        out = capsys.readouterr().out
        assert "hot-swapped" in out and "+ Classify sentiment" in out
        kinds = [e.kind for e in app.events.recent(None, ["optimize.*"])]
        assert kinds[0] == "optimize.start" and kinds[-1] == "optimize.end"
        assert "optimize.progress" in kinds
        end = app.events.recent(1, ["optimize.end"])[0].payload
        assert end["swapped"] and end["path"].startswith(str(tmp_path))
        (row,) = list_modules(app)
        assert row["compiled"] and row["swaps"] == 1

        assert say("great job") == "pos"  # the next call uses the optimized prompt
        handle_command(app, "inspect", "clf")
        assert "Words like 'great' are pos." in capsys.readouterr().out
    finally:
        app.shutdown()
