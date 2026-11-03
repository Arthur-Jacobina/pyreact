"""SandboxInterpreter (dspy CodeInterpreter / RLM) and sandbox_tools (ReAct),
with the LocalDriver (no Docker) and scripted LMs (no network)."""

import time

import dspy
import pytest
from dspy.primitives.code_interpreter import CodeInterpreter, CodeInterpreterError, FinalOutput

from integrations import DSPyProvider, use_agent
from integrations.claude_cli_lm import _wrap
from integrations.rlm import use_rlm
from pyreact.boot import bootstrap
from pyreact.core import component, hooks, use_turn
from pyreact_sandbox import LocalDriver, SandboxSpec, sandbox_tools
from pyreact_sandbox.interpreter import SandboxInterpreter
from pyreact_sandbox.react import SandboxProvider, use_sandbox, wait_closed


@pytest.fixture
def local(tmp_path):
    return LocalDriver(root=tmp_path / "sbx")


def test_interpreter_protocol_state_tools_submit_and_errors(local):
    sb = local.create_sync(SandboxSpec())
    interp = SandboxInterpreter(sb, tools={"double": lambda n: n * 2},
                                output_fields=[{"name": "answer", "type": "str"}], timeout=5)
    assert isinstance(interp, CodeInterpreter)
    try:
        assert interp.execute("x = 20") is None
        assert interp.execute("print(x + 1)") == "21\n"          # namespace persists
        assert interp.execute("x * 2") == "40"                   # last expression
        assert interp.execute("print(double(n))", {"n": 21}) == "42\n"  # host tool + variable
        assert interp.execute("open('f.txt', 'w').write('hi'); import os; "
                              "print(os.listdir('.'))") == "['f.txt']\n"
        out = interp.execute("SUBMIT(f'answer is {x}')")
        assert out == FinalOutput({"answer": "answer is 20"})
        with pytest.raises(CodeInterpreterError, match="NameError"):
            interp.execute("print(nope)")
        with pytest.raises(SyntaxError):
            interp.execute("def (")
        with pytest.raises(CodeInterpreterError, match="boom"):
            interp.execute("double2 = None\nraise ValueError('boom')")
        assert interp.execute("import subprocess; subprocess.run(['echo', 'stray'])\n"
                              "print('clean')") == "clean\n"  # fd-level output can't corrupt
    finally:
        interp.shutdown()
    # files persist across interpreters (RLM makes one per call); the namespace does not
    with SandboxInterpreter(sb, timeout=5) as again:
        assert again.execute("print(open('f.txt').read())") == "hi\n"
        with pytest.raises(CodeInterpreterError):
            again.execute("x")
    t = SandboxInterpreter(sb, timeout=0.5)
    t0 = time.perf_counter()
    with pytest.raises(CodeInterpreterError, match="timed out"):
        t.execute("import time; time.sleep(10)")
    assert time.perf_counter() - t0 < 5
    sb.destroy_sync()


class ScriptLM(dspy.BaseLM):
    """Answers each call with the next scripted {field: value} dict."""

    def __init__(self, steps):
        super().__init__(model="script/lm", cache=False)
        self.steps, self.n = list(steps), 0

    def forward(self, prompt=None, messages=None, **kw):
        fields = self.steps[min(self.n, len(self.steps) - 1)]
        self.n += 1
        text = "\n\n".join(f"[[ ## {k} ## ]]\n{v}" for k, v in fields.items())
        return _wrap(text + "\n\n[[ ## completed ## ]]", self.model)

    async def aforward(self, prompt=None, messages=None, **kw):
        return self.forward(prompt=prompt, messages=messages, **kw)


def _app(agent, driver):
    @component
    def App():
        return [SandboxProvider(key="sbx", driver=driver, spec=SandboxSpec(),
                                children=[agent(key="agent")])]

    return bootstrap(App, default_session=False)


def test_rlm_runs_its_repl_in_the_sandbox(local):
    lm = ScriptLM([
        {"reasoning": "count", "code": "n = len(context.split())\nprint(n)"},
        {"reasoning": "save + submit",
         "code": "open('count.txt', 'w').write(str(n))\nSUBMIT(answer=str(n))"},
    ])

    @component
    def Agent():
        sb = use_sandbox()
        run, (result, ver), _loading, error = use_rlm(
            "context, question -> answer", lm=lm, max_iterations=4,
            interpreter=lambda: SandboxInterpreter(sb, timeout=20))
        turn = use_turn()
        bus = hooks.get_service("event_bus", lambda: None)
        hooks.use_effect(lambda: turn and run(context=turn.text, question="how many words?"),
                         [turn and turn.id])

        def report():
            if ver:
                bus.publish("message", {"text": result.answer if result else repr(error)})

        hooks.use_effect(report, [ver])
        return []

    app = _app(Agent, local)
    try:
        s = app.open_session("s")
        events = []
        s.events.subscribe(events.append, ["message", "sandbox.exec"], replay=True)
        s.invoke("one two three four", wait=True, timeout=30)
        s.wait_idle(timeout=30)
        (box,) = local.list_sync()
        written = (local.root / box / "fs" / "count.txt").read_text()  # before close destroys it
    finally:
        app.shutdown()
    msgs = [e.payload["text"] for e in events if e.kind == "message"]
    assert msgs == ["4"]
    repl = [e for e in events if e.kind == "sandbox.exec"]
    assert [e.payload["op"] for e in repl] == ["repl", "repl"]
    assert all(e.payload["name"] == "python" and e.payload["error"] is None for e in repl)
    assert written == "4"  # really written inside the sandbox
    assert wait_closed(10) and local.list_sync() == []


def test_sandbox_tools_drive_a_react_agent(local):
    lm = ScriptLM([
        {"next_thought": "write a script", "next_tool_name": "write_file",
         "next_tool_args": '{"path": "calc.py", "content": "print(sum(range(1, 11)))"}'},
        {"next_thought": "run it", "next_tool_name": "run_shell",
         "next_tool_args": '{"command": "python3 calc.py"}'},
        {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": "{}"},
        {"reasoning": "the script printed 55", "answer": "55"},
    ])

    @component
    def Agent():
        sb = use_sandbox()
        tools = hooks.use_memo(lambda: sandbox_tools(sb, timeout=10), [id(sb)])
        call = use_agent("question -> answer", tools=tools, max_iters=5)
        turn = use_turn()
        bus = hooks.get_service("event_bus", lambda: None)
        hooks.use_effect(lambda: turn and call.run(question=turn.text), [turn and turn.id])

        def report():
            if call.ver:
                bus.publish("message", {"text": call.result.answer if call.result
                                        else repr(call.error)})

        hooks.use_effect(report, [call.ver])
        return []

    @component
    def WithLM(key=None):
        return [DSPyProvider(key="dspy", lm=lm, children=[Agent(key="a")])]

    app = _app(WithLM, local)
    try:
        s = app.open_session("s")
        events = []
        s.events.subscribe(events.append, ["message", "sandbox.exec", "tool.*"], replay=True)
        s.invoke("sum 1..10 with a script", wait=True, timeout=30)
        s.wait_idle(timeout=30)
    finally:
        app.shutdown()
        wait_closed(10)
    assert [e.payload["text"] for e in events if e.kind == "message"] == ["55"]
    ex = [e.payload for e in events if e.kind == "sandbox.exec"]
    assert [(p["op"], p["name"]) for p in ex] == [("write_file", "write_file"),
                                                  ("exec", "python3")]
    assert ex[1]["stdout"] == "55\n"
    tool_ends = [e.payload for e in events if e.kind == "tool.end"]
    assert [p["tool"] for p in tool_ends] == ["write_file", "run_shell"]


def test_sandbox_tools_options():
    from pyreact_sandbox.react import SandboxHandle
    from pyreact_sandbox.testing import FakeDriver

    h = SandboxHandle(key="k", store_key="k", driver=FakeDriver(), spec=SandboxSpec(),
                      store=None, on_close="destroy", autosnapshot=None, bus=None)
    names = [t.name for t in sandbox_tools(h, prefix="sb_", include=["run_shell", "read_file"])]
    assert names == ["sb_run_shell", "sb_read_file"]
    t = sandbox_tools(h, include=["run_shell"])[0]
    assert set(t.parameters["properties"]) == {"command"} and "/workspace" in t.description
    with pytest.raises(ValueError):
        sandbox_tools(h, include=["rm_rf"])
