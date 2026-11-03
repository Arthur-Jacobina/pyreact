"""Resuming turns that started but never ended (TurnStore, DurableTurnStore)."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from pyreact.boot import bootstrap
from pyreact.core import Turn, component, hooks, use_turn

ROOT = Path(__file__).resolve().parents[1]


class MemoryStore:
    def __init__(self, pending=()):
        self.started, self.ended = [], []
        self._pending = list(pending)

    def on_turn_start(self, turn):
        self.started.append(turn)

    def on_turn_end(self, turn, result):
        self.ended.append((turn.id, result["status"]))
        self._pending = [t for t in self._pending if t.id != turn.id]

    def pending(self):
        return list(self._pending)

    def get(self, turn_id):
        return next((t for t in self.started if t.id == turn_id), None)


def test_resume_pending_reruns_with_the_same_id_and_text():
    seen = []

    @component
    def Echo():
        turn = use_turn()
        hooks.use_effect(lambda: seen.append((turn.id, turn.text, turn.resumed)) if turn else None,
                         [turn and turn.id])
        return []

    store = MemoryStore([Turn(id="t7", index=7, text="unfinished", source="web")])
    app = bootstrap(Echo)
    starts = []
    app.event_bus.subscribe(starts.append, ["turn.start"])
    app.default_session.turns.configure(store=store)
    try:
        assert [t.id for t in app.default_session.pending_turns()] == ["t7"]
        assert app.resume_pending(wait=True, timeout=5) == {"default": ["t7"]}
        assert seen == [("t7", "unfinished", True)]
        assert starts[-1].payload == {"index": 7, "text": "unfinished", "source": "web",
                                      "resumed": True}
        assert store.ended == [("t7", "done")] and store.pending() == []
        # numbering continues after the resumed turn
        assert app.invoke("next", wait=True, timeout=5) == "t8"
        # explicit resume of an ended turn the store knows; unknown -> KeyError
        assert app.resume_turn("t8", wait=True, timeout=5) == "t8"
        with pytest.raises(KeyError):
            app.resume_turn("t99")
        assert app.resume_pending() == {}
    finally:
        app.shutdown()


CHILD = textwrap.dedent('''
    import asyncio, os, sys
    from types import SimpleNamespace
    import dspy
    from integrations import DSPyProvider, invoke_module
    from integrations.dspy_integration import use_dspy_env
    from pyreact.boot import bootstrap
    from pyreact.core import component, hooks, use_turn
    from pyreact_durable import Durable, use_durable_lm, use_persistent_state

    LOG, COUNT, CRASH = sys.argv[1], sys.argv[2], sys.argv[3] == "crash"

    class LM(dspy.BaseLM):
        def __init__(self):
            super().__init__(model="fake/det", cache=False)

        def forward(self, prompt=None, messages=None, **kw):
            with open(COUNT, "a") as f:
                f.write("x")
            body = messages[-1]["content"].split("## question ## ]]")[-1]
            q = [ln for ln in body.splitlines() if ln.strip()][0].strip()
            text = f"[[ ## answer ## ]]\\nA({q})\\n\\n[[ ## completed ## ]]"
            msg = SimpleNamespace(content=text, tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                                   usage={}, model=self.model)

    @component
    def Agent():
        turn = use_turn()
        env = use_dspy_env()
        answer, set_answer = use_persistent_state(None, key="answer")
        mod = hooks.use_memo(lambda: dspy.Predict("question -> answer"), [])

        async def work():
            lm = env.lm_for()
            r1 = await invoke_module(mod, {"question": turn.text}, lm=lm)
            print("STEP1", r1[0].answer, flush=True)
            if CRASH:
                await asyncio.sleep(60)  # killed here
            r2 = await invoke_module(mod, {"question": r1[0].answer}, lm=lm)
            set_answer(r2[0].answer)
            print("FINAL", r2[0].answer, flush=True)

        hooks.use_effect(lambda: work() if turn else None, [turn and turn.id])
        return []

    @component
    def LMLayer(children=None):
        return [DSPyProvider(key="d", lm=use_durable_lm(LM()), children=children)]

    @component
    def App():
        return [Durable(key="durable", log=LOG, children=[
            LMLayer(key="lm", children=[Agent(key="agent")])])]

    app = bootstrap(App)
    if CRASH:
        app.invoke("hello")
        import time
        time.sleep(60)
    else:
        print("RESUMED", app.resume_pending(wait=True, timeout=20), flush=True)
        print("NEXT", app.invoke("again", wait=True, timeout=20), flush=True)
        app.shutdown()
''')


@pytest.mark.skipif(os.name != "posix", reason="kill -9")
def test_kill_9_mid_turn_then_resume_replays_the_journaled_llm_call(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    log, count = tmp_path / "s.jsonl", tmp_path / "calls.txt"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    args = [sys.executable, str(script), str(log), str(count)]

    proc = subprocess.Popen([*args, "crash"], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        line = ""
        deadline = time.time() + 60
        while "STEP1" not in line and time.time() < deadline:
            line = proc.stdout.readline()
            assert line or proc.poll() is None, "child exited early"
        assert line.startswith("STEP1 A(hello)")
        os.kill(proc.pid, signal.SIGKILL)  # kill -9 mid-turn
        proc.wait(10)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == -signal.SIGKILL
    assert count.read_text() == "x"  # one model call before the crash

    out = subprocess.run([*args, "resume"], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = out.stdout.splitlines()
    assert "STEP1 A(hello)" in lines and "FINAL A(A(hello))" in lines
    assert "RESUMED {'default': ['t1']}" in lines
    assert "NEXT t2" in lines  # ids continue after the journaled turn
    # resume: step 1 replayed from the log (no call), step 2 called once;
    # the next turn ("again") makes its own 2 calls
    assert count.read_text() == "x" * 4

    from pyreact_durable import SessionLog

    with SessionLog.open(log, readonly=True) as slog:
        ends = [(e["turn_id"], e["payload"]["status"]) for e in slog.events("turn.end")]
        starts = [e["payload"].get("resumed", False) for e in slog.events("turn.start")]
        llm = [e for e in slog.entries if e.get("type") == "llm"]
    assert ends == [("t1", "done"), ("t2", "done")]
    assert starts == [False, True, False]
    assert len(llm) == 4  # each distinct call journaled exactly once
