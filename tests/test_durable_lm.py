import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import dspy
import pytest

from pyreact_durable import SessionLog
from pyreact_durable.lm import ReplayMissError, durable_lm, llm_entries


class FakeLM(dspy.BaseLM):
    """Counts calls; answers answer-1, answer-2, ... in DSPy chat format."""

    def __init__(self):
        super().__init__(model="fake/model", cache=False)
        self.calls = 0

    def forward(self, prompt=None, messages=None, **kwargs):
        self.calls += 1
        text = f"[[ ## answer ## ]]\nanswer-{self.calls}\n\n[[ ## completed ## ]]"
        msg = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            model=self.model,
        )


def _ask(lm, q="what?"):
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        return dspy.Predict("question -> answer")(question=q).answer


def test_record_then_replay_after_restart(tmp_path):
    p = tmp_path / "lm.jsonl"
    inner = FakeLM()
    with SessionLog.open(p) as log:
        dlm = durable_lm(inner, log)
        assert _ask(dlm, "q1") == "answer-1"
        assert inner.calls == 1 and dlm.stats["recorded"] == 1
        [e] = llm_entries(log)
        assert e["occurrence"] == 0 and e["model"] == "fake/model"
        assert e["request"]["messages"][-1]["role"] == "user"

    # "crash" + restart: a fresh process re-runs the same call on the branch
    inner2 = FakeLM()
    with SessionLog.open(p) as log:
        dlm = durable_lm(inner2, log, mode="replay")
        assert _ask(dlm, "q1") == "answer-1"
        assert inner2.calls == 0 and dlm.stats == {"replayed": 1, "recorded": 0,
                                                   "inner_calls": 0}
        with pytest.raises(ReplayMissError):
            dlm(messages=[{"role": "user", "content": "never asked"}])
        assert len(llm_entries(log)) == 1


def test_occurrences_branches_and_modes(tmp_path):
    p = tmp_path / "occ.jsonl"
    inner = FakeLM()
    with SessionLog.open(p) as log:
        start = log.set_state("App", "x", 0)
        dlm = durable_lm(inner, log)
        assert [_ask(dlm), _ask(dlm)] == ["answer-1", "answer-2"]  # same key, 2 occurrences
        assert [e["occurrence"] for e in llm_entries(log)] == [0, 1]

    inner2 = FakeLM()
    with SessionLog.open(p) as log:
        dlm = durable_lm(inner2, log)
        assert [_ask(dlm), _ask(dlm), _ask(dlm)] == ["answer-1", "answer-2", "answer-1"]
        assert inner2.calls == 1 and dlm.stats["replayed"] == 2
        # checkout before the journaled calls: the cursor resets, new branch records
        log.checkout(start["id"])
        assert _ask(dlm) == "answer-2" and inner2.calls == 2
        # record mode always calls the model
        rec = durable_lm(inner2, log, mode="record")
        _ask(rec)
        assert inner2.calls == 3
        pt = durable_lm(inner2, log, mode="passthrough")
        n = len(log.entries)
        _ask(pt)
        assert inner2.calls == 4 and len(log.entries) == n
    with pytest.raises(ValueError):
        durable_lm(FakeLM(), SessionLog.open(tmp_path / "x.jsonl"), mode="bogus")


def test_key_depends_on_request_not_transport(tmp_path):
    inner = FakeLM()
    with SessionLog.open(tmp_path / "k.jsonl") as log:
        dlm = durable_lm(inner, log)
        msgs = [{"role": "user", "content": "hi"}]
        dlm(messages=msgs, temperature=0.0)
        dlm.reset()
        dlm(messages=msgs, temperature=0.0, num_retries=9)  # transport kwarg ignored
        assert inner.calls == 1
        dlm.reset()
        dlm(messages=msgs, temperature=0.7)  # sampling kwarg changes the key
        assert inner.calls == 2


def test_async_path_and_copy(tmp_path):
    inner = FakeLM()
    with SessionLog.open(tmp_path / "a.jsonl") as log:
        dlm = durable_lm(inner, log)
        out = asyncio.run(dlm.acall(messages=[{"role": "user", "content": "x"}]))
        assert "answer-1" in out[0] and inner.calls == 1
        twin = dlm.copy(temperature=0.3)  # dspy deep-copies LMs
        assert twin.log is log and twin.inner is inner and twin.kwargs["temperature"] == 0.3
        dlm.reset()
        out = asyncio.run(dlm.acall(messages=[{"role": "user", "content": "x"}]))
        assert "answer-1" in out[0] and inner.calls == 1


def test_pyreact_never_imports_pyreact_durable():
    root = Path(__file__).resolve().parents[1]
    hits = [str(p) for p in (root / "pyreact").rglob("*.py")
            if "pyreact_durable" in p.read_text(encoding="utf-8")]
    assert hits == []
    code = (
        "import sys, pyreact, pyreact.core, pyreact.boot, pyreact.core.hook\n"
        "assert not any(m.startswith('pyreact_durable') for m in sys.modules), sys.modules\n"
    )
    assert subprocess.run([sys.executable, "-c", code], cwd=root).returncode == 0
    # and the storage layer itself is free of pyreact / dspy
    code2 = (
        "import sys; import pyreact_durable, pyreact_durable.log, pyreact_durable.__main__\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('pyreact', 'dspy')]\n"
        "assert not bad, bad\n"
    )
    assert subprocess.run([sys.executable, "-c", code2], cwd=root).returncode == 0
