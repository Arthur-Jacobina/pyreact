"""python -m integrations.learn: arg parsing and commands (fake LM, no network)."""

import json
import re

import dspy
import pytest

from integrations.claude_cli_lm import _wrap
from integrations.learn import cli


class RuleLM(dspy.BaseLM):
    def __init__(self):
        super().__init__(model="fake/rule", cache=False)

    def forward(self, prompt=None, messages=None, **kw):
        found = re.findall(r"\[\[ ## text ## \]\]\n(.*)", messages[-1]["content"])
        lab = "pos" if found and "good" in found[-1] else "neg"
        return _wrap(f"[[ ## label ## ]]\n{lab}\n\n[[ ## completed ## ]]", self.model)


def test_parser_defaults_and_types():
    p = cli.build_parser()
    a = p.parse_args(["optimize", "pkg.mod:prog", "--data", "a.jsonl", "--data", "b.jsonl",
                      "--module-key", "router", "--budget", "40"])
    assert a.command == "optimize" and a.data == ["a.jsonl", "b.jsonl"]
    assert a.budget == 40 and a.optimizer == "gepa" and a.lm == "claude:haiku"
    assert a.val == 0.3 and a.out is None and not a.resume
    assert p.parse_args(["optimize", "q -> a", "--data", "x", "--module-key", "k",
                         "--budget", "light"]).budget == "light"
    assert p.parse_args(["diff", "a.json", "b.json"]).b == "b.json"
    assert p.parse_args(["inspect", "a.json"]).path == "a.json"
    with pytest.raises(SystemExit):
        p.parse_args(["optimize", "x"])  # --data / --module-key required


def test_load_program_forms(tmp_path, monkeypatch):
    (tmp_path / "progmod.py").write_text(
        "import dspy\n"
        "class Sig(dspy.Signature):\n    text: str = dspy.InputField()\n"
        "    label: str = dspy.OutputField()\n"
        "prog = dspy.ChainOfThought('text -> label')\n"
        "def make():\n    return dspy.Predict('text -> label')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert isinstance(cli.load_program("text -> label", "cot"), dspy.ChainOfThought)
    assert isinstance(cli.load_program("progmod:prog"), dspy.ChainOfThought)
    assert isinstance(cli.load_program("progmod:make"), dspy.Predict)
    assert isinstance(cli.load_program("progmod:Sig", "cot"), dspy.ChainOfThought)
    with pytest.raises(SystemExit):
        cli.load_program("no_colon")


def _write_events(path):
    rows = [("good day", "pos", 1.0), ("bad day", "neg", 1.0), ("good food", "pos", 1.0),
            ("awful", "neg", 1.0)]
    with open(path, "w") as f:
        for i, (text, label, score) in enumerate(rows):
            cid = f"c{i}"
            f.write(json.dumps({"kind": "llm.start", "payload": {
                "call_id": cid, "module_key": "Predict:text -> label:App#clf",
                "inputs": {"text": text}}}) + "\n")
            f.write(json.dumps({"kind": "llm.end", "payload": {
                "call_id": cid, "module_key": "Predict:text -> label:App#clf",
                "outputs": {"label": label}}}) + "\n")
            f.write(json.dumps({"kind": "feedback", "payload": {
                "target": cid, "score": score, "text": ""}}) + "\n")


def test_cli_optimize_inspect_diff(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_lm", lambda spec: RuleLM() if spec else None)
    events = tmp_path / "events.jsonl"
    _write_events(events)
    out = tmp_path / "res.json"
    rc = cli.main(["optimize", "text -> label", "--data", str(events), "--module-key", "clf",
                   "--optimizer", "bootstrap", "--budget", "2", "--min-score", "1",
                   "--threads", "1", "--out", str(out)])
    assert rc == 0 and out.exists()
    text = capsys.readouterr().out
    assert "3 train / 1 val" in text and "bootstrap" in text and "demos: 0 -> 2" in text

    assert cli.main(["inspect", str(out)]) == 0
    assert "optimizer   bootstrap" in capsys.readouterr().out

    base = tmp_path / "base.json"
    base.write_text(json.dumps(dspy.Predict("text -> label").dump_state()))
    assert cli.main(["diff", str(base), str(out)]) == 0
    d = capsys.readouterr().out
    assert "demos: 0 -> 2" in d and "b: bootstrap" in d

    assert cli.main(["optimize", "text -> label", "--data", str(events),
                     "--module-key", "missing"]) == 2

    assert cli.main(["list"]) == 0
    listing = capsys.readouterr().out
    assert "rlm" in listing and "max_metric_calls" in listing
