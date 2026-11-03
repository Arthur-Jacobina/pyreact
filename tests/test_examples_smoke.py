"""Every example runs end to end offline (PYREACT_EXAMPLE_LM=offline).

Examples opt in by declaring a module-level ``SMOKE`` dict (read with ``ast``,
so nothing is imported here):

    SMOKE = {"args": [...], "stdin": "line\\n:q\\n", "timeout": 120}

``{tmp}`` in args is replaced by a per-test temporary directory, and artifacts
go there too (``PYREACT_EXAMPLE_OUT``). The run must exit 0, print no traceback
and leave the repository untouched (offline output must never overwrite real
artifacts). Answers are meaningless offline: this guards
against examples rotting, not against bad prompts.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _specs():
    out = []
    for path in sorted(EXAMPLES.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "SMOKE" for t in node.targets)
            ):
                module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
                out.append(pytest.param(module, ast.literal_eval(node.value), id=module))
    return out


SPECS = _specs()


def _repo_status():
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return out.stdout if out.returncode == 0 else None


def test_examples_declare_smoke_specs():
    assert len(SPECS) >= 8, "examples lost their SMOKE specs"


@pytest.mark.examples
@pytest.mark.parametrize("module,spec", SPECS)
def test_example_runs_offline(module, spec, tmp_path):
    args = [a.replace("{tmp}", str(tmp_path)) for a in spec.get("args", [])]
    env = {
        **os.environ,
        "PYREACT_EXAMPLE_LM": "offline",
        "PYREACT_EXAMPLE_OUT": str(tmp_path / "out"),
        "PYTHONUNBUFFERED": "1",
    }
    env.pop("OPENAI_API_KEY", None)
    before = _repo_status()
    proc = subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=tmp_path if spec.get("cwd") == "tmp" else ROOT,
        env=env,
        input=spec.get("stdin", ""),
        capture_output=True,
        text=True,
        timeout=spec.get("timeout", 180),
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"{module} exited {proc.returncode}\n{output[-4000:]}"
    assert "Traceback (most recent call last)" not in output, output[-4000:]
    if before is not None:
        assert _repo_status() == before, f"{module} wrote into the repository"
