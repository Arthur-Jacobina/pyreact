"""RLM agent: answer questions over a context too large for one prompt.

The context is the concatenated source of this repository (~hundreds of KB).
``use_rlm`` hands it to ``dspy.RLM`` as a REPL variable: the root LM writes
Python to grep/slice it and asks a sub-LM (``sub_model="fast"``) about the
relevant pieces, instead of stuffing everything into a prompt.

    uv run python -m examples.rlm_agent "Where are rerenders deduplicated?"

LM: ``examples._lm`` (``claude -p`` by default; skipped offline). Requires Deno for the
default sandbox (see ``integrations/rlm.py``).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, List, Optional

from examples._devtools import collect
from examples._lm import OFFLINE, example_models
from examples.message import Message
from integrations.dspy_integration import DSPyProvider
from integrations.rlm import use_rlm
from pyreact.boot import create_root
from pyreact.core import use_turn
from pyreact.core.core import component, hooks

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"args": []}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIRS = ("pyreact", "integrations", "examples")


def repo_source(root: str = REPO, dirs: Iterable[str] = SOURCE_DIRS) -> str:
    """Concatenate the repo's Python sources with ``# FILE: path`` headers."""
    parts: List[str] = []
    for d in dirs:
        for base, _sub, files in sorted(os.walk(os.path.join(root, d))):
            if "__pycache__" in base:
                continue
            for f in sorted(files):
                if f.endswith(".py"):
                    p = os.path.join(base, f)
                    with open(p, encoding="utf-8") as fh:
                        parts.append(f"# FILE: {os.path.relpath(p, root)}\n{fh.read()}")
    return "\n\n".join(parts)


class RepoQA:
    """Signature: question over a huge ``context`` string."""

    signature = "context: str, question: str -> answer: str"


@component
def RepoAgent(question: str = "", turn: int = 0, context: str = "",
              on_answer=None, max_iterations: int = 6, max_llm_calls: int = 6):
    run, (result, ver), loading, error = use_rlm(
        RepoQA.signature,
        sub_model="fast",
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        name="repo_qa",
    )

    def _ask():
        if question.strip():
            run(context=context, question=question)

    hooks.use_effect(_ask, [question, turn])

    def _report():
        if on_answer is not None and ver > 0:
            on_answer(question, None if result is None else result.answer, error)

    hooks.use_effect(_report, [ver])

    if loading:
        return [Message(key="thinking", text="Explorando o código (RLM)...",
                        sender="system", message_type="info")]
    if error is not None:
        return [Message(key="error", text=f"Erro: {error}", sender="system",
                        message_type="error")]
    if result is not None:
        return [Message(key="answer", text=str(result.answer), sender="assistant")]
    return []


def make_app(models: Dict[str, Any], context: str, on_answer=None, **agent_kw):
    @component
    def Home():
        turn = use_turn()  # a new turn per input, even when the text repeats
        return [
            RepoAgent(key="rlm", question=turn.text if turn else "",
                      turn=turn.index if turn else 0, context=context,
                      on_answer=on_answer, **agent_kw),
        ]

    @component
    def App():
        return [DSPyProvider(key="dspy", models=models, children=[Home(key="home")])]

    return App


def ask(question: str, *, models: Optional[Dict[str, Any]] = None,
        timeout: float = 600, show: bool = False, **agent_kw) -> Dict[str, Any]:
    """One question, one root. Returns ``{"context_chars", "answers": [(q, a,
    error)], "events": [llm.* / rlm.* events]}``."""
    if models is None:
        models = example_models()
    context = repo_source()
    out: Dict[str, Any] = {"context_chars": len(context), "answers": []}

    def on_answer(q, answer, error):  # a callback prop, like onAnswer in React
        out["answers"].append((q, answer, error))

    with create_root(make_app(models, context, on_answer, **agent_kw),
                     session_id="rlm") as root:
        out["events"] = collect(root, ["llm.*", "rlm.*"])  # devtools: the RLM's trace
        if show:
            root.render_to("console")
        root.send(question, wait=True, timeout=timeout)
    return out


if __name__ == "__main__":
    if OFFLINE:  # the RLM needs a real model writing code in the sandbox
        print("rlm_agent: skipped with PYREACT_EXAMPLE_LM=offline")
        sys.exit(0)
    q = " ".join(sys.argv[1:]) or (
        "Which function schedules component rerenders, and which data structure "
        "prevents a component from being enqueued twice?"
    )
    out = ask(q, show=True)
    print("context chars:", out["context_chars"])
    for ev in out.get("events", []):
        print(ev.kind, {k: v for k, v in ev.payload.items() if k != "error"} | (
            {"error": ev.payload["error"][:200]} if "error" in ev.payload else {}))
    for question, answer, error in out.get("answers", []):
        print("Q:", question)
        print("A:", answer if error is None else f"ERROR {error}")
