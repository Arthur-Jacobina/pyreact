"""RLM agent: answer questions over a context too large for one prompt.

The context is the concatenated source of this repository (~hundreds of KB).
``use_rlm`` hands it to ``dspy.RLM`` as a REPL variable: the root LM writes
Python to grep/slice it and asks a sub-LM (``sub_model="fast"``) about the
relevant pieces, instead of stuffing everything into a prompt.

    uv run python -m examples.rlm_agent "Where are rerenders deduplicated?"

Uses the ``claude -p`` CLI LM (no API key needed). Requires Deno for the
default sandbox (see ``integrations/rlm.py``).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, List, Optional

from examples.message import Message
from integrations.dspy_integration import DSPyProvider
from integrations.rlm import use_rlm
from pyreact.components.keystroke import Keystroke
from pyreact.core.core import component, hooks
from pyreact.core.events import EventBus

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


def make_app(models: Dict[str, Any], context: str, sink: Dict[str, Any], **agent_kw):
    @component
    def Home():
        (turn, q), set_q = hooks.use_state((0, ""))

        def on_answer(question, answer, error):
            sink.setdefault("answers", []).append((question, answer, error))

        return [
            Keystroke(key="in", on_submit=lambda line: set_q(lambda p: (p[0] + 1, line))),
            RepoAgent(key="rlm", question=q, turn=turn, context=context,
                      on_answer=on_answer, **agent_kw),
        ]

    @component
    def App():
        bus = hooks.get_service("event_bus", EventBus)
        hooks.use_effect(
            lambda: bus.subscribe(lambda ev: sink.setdefault("events", []).append(ev),
                                  kinds=["llm.*", "rlm.*"]),
            [],
        )
        return [DSPyProvider(key="dspy", models=models, children=[Home(key="home")])]

    return App


def ask(question: str, *, models: Optional[Dict[str, Any]] = None,
        timeout: float = 600, **agent_kw) -> Dict[str, Any]:
    from pyreact.boot import bootstrap

    if models is None:
        from integrations.claude_cli_lm import ClaudeCLILM

        models = {"default": ClaudeCLILM("haiku"), "fast": ClaudeCLILM("haiku")}
    context = repo_source()
    sink: Dict[str, Any] = {"context_chars": len(context)}
    app = bootstrap(make_app(models, context, sink, **agent_kw))
    try:
        app.invoke(question, wait=True, timeout=timeout)
    finally:
        app.shutdown()
    return sink


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or (
        "Which function schedules component rerenders, and which data structure "
        "prevents a component from being enqueued twice?"
    )
    out = ask(q)
    print("context chars:", out["context_chars"])
    for ev in out.get("events", []):
        print(ev.kind, {k: v for k, v in ev.payload.items() if k != "error"} | (
            {"error": ev.payload["error"][:200]} if "error" in ev.payload else {}))
    for question, answer, error in out.get("answers", []):
        print("Q:", question)
        print("A:", answer if error is None else f"ERROR {error}")
