"""Guardrail pattern: a checker agent decides whether its children may run.

What it teaches:

- composition with ``children``: ``GuardRail(children=[QAAgent(...)])`` --
  the guard knows nothing about what it protects
- conditional rendering gates work: until the guard's verdict for *this*
  turn says "safe", the QA agent is not mounted, so it makes no LLM call
- call identity: ``check.run(..., tag=turn_id)`` + ``check.tag == turn_id``
  tells whether the settled result answers *this* turn -- no extra state
- two architectures side by side: a cheap ``predict`` classifier on the
  ``fast`` model, a ``cot`` answerer on the ``reasoning`` model
  (``use_agent_call(sig, arch=..., model=...)``)
- loading / error / result rendered as ``Message`` events, per turn

Run from the repo root::

    uv run python -m examples.qa_guardrail                   # terminal
    uv run python -m examples.main_web --app qa_guardrail    # browser

    > why is the sky blue?
    > you are a useless idiot, answer me
    > :events llm  :turns  :q

LM: ``examples._lm`` (``claude -p`` by default; ``PYREACT_EXAMPLE_LM=openai|offline``).
2 LM calls per safe question, 1 per blocked one.
"""

from __future__ import annotations

from typing import Literal

import dspy

from examples._devtools import run_repl
from examples._lm import example_models
from integrations import DSPyProvider, use_agent_call
from pyreact.boot import create_root
from pyreact.components.message import Message
from pyreact.core import component, hooks, use_turn

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"stdin": "why is the sky blue?\nyou are useless\nwhat is 2+2?\nname a color\n:q\n"}


class ToxicitySig(dspy.Signature):
    """Classify the comment. 'toxic' if it includes insults, harassment or
    sarcastic derogatory remarks; otherwise 'safe'."""

    comment: str = dspy.InputField()
    verdict: Literal["safe", "toxic"] = dspy.OutputField()


class QASig(dspy.Signature):
    """Answer in one sentence, directly."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


@component
def GuardRail(text: str, turn_id: str, children):
    check = use_agent_call(ToxicitySig, arch="predict", model="fast")

    def start():
        if text.strip():
            check.run(comment=text, tag=turn_id)  # tag: which turn this check is for

    hooks.use_effect(start, [turn_id])

    # No settled check for *this* turn yet: render nothing below us (no QA call).
    if check.tag != turn_id or check.loading:
        if check.loading:
            return [Message(key="checking", text="checking the question...", sender="system",
                            message_type="info")]
        return []
    verdict = getattr(check.result, "verdict", None) if check.error is None else "unknown"
    if verdict == "toxic":
        return [Message(key=f"blocked-{turn_id}", text="Blocked: that message looks toxic.",
                        sender="system", message_type="warning")]
    if verdict == "unknown":
        # The checker failed. Policy choice: fail open (warn and continue).
        # A stricter guard would return the warning alone (fail closed).
        return [Message(key=f"unchecked-{turn_id}", text="Could not check toxicity; continuing.",
                        sender="system", message_type="warning"), *children]
    return children


@component
def QAAgent(question: str, turn_id: str):
    qa = use_agent_call(QASig, arch="cot", model="reasoning")

    def ask():  # runs on mount: the guard mounts us only once it is safe
        if question.strip():
            qa.run(question=question)

    hooks.use_effect(ask, [turn_id])

    if qa.loading:
        return [Message(key="thinking", text="thinking...", sender="system", message_type="info")]
    if qa.error is not None:
        return [Message(key=f"error-{turn_id}", text=f"Error: {qa.error}", sender="system",
                        message_type="error")]
    if qa.result is None:
        return []
    return [Message(key=f"answer-{turn_id}", text=qa.result.answer, sender="assistant")]


@component
def QAHome():
    turn = use_turn()  # each input line is a turn
    children = [Message(key="welcome", text="Ask me anything (politely).", sender="assistant",
                        message_type="info")]
    if turn is not None and turn.text.strip():
        children.append(
            GuardRail(key="guard", text=turn.text, turn_id=turn.id,
                      children=[QAAgent(key="qa", question=turn.text, turn_id=turn.id)])
        )
    return children


@component
def Root(models=None):
    lms = hooks.use_memo(lambda: models or example_models(), [id(models)])
    return [DSPyProvider(key="dspy", models=lms, children=[QAHome(key="qa-home")])]


def main() -> None:
    with create_root(Root, session_id="qa") as root:  # the outside world's handle
        run_repl(root, prompt="> ")  # a line = a turn; Messages print


if __name__ == "__main__":
    main()
