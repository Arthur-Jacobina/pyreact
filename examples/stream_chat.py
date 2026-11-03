"""Streaming multi-turn chat.

- conversation memory is component state: a ``dspy.History`` in ``use_state``
- the answer streams via ``use_stream`` (dspy.streamify + StreamListener);
  the finished answer is a ``Message`` (output is components)
- the live tokens are a devtools view: ``trace(root, ["llm.chunk"], ...)``
  prints each ``llm.chunk`` event as it is published

Run from the repo root:

    uv run python -m examples.stream_chat

LM: ``examples._lm`` (``claude -p`` by default; ``PYREACT_EXAMPLE_LM=openai|offline``).
OpenAI streams token by token; the ``claude -p`` dev LM doesn't stream, so the
answer appears at once.
"""

import dotenv
import dspy

from examples._devtools import run_repl, trace
from examples._lm import example_lm
from integrations import DSPyProvider, use_chain_of_thought, use_stream
from pyreact.boot import create_root
from pyreact.components.message import Message
from pyreact.core import component, hooks, use_on_turn

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"stdin": "hi\nand again\n:q\n"}


class ChatSig(dspy.Signature):
    """You are a friendly, concise assistant. Use the conversation history."""

    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()


@component
def Chat():
    # Multi-turn memory lives in component state (immutable updates).
    history, set_history = hooks.use_state(dspy.History(messages=[]))

    chat = use_chain_of_thought(ChatSig)
    stream = use_stream(chat, listen=["answer"])

    def ask(turn):  # once per input, even if the same line is sent twice
        if turn.text.strip():
            stream.run(question=turn.text.strip(), history=history)

    use_on_turn(ask)

    def _remember():
        # ver bumps once per settled call: append the finished turn.
        if stream.ver and stream.result is not None:
            # stream.inputs: the inputs of the call that just settled
            pair = {"question": stream.inputs["question"], "answer": stream.result.answer}
            set_history(lambda h: dspy.History(messages=[*h.messages, pair]))

    hooks.use_effect(_remember, [stream.ver])

    if stream.error is not None:
        return [Message(key=f"error-{stream.ver}", text=f"Error: {stream.error}",
                        sender="system", message_type="error")]
    if stream.ver and stream.result is not None:  # a new key per answer: printed once
        return [Message(key=f"answer-{stream.ver}", text=stream.result.answer,
                        sender="assistant")]
    return []


@component
def Root(lm=None):
    return [DSPyProvider(key="dspy", lm=lm, children=[Chat(key="chat")])]


def _default_lm():
    dotenv.load_dotenv()
    return example_lm()


def live_tokens():
    """Devtools: the answer's tokens as they stream (OpenAI streams; the
    ``claude -p`` dev LM doesn't, so nothing prints before the Message)."""
    streamed = set()

    def fmt(ev):
        p = ev.payload
        if ev.kind == "llm.chunk" and p.get("field") == "answer":
            first = p["call_id"] not in streamed
            streamed.add(p["call_id"])
            return ("\x1b[90m  ... " if first else "") + p["chunk"]
        if ev.kind == "llm.end" and p["call_id"] in streamed:
            return "\x1b[0m\n"
        return None

    return fmt


def main():
    lm = _default_lm()

    @component
    def Boot():
        return [Root(key="root", lm=lm)]

    with create_root(Boot, session_id="chat") as root:
        trace(root, ["llm.chunk", "llm.end"], live_tokens(), end="")
        run_repl(root, prompt="> ", console_kinds=("message", "error"))


if __name__ == "__main__":
    main()
