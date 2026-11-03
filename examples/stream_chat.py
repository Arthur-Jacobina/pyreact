"""Streaming multi-turn chat.

- conversation memory is component state: a ``dspy.History`` in ``use_state``
- the answer streams via ``use_stream`` (dspy.streamify + StreamListener)
- the terminal prints tokens live by subscribing to ``llm.chunk`` events

Run from the repo root:

    uv run python -m examples.stream_chat

Uses OpenAI (streams token by token) when OPENAI_API_KEY is set, otherwise
the ``claude -p`` dev LM (no streaming: the answer appears at once).
"""

import asyncio

import dotenv
import dspy

from integrations import DSPyProvider, use_chain_of_thought, use_stream
from pyreact.boot import bootstrap, read_terminal_and_invoke
from pyreact.components.keystroke import Keystroke
from pyreact.core import EventBus, HookContext, component, hooks

from examples._lm import example_lm
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
    pending, set_pending = hooks.use_state(None)  # (turn, question)

    chat = use_chain_of_thought(ChatSig)
    stream = use_stream(chat, listen=["answer"])

    def on_submit(line: str):
        if line.strip():
            set_pending(lambda p: ((p[0] + 1) if p else 1, line.strip()))

    def _ask():
        if pending is not None:
            stream.run(question=pending[1], history=history)

    hooks.use_effect(_ask, [pending])

    def _remember():
        # ver bumps once per settled call: append the finished turn.
        if stream.ver and stream.result is not None and pending is not None:
            turn = {"question": pending[1], "answer": stream.result.answer}
            set_history(lambda h: dspy.History(messages=[*h.messages, turn]))

    hooks.use_effect(_remember, [stream.ver])

    return [Keystroke(key="input", on_submit=on_submit)]


@component
def Root(lm=None):
    return [DSPyProvider(key="dspy", lm=lm, children=[Chat(key="chat")])]


def _default_lm():
    dotenv.load_dotenv()
    return example_lm()


def _print_stream_events():
    """Terminal 'renderer' for the stream: tokens live, then a newline."""
    streamed = set()

    def on_event(ev):
        p = ev.payload
        if ev.kind == "llm.chunk" and p.get("field") == "answer":
            if p["call_id"] not in streamed:
                streamed.add(p["call_id"])
                print("\x1b[32m[ASSISTANT] ", end="")
            print(p["chunk"], end="", flush=True)
        elif ev.kind == "llm.end":
            if p["call_id"] in streamed:
                print("\x1b[0m")
            else:  # LM didn't stream: print the final answer
                print(f"\x1b[32m[ASSISTANT] {p['outputs'].get('answer')}\x1b[0m")
        elif ev.kind == "llm.error":
            print(f"\x1b[31m[ERROR] {p['error']}\x1b[0m")

    HookContext.get_service("event_bus", EventBus).subscribe(on_event, ["llm.*"])


async def main():
    lm = _default_lm()

    @component
    def Boot():
        return [Root(key="root", lm=lm)]

    _print_stream_events()
    app = bootstrap(Boot)
    try:
        await read_terminal_and_invoke(app, prompt="> ", wait=True)
    finally:
        app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
