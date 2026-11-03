"""04 - Everything together: an LLM chat with durable, branchable memory.

The "full" example (01-03 teach the pieces one at a time):

- the transcript is ``use_conversation()`` — persisted in a pi-style JSONL
  session log through the ``Durable`` provider (``pyreact_durable``)
- every LM call goes through ``durable_lm``: re-running a turn on the same
  branch replays the journaled answer instead of calling the model again
- commands (handled by the agent itself, via ``Keystroke``):

    /label <name>        bookmark the current position
    /fork <label|id>     branch from a bookmark (next message starts a branch)
    /checkout <label|id> switch to another branch / position
    /history             print the conversation on the current branch
    /tree                print the session tree
    /help

Run from the repo root (Ctrl+C / ``:q`` to quit; kill it any way you like,
then start it again — it remembers):

    uv run python -m examples.durable.04_llm_chat [--log runs/durable/chat.jsonl]

Inspect the file from another terminal:

    uv run python -m pyreact_durable show runs/durable/chat.jsonl
    uv run python -m pyreact_durable events runs/durable/chat.jsonl --kind llm.*

End-to-end check (kill -9, branches, LM replay): ``uv run python scripts/e2e_durable.py``.

LM: ``examples._lm`` (``claude -p`` by default; ``PYREACT_EXAMPLE_LM=openai|offline``).
"""

from __future__ import annotations

import argparse
import asyncio

import dspy

from integrations import DSPyProvider, use_call, use_predict
from pyreact.boot import bootstrap, read_terminal_and_invoke
from pyreact.components.keystroke import Keystroke
from pyreact.core import EventBus, component, hooks
from pyreact_durable import (
    Durable,
    SessionLogError,
    use_conversation,
    use_durable,
    use_durable_lm,
)

from examples._lm import example_lm

# Offline smoke run (tests/test_examples_smoke.py, PYREACT_EXAMPLE_LM=offline).
SMOKE = {"args": ["--log", "{tmp}/chat.jsonl"], "stdin": "my name is Ada\n/label base\nwhat is my name?\n/history\n/tree\n:q\n"}

DEFAULT_LOG = "runs/durable/chat.jsonl"


class ChatSig(dspy.Signature):
    """You are a friendly, concise assistant with a long memory. Use the
    conversation history. Answer in at most two sentences."""

    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()


@component
def DurableChat():
    conv = use_conversation()  # persisted transcript (this branch)
    log = use_durable()
    events = hooks.get_service("event_bus", EventBus)
    chat = use_predict(ChatSig)
    call = use_call(chat)
    pending, set_pending = hooks.use_state(None)  # (seq, question), in-memory
    hooks_path = hooks.path

    def say(text: str, sender: str = "assistant") -> None:
        events.publish("message", {"text": text, "sender": sender, "message_type": "chat"})

    def command(line: str) -> None:
        cmd, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        try:
            if cmd == "label" and arg:
                e = log.label(arg)
                say(f"labelled {e['targetId']} as {arg!r}", "system")
            elif cmd in ("fork", "checkout") and arg:
                leaf = log.checkout(arg)
                verb = "branching from" if cmd == "fork" else "switched to"
                say(f"{verb} {arg!r} ({leaf}); conversation has "
                    f"{len(log.get_state(hooks_path, 'history', []))} message(s)", "system")
            elif cmd == "history":
                lines = [f"{m['role']}: {m['content']}" for m in conv.messages]
                say("\n".join(lines) if lines else "(empty)", "system")
            elif cmd == "tree":
                say(log.render(), "system")
            else:
                say("commands: /label <name>, /fork <label>, /checkout <label|id>, "
                    "/history, /tree", "system")
        except SessionLogError as exc:
            say(f"error: {exc}", "system")

    def on_submit(text: str) -> None:
        line = text.strip()
        if not line:
            return
        if line.startswith("/"):
            command(line)
        else:
            set_pending(lambda p: ((p[0] + 1) if p else 1, line))

    def _ask():
        if pending is not None:
            call.run(question=pending[1], history=conv.as_dspy_history())

    hooks.use_effect(_ask, [pending])

    def _remember():
        if call.ver and call.result is not None and pending is not None:
            answer = call.result.answer
            # one state entry for the pair: a crash never leaves half a turn
            conv.extend([{"role": "user", "content": pending[1]},
                         {"role": "assistant", "content": answer}])
            say(answer)
        elif call.ver and call.error is not None:
            say(f"LM error: {call.error}", "system")

    hooks.use_effect(_remember, [call.ver])

    return [Keystroke(key="input", on_submit=on_submit)]


@component
def LMLayer(lm=None, lm_mode="replay_or_record", children=None):
    dlm = use_durable_lm(lm, mode=lm_mode)  # journaled / replayed LM calls
    return [DSPyProvider(key="dspy", lm=dlm, children=children or [])]


@component
def DurableChatApp(log=None, lm=None, lm_mode="replay_or_record"):
    return [
        Durable(key="durable", log=log, children=[
            LMLayer(key="lm", lm=lm, lm_mode=lm_mode, children=[DurableChat(key="chat")]),
        ])
    ]


def default_lm():
    return example_lm()


def start(log, lm, lm_mode="replay_or_record"):
    """Boot the app on ``log`` (path or SessionLog); returns the AppRunner."""

    @component
    def App():
        return [DurableChatApp(key="app", log=log, lm=lm, lm_mode=lm_mode)]

    return bootstrap(App)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    args = ap.parse_args()
    try:
        import dotenv

        dotenv.load_dotenv()
    except ImportError:  # pragma: no cover
        pass

    app = start(args.log, default_lm())
    print(f"durable chat on {args.log} — /help for commands, :q to quit")
    try:
        await read_terminal_and_invoke(app, prompt="> ")  # "/..." lines reach the app
    finally:
        app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
