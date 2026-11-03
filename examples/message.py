"""``Message``: an agent's chat output is an *event*, not a print.

``Message`` lives in :mod:`pyreact.components.message`; this module re-exports
it (``from examples.message import Message`` keeps working) and shows what it
does. Rendering ``Message(text=..., sender=..., message_type=...)`` publishes

    message  {"text", "sender", "message_type", "timestamp"}

on the session's ``EventBus`` (with the turn id and component path). Who
displays it is the transport's business: the terminal's ``ConsoleSink``
(``run_repl``), the web UI over its WebSocket, a JSONL log -- below, also a
devtools trace of the raw event (``examples/_devtools.py``).

Run from the repo root::

    uv run python -m examples.message
    > hi
    > :q

No LM.
"""

from __future__ import annotations

from examples._devtools import run_repl, trace
from pyreact.boot import create_root
from pyreact.components.message import Message, set_legacy_message_bridge
from pyreact.core import component, use_turn

__all__ = ["Message", "set_legacy_message_bridge"]

# Offline smoke run (tests/test_examples_smoke.py).
SMOKE = {"stdin": "hi\nhi\n:q\n"}


@component
def Greeter():
    turn = use_turn()
    if turn is None:
        # trigger="mount": publish once; the default "change" republishes when text changes
        return [Message(key="hello", text="Say something.", sender="assistant",
                        message_type="info", trigger="mount")]
    # A new key per turn mounts a new Message, so the same text is sent again.
    return [Message(key=f"reply-{turn.index}", text=f"you said {turn.text!r} "
                    f"({len(turn.text)} chars)", sender="assistant")]


def raw(ev) -> str:  # devtools view of the event behind each Message
    return (f"  [devtools] {ev.kind} turn={ev.turn_id} from {ev.source}: "
            f"{ev.payload['sender']}/{ev.payload['message_type']}")


def main() -> None:
    with create_root(Greeter, session_id="message") as root:  # mounts the tree; unmounts on exit
        trace(root, ["message"], raw)
        run_repl(root, prompt="> ")  # each line -> root.send(line); Messages print


if __name__ == "__main__":
    main()
