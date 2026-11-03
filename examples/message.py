"""``Message``: an agent's chat output is an *event*, not a print.

``Message`` lives in :mod:`pyreact.components.message`; this module re-exports
it (``from examples.message import Message`` keeps working) and shows what it
does. Rendering ``Message(text=..., sender=..., message_type=...)`` publishes

    message  {"text", "sender", "message_type", "timestamp"}

on the session's ``EventBus`` (with the turn id and component path). Who
displays it is the transport's business: the terminal's ``ConsoleSink``, the
web UI over its WebSocket, a JSONL log, your own subscriber -- below, a plain
``bus.subscribe`` callback.

Run from the repo root::

    uv run python -m examples.message
    > hi
    > :q

No LM.
"""

from __future__ import annotations

import asyncio

from pyreact.boot import bootstrap, read_terminal_and_invoke
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


async def main() -> None:
    app = bootstrap(Greeter)
    # Any subscriber can render messages; this one prints the raw event. It
    # attaches after mount, so it misses the welcome (ConsoleSink replays the log).
    app.event_bus.subscribe(
        lambda ev: print(f"  [subscriber] {ev.kind} turn={ev.turn_id} from {ev.source}: "
                         f"{ev.payload['sender']}/{ev.payload['message_type']}"),
        ["message"],
    )
    try:
        await read_terminal_and_invoke(app, prompt="> ")  # ConsoleSink renders them too
    finally:
        app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
