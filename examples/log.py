"""Write your own output component: ``Log`` publishes a custom ``log`` event.

What it teaches:

- an output component renders nothing (``return []``); its effect *publishes
  an event* on the session's ``EventBus`` -- the same way the built-in
  ``Message`` publishes ``message`` events (and DOM events reach listeners)
- ``trigger="mount"`` (publish once) vs ``"change"`` (publish whenever
  ``text`` changes) is just the effect's deps list
- events carry the turn id and the component path, so ``:events log`` and
  ``:turns`` attribute them; a ``ConsoleSink`` decides what the terminal shows

Run from the repo root::

    uv run python -m examples.log
    > hello
    > :events log   :turns   :q

No LM.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pyreact.boot import bootstrap, read_terminal_and_invoke
from pyreact.components.message import Message
from pyreact.core import component, hooks, use_emit, use_turn

# Offline smoke run (tests/test_examples_smoke.py).
SMOKE = {"stdin": "hello\nhello\n:events log\n:q\n"}


@component
def Log(text: str, trigger: Literal["change", "mount"] = "change"):
    """Publish ``text`` as a ``log`` event ({"text"}) on mount or on change."""
    # publishes on the session bus with this component's stable path as the
    # source (e.g. Echo/Log[seen-1]) and the current turn id
    emit = use_emit()

    def _publish():
        emit("log", {"text": text})

    hooks.use_effect(_publish, [] if trigger == "mount" else [text])
    return []


@component
def Echo():
    turn = use_turn()
    out = [Log(key="boot", text="app mounted", trigger="mount")]  # once, stays mounted
    if turn is not None:
        # keyed per turn, so a repeated line is logged again
        out.append(Log(key=f"seen-{turn.index}", text=f"turn {turn.index}: {turn.text!r}"))
        out.append(Message(key=f"echo-{turn.index}", text=f"echo: {turn.text}",
                           sender="assistant"))
    return out


async def main() -> None:
    app = bootstrap(Echo)
    try:
        # Show our custom kind next to chat messages.
        await read_terminal_and_invoke(app, prompt="> ", console_kinds=("message", "log"))
    finally:
        app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
