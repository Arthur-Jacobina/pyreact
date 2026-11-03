"""Hooks built on turns: handle each input exactly once.

``use_state`` drops a value equal to the current one, so storing the
submitted *text* misses a repeated input ("yes", "yes"). Turns are the
identity of an input: every submission is a new :class:`~pyreact.core.turns.Turn`
with a new ``id`` even when the text repeats. Key work on the turn::

    @component
    def Chat():
        call = use_agent_call(ChatSig)
        use_on_turn(lambda turn: call.run(question=turn.text))   # once per input
        ...

or, equivalently, ``turn = use_turn()`` + ``use_effect(fn, [turn and turn.id])``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .core import hooks
from .turns import Turn, use_turn

__all__ = ["use_on_turn"]


def use_on_turn(fn: Callable[[Turn], Any], **use_turn_kwargs: Any) -> Optional[Turn]:
    """Call ``fn(turn)`` once per turn, after commit (like an effect keyed on
    ``turn.id``), also when two inputs have the same text. Returns the turn.

    ``fn`` always sees the latest closure (it may change every render). Like
    an effect, it may return a cleanup, or be ``async`` (run as a tracked task
    of the turn; cancelled if the component unmounts or the next turn fires
    while it is still running). Extra keyword arguments go to :func:`use_turn`;
    ``use_on_turn(fn, since_mount=True)`` skips the turn that was in flight
    (or last ran) when the component mounted and only handles later ones.
    """
    turn = use_turn(**use_turn_kwargs)
    latest = hooks.use_ref(fn)
    latest.current = fn

    def _fire():
        if turn is not None:
            return latest.current(turn)
        return None

    hooks.use_effect(_fire, [turn.id if turn is not None else None])
    return turn
