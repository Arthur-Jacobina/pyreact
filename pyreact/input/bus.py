from typing import Callable, TypedDict, Literal


class Event(TypedDict, total=False):
    type: Literal["text", "submit", "key"]  # expand if desired
    value: str
    source: Literal["web", "term"]
    ts: float
    turn_id: str  # id of the Turn this input started (see pyreact.core.turns)


Subscriber = Callable[[Event], None]


class InputBus:
    """Input bus (thread-safe enough for use with ``asyncio``).

    Inputs are normally emitted by the ``TurnManager`` when a turn starts
    (``AppRunner.invoke`` → turn → ``text`` + ``submit`` events carrying
    ``turn_id``). Emitting directly bypasses turn tracking.
    """

    def __init__(self):
        self._subs: list[Subscriber] = []

    def subscribe(self, fn: Subscriber):
        if fn not in self._subs:
            self._subs.append(fn)

        def unsubscribe():
            try:
                self._subs.remove(fn)
            except ValueError:
                pass

        return unsubscribe

    def emit(self, ev: Event) -> None:
        for fn in list(self._subs):
            try:
                fn(ev)
            except Exception:
                # don't let a bad subscriber kill the others
                pass
