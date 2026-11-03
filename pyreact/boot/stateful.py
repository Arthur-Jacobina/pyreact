"""Session participants: the generic checkpoint protocol.

An *agent* is a session's whole state: durable component state, the
conversation, its sandboxes, pending turns, learned modules... Each piece is
owned by a library at the border (durable memory, sandboxes, ...), so the
core only defines how they take part in a checkpoint: objects registered on
a session (``session.participants.add(p)``) with::

    class MyParticipant:
        name = "mine"                 # key of its fragment in a checkpoint

        async def suspend(self, session) -> dict: ...         # JSON-able fragment
        async def resume(self, session, fragment) -> None: ... # before first render

Participants are registered when a session opens (before it mounts), usually
from a factory given to the runner::

    app = AppRunner(App, participants=[MyParticipant])
    # == app.on_session(open=lambda s: s.participants.add(MyParticipant(s)))

Nothing here imports a persistence library; they plug in through this
protocol (libraries at the border).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterator, List, Optional

__all__ = ["Participants"]


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------
class Participants:
    """The checkpoint participants of one session (``session.participants``).

    Ordered by registration: ``suspend`` and ``resume`` are called in that
    order. Names are unique (they key the manifest's ``participants``).
    """

    def __init__(self) -> None:
        self._items: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def add(self, participant: Any, *, replace: bool = False) -> Any:
        """Register ``participant`` (see the module docs); returns it."""
        name = getattr(participant, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(f"participant {participant!r} needs a non-empty str .name")
        for meth in ("suspend", "resume"):
            if not callable(getattr(participant, meth, None)):
                raise TypeError(f"participant {name!r} has no {meth}() method")
        with self._lock:
            if name in self._items and not replace and self._items[name] is not participant:
                raise ValueError(f"a participant named {name!r} is already registered")
            self._items[name] = participant
        return participant

    def remove(self, participant_or_name: Any) -> bool:
        name = (participant_or_name if isinstance(participant_or_name, str)
                else getattr(participant_or_name, "name", None))
        with self._lock:
            return self._items.pop(name, None) is not None

    def get(self, name: str) -> Optional[Any]:
        return self._items.get(name)

    def names(self) -> List[str]:
        return list(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[Any]:
        with self._lock:
            return iter(list(self._items.values()))

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"<Participants {self.names()}>"
