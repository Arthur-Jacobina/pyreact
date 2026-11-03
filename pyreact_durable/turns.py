"""Resume interrupted turns from the session log (``TurnStore`` adapter).

``pyreact.core.turns.TurnStore`` is the core's generic hook; this module
implements it on top of the ``turn.*`` events that :func:`~pyreact_durable.Durable`
journals (``DEFAULT_JOURNAL`` includes them), so nothing extra is written:

- a turn is *pending* when its last ``turn.start`` on the current branch
  (root → leaf) has no ``turn.end`` after it — the process died mid-turn;
- ``get(turn_id)`` returns the last journaled start of any turn (e.g. to
  re-run one that ended ``interrupted``).

``Durable`` attaches a ``DurableTurnStore`` to its session's ``TurnManager``
on mount (``resume_turns=True``), so after a crash::

    app = bootstrap(App)                     # mounts Durable(log=...)
    app.resume_pending(wait=True)            # re-runs the unfinished turns

Combined with :func:`~pyreact_durable.durable_lm`, the LLM calls the turn
already made before the crash are replayed from the log (no model call, no
cost); only the calls it had not reached run for real.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .log import SessionLog

__all__ = ["DurableTurnStore"]


def _turn(entry: dict, resumed: bool = False):
    from pyreact.core.turns import Turn

    p = entry.get("payload") or {}
    return Turn(id=str(entry.get("turn_id")), index=int(p.get("index") or 0),
                text=str(p.get("text") or ""), source=str(p.get("source") or "term"),
                ts=float(entry.get("ts") or 0.0), resumed=resumed)


class DurableTurnStore:
    """A :class:`pyreact.core.turns.TurnStore` reading ``turn.*`` events
    journaled in ``log`` (read-only: the ``Durable`` journal writes them)."""

    def __init__(self, log: SessionLog) -> None:
        self.log = log

    # the journal already records turn.start / turn.end
    def on_turn_start(self, turn: Any) -> None:
        return None

    def on_turn_end(self, turn: Any, result: Dict[str, Any]) -> None:
        return None

    def _starts(self) -> "Dict[str, dict]":
        open_: Dict[str, dict] = {}
        for e in self.log.events(["turn.start", "turn.end"]):
            tid = e.get("turn_id")
            if tid is None:
                continue
            if e.get("kind") == "turn.start":
                open_.pop(tid, None)  # keep insertion order = latest start
                open_[tid] = e
            else:
                open_.pop(tid, None)
        return open_

    def pending(self) -> List[Any]:
        """Turns started and never ended on the current branch, oldest first."""
        if self.log.closed:
            return []
        return [_turn(e) for e in self._starts().values()]

    def get(self, turn_id: str) -> Optional[Any]:
        if self.log.closed:
            return None
        starts = [e for e in self.log.events("turn.start") if e.get("turn_id") == turn_id]
        return _turn(starts[-1]) if starts else None

    def __repr__(self) -> str:
        return f"<DurableTurnStore {self.log.path}>"
