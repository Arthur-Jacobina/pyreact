"""Checkpoint participant: a session's durable logs in its suspend manifest.

The core's stateful sessions (``Session.suspend`` / ``AppRunner.resume``,
``pyreact.boot.stateful``) ask every registered participant for a JSON
fragment. This one records, for each :func:`~pyreact_durable.Durable`
provider mounted in the session, *where its log is and where its leaf was*::

    "durable": {"logs": [{"path": "/abs/runs/agent.jsonl", "log_id": "<header id>",
                          "leaf": "a1b2c3d4", "entries": 42, "version": 1,
                          "pending_turns": []}]}

On resume (before the first render) it checks each log is still there and
is the same log (header id), and tells the ``Durable`` provider that opens it
to put the leaf back where it was. So persistent state and the conversation
restore exactly as suspended, the ``DurableTurnStore`` finds turns a crash cut
short (``pending_turns``; ``AppRunner.resume`` re-runs them), and
``durable_lm`` replays the LLM calls a re-run turn had already made.

    app = AppRunner(App, participants=[DurableParticipant],
                    manifest_store=DirectoryManifestStore("runs/manifests"))

Failure semantics: a missing or replaced log, or a leaf that is no longer in
it, is reported with ``session.degrade("durable", reason, path=...)``
(``session.resume.degraded``); the provider then opens the file as usual
(a fresh log if it is gone). The log itself is never modified at suspend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .log import SessionLog

__all__ = ["DurableParticipant"]

# Must match pyreact_durable.react (kept here so this module loads without pyreact).
LOGS_SERVICE = "pyreact_durable.logs"
RESTORE_SERVICE = "pyreact_durable.restore"


class DurableParticipant:
    """``name = "durable"``. Construct with the session (a participant
    factory) or without arguments."""

    name = "durable"
    version = "1"

    def __init__(self, session: Any = None) -> None:
        self.session = session

    async def suspend(self, session: Any) -> Dict[str, Any]:
        logs: List[Dict[str, Any]] = []
        for key, slog in list((session.services.get(LOGS_SERVICE) or {}).items()):
            if slog.closed:
                continue
            pending: List[str] = []
            store = getattr(session.turns, "store", None)
            if getattr(store, "log", None) is slog:
                pending = [t.id for t in store.pending()]
            logs.append({"path": key, "log_id": slog.id, "leaf": slog.leaf,
                         "entries": len(slog.entries), "version": slog.version,
                         "pending_turns": pending})
        return {"logs": logs}

    async def resume(self, session: Any, fragment: Dict[str, Any]) -> None:
        restore = session.services.setdefault(RESTORE_SERVICE, {})
        for rec in fragment.get("logs") or []:
            path = Path(rec.get("path") or "")
            if not path.exists():
                session.degrade(self.name, "log missing", path=str(path))
                continue
            try:
                log = SessionLog.open(path, readonly=True)
            except Exception as exc:
                session.degrade(self.name, "log unreadable", path=str(path),
                                error=f"{type(exc).__name__}: {exc}")
                continue
            try:
                if rec.get("log_id") and log.id != rec["log_id"]:
                    session.degrade(self.name, "log replaced", path=str(path),
                                    expected=rec["log_id"], found=log.id)
                    continue
                leaf = rec.get("leaf")
                if leaf and leaf not in log.by_id:
                    session.degrade(self.name, "leaf missing", path=str(path), leaf=leaf)
                    leaf = None
            finally:
                log.close()
            restore[str(path.resolve())] = {"leaf": leaf}

    def __repr__(self) -> str:
        return "<DurableParticipant>"
