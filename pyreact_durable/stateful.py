"""Deprecated: ``DurableParticipant`` (kept for one release).

:func:`~pyreact_durable.Durable` now dehydrates its own state through the
core's ``use_dehydrate`` (ADR 0001, ``docs/adr/0001-look-like-react.md``):
its slice in a snapshot is the reference to its log::

    "slices": {"App/Durable[durable]": {"log": {
        "path": "/abs/runs/agent.jsonl", "log_id": "<header id>", "leaf": "a1b2c3d4",
        "entries": 42, "version": 1, "pending_turns": []}}}

and ``hydrate_root`` reopens the log at that leaf before the children's
first render. So a root carries it with no participant::

    root = create_root(App, session_id="agent", store=DirectorySnapshots("state/"))
    root.suspend()
    root = hydrate_root(App, session_id="agent", store=DirectorySnapshots("state/"))

The legacy ``Session.suspend`` manifest carries the same slices, so this
participant has nothing left to do: its fragment is empty. It still reads
the ``logs`` fragment of a manifest written by an older pyreact (the leaf
is put back on resume).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict

__all__ = ["DurableParticipant"]

# Must match pyreact_durable.react (kept here so this module loads without pyreact).
RESTORE_SERVICE = "pyreact_durable.restore"

_ADR = "docs/adr/0001-look-like-react.md"


class DurableParticipant:
    """Deprecated (``name = "durable"``): ``Durable`` dehydrates itself."""

    name = "durable"
    version = "2"

    def __init__(self, session: Any = None) -> None:
        warnings.warn(
            "DurableParticipant is deprecated: Durable dehydrates its log reference "
            "itself (use_dehydrate); suspend with root.suspend() and resume with "
            f"hydrate_root(...), no participants needed (see {_ADR})",
            DeprecationWarning, stacklevel=2)
        self.session = session

    async def suspend(self, session: Any) -> Dict[str, Any]:
        return {}  # the log reference rides in the manifest's "slices"

    async def resume(self, session: Any, fragment: Dict[str, Any]) -> None:
        # a manifest from an older pyreact: no Durable slice, the leaf is here
        restore = None
        for rec in fragment.get("logs") or []:
            if restore is None:
                restore = session.services.setdefault(RESTORE_SERVICE, {})
            path = Path(rec.get("path") or "")
            if path.exists():
                restore[str(path.resolve())] = {"leaf": rec.get("leaf")}
            else:
                session.degrade(self.name, "log missing", path=str(path))

    def __repr__(self) -> str:
        return "<DurableParticipant (deprecated)>"
