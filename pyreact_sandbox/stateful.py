"""Checkpoint participant: a session's sandboxes in its suspend manifest.

The core's stateful sessions (``Session.suspend`` / ``AppRunner.resume``,
``pyreact.boot.stateful``) ask every registered participant for a JSON
fragment. This one covers every sandbox handle the session created with
:func:`~pyreact_sandbox.use_sandbox`::

    "sandbox": {"mode": "snapshot", "sandboxes": [
        {"key": "default", "store_key": "agent-1/default", "driver": "docker",
         "sandbox_id": null, "snapshot": {...SnapshotRef...}, "spec": {...}}]}

``mode`` -- what a suspend does to a running sandbox:

- ``"snapshot"`` (default): snapshot it (``sandbox.snapshot``), then destroy it
  when the tree unmounts. Resume boots a new sandbox from the snapshot.
  Survives a machine restart; costs a snapshot (``docker commit``).
- ``"suspend"``: pause it (``docker pause``) and keep it. Resume reattaches
  (and unpauses) -- fast, same process tree -- and falls back to the last
  snapshot it has, if any.
- ``"keep"``: leave it running; resume reattaches.

On resume (before the first render) the records are handed to the handles
``use_sandbox`` creates for the same keys; they win over the provider's
``SnapshotStore``. Failure semantics: if the snapshot is gone (or the
sandbox, or the driver changed), a **fresh sandbox** boots and
``session.resume.degraded`` is published (``participant: "sandbox"``,
``reason: "snapshot missing" | "restore failed" | "sandbox gone" |
"driver changed"``) -- the agent goes on, without its files.

    from pyreact_sandbox import SandboxParticipant
    app = AppRunner(App, participants=[DurableParticipant, SandboxParticipant])
    # or a mode: participants=[lambda s: SandboxParticipant(s, mode="suspend")]

Sandbox closes are asynchronous (a private worker loop); ``suspended()``
waits for them, so when ``session.suspend()`` returns the sandboxes are in
their final state.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from .react import HANDLES_SERVICE, RESTORE_SERVICE

__all__ = ["SandboxParticipant", "MODES"]

MODES = ("snapshot", "suspend", "keep")
_POLICY = {"snapshot": "destroy", "suspend": "suspend", "keep": "keep"}


class SandboxParticipant:
    """``name = "sandbox"``. ``SandboxParticipant(session=None, mode="snapshot")``."""

    name = "sandbox"
    version = "1"

    def __init__(self, session: Any = None, *, mode: str = "snapshot",
                 close_timeout: float = 120.0) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.session = session
        self.mode = mode
        self.close_timeout = close_timeout
        self._handles: List[Any] = []

    async def suspend(self, session: Any) -> Dict[str, Any]:
        entries: List[Dict[str, Any]] = []
        self._handles = []
        for store_key, h in list((session.services.get(HANDLES_SERVICE) or {}).items()):
            if h.closed:
                continue
            base = {"key": h.key, "store_key": store_key, "driver": h.driver.name,
                    "spec": h.spec.to_dict()}
            if h.sandbox is None:
                # never booted in this run (lazy): carry its pending record over
                rec = h.restore_record or (h.store.get(store_key) if h.store else None)
                if rec:
                    entries.append({**base, "sandbox_id": rec.get("sandbox_id"),
                                    "snapshot": rec.get("snapshot")})
                continue
            snapshot = h.last_snapshot.to_dict() if h.last_snapshot is not None else None
            if self.mode == "snapshot":
                ref = await h.snapshot("suspend")
                entries.append({**base, "sandbox_id": None, "snapshot": ref.to_dict()})
            else:
                entries.append({**base, "sandbox_id": h.id, "snapshot": snapshot})
            h.suspend_policy = _POLICY[self.mode]
            self._handles.append(h)
        return {"mode": self.mode, "sandboxes": entries}

    async def suspended(self, session: Any) -> None:
        """After unmount: wait until the handles' close policies are applied."""
        handles, self._handles = self._handles, []
        futs = [asyncio.wrap_future(h.close_future) for h in handles
                if h.close_future is not None]
        if futs:
            await asyncio.wait_for(asyncio.gather(*futs, return_exceptions=True),
                                   self.close_timeout)

    async def resume(self, session: Any, fragment: Dict[str, Any]) -> None:
        restore = session.services.setdefault(RESTORE_SERVICE, {})
        for e in fragment.get("sandboxes") or []:
            restore[e["store_key"]] = {"driver": e.get("driver"),
                                       "sandbox_id": e.get("sandbox_id"),
                                       "snapshot": e.get("snapshot"), "spec": e.get("spec")}

    def __repr__(self) -> str:
        return f"<SandboxParticipant mode={self.mode!r}>"
