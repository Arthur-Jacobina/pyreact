"""Deprecated: ``SandboxParticipant`` (kept for one release).

:func:`~pyreact_sandbox.use_sandbox` / :func:`~pyreact_sandbox.SandboxProvider`
now dehydrate their own state through the core's ``use_dehydrate`` (ADR 0001,
``docs/adr/0001-look-like-react.md``): on ``root.suspend()`` the effect
cleanup (``use_cleanup_reason() == "suspend"``) snapshots or pauses each
sandbox per ``on_close`` (``resolve_policy(on_close, "suspend")``: never an
implicit destroy) and the provider's slice records how to bring it back::

    "slices": {"App/SandboxProvider[sandbox]": {"sandboxes": {
        "default": {"driver": "docker", "spec": {...}, "sandbox_id": null,
                    "snapshot": {...SnapshotRef...}}}}}

``hydrate_root`` restores from it (a missing snapshot boots a fresh sandbox
and reports a ``hydration.mismatch`` + ``session.resume.degraded``).

This participant still works with ``Session.suspend`` / ``AppRunner.resume``:
``mode`` -- ``"snapshot"`` (snapshot, then destroy), ``"suspend"`` (pause and
keep) or ``"keep"`` -- overrides the handles' suspend policy, and its
fragment lists the sandboxes as before.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any, Dict, List

from .react import HANDLES_SERVICE, RESTORE_SERVICE

__all__ = ["SandboxParticipant", "MODES"]

MODES = ("snapshot", "suspend", "keep")
_POLICY = {"snapshot": "destroy", "suspend": "suspend", "keep": "keep"}


class SandboxParticipant:
    """Deprecated. ``name = "sandbox"``. ``SandboxParticipant(session=None, mode="snapshot")``."""

    name = "sandbox"
    version = "1"

    def __init__(self, session: Any = None, *, mode: str = "snapshot",
                 close_timeout: float = 120.0) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        warnings.warn(
            "SandboxParticipant is deprecated: use_sandbox / SandboxProvider dehydrate "
            "their snapshot ref themselves (use_dehydrate); suspend with root.suspend() "
            "and set the policy with on_close={'suspend': ...}; no participants needed "
            "(see docs/adr/0001-look-like-react.md)", DeprecationWarning, stacklevel=2)
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
        return f"<SandboxParticipant mode={self.mode!r} (deprecated)>"
