"""Adapter: keep sandbox restart records in a ``pyreact_durable`` session log.

Neither library imports the other; this module imports both, lazily, and is
only loaded if you use it::

    from pyreact_sandbox.contrib.durable import DurableSnapshotStore, use_durable_snapshot_store

    @component
    def Agent():
        store = use_durable_snapshot_store()        # bound to the nearest Durable log
        sb = use_sandbox(store=store, autosnapshot="turn")
        ...

    Durable(log="runs/agent.jsonl", children=[
        SandboxProvider(driver="docker", spec=spec, children=[Agent(key="agent")])])

Records are ``state`` entries at path ``"pyreact_sandbox"`` (key = the store
key), so they live on the log's *tree*: ``checkout`` a label and the record
— hence the snapshot the sandbox restores from — is the one at that point.
Branching the conversation branches the sandbox too.

Ordering note: on app shutdown ``Durable`` (the parent) closes its log before
the sandbox's close policy runs, so records written *at close*
(``on_close="snapshot"``) are dropped with a warning. Use
``autosnapshot="turn"`` (a snapshot after every turn that touched the
sandbox, written while the log is open) with this store.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

from ..store import Record, SnapshotStore

__all__ = ["DurableSnapshotStore", "use_durable_snapshot_store", "STATE_PATH"]

STATE_PATH = "pyreact_sandbox"


class DurableSnapshotStore(SnapshotStore):
    """SnapshotStore over a ``pyreact_durable.SessionLog`` (state entries)."""

    def __init__(self, log: Any, path: str = STATE_PATH) -> None:
        self.log = log
        self.path = path

    def get(self, key: str) -> Optional[Record]:
        if self.log is None or getattr(self.log, "closed", False):
            return None
        rec = self.log.get_state(self.path, key, None)
        return dict(rec) if rec else None

    def set(self, key: str, record: Record) -> None:
        if self.log is None or getattr(self.log, "closed", False):
            warnings.warn(f"DurableSnapshotStore: log closed, sandbox record {key!r} not "
                          "saved (use autosnapshot='turn')", RuntimeWarning, stacklevel=2)
            return
        self.log.set_state(self.path, key, record)

    def delete(self, key: str) -> None:
        if self.log is not None and not getattr(self.log, "closed", False):
            self.log.delete_state(self.path, key)


def use_durable_snapshot_store(path: str = STATE_PATH) -> Optional[DurableSnapshotStore]:
    """Hook: a :class:`DurableSnapshotStore` on the nearest ``Durable`` log
    (``None`` without one, so ``use_sandbox(store=None)`` falls back to the
    provider's store)."""
    from pyreact.core import hooks
    from pyreact_durable import use_durable

    log = use_durable()
    return hooks.use_memo(lambda: DurableSnapshotStore(log, path) if log is not None else None,
                          [id(log), path])
