"""Snapshot stores: where ``root.suspend()`` puts a dehydrated tree and
``hydrate_root`` finds it, keyed by session id.

Any object with ``save / load / delete / list`` (the :class:`SnapshotStore`
protocol) works: a database, an object store... The core ships two:

- :class:`MemorySnapshots`: in-process (suspend / hydrate within one process);
- :class:`DirectorySnapshots`: one ``<dir>/<session id>.json`` per session
  (id percent-encoded), written atomically; survives restarts and is
  readable by any process.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable
from urllib.parse import quote, unquote

from pyreact.core.dehydrate import Snapshot, check_snapshot

__all__ = ["SnapshotStore", "MemorySnapshots", "DirectorySnapshots", "load_snapshot"]


@runtime_checkable
class SnapshotStore(Protocol):
    """Persists snapshots by ``snapshot["session_id"]``.

    ``save`` returns a reference (a path, a key...) published in the
    ``session.suspended`` event; ``load`` returns ``None`` when there is no
    snapshot for the session.
    """

    def save(self, snapshot: Snapshot) -> str: ...

    def load(self, session_id: str) -> Optional[Snapshot]: ...

    def delete(self, session_id: str) -> None: ...

    def list(self) -> List[str]: ...


def _session_id(snapshot: Dict[str, Any]) -> str:
    sid = snapshot.get("session_id")
    if not isinstance(sid, str) or not sid:
        raise ValueError("a snapshot needs a session_id to be stored")
    return sid


class MemorySnapshots:
    """In-process snapshot store (deep copies in and out)."""

    def __init__(self) -> None:
        self._data: Dict[str, Snapshot] = {}
        self._lock = threading.Lock()

    def save(self, snapshot: Snapshot) -> str:
        sid = _session_id(snapshot)
        with self._lock:
            self._data[sid] = copy.deepcopy(snapshot)
        return f"memory:{sid}"

    def load(self, session_id: str) -> Optional[Snapshot]:
        with self._lock:
            snap = self._data.get(session_id)
            return copy.deepcopy(snap) if snap is not None else None

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def list(self) -> List[str]:
        with self._lock:
            return sorted(self._data)

    def __contains__(self, session_id: object) -> bool:
        with self._lock:
            return session_id in self._data

    def __repr__(self) -> str:
        return f"<MemorySnapshots {len(self._data)}>"


class DirectorySnapshots:
    """One JSON file per session in ``directory``, written atomically
    (temp file in the same directory, fsync, rename)."""

    def __init__(self, directory: Union[str, "os.PathLike[str]"]) -> None:
        self.dir = Path(directory)

    def path_for(self, session_id: str) -> Path:
        return self.dir / f"{quote(str(session_id), safe='')}.json"

    def save(self, snapshot: Snapshot) -> str:
        path = self.path_for(_session_id(snapshot))
        data = json.dumps(snapshot, indent=2, allow_nan=False)
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=f".{path.stem}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return str(path)

    def load(self, session_id: str) -> Optional[Snapshot]:
        try:
            text = self.path_for(session_id).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        return json.loads(text)

    def delete(self, session_id: str) -> None:
        try:
            self.path_for(session_id).unlink()
        except FileNotFoundError:
            pass

    def list(self) -> List[str]:
        if not self.dir.is_dir():
            return []
        return sorted(unquote(p.stem) for p in self.dir.glob("*.json"))

    def __contains__(self, session_id: object) -> bool:
        return isinstance(session_id, str) and self.path_for(session_id).exists()

    def __repr__(self) -> str:
        return f"<DirectorySnapshots {self.dir}>"


def load_snapshot(path: Union[str, "os.PathLike[str]"]) -> Snapshot:
    """Read and validate a snapshot JSON file."""
    with open(path, encoding="utf-8") as f:
        return check_snapshot(json.load(f))
