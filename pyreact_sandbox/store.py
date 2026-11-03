"""Snapshot stores: where a sandbox's restart record lives.

A store maps a key (``"<session>/<sandbox key>"``) to a small JSON record so
a restarted process can find its sandbox again::

    {"driver": "docker", "sandbox_id": "pyrsbx-1a2b...",       # reattach if alive
     "snapshot": {...SnapshotRef.to_dict()} | None,             # else restore from it
     "spec": {...}, "updated_at": 1793680801.1}

The interface is three methods; anything that can persist JSON works (a
file, Redis, a database row, the pyreact_durable session log — see
``pyreact_sandbox.contrib.durable``). Calls may come from a worker thread.
"""

from __future__ import annotations

import abc
import copy
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["SnapshotStore", "MemorySnapshotStore", "FileSnapshotStore"]

Record = Dict[str, Any]


class SnapshotStore(abc.ABC):
    """get / set / delete a JSON record by key."""

    @abc.abstractmethod
    def get(self, key: str) -> Optional[Record]: ...

    @abc.abstractmethod
    def set(self, key: str, record: Record) -> None: ...

    def delete(self, key: str) -> None:
        return None


class MemorySnapshotStore(SnapshotStore):
    """In-process dict (default; does not survive restarts by itself)."""

    def __init__(self) -> None:
        self._data: Dict[str, Record] = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            rec = self._data.get(key)
            return copy.deepcopy(rec) if rec is not None else None

    def set(self, key, record):
        with self._lock:
            self._data[key] = copy.deepcopy(record)

    def delete(self, key):
        with self._lock:
            self._data.pop(key, None)

    def __len__(self) -> int:
        return len(self._data)


class FileSnapshotStore(SnapshotStore):
    """One JSON file (``{key: record}``), rewritten atomically on each set."""

    def __init__(self, path: os.PathLike) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, Record]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except ValueError:
            return {}

    def _save(self, data: Dict[str, Record]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, key):
        with self._lock:
            return self._load().get(key)

    def set(self, key, record):
        with self._lock:
            data = self._load()
            data[key] = record
            self._save(data)

    def delete(self, key):
        with self._lock:
            data = self._load()
            if data.pop(key, None) is not None:
                self._save(data)
