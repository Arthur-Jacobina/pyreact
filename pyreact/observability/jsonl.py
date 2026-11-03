"""Append every event as one JSON line, rotating by size."""

from __future__ import annotations

import json
import os
import threading
from typing import Iterable, Optional

from pyreact.core.events import Event

from ._base import Sink


class JSONLSink(Sink):
    """Writes ``Event.to_dict()`` lines to ``path`` (append mode).

    When the file would exceed ``max_bytes`` it is rotated to ``path.1``
    (``path.1`` -> ``path.2`` ... up to ``backups``; the oldest is dropped).
    ``max_bytes=0`` disables rotation. Non-JSON payload values are stringified.
    """

    def __init__(
        self,
        path: str,
        *,
        kinds: Optional[Iterable[str]] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backups: int = 3,
    ) -> None:
        super().__init__(kinds)
        self.path = os.fspath(path)
        self.max_bytes = int(max_bytes)
        self.backups = max(0, int(backups))
        self._lock = threading.Lock()
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def _rotate(self) -> None:
        self._fh.close()
        if self.backups == 0:
            os.remove(self.path)
        else:
            for i in range(self.backups - 1, 0, -1):
                src = f"{self.path}.{i}"
                if os.path.exists(src):
                    os.replace(src, f"{self.path}.{i + 1}")
            os.replace(self.path, f"{self.path}.1")
        self._fh = open(self.path, "a", encoding="utf-8")

    def handle(self, ev: Event) -> None:
        line = json.dumps(ev.to_dict(), default=str, ensure_ascii=False) + "\n"
        data = line.encode("utf-8")
        with self._lock:
            if self._fh.closed:
                return
            if self.max_bytes > 0:
                size = self._fh.tell()
                if size > 0 and size + len(data) > self.max_bytes:
                    self._rotate()
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        self.detach()
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "JSONLSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: str) -> list:
    """Load events written by :class:`JSONLSink` (as dicts)."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
