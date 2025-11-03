from collections import deque
from threading import RLock
from typing import Callable, Deque, List, Optional


class MessageBuffer:
    """
    Console buffer with a character-count ring buffer.

    - `append(text)` adds text to the end
    - `dump()` returns the entire current content (concatenation of chunks).
    - `subscribe(cb)`/`unsubscribe(cb)` register callbacks invoked on each append.
    - Implemented as a SINGLETON to be shared between the server and the hook
      that intercepts stdout/stderr.
    """

    _instance: Optional["MessageBuffer"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._chunks: Deque[str] = deque()
        self._length: int = 0
        self._subs: List[Callable[[str], None]] = []
        self._lock: RLock = RLock()

        self._initialized = True

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._chunks.append(text)
            self._length += len(text)

        for cb in list(self._subs):
            try:
                cb(text)
            except Exception:
                pass

    # ---------------- Public API ----------------
    def dump(self) -> str:
        with self._lock:
            return "".join(self._chunks)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._length = 0

    def length(self) -> int:
        with self._lock:
            return self._length

    def subscribe(self, cb: Callable[[str], None]) -> None:
        with self._lock:
            if cb not in self._subs:
                self._subs.append(cb)

    def unsubscribe(self, cb: Callable[[str], None]) -> None:
        with self._lock:
            try:
                self._subs.remove(cb)
            except ValueError:
                pass
