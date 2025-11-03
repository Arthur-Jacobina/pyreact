from __future__ import annotations

import sys

from pyreact.core import MessageBuffer


__all__ = [
    "enable_web_print",
    "disable_web_print",
]


# -----------------------------------------------------------------------------
# Redirection of stdout/stderr
# -----------------------------------------------------------------------------
_original_stdout = None
_original_stderr = None
_patched = False


class _WebStream:
    """
    File-like that duplicates writes:
      - optionally echoes to the original stream (for server logs)
      - always writes to the MessageBuffer singleton
    """

    def __init__(self, console: MessageBuffer, original) -> None:
        self._console = console
        self._original = original
        self.encoding = getattr(original, "encoding", "utf-8")

    def write(self, s: str) -> int:
        if self._original is not None:
            self._original.write(s)
            self._original.flush()

        if self._console is not None:
            self._console.append(s)

        return len(s)

    def flush(self) -> None:
        if self._original is not None:
            self._original.flush()


def enable_web_print() -> None:
    """
    Redirects `sys.stdout` and `sys.stderr` to the MessageBuffer singleton,
    with an option to also write to the server's original streams.

    - `echo_to_server_stdout=True`: keeps logs in the terminal/uvicorn.
    - `max_chars`: optionally changes the ring buffer size at this time.

    Multiple calls are idempotent.
    """
    global _original_stdout, _original_stderr, _patched

    console = MessageBuffer()

    _original_stdout = sys.stdout
    _original_stderr = sys.stderr

    sys.stdout = _WebStream(console, _original_stdout)  # type: ignore[assignment]
    sys.stderr = _WebStream(console, _original_stderr)  # type: ignore[assignment]

    _patched = True


def disable_web_print() -> None:
    """
    Restores the original `sys.stdout` and `sys.stderr`.
    No-op if already disabled.
    """
    global _original_stdout, _original_stderr, _patched

    if not _patched:
        return

    try:
        if _original_stdout is not None:
            sys.stdout = _original_stdout  # type: ignore[assignment]
    except Exception:
        pass

    try:
        if _original_stderr is not None:
            sys.stderr = _original_stderr  # type: ignore[assignment]
    except Exception:
        pass

    _original_stdout = None
    _original_stderr = None
    _patched = False
