"""Small shared helpers (stdlib only)."""

from __future__ import annotations

import os
import re
import socket
from typing import Mapping, Optional


def owner_tag() -> str:
    """``host:pid`` of this process (who created a sandbox; used by gc)."""
    return f"{socket.gethostname()}:{os.getpid()}"


def owner_alive(tag: Optional[str]) -> bool:
    """False only when ``tag`` names a dead process on *this* host."""
    if not tag or ":" not in tag:
        return True
    host, _, pid = tag.rpartition(":")
    if host != socket.gethostname():
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError):
        return True
    return True


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_tag(text: str, limit: int = 60) -> str:
    """A Docker-tag / filename-safe slug."""
    return (_SAFE.sub("-", text).strip("-.") or "x")[:limit]


def labels_match(have: Mapping[str, str], want: Optional[Mapping[str, str]]) -> bool:
    return not want or all(have.get(k) == str(v) for k, v in want.items())


def truncate(text: str, limit: int) -> str:
    if limit and len(text) > limit:
        cut = len(text) - limit
        return text[: limit // 2] + f"\n... [{cut} chars truncated] ...\n" + text[-limit // 2:]
    return text
