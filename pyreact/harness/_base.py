"""Shared base of the built-in guard / verify middleware."""

from __future__ import annotations

import inspect
from typing import Any, Iterable, Optional, Tuple, Union

Kinds = Union[str, Iterable[str], None]


def norm_kinds(kinds: Kinds) -> Optional[Tuple[str, ...]]:
    if kinds is None:
        return None
    return (kinds,) if isinstance(kinds, str) else tuple(kinds)


async def resolve(value: Any) -> Any:
    """``await`` ``value`` if it is awaitable (sync-or-async callbacks)."""
    return await value if inspect.isawaitable(value) else value


class HarnessMiddleware:
    """A middleware object with ``kinds`` (the harness kind filter) and
    ``harness_name`` (shown in ``Blocked`` messages and events).
    ``use_guard`` / ``use_verify`` register such objects as they are."""

    __pyreact_harness__ = True
    default_kinds: Optional[Tuple[str, ...]] = None

    def __init__(self, *, kinds: Kinds = None, name: Optional[str] = None) -> None:
        self.kinds = norm_kinds(kinds) if kinds is not None else self.default_kinds
        self.harness_name = name or type(self).__name__

    @property
    def __name__(self) -> str:  # so logs / _name(mw) read naturally
        return self.harness_name

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.harness_name!r} kinds={self.kinds}>"
