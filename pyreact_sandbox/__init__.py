"""pyreact_sandbox: pluggable, bootstrappable sandboxes for pyreact agents.

A separate library at the border of the flow (like ``pyreact_durable``):
nothing in ``pyreact`` imports it.

Layers (``import pyreact_sandbox`` loads only the stdlib-only core; the rest
is resolved lazily on first access):

- core: :class:`SandboxSpec`, :class:`Sandbox`, :class:`SandboxDriver`,
  :class:`ExecResult`, :class:`SnapshotRef`, ``register_driver`` /
  ``get_driver`` / :func:`gc`, :class:`LocalDriver` (NOT isolated),
  :class:`DockerDriver`, snapshot stores.
- React border (``pyreact_sandbox.react``): ``SandboxProvider``,
  ``use_sandbox``, :class:`SandboxHandle`.
- stateful sessions: ``SandboxParticipant`` (sandboxes in a session's
  suspend manifest; ``pyreact_sandbox.stateful``).
- integrations: ``SandboxInterpreter`` (dspy ``CodeInterpreter`` for RLM),
  ``sandbox_tools(handle)`` (tools for ``use_agent`` / ``ToolProvider``).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import (
    BlockingDriver,
    BlockingSandbox,
    ExecResult,
    NotSupported,
    Sandbox,
    SandboxDriver,
    SandboxError,
    SandboxNotFound,
    SandboxProcess,
    SandboxSetupError,
    SandboxSpec,
    SnapshotRef,
    Status,
)
from .registry import gc, get_driver, list_drivers, register_driver, unregister_driver

__all__ = [
    "BlockingDriver",
    "BlockingSandbox",
    "DockerDriver",
    "ExecResult",
    "FileSnapshotStore",
    "LocalDriver",
    "MemorySnapshotStore",
    "NotSupported",
    "Sandbox",
    "SandboxDriver",
    "SandboxError",
    "SandboxHandle",
    "SandboxInterpreter",
    "SandboxNotFound",
    "SandboxParticipant",
    "SandboxProcess",
    "SandboxProvider",
    "SandboxSetupError",
    "SandboxSpec",
    "SnapshotRef",
    "SnapshotStore",
    "Status",
    "gc",
    "get_driver",
    "list_drivers",
    "register_driver",
    "sandbox_tools",
    "unregister_driver",
    "use_sandbox",
    "wait_closed",
]

_LAZY = {
    "DockerDriver": ".docker",
    "LocalDriver": ".local",
    "FileSnapshotStore": ".store",
    "MemorySnapshotStore": ".store",
    "SnapshotStore": ".store",
    "SandboxProvider": ".react",
    "SandboxHandle": ".react",
    "use_sandbox": ".react",
    "wait_closed": ".react",
    "SandboxInterpreter": ".interpreter",
    "sandbox_tools": ".tools",
    "SandboxParticipant": ".stateful",
}


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'pyreact_sandbox' has no attribute {name!r}")
    value = getattr(import_module(mod, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
