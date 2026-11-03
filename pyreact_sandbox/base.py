"""The driver protocol: specs, results, sandboxes and drivers.

Stdlib only. A *driver* (Docker, local, E2B, Modal, ...) creates and attaches
:class:`Sandbox` objects from a :class:`SandboxSpec`. The API is async (the
pyreact runtime is asyncio); most vendor SDKs are blocking, so
:class:`BlockingSandbox` / :class:`BlockingDriver` let a driver implement
plain synchronous ``*_sync`` methods and get the async API for free (each call
runs in a worker thread via :func:`asyncio.to_thread`).

    driver = get_driver("docker")
    sb = await driver.create(SandboxSpec(image="python:3.12-slim"))
    r = await sb.exec("echo hi")             # ExecResult(exit_code=0, stdout="hi\\n", ...)
    ref = await sb.snapshot("after-setup")   # SnapshotRef (JSON-able)
    await sb.destroy()
    sb2 = await driver.create(spec, snapshot=ref)   # restore elsewhere / later
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import shlex
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "BlockingDriver",
    "BlockingSandbox",
    "Command",
    "ExecResult",
    "SandboxDriver",
    "SandboxError",
    "SandboxNotFound",
    "SandboxProcess",
    "SandboxSetupError",
    "Sandbox",
    "SandboxSpec",
    "SnapshotRef",
    "Status",
    "NotSupported",
    "command_name",
    "new_sandbox_id",
]

Command = Union[str, Sequence[str]]


class SandboxError(RuntimeError):
    """A sandbox operation failed (driver/transport error, not a non-zero exit)."""


class SandboxNotFound(SandboxError):
    """``attach`` / an operation referenced a sandbox that no longer exists."""


class SandboxSetupError(SandboxError):
    """A ``spec.setup`` command failed while bootstrapping."""

    def __init__(self, message: str, result: "ExecResult" = None) -> None:
        super().__init__(message)
        self.result = result


class NotSupported(SandboxError):
    """The driver does not implement this capability."""


class Status:
    """Sandbox states (plain strings)."""

    CREATING = "creating"
    RUNNING = "running"
    SUSPENDED = "suspended"
    STOPPED = "stopped"
    DESTROYED = "destroyed"
    UNKNOWN = "unknown"


def new_sandbox_id(prefix: str = "pyrsbx") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def command_name(cmd: Command) -> str:
    """Short name for a command (harness boundary name): argv[0]'s basename or
    the first word of a shell string."""
    if isinstance(cmd, str):
        try:
            parts = shlex.split(cmd)
        except ValueError:
            parts = cmd.split()
        first = parts[0] if parts else "sh"
    else:
        first = str(cmd[0]) if len(cmd) else "exec"
    return first.rsplit("/", 1)[-1] or "exec"


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SandboxSpec:
    """What to boot.

    - ``image``: base image (Docker / OCI ref; vendors map it to theirs).
    - ``template``: a named, pre-bootstrapped environment. Vendors: their
      template / snapshot id (E2B template, Daytona snapshot, Modal image).
      Docker: the tag name of the bootstrapped image
      (``pyreact-sandbox-tpl:<template>``) instead of the spec hash.
    - ``env``: environment for every exec.
    - ``cpu`` (cores, float), ``memory`` (``"512m"``, ``"2g"``).
    - ``network``: ``False`` (default: no network), ``True`` (driver default
      network) or a network name.
    - ``workdir``: working directory inside the sandbox (default ``/workspace``).
    - ``ttl``: seconds the sandbox may live; drivers stop it after that and
      :func:`pyreact_sandbox.gc` collects it.
    - ``labels``: extra metadata (Docker labels, vendor metadata).
    - ``setup``: shell commands run once to bootstrap; drivers cache the
      result by :meth:`setup_key` so the second boot is fast.
    - ``setup_network``: network during ``setup`` only (``pip install`` ...).
    """

    image: Optional[str] = "python:3.12-slim"
    template: Optional[str] = None
    env: Mapping[str, str] = field(default_factory=dict)
    cpu: Optional[float] = 1.0
    memory: Optional[str] = "512m"
    network: Union[bool, str] = False
    workdir: str = "/workspace"
    ttl: Optional[float] = None
    labels: Mapping[str, str] = field(default_factory=dict)
    setup: Tuple[str, ...] = ()
    setup_network: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.setup, str):
            object.__setattr__(self, "setup", (self.setup,))
        else:
            object.__setattr__(self, "setup", tuple(self.setup))

    def with_(self, **changes: Any) -> "SandboxSpec":
        """A copy with ``changes`` applied (``env`` / ``labels`` are merged)."""
        for name in ("env", "labels"):
            if name in changes and changes[name] is not None:
                changes[name] = {**getattr(self, name), **changes[name]}
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["env"], d["labels"], d["setup"] = dict(self.env), dict(self.labels), list(self.setup)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SandboxSpec":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def coerce(cls, spec: Union["SandboxSpec", Mapping[str, Any], None]) -> "SandboxSpec":
        if spec is None:
            return cls()
        if isinstance(spec, SandboxSpec):
            return spec
        return cls.from_dict(spec)

    def setup_key(self) -> str:
        """Hash of what the bootstrapped environment depends on (image, setup
        commands, setup env and network); the cache key for drivers."""
        data = json.dumps([self.image, list(self.setup), sorted(self.env.items()),
                           self.setup_network], sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """stdout followed by stderr (what a terminal would show)."""
        if self.stderr and self.stdout:
            sep = "" if self.stdout.endswith("\n") else "\n"
            return f"{self.stdout}{sep}{self.stderr}"
        return self.stdout or self.stderr

    def check(self) -> "ExecResult":
        """Raise :class:`SandboxError` unless the command succeeded."""
        if not self.ok:
            what = "timed out" if self.timed_out else f"exited {self.exit_code}"
            raise SandboxError(f"command {what}: {self.output[-2000:]}")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnapshotRef:
    """A restorable point-in-time copy of a sandbox. JSON-able (store it)."""

    driver: str
    id: str  # driver-specific (Docker: image tag; local: directory)
    sandbox_id: Optional[str] = None
    label: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["meta"] = dict(self.meta)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SnapshotRef":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _encode(data: Union[str, bytes, None]) -> Optional[bytes]:
    if data is None:
        return None
    return data.encode("utf-8") if isinstance(data, str) else bytes(data)


# ---------------------------------------------------------------------------
# Interactive processes (optional capability; used by SandboxInterpreter)
# ---------------------------------------------------------------------------
class SandboxProcess(abc.ABC):
    """A long-running process inside a sandbox with line-oriented, *blocking*
    stdin/stdout (bytes). Returned by :meth:`Sandbox.open_process_sync`."""

    @abc.abstractmethod
    def write(self, data: bytes) -> None: ...

    @abc.abstractmethod
    def readline(self) -> bytes:
        """One line from stdout (``b""`` at EOF)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close stdin and terminate the process."""

    def poll(self) -> Optional[int]:
        return None

    def read_stderr(self) -> str:
        return ""


class PopenProcess(SandboxProcess):
    """:class:`SandboxProcess` over a host ``subprocess.Popen`` (local
    subprocess, ``docker exec -i``, ``ssh``, ...)."""

    def __init__(self, popen: Any) -> None:
        self.popen = popen

    def write(self, data: bytes) -> None:
        self.popen.stdin.write(data)
        self.popen.stdin.flush()

    def readline(self) -> bytes:
        return self.popen.stdout.readline()

    def poll(self) -> Optional[int]:
        return self.popen.poll()

    def read_stderr(self) -> str:
        try:
            return (self.popen.stderr.read() or b"").decode("utf-8", "replace")
        except Exception:
            return ""

    def close(self) -> None:
        p = self.popen
        try:
            if p.stdin and not p.stdin.closed:
                p.stdin.close()
        except Exception:
            pass
        try:
            p.wait(timeout=3)
        except Exception:
            p.kill()
            try:
                p.wait(timeout=3)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
class Sandbox(abc.ABC):
    """One isolated execution environment. All methods are coroutines.

    Required: :meth:`exec`, :meth:`destroy`, :meth:`status`. The others have
    defaults built on :meth:`exec` (``run_python``, ``write_file``,
    ``read_file``) or raise :class:`NotSupported` (``snapshot``, ``restore``,
    ``suspend``, ``resume``, ``open_process_sync``).
    """

    id: str
    spec: SandboxSpec
    driver: "SandboxDriver"

    @abc.abstractmethod
    async def exec(self, cmd: Command, *, timeout: Optional[float] = None,
                   env: Optional[Mapping[str, str]] = None, cwd: Optional[str] = None,
                   stdin: Union[str, bytes, None] = None) -> ExecResult:
        """Run ``cmd`` (a shell string, or an argv list run without a shell)."""

    async def run_python(self, code: str, *, timeout: Optional[float] = None,
                         env: Optional[Mapping[str, str]] = None,
                         cwd: Optional[str] = None) -> ExecResult:
        """Run a Python snippet (``python3 -`` with the code on stdin)."""
        return await self.exec(["python3", "-"], timeout=timeout, env=env, cwd=cwd, stdin=code)

    async def write_file(self, path: str, data: Union[str, bytes]) -> None:
        r = await self.exec(["sh", "-c", 'mkdir -p "$(dirname "$1")" && cat > "$1"', "sh", path],
                            stdin=_encode(data))
        if not r.ok:
            raise SandboxError(f"write_file({path!r}) failed: {r.output}")

    async def read_file(self, path: str) -> bytes:
        r = await self.exec(["cat", path])
        if not r.ok:
            raise FileNotFoundError(f"{path}: {r.stderr.strip() or r.exit_code}")
        return r.stdout.encode("utf-8")

    async def snapshot(self, label: Optional[str] = None) -> SnapshotRef:
        raise NotSupported(f"{type(self).__name__} cannot snapshot")

    async def restore(self, ref: SnapshotRef) -> None:
        """Replace this sandbox's state with ``ref`` (same id). Drivers that
        cannot restore in place still support ``driver.create(spec, snapshot=ref)``."""
        raise NotSupported(f"{type(self).__name__} cannot restore in place")

    async def suspend(self) -> None:
        raise NotSupported(f"{type(self).__name__} cannot suspend")

    async def resume(self) -> None:
        raise NotSupported(f"{type(self).__name__} cannot resume")

    @abc.abstractmethod
    async def destroy(self) -> None: ...

    @abc.abstractmethod
    async def status(self) -> str: ...

    def open_process_sync(self, argv: Sequence[str], *, env: Optional[Mapping[str, str]] = None,
                          cwd: Optional[str] = None) -> SandboxProcess:
        """Start an interactive process (blocking API). Optional capability."""
        raise NotSupported(f"{type(self).__name__} has no interactive processes")

    async def __aenter__(self) -> "Sandbox":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.destroy()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {getattr(self, 'id', '?')}>"


class SandboxDriver(abc.ABC):
    """Creates and attaches sandboxes. ``name`` is what :class:`SnapshotRef`
    and stored records carry."""

    name: str = "driver"

    @abc.abstractmethod
    async def create(self, spec: SandboxSpec, *, snapshot: Optional[SnapshotRef] = None,
                     sandbox_id: Optional[str] = None) -> Sandbox:
        """Boot a sandbox from ``spec`` (bootstrapping ``spec.setup``), or from
        ``snapshot`` when given (restore after a restart)."""

    @abc.abstractmethod
    async def attach(self, sandbox_id: str) -> Sandbox:
        """Reconnect to a live (running or suspended) sandbox; raises
        :class:`SandboxNotFound` if it is gone."""

    async def list(self, labels: Optional[Mapping[str, str]] = None) -> List[str]:
        return []

    async def delete_snapshot(self, ref: SnapshotRef) -> None:
        return None

    def gc_sync(self, **kwargs: Any) -> Dict[str, List[str]]:
        """Remove orphaned / expired resources (see :func:`pyreact_sandbox.gc`)."""
        return {}

    def available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Blocking adapters: implement *_sync, get the async API
# ---------------------------------------------------------------------------
class BlockingSandbox(Sandbox):
    """Base for drivers built on blocking calls (CLIs, vendor SDKs).

    Implement ``exec_sync``, ``destroy_sync``, ``status_sync`` and optionally
    ``write_file_sync`` / ``read_file_sync`` / ``snapshot_sync`` /
    ``restore_sync`` / ``suspend_sync`` / ``resume_sync``. Every async method
    runs its sync twin in a worker thread.
    """

    def exec_sync(self, cmd: Command, *, timeout=None, env=None, cwd=None,
                  stdin=None) -> ExecResult:
        raise NotImplementedError

    def destroy_sync(self) -> None:
        raise NotImplementedError

    def status_sync(self) -> str:
        raise NotImplementedError

    def run_python_sync(self, code: str, *, timeout=None, env=None, cwd=None) -> ExecResult:
        return self.exec_sync(["python3", "-"], timeout=timeout, env=env, cwd=cwd, stdin=code)

    def write_file_sync(self, path: str, data: Union[str, bytes]) -> None:
        r = self.exec_sync(["sh", "-c", 'mkdir -p "$(dirname "$1")" && cat > "$1"', "sh", path],
                           stdin=_encode(data))
        if not r.ok:
            raise SandboxError(f"write_file({path!r}) failed: {r.output}")

    def read_file_sync(self, path: str) -> bytes:
        r = self.exec_sync(["cat", path])
        if not r.ok:
            raise FileNotFoundError(f"{path}: {r.stderr.strip() or r.exit_code}")
        return r.stdout.encode("utf-8")

    def snapshot_sync(self, label: Optional[str] = None) -> SnapshotRef:
        raise NotSupported(f"{type(self).__name__} cannot snapshot")

    def restore_sync(self, ref: SnapshotRef) -> None:
        raise NotSupported(f"{type(self).__name__} cannot restore in place")

    def suspend_sync(self) -> None:
        raise NotSupported(f"{type(self).__name__} cannot suspend")

    def resume_sync(self) -> None:
        raise NotSupported(f"{type(self).__name__} cannot resume")

    async def exec(self, cmd, *, timeout=None, env=None, cwd=None, stdin=None) -> ExecResult:
        return await asyncio.to_thread(self.exec_sync, cmd, timeout=timeout, env=env, cwd=cwd,
                                       stdin=stdin)

    async def run_python(self, code, *, timeout=None, env=None, cwd=None) -> ExecResult:
        return await asyncio.to_thread(self.run_python_sync, code, timeout=timeout, env=env,
                                       cwd=cwd)

    async def write_file(self, path, data) -> None:
        await asyncio.to_thread(self.write_file_sync, path, data)

    async def read_file(self, path) -> bytes:
        return await asyncio.to_thread(self.read_file_sync, path)

    async def snapshot(self, label=None) -> SnapshotRef:
        return await asyncio.to_thread(self.snapshot_sync, label)

    async def restore(self, ref) -> None:
        await asyncio.to_thread(self.restore_sync, ref)

    async def suspend(self) -> None:
        await asyncio.to_thread(self.suspend_sync)

    async def resume(self) -> None:
        await asyncio.to_thread(self.resume_sync)

    async def destroy(self) -> None:
        await asyncio.to_thread(self.destroy_sync)

    async def status(self) -> str:
        return await asyncio.to_thread(self.status_sync)


class BlockingDriver(SandboxDriver):
    """Driver twin of :class:`BlockingSandbox`: implement ``create_sync`` and
    ``attach_sync`` (and optionally ``list_sync`` / ``delete_snapshot_sync``)."""

    def create_sync(self, spec: SandboxSpec, *, snapshot=None, sandbox_id=None) -> Sandbox:
        raise NotImplementedError

    def attach_sync(self, sandbox_id: str) -> Sandbox:
        raise NotImplementedError

    def list_sync(self, labels=None) -> List[str]:
        return []

    def delete_snapshot_sync(self, ref: SnapshotRef) -> None:
        return None

    async def create(self, spec, *, snapshot=None, sandbox_id=None) -> Sandbox:
        return await asyncio.to_thread(self.create_sync, SandboxSpec.coerce(spec),
                                       snapshot=snapshot, sandbox_id=sandbox_id)

    async def attach(self, sandbox_id) -> Sandbox:
        return await asyncio.to_thread(self.attach_sync, sandbox_id)

    async def list(self, labels=None) -> List[str]:
        return await asyncio.to_thread(self.list_sync, labels)

    async def delete_snapshot(self, ref) -> None:
        await asyncio.to_thread(self.delete_snapshot_sync, ref)
