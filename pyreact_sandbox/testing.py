"""``FakeDriver``: an in-memory driver for tests (no processes, no Docker).

Files live in a dict; ``exec`` understands a tiny shell (``echo``, ``cat``,
``ls``, ``exit N``) and records every call in ``driver.calls``. Sandboxes
outlive their Python objects (they live in the driver, like containers in a
daemon), so ``attach`` after a simulated restart works when the same driver
object is reused.
"""

from __future__ import annotations

import copy
import shlex
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    ExecResult,
    Sandbox,
    SandboxDriver,
    SandboxError,
    SandboxNotFound,
    SandboxSpec,
    SnapshotRef,
    Status,
    new_sandbox_id,
)

__all__ = ["FakeDriver", "FakeSandbox"]


class FakeSandbox(Sandbox):
    def __init__(self, driver: "FakeDriver", sandbox_id: str, spec: SandboxSpec) -> None:
        self.driver = driver
        self.id = sandbox_id
        self.spec = spec

    @property
    def _state(self) -> Dict[str, Any]:
        st = self.driver.boxes.get(self.id)
        if st is None:
            raise SandboxNotFound(self.id)
        return st

    def _log(self, op: str, *args: Any) -> None:
        self.driver.calls.append((op, self.id, *args))

    async def exec(self, cmd, *, timeout=None, env=None, cwd=None, stdin=None) -> ExecResult:
        self._log("exec", cmd)
        st = self._state
        if st["status"] != Status.RUNNING:
            raise SandboxError(f"sandbox {self.id} is {st['status']}")
        argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
        files = st["files"]
        if not argv:
            return ExecResult(0)
        head, rest = argv[0], argv[1:]
        if head == "echo":
            return ExecResult(0, " ".join(rest) + "\n", "", 0.001)
        if head == "cat":
            if stdin is not None:
                return ExecResult(0, stdin if isinstance(stdin, str) else stdin.decode())
            missing = [p for p in rest if p not in files]
            if missing:
                return ExecResult(1, "", f"cat: {missing[0]}: No such file\n")
            return ExecResult(0, "".join(files[p].decode() for p in rest))
        if head == "ls":
            return ExecResult(0, "".join(f"{p}\n" for p in sorted(files)))
        if head == "exit":
            return ExecResult(int(rest[0]) if rest else 0)
        return ExecResult(0, "", "", 0.001)

    async def run_python(self, code, *, timeout=None, env=None, cwd=None) -> ExecResult:
        self._log("run_python", code)
        return ExecResult(0, "", "", 0.001)

    async def write_file(self, path, data) -> None:
        self._log("write_file", path)
        self._state["files"][path] = data.encode() if isinstance(data, str) else bytes(data)

    async def read_file(self, path) -> bytes:
        self._log("read_file", path)
        try:
            return self._state["files"][path]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def snapshot(self, label=None) -> SnapshotRef:
        self._log("snapshot", label)
        self.driver.n += 1
        snap_id = f"{self.id}-{label or self.driver.n}"
        self.driver.snapshots[snap_id] = copy.deepcopy(self._state["files"])
        return SnapshotRef(driver=self.driver.name, id=snap_id, sandbox_id=self.id, label=label)

    async def restore(self, ref) -> None:
        self._log("restore", ref.id)
        self._state["files"] = copy.deepcopy(self.driver.snapshots[ref.id])

    async def suspend(self) -> None:
        self._log("suspend")
        self._state["status"] = Status.SUSPENDED

    async def resume(self) -> None:
        self._log("resume")
        self._state["status"] = Status.RUNNING

    async def destroy(self) -> None:
        self._log("destroy")
        self.driver.boxes.pop(self.id, None)

    async def status(self) -> str:
        st = self.driver.boxes.get(self.id)
        return st["status"] if st else Status.DESTROYED


class FakeDriver(SandboxDriver):
    name = "fake"

    def __init__(self, *, create_delay: float = 0.0) -> None:
        self.boxes: Dict[str, Dict[str, Any]] = {}
        self.snapshots: Dict[str, Dict[str, bytes]] = {}
        self.calls: List[Tuple[Any, ...]] = []
        self.create_delay = create_delay
        self.n = 0

    async def create(self, spec, *, snapshot=None, sandbox_id=None) -> FakeSandbox:
        import asyncio

        if self.create_delay:
            await asyncio.sleep(self.create_delay)
        spec = SandboxSpec.coerce(spec)
        sid = sandbox_id or new_sandbox_id("fake")
        files: Dict[str, bytes] = {}
        if snapshot is not None:
            if snapshot.id not in self.snapshots:
                raise SandboxNotFound(f"snapshot {snapshot.id}")
            files = copy.deepcopy(self.snapshots[snapshot.id])
        self.boxes[sid] = {"status": Status.RUNNING, "files": files, "spec": spec,
                           "created_at": time.time()}
        self.calls.append(("create", sid, snapshot.id if snapshot else None))
        return FakeSandbox(self, sid, spec)

    async def attach(self, sandbox_id) -> FakeSandbox:
        st = self.boxes.get(sandbox_id)
        if st is None:
            raise SandboxNotFound(sandbox_id)
        self.calls.append(("attach", sandbox_id))
        return FakeSandbox(self, sandbox_id, st["spec"])

    async def list(self, labels=None) -> List[str]:
        return list(self.boxes)

    async def delete_snapshot(self, ref) -> None:
        self.snapshots.pop(ref.id, None)

    def ops(self, op: Optional[str] = None) -> List[Tuple[Any, ...]]:
        return [c for c in self.calls if op is None or c[0] == op]
