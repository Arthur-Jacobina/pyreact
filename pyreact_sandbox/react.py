"""The border between pyreact and sandboxes: ``SandboxProvider`` + ``use_sandbox``.

Built on pyreact's public surface only (hooks, ``hooks.provide``, the
session ``EventBus``, the harness ``through``); nothing in ``pyreact`` knows
this module exists.

    @component
    def App():
        return [SandboxProvider(driver="docker", spec=SandboxSpec(image="python:3.12-slim"),
                                on_close="destroy", children=[Coder(key="coder")])]

    @component
    def Coder():
        sb = use_sandbox()                       # session-scoped handle
        ...
        r = await sb.exec("python3 main.py", timeout=30)   # in an effect / tool / task

Lifecycle
---------
- **create**: on mount (``eager=True``, in the background) or on first use.
  If the :class:`~pyreact_sandbox.store.SnapshotStore` has a record for this
  sandbox (``"<session id>/<key>"``) it is *reattached* (still alive:
  running or suspended) or *restored* (from its snapshot) instead.
- **scope**: ``"session"`` (default) — one sandbox per ``key`` per session
  (provider), shared by every component that asks, closed when the
  provider unmounts / the session closes. ``"component"`` — owned by the
  calling component, closed on unmount.
- **close** policy (``on_close``): ``"destroy"`` (default), ``"suspend"``
  (pause it; a restart reattaches), ``"snapshot"`` (snapshot, record, then
  destroy; a restart restores) or ``"keep"`` (leave it running). It may
  depend on *why* it closes: a mapping from the session's ``close_reason``
  (``"shutdown"``, ``"idle"``, ``"closed"``; ``"unmount"`` when only the
  component/provider goes away) to a policy, with ``"*"`` as the fallback,
  e.g. ``{"shutdown": "suspend", "*": "destroy"}`` (resume after a restart,
  clean up when a user's session ends) — or a callable ``reason -> policy``.
- **autosnapshot="turn"**: at the end of every turn that touched the
  sandbox, snapshot it and record the ref. The snapshot runs as a task of
  that turn, after the turn's other work: ``turn.end`` (and
  ``invoke(wait=True)``) come after the ref is recorded, so it survives a
  crash right after the turn, and the ``pyreact_durable`` store (whose log
  closes before the sandbox on shutdown) gets it while the log is open.

Sandbox operations run on a private background event loop (one thread per
process), so they are independent from the render loop's lifecycle: closing
a session never strands a half-created container, and :func:`wait_closed`
(also run at interpreter exit) waits for pending closes/snapshots.

Every exec-like call (``exec``, ``run_python``, ``write_file``,
``read_file``) runs through the harness boundary ``sandbox.exec`` (name: the
command, ``python``, ``write_file``, ``read_file``) so middleware can guard,
rewrite, retry or trace it. Events on the session ``EventBus``::

    sandbox.created    {key, sandbox_id, driver, image, duration}
    sandbox.restored   {key, sandbox_id, driver, mode: "attach"|"snapshot", snapshot}
    sandbox.exec       {key, sandbox_id, op, name, command, exit_code, duration,
                        timed_out, error, boundary_id}
    sandbox.snapshot   {key, sandbox_id, snapshot, label, auto}
    sandbox.suspended / sandbox.resumed / sandbox.destroyed   {key, sandbox_id}
    sandbox.error      {key, phase, error}
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Union

from pyreact.core import (EventBus, component, create_context, hooks,
                          report_hydration_mismatch, runtime, use_dehydrate,
                          use_imperative_handle, use_path)
from pyreact.core.harness import through

from ._util import truncate
from .base import (
    Command,
    ExecResult,
    NotSupported,
    Sandbox,
    SandboxDriver,
    SandboxError,
    SandboxNotFound,
    SandboxSpec,
    SnapshotRef,
    Status,
    command_name,
)
from .registry import get_driver
from .store import MemorySnapshotStore, SnapshotStore

__all__ = [
    "ON_CLOSE",
    "resolve_policy",
    "SandboxEnv",
    "SandboxHandle",
    "SandboxProvider",
    "use_sandbox",
    "use_sandbox_env",
    "wait_closed",
]

ON_CLOSE = ("destroy", "suspend", "snapshot", "keep")
_PREVIEW = 400

# Deprecated session services (pyreact_sandbox.stateful.SandboxParticipant,
# kept for one release): handles now dehydrate through use_dehydrate.
HANDLES_SERVICE = "pyreact_sandbox.handles"  # {store key: SandboxHandle} of the session
RESTORE_SERVICE = "pyreact_sandbox.restore"  # {store key: record} from a legacy manifest

# How long a dehydrate waits for the suspend close (snapshot / pause) of a handle.
SUSPEND_CLOSE_TIMEOUT = 600.0


# ---------------------------------------------------------------------------
# Private worker loop: sandbox I/O independent from the render loop
# ---------------------------------------------------------------------------
class _Worker:
    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()
        self._pending: set = set()

    def loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="pyreact-sandbox",
                                 daemon=True).start()
                self._loop = loop
            return self._loop

    def submit(self, coro: Awaitable, *, track: bool = False) -> concurrent.futures.Future:
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop())
        if track:
            with self._lock:
                self._pending.add(fut)
            fut.add_done_callback(self._done)
        return fut

    def _done(self, fut) -> None:
        with self._lock:
            self._pending.discard(fut)

    def drain(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                pending = list(self._pending)
            if not pending:
                return True
            left = None if deadline is None else max(deadline - time.monotonic(), 0)
            done, not_done = concurrent.futures.wait(pending, timeout=left)
            if not_done and deadline is not None and time.monotonic() >= deadline:
                return False


_worker = _Worker()


def wait_closed(timeout: Optional[float] = 120) -> bool:
    """Block until pending sandbox closes / snapshots finish (also runs at
    interpreter exit). Returns False on timeout."""
    return _worker.drain(timeout)


atexit.register(wait_closed, 120)


class _RWLock:
    """Many concurrent ops (exec) or one exclusive op (snapshot/restore/close).
    Lives on the worker loop."""

    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._readers = 0
        self._writer = False

    async def acquire(self, exclusive: bool) -> None:
        async with self._cond:
            if exclusive:
                await self._cond.wait_for(lambda: not self._writer and self._readers == 0)
                self._writer = True
            else:
                await self._cond.wait_for(lambda: not self._writer)
                self._readers += 1

    async def release(self, exclusive: bool) -> None:
        async with self._cond:
            if exclusive:
                self._writer = False
            else:
                self._readers -= 1
            self._cond.notify_all()


def _check_policy(on_close: Any) -> None:
    if callable(on_close):
        return
    values = on_close.values() if isinstance(on_close, Mapping) else [on_close]
    bad = [v for v in values if v not in ON_CLOSE]
    if bad:
        raise ValueError(f"on_close must be one of {ON_CLOSE} (or a mapping / callable), "
                         f"got {bad[0]!r}")


def resolve_policy(on_close: Any, reason: str) -> str:
    """The close policy for ``reason`` (``"shutdown"``, ``"idle"``,
    ``"closed"``, ``"unmount"``, ``"suspend"``).

    ``"suspend"`` (``root.suspend()``: the tree is dehydrated, the agent will
    be hydrated again) never destroys implicitly: a ``"destroy"`` that comes
    from a plain policy or a mapping's ``"*"`` becomes ``"snapshot"``. Name
    ``"suspend"`` in a mapping (or decide in a callable) to choose otherwise,
    e.g. ``{"suspend": "suspend", "*": "destroy"}`` pauses instead."""
    if callable(on_close):
        policy = on_close(reason)
        if reason == "suspend" and policy not in ON_CLOSE:
            return "snapshot"
        return policy
    if isinstance(on_close, Mapping):
        if reason == "suspend" and "suspend" in on_close:
            return on_close["suspend"]
        policy = on_close.get(reason, on_close.get("*", "destroy"))
    else:
        policy = on_close
    if reason == "suspend" and policy == "destroy":
        return "snapshot"
    return policy


# ---------------------------------------------------------------------------
# Handle
# ---------------------------------------------------------------------------
class SandboxHandle:
    """What :func:`use_sandbox` returns. Async methods, safe from any loop
    (the render loop, tools, effects) — and :meth:`run_sync` for worker
    threads (e.g. an RLM interpreter).

    ``handle.sandbox`` is the underlying driver :class:`Sandbox` once created
    (``None`` before); ``handle.id`` its id.
    """

    def __init__(self, *, key: str, store_key: str, driver: SandboxDriver, spec: SandboxSpec,
                 store: Optional[SnapshotStore], on_close: Any, autosnapshot: Optional[str],
                 bus: Optional[EventBus], owner: Any = None, source: Optional[str] = None,
                 home_loop: Optional[asyncio.AbstractEventLoop] = None,
                 session: Any = None) -> None:
        _check_policy(on_close)
        if autosnapshot not in (None, "turn"):
            raise ValueError("autosnapshot must be None or 'turn'")
        self.key = key
        self.store_key = store_key
        self.driver = driver
        self.spec = spec
        self.store = store
        self.on_close = on_close
        self.autosnapshot = autosnapshot
        self.bus = bus
        self.owner = owner
        self.source = source
        self.home_loop = home_loop
        self.session = session
        self.sandbox: Optional[Sandbox] = None
        self.restored: Optional[str] = None  # "attach" | "snapshot" | None
        self.last_snapshot: Optional[SnapshotRef] = None
        self.error: Optional[BaseException] = None
        self.closed = False
        self.dirty = False
        # set by hydration (a snapshot's slice): tried before the store's record
        self.restore_record: Optional[Dict[str, Any]] = None
        # where that slice was (path, key): for hydration.mismatch reports
        self.restore_origin: Optional[tuple] = None
        # close policy forced for a suspend (deprecated SandboxParticipant)
        self.suspend_policy: Optional[str] = None
        self.closed_policy: Optional[str] = None  # the policy close applied
        self.close_future: Optional[concurrent.futures.Future] = None
        self._create_lock: Optional[asyncio.Lock] = None
        self._rw: Optional[_RWLock] = None
        self._unsubs: list = []
        self._snap_turns: set = set()
        if autosnapshot == "turn" and bus is not None:
            self._unsubs.append(bus.subscribe(self._on_turn_end, ["turn.end"]))

    def __repr__(self) -> str:
        state = "closed" if self.closed else (self.id or "pending")
        return f"<SandboxHandle {self.key!r} {self.driver.name}:{state}>"

    @property
    def id(self) -> Optional[str]:
        return self.sandbox.id if self.sandbox is not None else None

    # ------------------------------------------------------------- plumbing
    def _publish(self, kind: str, payload: Dict[str, Any], turn_id: Optional[str] = None) -> None:
        if self.bus is None:
            return
        base = {"key": self.key, "sandbox_id": self.id, "driver": self.driver.name}
        try:
            self.bus.publish(kind, {**base, **payload}, turn_id=turn_id, source=self.source)
        except Exception:
            pass

    async def _await(self, coro: Awaitable) -> Any:
        """Run ``coro`` on the worker loop and await it from the caller's loop."""
        fut = _worker.submit(coro)
        return await asyncio.wrap_future(fut)

    def _locks(self) -> None:
        if self._create_lock is None:
            self._create_lock = asyncio.Lock()
            self._rw = _RWLock()

    def _record(self, **changes: Any) -> Dict[str, Any]:
        rec = (self.store.get(self.store_key) if self.store is not None else None) or {}
        rec.update({"driver": self.driver.name, "spec": self.spec.to_dict(),
                    "updated_at": time.time(), **changes})
        return rec

    def _save(self, **changes: Any) -> None:
        if self.store is None:
            return
        try:
            self.store.set(self.store_key, self._record(**changes))
        except Exception as exc:  # a store failure must not break the sandbox
            self._publish("sandbox.error", {"phase": "store", "error": repr(exc)})

    # ------------------------------------------------- create/attach/restore
    async def _ensure(self) -> Sandbox:  # worker loop
        self._locks()
        if self.sandbox is not None:
            return self.sandbox
        async with self._create_lock:
            if self.sandbox is not None:
                return self.sandbox
            if self.closed:
                raise SandboxError(f"sandbox {self.key!r} is closed")
            try:
                sb = await self._boot()
            except BaseException as exc:
                self.error = exc
                self._publish("sandbox.error", {"phase": "create", "error": repr(exc)})
                raise
            self.error = None
            self.sandbox = sb
            self._save(sandbox_id=sb.id)
            return sb

    def _degrade(self, reason: str, **info: Any) -> None:
        """A snapshot (or legacy manifest) said how to bring this sandbox
        back and it could not be done: a ``hydration.mismatch`` (on the
        slice it came from) and ``session.resume.degraded``."""
        fn = getattr(self.session, "degrade", None)
        if callable(fn):
            try:
                fn("sandbox", reason, key=self.key, **info)
            except Exception:
                pass
        path, slice_key = self.restore_origin or (self.source, "sandboxes")
        try:
            report_hydration_mismatch(
                slice_key, f"sandbox {self.key!r}: {reason}; a fresh sandbox boots",
                ctx=self.owner, path=path, reason=reason, sandbox=self.key, **info)
        except Exception:
            pass

    # ------------------------------------------------------------ dehydrate
    def dehydrate(self) -> Optional[Dict[str, Any]]:
        """This sandbox's slice record: how to bring it back
        (``{driver, spec, sandbox_id, snapshot}``), or ``None`` (destroyed).

        After a suspend close: ``"snapshot"`` -> the snapshot ref only,
        ``"suspend"`` / ``"keep"`` -> its id (reattach) and its last
        snapshot, ``"destroy"`` -> ``None``. Live (``root.dehydrate()``): its
        id and last snapshot. Never booted: the record it was given."""
        base = {"driver": self.driver.name, "spec": self.spec.to_dict()}
        if self.sandbox is None:
            rec = self.restore_record or (self.store.get(self.store_key)
                                          if self.store is not None else None)
            if not rec or not (rec.get("sandbox_id") or rec.get("snapshot")):
                return None
            return {**base, "driver": rec.get("driver") or self.driver.name,
                    "sandbox_id": rec.get("sandbox_id"), "snapshot": rec.get("snapshot")}
        policy = self.closed_policy
        if policy == "destroy":
            return None
        snap = self.last_snapshot.to_dict() if self.last_snapshot is not None else None
        sid = self.id if policy in (None, "suspend", "keep") else None
        return {**base, "sandbox_id": sid, "snapshot": snap}

    def suspend_close_policy(self) -> str:
        """The close policy a suspend applies (see :func:`resolve_policy`)."""
        if self.suspend_policy is not None:
            return self.suspend_policy
        return resolve_policy(self.on_close, "suspend")

    async def adehydrate(self, timeout: float = SUSPEND_CLOSE_TIMEOUT) -> Optional[Dict[str, Any]]:
        """:meth:`dehydrate` once a close in flight (the suspend's snapshot /
        pause) has finished."""
        fut = self.close_future
        if fut is not None and not fut.done():
            try:
                await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(fut)), timeout)
            except Exception:
                pass
        return self.dehydrate()

    async def _boot(self) -> Sandbox:
        t0 = time.perf_counter()
        from_manifest, self.restore_record = self.restore_record, None
        rec = from_manifest or (self.store.get(self.store_key) if self.store is not None else None)
        if from_manifest and from_manifest.get("driver") != self.driver.name:
            self._degrade("driver changed", expected=from_manifest.get("driver"),
                          driver=self.driver.name)
        if rec and rec.get("driver") == self.driver.name:
            sid = rec.get("sandbox_id")
            if sid:
                try:
                    sb = await self.driver.attach(sid)
                    st = await sb.status()
                    if st in (Status.SUSPENDED, Status.STOPPED):
                        await sb.resume()
                        st = await sb.status()
                    if st == Status.RUNNING:
                        self.restored = "attach"
                        self.sandbox = sb
                        self._publish("sandbox.restored", {
                            "mode": "attach", "snapshot": None,
                            "duration": time.perf_counter() - t0})
                        return sb
                except (SandboxNotFound, NotSupported, SandboxError):
                    pass
            snap = rec.get("snapshot")
            if snap:
                ref = SnapshotRef.from_dict(snap)
                try:
                    sb = await self.driver.create(self.spec, snapshot=ref)
                except SandboxError as exc:
                    self._publish("sandbox.error", {"phase": "restore", "error": repr(exc),
                                                    "snapshot": ref.id})
                    if from_manifest:
                        self._degrade("snapshot missing" if isinstance(exc, SandboxNotFound)
                                      else "restore failed", snapshot=ref.id, error=repr(exc))
                else:
                    self.restored = "snapshot"
                    self.last_snapshot = ref
                    self.sandbox = sb
                    self._publish("sandbox.restored", {
                        "mode": "snapshot", "snapshot": ref.id,
                        "duration": time.perf_counter() - t0})
                    return sb
            elif from_manifest and sid:
                self._degrade("sandbox gone", sandbox_id=sid)
        sb = await self.driver.create(self.spec)
        self.sandbox = sb
        self._publish("sandbox.created", {"image": self.spec.image,
                                          "template": self.spec.template,
                                          "duration": time.perf_counter() - t0})
        return sb

    async def _op(self, fn: Callable[[Sandbox], Awaitable], *, exclusive: bool = False,
                  dirty: bool = False) -> Any:  # worker loop
        sb = await self._ensure()
        await self._rw.acquire(exclusive)
        try:
            if self.closed and self.sandbox is None:
                raise SandboxError(f"sandbox {self.key!r} is closed")
            if dirty:
                self.dirty = True
            return await fn(sb)
        finally:
            await self._rw.release(exclusive)

    def start_nowait(self) -> concurrent.futures.Future:
        """Begin creating/attaching/restoring in the background."""
        fut = _worker.submit(self._ensure())
        fut.add_done_callback(lambda f: f.cancelled() or f.exception())  # mark retrieved
        return fut

    async def start(self) -> Sandbox:
        """Create (or reattach / restore) the sandbox now; returns it."""
        return await self._await(self._ensure())

    # ------------------------------------------------------------- boundary
    async def _boundary(self, op: str, name: str, inputs: Dict[str, Any],
                        run: Callable[[Dict[str, Any]], Awaitable[Any]]) -> Any:
        """Run one exec-like call through the harness (``sandbox.exec``) and
        publish ``sandbox.exec``. ``run(inputs)`` does the work (on the
        worker loop) with the inputs as transformed by middleware."""
        ctx = runtime.current_component() or self.owner
        seen: Dict[str, Any] = {}

        async def call(b):
            seen["b"] = b
            return await run(b.inputs)

        t0 = time.perf_counter()
        error = None
        result = None
        try:
            result = await through("sandbox.exec", name, {"op": op, "sandbox": self.key, **inputs},
                                   call, ctx=ctx, sandbox_id=self.id, driver=self.driver.name)
            return result
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            b = seen.get("b")
            payload: Dict[str, Any] = {
                "op": op, "name": name,
                "command": truncate(str(inputs.get("cmd") or inputs.get("code")
                                        or inputs.get("path") or ""), _PREVIEW),
                "duration": time.perf_counter() - t0, "error": error,
                "boundary_id": getattr(b, "id", None),
            }
            if isinstance(result, ExecResult):
                payload.update(exit_code=result.exit_code, timed_out=result.timed_out,
                               stdout=truncate(result.stdout, _PREVIEW),
                               stderr=truncate(result.stderr, _PREVIEW))
            turn = getattr(b, "turn_id", None)
            if turn is None and ctx is not None:
                turn = runtime.current_turn_id(ctx)
            self._publish("sandbox.exec", payload, turn_id=turn)
            if error is None and op != "read_file":
                self._schedule_turn_snapshot()

    # ------------------------------------------------------------ public ops
    async def exec(self, cmd: Command, *, timeout: Optional[float] = None,
                   env: Optional[Mapping[str, str]] = None, cwd: Optional[str] = None,
                   stdin: Union[str, bytes, None] = None) -> ExecResult:
        """Run a command (shell string or argv) in the sandbox."""
        async def run(i):
            return await self._await(self._op(
                lambda sb: sb.exec(i["cmd"], timeout=i.get("timeout"), env=i.get("env"),
                                   cwd=i.get("cwd"), stdin=i.get("stdin")), dirty=True))

        cmd_in = cmd if isinstance(cmd, str) else list(cmd)
        return await self._boundary("exec", command_name(cmd), {
            "cmd": cmd_in, "timeout": timeout, "env": dict(env) if env else None, "cwd": cwd,
            "stdin": stdin}, run)

    async def run_python(self, code: str, *, timeout: Optional[float] = None,
                         env: Optional[Mapping[str, str]] = None,
                         cwd: Optional[str] = None) -> ExecResult:
        async def run(i):
            return await self._await(self._op(
                lambda sb: sb.run_python(i["code"], timeout=i.get("timeout"), env=i.get("env"),
                                         cwd=i.get("cwd")), dirty=True))

        return await self._boundary("run_python", "python", {
            "code": code, "timeout": timeout, "env": dict(env) if env else None, "cwd": cwd}, run)

    async def write_file(self, path: str, data: Union[str, bytes]) -> None:
        async def run(i):
            return await self._await(self._op(lambda sb: sb.write_file(i["path"], i["data"]),
                                              dirty=True))

        await self._boundary("write_file", "write_file", {"path": path, "data": data}, run)

    async def read_file(self, path: str) -> bytes:
        async def run(i):
            return await self._await(self._op(lambda sb: sb.read_file(i["path"])))

        return await self._boundary("read_file", "read_file", {"path": path}, run)

    async def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return (await self.read_file(path)).decode(encoding, "replace")

    async def status(self) -> str:
        if self.sandbox is None:
            return Status.DESTROYED if self.closed else Status.CREATING
        return await self._await(self.sandbox.status())

    async def snapshot(self, label: Optional[str] = None) -> SnapshotRef:
        """Snapshot now and record it in the store (a restart restores it)."""
        return await self._await(self._snapshot(label, auto=False))

    async def _snapshot(self, label: Optional[str], *, auto: bool) -> SnapshotRef:  # worker
        async def do(sb):
            # auto snapshots reuse one label, so each sandbox keeps one; older
            # ones (other sandboxes, e.g. before a restart) stay: a durable
            # store may still point at them from another branch. gc() them.
            ref = await sb.snapshot(label)
            self.last_snapshot = ref
            self.dirty = False
            self._save(sandbox_id=sb.id, snapshot=ref.to_dict())
            self._publish("sandbox.snapshot", {"snapshot": ref.id, "label": label, "auto": auto})
            return ref

        return await self._op(do, exclusive=True)

    async def restore(self, ref: SnapshotRef) -> None:
        """Roll the sandbox back to ``ref`` (in place when the driver can,
        else a new sandbox from it replaces this one)."""
        async def do(sb):
            try:
                await sb.restore(ref)
            except NotSupported:
                new = await self.driver.create(self.spec, snapshot=ref)
                await sb.destroy()
                self.sandbox = new
            self._save(sandbox_id=self.id, snapshot=ref.to_dict())
            self._publish("sandbox.restored", {"mode": "snapshot", "snapshot": ref.id})

        await self._await(self._op(do, exclusive=True))

    async def suspend(self) -> None:
        async def do(sb):
            await sb.suspend()
            self._publish("sandbox.suspended", {})

        await self._await(self._op(do, exclusive=True))

    async def resume(self) -> None:
        async def do(sb):
            await sb.resume()
            self._publish("sandbox.resumed", {})

        await self._await(self._op(do, exclusive=True))

    async def destroy(self) -> None:
        """Destroy now (same as ``close("destroy")``)."""
        await self.close("destroy")

    # ---------------------------------------------------------------- close
    async def close(self, policy: Optional[str] = None) -> None:
        """Apply the close policy (default: ``on_close``) and wait for it."""
        await asyncio.wrap_future(self.close_nowait(policy))

    def close_nowait(self, policy: Optional[str] = None) -> concurrent.futures.Future:
        """Start closing in the background (safe from cleanups / any thread);
        :func:`wait_closed` waits for it. Without ``policy``, ``on_close`` is
        resolved against the session's ``close_reason``."""
        if policy is None:
            reason = getattr(self.session, "close_reason", None) or "unmount"
            if reason == "suspend":
                policy = self.suspend_close_policy()
            else:
                policy = resolve_policy(self.on_close, reason)
        fut = _worker.submit(self._aclose(policy), track=True)
        if self.close_future is None:
            self.close_future = fut
        return fut

    async def _aclose(self, policy: str) -> None:  # worker loop
        if policy not in ON_CLOSE:
            raise ValueError(f"on_close must be one of {ON_CLOSE}")
        if self.closed:
            return
        self.closed = True
        self.closed_policy = policy
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()
        self._locks()
        async with self._create_lock:  # wait for an in-flight create
            pass
        sb = self.sandbox
        if sb is None:
            return
        await self._rw.acquire(True)
        try:
            await self._apply_close(sb, policy)
        except Exception as exc:
            self._publish("sandbox.error", {"phase": f"close:{policy}", "error": repr(exc)})
        finally:
            await self._rw.release(True)

    async def _apply_close(self, sb: Sandbox, policy: str) -> None:
        if policy == "keep":
            return
        if policy == "suspend":
            try:
                await sb.suspend()
                self._publish("sandbox.suspended", {})
            except NotSupported:
                pass
            self._save(sandbox_id=sb.id)
            return
        if policy == "snapshot":
            ref = await sb.snapshot("close")
            self.last_snapshot = ref
            self._save(sandbox_id=None, snapshot=ref.to_dict())
            self._publish("sandbox.snapshot", {"snapshot": ref.id, "label": "close",
                                               "auto": True})
        await sb.destroy()
        if self.store is not None and policy == "destroy":
            rec = self.store.get(self.store_key) or {}
            if rec.get("snapshot"):
                self._save(sandbox_id=None)
            else:
                try:
                    self.store.delete(self.store_key)
                except Exception:
                    pass
        self._publish("sandbox.destroyed", {"policy": policy})

    # ----------------------------------------------------------- autosnapshot
    def _schedule_turn_snapshot(self) -> None:
        """autosnapshot="turn": snapshot as the last task of the current turn."""
        if self.autosnapshot != "turn" or self.closed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        tid = runtime.current_turn_id()
        if tid is None or loop is not self.home_loop or tid in self._snap_turns:
            return
        self._snap_turns.add(tid)
        try:
            runtime.spawn(self._turn_snapshot(tid), owner=self.owner, turn=tid,
                          name="sandbox:autosnapshot")
        except Exception:
            self._snap_turns.discard(tid)

    async def _turn_snapshot(self, tid: str) -> None:
        try:
            while True:  # wait for the rest of the turn (other snapshots excluded)
                others = [t for t in runtime.pending_turn_tasks(tid)
                          if t.get_name() != "sandbox:autosnapshot"]
                if not others and not runtime.renders_pending():
                    break
                await asyncio.sleep(0.02)
            if self.dirty and not self.closed:
                await self._await(self._snapshot("auto", auto=True))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._publish("sandbox.error", {"phase": "autosnapshot", "error": repr(exc)},
                          turn_id=tid)
        finally:
            self._snap_turns.discard(tid)

    def _on_turn_end(self, ev) -> None:
        """Fallback for work outside a turn task (e.g. another loop)."""
        if self.closed or not self.dirty or self.sandbox is None:
            return
        self.dirty = False

        async def snap():
            try:
                await self._snapshot("auto", auto=True)
            except Exception as exc:
                self._publish("sandbox.error", {"phase": "autosnapshot", "error": repr(exc)})

        _worker.submit(snap(), track=True)

    # ---------------------------------------------------------------- sync
    def run_sync(self, fn: Callable[[], Awaitable[Any]], timeout: Optional[float] = None) -> Any:
        """Call an async method from a thread without a running loop, e.g.
        ``handle.run_sync(lambda: handle.exec("ls"))``. It runs on the app's
        render loop when that is alive (so harness middleware / owner
        attribution behave as in components), else on a temporary loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("run_sync() called from a running event loop; await instead")
        loop = self.home_loop
        if loop is not None and loop.is_running() and not loop.is_closed():
            return asyncio.run_coroutine_threadsafe(fn(), loop).result(timeout)
        return asyncio.run(fn())


# ---------------------------------------------------------------------------
# Provider + hook
# ---------------------------------------------------------------------------
@dataclass
class SandboxEnv:
    """Value provided by :func:`SandboxProvider` (defaults for its subtree)."""

    driver: SandboxDriver
    spec: SandboxSpec
    store: Optional[SnapshotStore]
    on_close: Any = "destroy"
    autosnapshot: Optional[str] = None
    handles: Dict[str, SandboxHandle] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    # hydration: {key: record} from the provider's slice, taken by use_sandbox
    restore: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None  # the provider's path (its slice)
    # handles a suspend closed (snapshot / pause in flight): still dehydrated
    suspended: Dict[str, SandboxHandle] = field(default_factory=dict)

    def close_all(self, reason: Optional[str] = None) -> None:
        """Close every handle; ``reason="suspend"`` applies each handle's
        suspend policy (snapshot / pause, never an implicit destroy) and
        keeps the handles for :meth:`adehydrate`."""
        with self.lock:
            handles, self.handles = list(self.handles.values()), {}
            if reason == "suspend":
                self.suspended.update({h.key: h for h in handles})
        for h in handles:
            h.close_nowait(h.suspend_close_policy() if reason == "suspend" else None)

    async def adehydrate(self) -> Dict[str, Any]:
        """The provider's slice: ``{key: record}`` (see :meth:`SandboxHandle.dehydrate`)
        for its handles, after a suspend's closes finished; records hydrated
        but never asked for in this run are carried over."""
        with self.lock:
            handles = {**self.suspended, **self.handles}
        out: Dict[str, Any] = {k: v for k, v in self.restore.items() if v}
        for k, h in handles.items():
            rec = await h.adehydrate()
            if rec is not None:
                out[k] = rec
            else:
                out.pop(k, None)
        return out


_SandboxCtx = create_context(default=None, name="Sandbox")


def _driver_key(driver: Any) -> Any:
    return driver if isinstance(driver, str) or driver is None else ("obj", id(driver))


@component(transparent=True)
def SandboxProvider(driver: Union[str, SandboxDriver] = "docker", spec=None, store=None,
                    on_close: Any = "destroy", autosnapshot: Optional[str] = None,
                    driver_options: Optional[Mapping[str, Any]] = None, ref=None,
                    children=None):
    """Provide a sandbox driver, default spec and policies to the subtree.

    - ``driver``: a registered name (``"docker"``, ``"local"``, a vendor) or
      a :class:`SandboxDriver` instance; ``driver_options`` go to its factory.
    - ``spec``: default :class:`SandboxSpec` (or dict) for ``use_sandbox``.
    - ``store``: :class:`SnapshotStore` for restart records (default: one
      in-memory store per provider).
    - ``on_close`` / ``autosnapshot``: defaults for handles (see module docs).

    - ``ref``: exposes the provider's :class:`SandboxEnv` (its ``handles``
      by key) through :func:`~pyreact.core.use_imperative_handle` -- a
      ``Ref``, a callback, or a name for the session's refs registry.

    Session-scoped sandboxes created under it are closed when it unmounts
    (session close / app shutdown included).
    """
    drv = hooks.use_memo(lambda: get_driver(driver, **dict(driver_options or {})),
                         [_driver_key(driver), tuple(sorted((driver_options or {}).items()))])
    default_store = hooks.use_memo(MemorySnapshotStore, [])
    path = use_path()
    current = hooks.use_ref(None)
    # The provider's own state (ADR 0001): how to bring each sandbox back.
    hydrated = use_dehydrate("sandboxes", dehydrate=lambda: current.current.adehydrate(),
                             hydrate=_hydrate_records)
    env = hooks.use_memo(lambda: SandboxEnv(driver=drv, spec=SandboxSpec.coerce(spec),
                                            store=store if store is not None else default_store,
                                            on_close=on_close, autosnapshot=autosnapshot,
                                            restore=dict(hydrated or {}), path=path),
                         [id(drv)])
    current.current = env
    env.spec = SandboxSpec.coerce(spec)
    env.store = store if store is not None else default_store
    env.on_close, env.autosnapshot = on_close, autosnapshot

    def _lifecycle():
        def _close(reason):
            if reason == "hidden":  # Activity: the sandboxes keep running
                return
            env.close_all(reason)

        return _close

    hooks.use_effect(_lifecycle, [id(env)])
    use_imperative_handle(ref, lambda: env, [id(env)])
    hooks.provide(_SandboxCtx, env)
    return list(children or [])


def _hydrate_records(slice_: Any) -> Dict[str, Any]:
    """``hydrate`` of a sandbox slice: ``{key: record}`` (records are checked
    when the sandbox boots: a missing snapshot boots a fresh one)."""
    if not isinstance(slice_, Mapping):
        raise TypeError(f"a sandbox slice is a dict {{key: record}}, got {type(slice_).__name__}")
    return {str(k): dict(v) for k, v in slice_.items() if isinstance(v, Mapping)}


def use_sandbox_env() -> Optional[SandboxEnv]:
    """The nearest :func:`SandboxProvider`'s env (``None`` without one)."""
    return hooks.use_context(_SandboxCtx)


def _session_env(bus: EventBus) -> SandboxEnv:
    """Provider-less default: one env per session (docker driver, in-memory
    store), closed on ``session.closed``."""

    def make():
        env = SandboxEnv(driver=get_driver("docker"), spec=SandboxSpec(),
                         store=MemorySnapshotStore())
        bus.subscribe(lambda _ev: env.close_all(), ["session.closed"])
        return env

    return hooks.get_service("pyreact_sandbox", make)


def use_sandbox(spec: Union[SandboxSpec, Mapping[str, Any], None] = None, *,
                key: Optional[str] = None, scope: str = "session",
                driver: Union[str, SandboxDriver, None] = None,
                store: Optional[SnapshotStore] = None, on_close: Any = None,
                autosnapshot: Optional[str] = None, eager: bool = True,
                ref: Any = None) -> SandboxHandle:
    """A :class:`SandboxHandle` for this component.

    - ``spec``: a :class:`SandboxSpec` replaces the provider's; a dict
      overrides some of its fields (``{"memory": "1g"}``).
    - ``key``: sandbox name. ``scope="session"``: components asking for the
      same key (default ``"default"``) share one sandbox. ``scope="component"``:
      default key is the component path; closed when it unmounts.
    - ``driver`` / ``store`` / ``on_close`` / ``autosnapshot``: override the
      provider's (without a provider: docker, in-memory store, destroy).
    - ``eager``: start creating on mount (else on first use).
    - ``ref``: also expose the handle through
      :func:`~pyreact.core.use_imperative_handle` -- a ``Ref`` /
      callback, or a name: ``use_sandbox(ref="sandbox")`` registers it in
      the session's refs registry (``root.ref("sandbox")``) instead of
      reaching into ``session.services[HANDLES_SERVICE]``.

    The record key in the store is ``"<session id>/<key>"``.

    Suspend / hydrate (ADR 0001): on ``root.suspend()`` the effect cleanup
    (``use_cleanup_reason() == "suspend"``) snapshots or pauses the sandbox
    per ``on_close`` (:func:`resolve_policy`: never an implicit destroy) and
    its record (snapshot ref, id) is dehydrated -- by the
    :func:`SandboxProvider` for session-scoped sandboxes, by this hook
    (slice ``"sandbox"`` / ``"sandbox:<key>"`` of the calling component) for
    ``scope="component"`` or without a provider. ``hydrate_root`` hands the
    record back, so the sandbox restores from its snapshot (or reattaches);
    if that fails a fresh sandbox boots and a ``hydration.mismatch`` (plus
    ``session.resume.degraded``) is reported.
    """
    if scope not in ("session", "component"):
        raise ValueError("scope must be 'session' or 'component'")
    bus = hooks.get_service("event_bus", EventBus)
    session = hooks.get_service("session", lambda: None)
    sid = str(getattr(session, "id", None) or "default")
    env = hooks.use_context(_SandboxCtx)
    provided = env is not None
    if env is None:
        env = _session_env(bus)
    ctx = runtime.current_component()
    path = use_path()
    k = key or ("default" if scope == "session" else path)
    # who dehydrates this sandbox: the provider (session scope under one), else
    # this hook, in the calling component (stable: the tree shape decides)
    owns_slice = scope == "component" or not provided
    slice_key = "sandbox" if key is None else f"sandbox:{key}"
    current = hooks.use_ref(None)
    hydrated = None
    if owns_slice:
        hydrated = use_dehydrate(
            slice_key, hydrate=lambda rec: dict(rec) if isinstance(rec, Mapping) else None,
            dehydrate=lambda: current.current.adehydrate() if current.current else None)

    if isinstance(spec, SandboxSpec):
        eff_spec = spec
    elif spec:
        eff_spec = env.spec.with_(**dict(spec))
    else:
        eff_spec = env.spec
    drv = get_driver(driver) if driver is not None else env.driver

    registry = hooks.get_service(HANDLES_SERVICE, dict)
    restore = hooks.get_service(RESTORE_SERVICE, dict)

    def make() -> SandboxHandle:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        h = SandboxHandle(
            key=k, store_key=f"{sid}/{k}", driver=drv, spec=eff_spec,
            store=store if store is not None else env.store,
            on_close=on_close or env.on_close,
            autosnapshot=autosnapshot if autosnapshot is not None else env.autosnapshot,
            bus=bus, owner=ctx, source=path, home_loop=loop, session=session)
        legacy = restore.pop(h.store_key, None)  # a deprecated SandboxParticipant
        if owns_slice:
            rec, origin = hydrated, (path, slice_key)
        else:
            rec, origin = env.restore.pop(k, None), (env.path, "sandboxes")
        h.restore_record = rec or legacy
        h.restore_origin = origin if rec else None
        registry[h.store_key] = h  # deprecated: for a SandboxParticipant
        return h

    local = hooks.use_ref(None)
    if scope == "session":
        with env.lock:
            handle = env.handles.get(k)
            if handle is None or handle.closed:
                handle = env.handles[k] = make()
    else:
        handle = local.current
        if handle is None or handle.closed:
            handle = local.current = make()

    current.current = handle

    def _lifecycle():
        if eager:
            handle.start_nowait()

        def _cleanup(reason):
            if reason == "hidden":  # Activity: keep it running
                return
            if reason == "suspend":
                if owns_slice:  # else the provider closes it
                    handle.close_nowait(handle.suspend_close_policy())
                return
            if scope == "component":
                handle.close_nowait()

        return _cleanup

    hooks.use_effect(_lifecycle, [id(handle)])
    use_imperative_handle(ref, lambda: handle, [id(handle)])
    return handle
