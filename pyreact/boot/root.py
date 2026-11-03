"""Roots: the outside world's handle on a tree (cf. React ``createRoot`` /
``hydrateRoot``).

::

    root = create_root(App, session_id="coder-1", store=DirectorySnapshots("state/"))
    root.send("hello", wait=True)          # a new turn (was session.invoke)
    root.suspend(on_active="interrupt")    # hide (cleanups: "suspend") -> dehydrate -> store
    ...
    root = hydrate_root(App, session_id="coder-1", store=DirectorySnapshots("state/"))

A root sits on top of an :class:`~pyreact.boot.app_runner.AppRunner` (the
loop) and one :class:`~pyreact.boot.session.Session` (the tree and its
services). ``runner=None`` creates a runner owned by the root (shut down
with it); pass ``runner=`` to host many roots on one loop.

Outside code talks to the tree only through the root (ADR 0001): input with
:meth:`Root.send`, imperative handles with :meth:`Root.ref` (named refs,
``use_imperative_handle(ref, factory, name=...)``), lifecycle with
:meth:`Root.suspend` / :meth:`Root.dehydrate` / :meth:`Root.unmount`,
crash recovery with :meth:`Root.resume_pending`.
:meth:`Root.render_to` renders the tree's ``message`` output to a sink;
:attr:`Root.events` is the session's event log, for devtools.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import time
import uuid
import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Union

from pyreact.core import runtime
from pyreact.core.activity import hide_tree, reveal_tree
from pyreact.core.dehydrate import (
    HydrationWarning,
    Snapshot,
    SnapshotError,
    adehydrate_tree,
    check_snapshot,
    finish_hydration,
    hydration_of,
)
from pyreact.core.refs import REFS_SERVICE

from .session import Session
from .snapshots import SnapshotStore, load_snapshot
from .stateful import SuspendError

if TYPE_CHECKING:  # pragma: no cover
    from .app_runner import AppRunner

__all__ = ["Root", "create_root", "hydrate_root", "acreate_root", "ahydrate_root"]

# Upper bound on waiting for async effect cleanups during a suspend.
CLEANUP_TIMEOUT = 30.0


def _new_runner(app: Callable) -> "AppRunner":
    from .app_runner import AppRunner

    return AppRunner(app, default_session=False)


class Root:
    """A mounted tree. Built by :func:`create_root` / :func:`hydrate_root`."""

    def __init__(self, runner: "AppRunner", session: Session, *,
                 store: Optional[SnapshotStore] = None, owns_runner: bool = False) -> None:
        self.runner = runner
        self.session = session  # low-level escape hatch
        self.store = store
        self._owns_runner = owns_runner
        self._released = False

    def __repr__(self) -> str:
        return f"<Root {self.id!r}{' unmounted' if self.closed else ''}>"

    def __enter__(self) -> "Root":
        return self

    def __exit__(self, *exc: Any) -> None:
        if not self.closed:
            self.unmount()
        self._release_runner()

    # ----------------------------------------------------------------- state
    @property
    def id(self) -> str:
        return self.session.id

    @property
    def closed(self) -> bool:
        return self.session.closed

    @property
    def events(self):
        """The session's event log (``EventLog``): devtools / observability."""
        return self.session.events

    @property
    def observer(self):
        """The session's Observer (``events`` / ``usage`` / ``timeline``)."""
        return self.session.observer

    @property
    def event_bus(self):
        """The session's EventBus (subscribe for devtools; not for rendering)."""
        return self.session.event_bus

    # ---------------------------------------------------------------- output
    def render_to(self, sink: Any = "console", *, kinds: Optional[Iterable[str]] = None,
                  replay: bool = True) -> Callable[[], None]:
        """Render this root's output -- ``message`` events (what ``Message``
        components publish) and ``error`` events -- to ``sink``; returns the
        unsubscribe.

        ``sink``: ``"console"`` (a :class:`~pyreact.observability.ConsoleSink`
        on stdout), a sink object (``Sink`` subclass: it applies its own
        ``kinds`` filter) or any callable ``fn(event)``. ``kinds`` overrides
        what is rendered (default ``("message", "error")``; for a ``Sink`` the
        default is its own filter). With ``replay`` (default) the recent
        matching events, e.g. those of the first render, are rendered first.
        The root-level renderer of ADR 0001; bus subscriptions stay for
        devtools."""
        from pyreact.observability import ConsoleSink, Sink

        if isinstance(sink, str) and sink == "console":
            sink = ConsoleSink(kinds=tuple(kinds) if kinds is not None else ConsoleSink.DEFAULT_KINDS)
            kinds = None  # the sink filters
        elif isinstance(sink, str):
            raise ValueError(f"unknown sink {sink!r} (use 'console', a Sink or a callable)")
        elif not callable(sink):
            raise TypeError(f"render_to needs 'console', a Sink or a callable, got {sink!r}")
        elif kinds is None and not isinstance(sink, Sink):
            kinds = ConsoleSink.DEFAULT_KINDS
        return self.events.subscribe(sink, tuple(kinds) if kinds is not None else None,
                                     replay=replay)

    @property
    def mismatches(self) -> List[Dict[str, Any]]:
        """Hydration mismatches of this root (empty for :func:`create_root`)."""
        state = hydration_of(self.session.root) if self.session.root is not None else None
        return list(state.mismatches) if state is not None else []

    # ----------------------------------------------------------------- input
    def send(self, text: str, *, wait: bool = False, timeout: Optional[float] = None,
             source: str = "term") -> str:
        """Submit ``text`` as a new turn; returns its id. ``wait=True`` blocks
        until it ends (``TimeoutError`` after ``timeout`` s). From inside an
        event loop use :meth:`asend`."""
        return self.session.invoke(text, wait=wait, timeout=timeout, source=source)

    async def asend(self, text: str, *, wait: bool = False, timeout: Optional[float] = None,
                    source: str = "term") -> str:
        """Async :meth:`send`."""
        return await self.session.ainvoke(text, wait=wait, timeout=timeout, source=source)

    def wait_turn(self, turn_id: str, timeout: Optional[float] = None) -> dict:
        """Block until ``turn_id`` ends; returns ``{"status", "duration"}``."""
        return self.session.wait_turn(turn_id, timeout=timeout)

    def wait_idle(self, timeout: Optional[float] = None) -> None:
        """Block until the tree has no queued renders or pending tasks."""
        self.session.wait_idle(timeout=timeout)

    async def await_idle(self, timeout: Optional[float] = None) -> None:
        await self.session.await_idle(timeout=timeout)

    def resume_pending(self, *, wait: bool = False,
                       timeout: Optional[float] = None) -> List[str]:
        """Re-run the turns this root's ``TurnStore`` saw start but never end
        (crash recovery without a snapshot: e.g. a ``Durable`` log from a
        process that died mid-turn). Same ids and text, ``resumed=True``,
        oldest first, queued like new input. Returns their ids (``[]``
        without a store or with nothing pending); ``wait=True`` blocks until
        they end (``TimeoutError`` after ``timeout`` s).

        Waits for the tree to be idle first, so a store attached on mount
        (by an effect) is in place. :func:`hydrate_root` already resumes
        pending turns; this is for a root made with :func:`create_root`."""
        self.session.wait_idle(timeout=timeout)
        return self.session.resume_pending(wait=wait, timeout=timeout)

    # ------------------------------------------------------------------ refs
    def ref(self, name: str) -> Any:
        """The handle a component exposed under ``name``
        (``use_imperative_handle(ref, factory, name=name)`` or a ``ref="name"``
        prop). ``KeyError`` if no mounted component holds that name (handles
        are assigned after the first commit)."""
        registry = self.session.services.get(REFS_SERVICE) or {}
        try:
            return registry[name]
        except KeyError:
            known = sorted(registry)
            raise KeyError(f"no ref {name!r} on root {self.id!r} (known: {known})") from None

    # ------------------------------------------------------------- lifecycle
    def dehydrate(self) -> Snapshot:
        """Snapshot of the live tree (nothing is hidden or unmounted)."""
        return self._call(self._dehydrate_on_loop())

    async def adehydrate(self) -> Snapshot:
        return await self._acall(self._dehydrate_on_loop())

    def suspend(self, *, on_active: str = "wait", timeout: Optional[float] = None) -> Snapshot:
        """Suspend the tree; returns its snapshot (saved to ``store``, if any).

        1. ``session.suspending``; new input is refused, queued turns held;
        2. the active turn ends: ``on_active="wait"`` lets it finish (after
           ``timeout`` s it is interrupted), ``"interrupt"`` cancels it now; a
           cut turn is carried (re-run by :func:`hydrate_root`);
        3. the tree is hidden: every effect cleanup runs with the reason
           ``"suspend"`` (pause / snapshot instead of destroy), async ones
           are awaited;
        4. the tree is dehydrated (``use_dehydrate``) and saved to ``store``;
           ``session.suspended {ref, ...}``;
        5. the tree unmounts (and an owned runner stops).

        If 3-4 fail nothing is unmounted: effects re-run, held turns start,
        a cut turn is re-queued, ``session.suspend.failed`` is published and
        :class:`~pyreact.boot.stateful.SuspendError` raised.
        """
        if self.runner._on_loop_thread():
            raise RuntimeError("Root.suspend() blocks; use 'await root.asuspend()' on the loop")
        snap = self._call(self._suspend_on_loop(on_active, timeout))
        self._release_runner()
        return snap

    async def asuspend(self, *, on_active: str = "wait",
                       timeout: Optional[float] = None) -> Snapshot:
        """Async :meth:`suspend`."""
        snap = await self._acall(self._suspend_on_loop(on_active, timeout))
        await self._arelease_runner()
        return snap

    def unmount(self) -> None:
        """Unmount the tree (cleanups: ``"unmount"``); no snapshot is taken."""
        if not self.closed and self.runner.is_running:
            self.runner.close_session(self.id)
        self._release_runner()

    async def aunmount(self) -> None:
        if not self.closed and self.runner.is_running:
            await self.runner.aclose_session(self.id)
        await self._arelease_runner()

    # -------------------------------------------------------------- internals
    def _call(self, coro):
        return self.runner._submit(coro).result()

    async def _acall(self, coro):
        return await asyncio.wrap_future(self.runner._submit(coro))

    def _release_runner(self) -> None:
        if not self._owns_runner or self._released:
            return
        self._released = True
        if self.runner._on_loop_thread():
            threading.Thread(target=self.runner.shutdown, daemon=True).start()
        else:
            self.runner.shutdown()

    async def _arelease_runner(self) -> None:
        if not self._owns_runner or self._released:
            return
        if self.runner._on_loop_thread():
            self._release_runner()
        else:
            self._released = True
            await asyncio.to_thread(self.runner.shutdown)

    async def _dehydrate_on_loop(self) -> Snapshot:
        sess = self.session
        if sess.closed or sess.root is None:
            raise RuntimeError(f"root {self.id!r} is unmounted")
        await sess._drain()
        return await adehydrate_tree(sess.root, session_id=sess.id)

    async def _suspend_on_loop(self, on_active: str, timeout: Optional[float]) -> Snapshot:
        store = self.store
        snap, _ref = await suspend_session(
            self.session, on_active=on_active, timeout=timeout, reason="suspend",
            save=store.save if store is not None else None)
        return snap


# ---------------------------------------------------------------------------
# The suspend / hydrate machinery (Root and the deprecated Session.suspend /
# AppRunner.resume wrappers share it)
# ---------------------------------------------------------------------------
async def suspend_session(sess: Session, *, on_active: str, timeout: Optional[float],
                          reason: str = "suspend",
                          save: Optional[Callable[[Snapshot, Dict[str, Any]], Any]] = None,
                          before_hide: Optional[Callable[[], Any]] = None,
                          payload: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                          ) -> tuple:
    """Suspend ``sess`` (see :meth:`Root.suspend`); returns ``(snapshot, ref)``.

    ``save(snapshot)`` persists it (its return value is the ``ref``; with
    ``before_hide`` it also gets that coroutine's result: ``save(snapshot,
    extra)``). ``before_hide()`` (async) runs once the active turn has
    settled, before the tree is hidden (legacy participants); ``payload(extra)``
    adds fields to ``session.suspended``. Must run on the runner loop."""
    if on_active not in ("wait", "interrupt"):
        raise ValueError(f"on_active must be 'wait' or 'interrupt', got {on_active!r}")
    sess._check_open()
    bus = sess.event_bus
    sess.suspending = True
    sess._suspend_carry = []
    bus.publish("session.suspending", {
        "session_id": sess.id, "reason": reason, "on_active": on_active})
    sess.turns.hold()
    t0 = time.time()
    hidden = False
    extra: Dict[str, Any] = {}
    try:
        await sess._settle_active(on_active, timeout)
        await sess._drain()
        if before_hide is not None:
            extra = await before_hide() or {}
        tasks = hide_tree(sess.root, "suspend")
        hidden = True
        if tasks:
            await asyncio.wait(tasks, timeout=CLEANUP_TIMEOUT)
        await sess._drain()
        try:
            snap = await adehydrate_tree(sess.root, session_id=sess.id,
                                         carry=sess._suspend_carry)
        except SnapshotError as exc:
            raise SuspendError(str(exc)) from exc
        ref = None
        if save is not None:
            try:
                ref = save(snap, extra) if before_hide is not None else save(snap)
            except Exception as exc:
                raise SuspendError(f"could not save the snapshot: "
                                   f"{type(exc).__name__}: {exc}") from exc
    except BaseException as exc:
        sess.suspending = False
        carry, sess._suspend_carry = sess._suspend_carry, []
        if hidden:
            reveal_tree(sess.root)
        sess.turns.release()
        for t in carry:
            try:
                sess.turns.enqueue(replace(t, resumed=True, ts=time.time()))
            except Exception:
                pass
        bus.publish("session.suspend.failed", {
            "session_id": sess.id, "reason": reason,
            "error": f"{type(exc).__name__}: {exc}"})
        if isinstance(exc, SuspendError) or not isinstance(exc, Exception):
            raise
        raise SuspendError(f"suspend of session {sess.id!r} failed: {exc}") from exc
    bus.publish("session.suspended", {
        "session_id": sess.id, "ref": ref,
        "turn_seq": int(snap["turns"].get("next_index", 1)) - 1,
        "slices": sum(len(v) for v in snap["slices"].values()),
        "carried_turns": [t["id"] for t in snap["turns"].get("carry", [])],
        **(payload(extra) if payload is not None else {}),
        "duration": round(time.time() - t0, 4)})
    sess.runner._close_on_loop(sess.id, "suspend")
    return snap, ref


# ---------------------------------------------------------------------------
# create / hydrate
# ---------------------------------------------------------------------------
def _check_free(runner: "AppRunner", session_id: str) -> None:
    cur = runner.get_session(session_id)
    if cur is not None and not cur.closed:
        raise RuntimeError(f"session {session_id!r} is already open on this runner")


def create_root(app: Callable, *, session_id: Optional[str] = None,
                store: Optional[SnapshotStore] = None, runner: Optional["AppRunner"] = None,
                **session_opts: Any) -> Root:
    """Mount ``app`` as a fresh tree (no hydration) and return its root.

    ``session_id`` defaults to a random hex id; ``store`` is where
    :meth:`Root.suspend` saves the snapshot; ``runner`` hosts the tree
    (default: a new runner owned by the root). ``session_opts`` go to
    ``AppRunner.open_session`` (e.g. ``turn_policy="interrupt"``). Blocks
    until the first render is committed.
    """
    owns = runner is None
    runner = runner if runner is not None else _new_runner(app)
    sid = session_id or uuid.uuid4().hex
    try:
        _check_free(runner, sid)
        sess = runner.open_session(sid, app=app, **session_opts)
    except BaseException:
        if owns:
            runner.shutdown()
        raise
    return Root(runner, sess, store=store, owns_runner=owns)


async def acreate_root(app: Callable, **kw: Any) -> Root:
    """:func:`create_root` without blocking the calling event loop."""
    return await asyncio.to_thread(create_root, app, **kw)


SnapshotSource = Union[Snapshot, str, "os.PathLike[str]", None]


def _resolve_snapshot(session_id: str, snapshot: SnapshotSource,
                      store: Optional[SnapshotStore], if_missing: str = "raise") -> tuple:
    """-> (snapshot, store it was loaded from or None); ``(None, None)`` when
    ``store`` has no snapshot and ``if_missing="create"``."""
    if snapshot is None:
        if store is None:
            raise ValueError("hydrate_root needs snapshot= or store=")
        found = store.load(session_id)
        if found is None:
            if if_missing == "create":
                return None, None
            raise KeyError(f"no snapshot for session {session_id!r} in {store!r}")
        return check_snapshot(found), store
    if isinstance(snapshot, dict):
        return check_snapshot(snapshot), None
    return load_snapshot(snapshot), None


async def _hydrate_on_loop(runner: "AppRunner", session_id: str, app: Callable,
                           snapshot: Snapshot, *, turn_policy: Optional[str],
                           resume_turns: bool, consume_from: Optional[SnapshotStore],
                           wait: bool, legacy: Any = None) -> Session:
    """Open ``session_id`` hydrated from ``snapshot`` (see :func:`hydrate_root`).

    ``legacy`` (the deprecated ``AppRunner.resume``): an object with async
    ``before_mount(sess)`` / ``after_hydrate(sess)`` and ``payload(sess) ->
    dict`` (extra ``session.resumed`` fields); a snapshot whose ``slices`` is
    ``None`` mounts without hydration (a manifest from before use_dehydrate)."""
    from pyreact.core.turns import POLICIES

    t0 = time.time()
    with runner._sessions_lock:
        cur = runner._sessions.get(session_id)
        if cur is not None and not cur.closed:
            raise RuntimeError(f"session {session_id!r} is already open on this runner")
        policy = turn_policy or (snapshot.get("turns") or {}).get("policy") or runner._turn_policy
        if policy not in POLICIES:
            policy = runner._turn_policy
        sess = runner._new_session(session_id, app, policy)
    sess.resumed_from = snapshot
    sess._resume_turns = bool(resume_turns)
    runner._opened(sess, resumed=True)
    carried = sess.turns.hydrate(snapshot.get("turns"))
    if legacy is not None:
        await legacy.before_mount(sess)
    slices = snapshot.get("slices")
    if legacy is None and slices is None:
        slices = {}
    sess._mount(slices=slices if isinstance(slices, dict) else None)
    # first render + async hydrates (which may mount more hydrating children)
    state = hydration_of(sess.root)
    while True:
        await sess._drain()
        pending = [t for t in state.pending if not t.done()] if state is not None else []
        if not pending:
            break
        await asyncio.wait(pending)
    await sess._drain()
    mismatches = finish_hydration(sess.root)
    if legacy is not None:
        await legacy.after_hydrate(sess)
    if resume_turns:  # once hydration completed: the tree is as suspended
        for t in carried:
            sess.turns.enqueue(t)
            sess._resumed_turns.append(t.id)
        for t in sess.turns.pending_turns():  # a TurnStore saw a crash mid-turn
            try:
                sess.turns.resume(t.id)
                sess._resumed_turns.append(t.id)
            except Exception as exc:
                runtime.report_error(exc, sess.root, "resume")
    if consume_from is not None:
        try:
            consume_from.delete(session_id)
        except Exception as exc:
            runtime.report_error(exc, sess.root, "hydrate")
    sess.event_bus.publish("session.resumed", {
        "session_id": session_id, "snapshot": legacy is None,
        "created_at": snapshot.get("created_at"),
        "slices": sum(len(v) for v in (slices or {}).values()),
        "mismatches": len(mismatches), "turns": list(sess._resumed_turns),
        **(legacy.payload(sess) if legacy is not None else {}),
        "duration": round(time.time() - t0, 4)})
    if wait:
        for tid in list(sess._resumed_turns):
            await sess.turns.wait(tid)
    return sess


def hydrate_root(app: Callable, *, session_id: str, store: Optional[SnapshotStore] = None,
                 snapshot: SnapshotSource = None, runner: Optional["AppRunner"] = None,
                 resume_turns: bool = True, consume: bool = True, wait: bool = False,
                 timeout: Optional[float] = None, turn_policy: Optional[str] = None,
                 if_missing: str = "create") -> Root:
    """Mount ``app`` hydrated from a snapshot and return its root.

    ``snapshot``: a snapshot dict, a path to one, or ``None`` to load
    ``session_id``'s from ``store``. When ``store`` has none,
    ``if_missing="create"`` (default; React's ``hydrateRoot`` on an empty
    container just renders) mounts a fresh tree, as :func:`create_root`, and
    reports it -- a :class:`~pyreact.core.HydrationWarning` and a
    ``session.resume.degraded {participant: "snapshot", reason: "no
    snapshot"}`` event (``root.session.degraded``); with ``resume_turns`` the
    turns its ``TurnStore`` saw start but never end still re-run.
    ``if_missing="raise"`` raises ``KeyError`` instead.
    ``use_dehydrate`` hooks get their slice **during their first render**;
    then (async hydrates resolved) mismatches are reported
    (``hydration.mismatch``, never fatal), turn numbering continues and, with
    ``resume_turns``, the carried turns re-run (same ids, ``resumed``).
    A snapshot loaded from ``store`` is deleted from it once hydrated
    (``consume``); the root suspends into ``store`` again. ``wait=True``
    blocks until the re-run turns end. Returns once hydrated.
    """
    if if_missing not in ("create", "raise"):
        raise ValueError(f"if_missing must be 'create' or 'raise', got {if_missing!r}")
    snap, src = _resolve_snapshot(session_id, snapshot, store, if_missing)
    if snap is None:
        return _create_missing(app, session_id=session_id, store=store, runner=runner,
                               resume_turns=resume_turns, wait=wait, timeout=timeout,
                               turn_policy=turn_policy)
    if snap.get("session_id") not in (None, session_id):
        raise SnapshotError(f"snapshot is for session {snap.get('session_id')!r}, "
                            f"not {session_id!r}")
    owns = runner is None
    runner = runner if runner is not None else _new_runner(app)
    try:
        if runner._on_loop_thread():
            raise RuntimeError("hydrate_root() blocks; call it off the runner's loop")
        fut = runner._submit(_hydrate_on_loop(
            runner, session_id, app, snap, turn_policy=turn_policy,
            resume_turns=resume_turns, consume_from=src if consume else None, wait=wait))
        try:
            sess = fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"root {session_id!r} not hydrated after {timeout}s") from None
    except BaseException:
        if owns:
            runner.shutdown()
        raise
    return Root(runner, sess, store=store, owns_runner=owns)


def _create_missing(app: Callable, *, session_id: str, store: SnapshotStore,
                    runner: Optional["AppRunner"], resume_turns: bool, wait: bool,
                    timeout: Optional[float], turn_policy: Optional[str]) -> Root:
    """``hydrate_root(if_missing="create")`` with nothing in ``store``."""
    warnings.warn(HydrationWarning(
        f"hydration mismatch: no snapshot for session {session_id!r} in {store!r}; "
        "mounting a fresh tree"), stacklevel=3)
    opts = {"turn_policy": turn_policy} if turn_policy is not None else {}
    root = create_root(app, session_id=session_id, store=store, runner=runner, **opts)
    try:
        root.session.degrade("snapshot", "no snapshot", store=repr(store))
        if resume_turns:
            root.resume_pending(wait=wait, timeout=timeout)
    except BaseException:
        root.unmount()
        raise
    return root


async def ahydrate_root(app: Callable, **kw: Any) -> Root:
    """:func:`hydrate_root` without blocking the calling event loop."""
    return await asyncio.to_thread(hydrate_root, app, **kw)
