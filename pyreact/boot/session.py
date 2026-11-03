# session.py ---------------------------------------------------------------
"""Sessions: independent app trees hosted by one :class:`AppRunner` loop.

The render runtime (``pyreact.core.runtime``) is process-global: one queue,
one loop. A *session* is one root :class:`HookContext` mounted on that loop
with its own services dict, so ``hooks.get_service(...)`` inside the session
resolves to the session's own services:

    ``input_bus``    InputBus
    ``nav_service``  NavService
    ``event_bus``    EventBus (observed by the session's ``observer``)
    ``turns``        TurnManager (own queue/active turn/policy)
    ``session``      this Session

Sessions never share state, input, turns or navigation; they only share the
loop (and process-global services resolved with ``HookContext.get_service``
as a class call).

Typical use::

    app = bootstrap(App)                      # opens the default session
    s = app.open_session("user-42")           # another, isolated tree
    tid = s.invoke("hi", wait=True, timeout=30)
    s.event_bus.subscribe(print, ["message", "turn.*"])
    s.usage, s.timeline, s.events             # per-session observability
    app.close_session("user-42")
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from pyreact.core import runtime
from pyreact.core.dehydrate import SnapshotError, adehydrate_tree, begin_hydration
from pyreact.core.events import EventBus
from pyreact.core.hook import HookContext
from pyreact.core.turns import TurnManager
from pyreact.input.bus import InputBus
from pyreact.observability import Observer
from pyreact.router.nav_service import NavService

from .stateful import (
    MANIFEST_FORMAT,
    ManifestStore,
    Participants,
    SuspendError,
    TurnsParticipant,
    _jsonable,
    versions,
)

if TYPE_CHECKING:  # pragma: no cover
    from .app_runner import AppRunner

DEFAULT_SESSION_ID = "default"

# Service keys pre-populated in every session's services dict.
INPUT_BUS = "input_bus"
NAV_SERVICE = "nav_service"
EVENT_BUS = "event_bus"
TURNS = "turns"
SESSION = "session"


def session_of(ctx: Any) -> Optional["Session"]:
    """Session owning the component ``ctx`` (a HookContext), if any."""
    root = getattr(ctx, "_root", None)
    services = getattr(root, "_session_services", None)
    if not services:
        return None
    sess = services.get(SESSION)
    return sess if isinstance(sess, Session) else None


def current_session() -> Optional["Session"]:
    """Session on whose behalf the calling code runs, if it can be inferred.

    Works during render, inside effects, inside tasks spawned from effects,
    during turn processing and inside session entry points. Returns ``None``
    for code not attributable to a session (e.g. server logs).
    """
    return session_of(runtime.current_component())


class Session:
    """One isolated app tree on a shared :class:`AppRunner` loop.

    Do not construct directly; use :meth:`AppRunner.open_session`.

    Attributes:
        id: session id.
        input_bus / nav_service / event_bus / turns: this session's services
            (the same objects ``hooks.get_service`` returns inside it).
        observer: EventLog + UsageAggregator + TurnTimeline over ``event_bus``
            (also ``events`` / ``usage`` / ``timeline``).
        services: the session's services dict.
        created_at / last_active: wall-clock timestamps.
        participants: checkpoint participants (see ``pyreact.boot.stateful``).

        resumed_from: the manifest this session was resumed from (or None).
        degraded: what could not be restored on resume (see :meth:`degrade`).

    Lifecycle events on ``event_bus``: ``session.opened`` (before the first
    render), ``session.suspending`` / ``session.suspended`` /
    ``session.suspend.failed`` (:meth:`suspend`), ``session.resumed`` and
    ``session.resume.degraded`` (``AppRunner.resume``), ``session.closed``
    (after the tree unmounted, ``{"reason"}``).
    """

    def __init__(
        self,
        runner: "AppRunner",
        session_id: str,
        app_component_fn: Callable,
        *,
        turn_policy: str = "queue",
        turn_id_prefix: str = "",
    ) -> None:
        self.id = session_id
        self.runner = runner
        self.app_component_fn = app_component_fn
        self.input_bus = InputBus()
        self.nav_service = NavService()
        self.event_bus = EventBus()
        self.turns = TurnManager(
            policy=turn_policy, input_bus=self.input_bus, event_bus=self.event_bus
        ).configure(policy=turn_policy, id_prefix=turn_id_prefix, owner_filter=self.owns)
        self.services: Dict[str, Any] = {
            INPUT_BUS: self.input_bus,
            NAV_SERVICE: self.nav_service,
            EVENT_BUS: self.event_bus,
            TURNS: self.turns,
            SESSION: self,
        }
        self.observer = Observer().attach(self.event_bus)
        self.participants = Participants()
        self.participants.add(TurnsParticipant())
        self.suspending = False
        self.resumed_from: Optional[Dict[str, Any]] = None
        self.degraded: List[Dict[str, Any]] = []
        self._suspend_carry: list = []  # turns a suspend cut short
        self._resumed_turns: List[str] = []
        self._resume_turns = True
        self._unsubs: list = []
        self.root: Optional[HookContext] = None
        self.created_at = self.last_active = time.time()
        self.closed = False
        # Why the session was closed (set before its tree unmounts, so
        # cleanups can read it): "idle" (reaped after its TTL), "closed"
        # (close_session), "shutdown" (the runner stopped) or "suspend"
        # (suspend(): its state lives on in a manifest). None while open.
        self.close_reason: Optional[str] = None

        def _on_nav(url: str) -> None:
            self.event_bus.publish("nav", {"path": url}, source="nav")

        self._nav_listener = _on_nav
        self.nav_service.subs.append(_on_nav)

    def __repr__(self) -> str:
        return f"<Session {self.id!r}{' closed' if self.closed else ''}>"

    # Observability shortcuts
    @property
    def events(self):
        return self.observer.log

    @property
    def usage(self):
        return self.observer.usage

    @property
    def timeline(self):
        return self.observer.timeline

    # ------------------------------------------------------------------ utils
    def touch(self) -> None:
        self.last_active = time.time()

    def owns(self, obj: Any) -> bool:
        """True if ``obj`` (a HookContext or task owner) belongs to this session."""
        root = self.root
        return root is not None and getattr(obj, "_root", None) is root

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError(f"session {self.id!r} is closed")
        if self.suspending:
            raise RuntimeError(f"session {self.id!r} is suspending")

    def degrade(self, participant: str, reason: str, **info: Any) -> None:
        """Record that ``participant`` could not restore everything on
        resume (the session goes on with what it has) and publish
        ``session.resume.degraded {session_id, participant, reason, **info}``.
        Participants call it themselves (e.g. a sandbox whose snapshot is gone
        boots fresh); ``AppRunner.resume`` calls it when one raises."""
        rec = {"participant": participant, "reason": reason, **info}
        self.degraded.append(rec)
        try:
            self.event_bus.publish("session.resume.degraded", {"session_id": self.id, **rec})
        except Exception:
            pass

    # ---------------------------------------------------------- loop-side ops
    def _mount(self, slices: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Create and schedule the root. Must run on the runner loop.

        ``slices`` (``{path: {key: value}}``): hydrate the tree from them
        (``use_dehydrate`` hooks get their slice during the first render)."""
        self.root = HookContext(
            self.app_component_fn.__name__,
            self.app_component_fn,
            services=self.services,
        )
        if slices is not None:
            begin_hydration(self.root, slices)
        self.turns.configure(owner=self.root, loop=asyncio.get_running_loop())
        runtime.schedule_rerender(self.root, reason=f"session {self.id} mount")

    async def _drain(self) -> None:
        """Commit renders queued for this session (not its background tasks)."""
        await asyncio.sleep(0)
        while runtime.has_queued(self.owns):
            await runtime.run_renders()
            await asyncio.sleep(0)

    def _unmount(self, reason: str = "closed") -> None:
        """Unmount the tree and stop its turns and tasks. Must run on the
        runner loop."""
        self.closed = True
        self.close_reason = reason
        try:
            self.nav_service.subs.remove(self._nav_listener)
        except ValueError:
            pass
        pump = getattr(self.turns, "_pump", None)
        if pump is not None and not pump.done():
            pump.cancel()
        # queued turns will never run here: release their waiters
        self.turns.abandon("suspended" if reason == "suspend" else "closed")
        runtime.set_active_turn(None, scope=self.turns)
        root = self.root
        if root is not None:
            try:
                # cleanups see use_cleanup_reason() == "suspend" | "unmount"
                root.unmount("suspend" if reason == "suspend" else "unmount")
            except Exception as exc:
                runtime.report_error(exc, root, "unmount")
            for t in runtime.pending_tasks(self.owns):
                t.cancel()
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs.clear()
        self.event_bus.publish(
            "session.closed", {"session_id": self.id, "reason": reason}
        )

    # ---------------------------------------------------------------- suspend
    async def _settle_active(self, on_active: str, timeout: Optional[float]) -> None:
        """Let the active turn end (or cut it short); remember a cut turn."""
        active = self.turns.active
        if active is None:
            return
        aborted = False
        if on_active == "interrupt":
            self.turns.abort_active()
            aborted = True
        elif timeout is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.turns.wait(active.id)), timeout)
            except asyncio.TimeoutError:
                self.turns.abort_active()
                aborted = True
        res = await self.turns.wait(active.id)
        if aborted and res.get("status") == "interrupted":
            self._suspend_carry.append(active)

    async def _suspend_on_loop(self, on_active: str, timeout: Optional[float],
                               store: Optional[ManifestStore], reason: str) -> Dict[str, Any]:
        if on_active not in ("wait", "interrupt"):
            raise ValueError(f"on_active must be 'wait' or 'interrupt', got {on_active!r}")
        self._check_open()
        store = store if store is not None else self.runner.manifest_store
        self.suspending = True
        self._suspend_carry = []
        self.event_bus.publish("session.suspending", {
            "session_id": self.id, "reason": reason, "on_active": on_active})
        self.turns.hold()
        t0 = time.time()
        try:
            await self._settle_active(on_active, timeout)
            await self._drain()
            fragments: Dict[str, Any] = {}
            for p in self.participants:
                try:
                    frag = await p.suspend(self)
                except Exception as exc:
                    raise SuspendError(f"participant {p.name!r} failed to suspend: "
                                       f"{type(exc).__name__}: {exc}") from exc
                fragments[p.name] = _jsonable(frag if frag is not None else {}, p.name)
            # use_dehydrate slices ride along (hydrated by AppRunner.resume)
            try:
                slices = (await adehydrate_tree(self.root, session_id=self.id))["slices"]
            except SnapshotError as exc:
                raise SuspendError(str(exc)) from exc
            app = self.app_component_fn
            manifest = {
                "format": MANIFEST_FORMAT,
                "session_id": self.id,
                "created_at": self.created_at,
                "suspended_at": time.time(),
                "reason": reason,
                "turn_seq": self.turns.next_index() - 1,
                "turn_policy": self.turns.policy,
                "app": f"{getattr(app, '__module__', '?')}:{getattr(app, '__qualname__', app)}",
                "participants": fragments,
                "slices": slices,
                "versions": versions(self.participants),
            }
            try:
                ref = store.save(manifest)
            except Exception as exc:
                raise SuspendError(f"could not save the manifest: {type(exc).__name__}: {exc}"
                                   ) from exc
        except BaseException as exc:
            # The session goes on: let queued turns start again and re-run
            # the turn we cut short.
            self.suspending = False
            carry, self._suspend_carry = self._suspend_carry, []
            self.turns.release()
            for t in carry:
                try:
                    self.turns.enqueue(replace(t, resumed=True, ts=time.time()))
                except Exception:
                    pass
            self.event_bus.publish("session.suspend.failed", {
                "session_id": self.id, "reason": reason, "error": f"{type(exc).__name__}: {exc}"})
            if isinstance(exc, SuspendError) or not isinstance(exc, Exception):
                raise
            raise SuspendError(f"suspend of session {self.id!r} failed: {exc}") from exc
        manifest["ref"] = ref
        self.event_bus.publish("session.suspended", {
            "session_id": self.id, "ref": ref, "turn_seq": manifest["turn_seq"],
            "participants": list(fragments), "carried_turns": [
                t["id"] for t in fragments.get("turns", {}).get("carry", [])],
            "duration": round(time.time() - t0, 4)})
        self.runner._close_on_loop(self.id, "suspend")
        for p in self.participants:
            fn = getattr(p, "suspended", None)
            if callable(fn):
                try:
                    await fn(self)
                except Exception as exc:
                    runtime.report_error(exc, None, "suspend")
        return manifest

    def suspend(self, *, on_active: str = "wait", timeout: Optional[float] = None,
                store: Optional[ManifestStore] = None, reason: str = "manual") -> Dict[str, Any]:
        """Suspend this agent: one explicit, observable operation.

        1. ``session.suspending``; new input is refused, queued turns are held;
        2. the active turn ends: ``on_active="wait"`` (default) lets it finish
           (after ``timeout`` seconds it is interrupted), ``"interrupt"``
           cancels it now -- an interrupted turn is *carried*: re-run on resume;
        3. every participant's ``suspend(session)`` returns its fragment;
        4. the manifest is saved to ``store`` (default: the runner's
           ``manifest_store``), ``session.suspended`` is published;
        5. the tree unmounts (``close_reason == "suspend"``, ``session.closed``),
           then participants' optional ``suspended(session)`` run.

        Returns the manifest (plus ``"ref"``: where the store put it). If a
        participant or the store fails, nothing is unmounted: the session
        goes on (held turns start, a cut turn is re-queued),
        ``session.suspend.failed`` is published and :class:`SuspendError`
        raised. Do not call from inside one of this session's turns with
        ``on_active="wait"`` (it would wait for itself).
        """
        if self.runner._on_loop_thread():
            raise RuntimeError("Session.suspend() blocks; use 'await session.asuspend()' on the loop")
        return self.runner._submit(self._suspend_on_loop(on_active, timeout, store, reason)).result()

    async def asuspend(self, *, on_active: str = "wait", timeout: Optional[float] = None,
                       store: Optional[ManifestStore] = None,
                       reason: str = "manual") -> Dict[str, Any]:
        """Async :meth:`suspend`."""
        fut = self.runner._submit(self._suspend_on_loop(on_active, timeout, store, reason))
        return await asyncio.wrap_future(fut)

    async def _enqueue_turn(self, turn, wait: bool) -> Dict[str, Any]:
        self.turns.enqueue(turn)
        if wait:
            return await self.turns.wait(turn.id)
        return {}

    async def _nav(self, dest: str, query, fragment) -> None:
        token = runtime.current_owner.set(self.root)
        try:
            self.nav_service.go(dest, None, query, fragment or "")
        except Exception as exc:
            runtime.report_error(exc, self.root, "nav")
        finally:
            runtime.current_owner.reset(token)

    # ------------------------------------------------------------- public API
    def invoke(
        self,
        text: str,
        *,
        wait: bool = False,
        timeout: Optional[float] = None,
        source: str = "term",
    ) -> str:
        """Submit ``text`` as a new turn of this session; returns the turn id.

        The turn is processed per this session's turn policy; its
        TurnManager emits the ``text``/``submit`` input events (with
        ``turn_id``). With ``wait=True`` blocks until that turn has ended
        (``TimeoutError`` after ``timeout`` s). Other sessions never delay or
        interrupt it. Use :meth:`ainvoke` from inside an event loop.
        """
        self._check_open()
        self.touch()
        turn = self.turns.new_turn(text, source=source)
        fut = self.runner._submit(self._enqueue_turn(turn, wait))
        if wait:
            try:
                fut.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"turn {turn.id} not done after {timeout}s") from None
        return turn.id

    async def ainvoke(
        self,
        text: str,
        *,
        wait: bool = False,
        timeout: Optional[float] = None,
        source: str = "term",
    ) -> str:
        """Async :meth:`invoke` for callers on another event loop."""
        self._check_open()
        self.touch()
        turn = self.turns.new_turn(text, source=source)
        fut = self.runner._submit(self._enqueue_turn(turn, wait))
        await asyncio.wait_for(asyncio.wrap_future(fut), timeout)
        return turn.id

    # ----------------------------------------------------------------- resume
    def pending_turns(self) -> list:
        """Turns this session's ``TurnStore`` saw start but never end (see
        :meth:`TurnManager.pending_turns`); ``[]`` without a store."""
        async def _get():
            return self.turns.pending_turns()

        return self.runner._submit(_get()).result()

    async def _resume(self, turn_ids, wait: bool) -> list:
        turns = [self.turns.resume(tid) for tid in turn_ids]
        if wait:
            for t in turns:
                await self.turns.wait(t.id)
        return [t.id for t in turns]

    def resume_turn(self, turn_id: str, *, wait: bool = False,
                    timeout: Optional[float] = None) -> str:
        """Re-run a recorded turn (same id/text; see
        :meth:`TurnManager.resume`). Returns the turn id; with ``wait=True``
        blocks until it ends."""
        self._check_open()
        self.touch()
        fut = self.runner._submit(self._resume([turn_id], wait))
        try:
            return fut.result(timeout=timeout)[0]
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"turn {turn_id} not done after {timeout}s") from None

    def resume_pending(self, *, wait: bool = False,
                       timeout: Optional[float] = None) -> list:
        """Re-run every turn that started but never ended (oldest first,
        queued like new input). Returns their ids."""
        self._check_open()
        ids = [t.id for t in self.pending_turns()]
        if not ids:
            return []
        fut = self.runner._submit(self._resume(ids, wait))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"resumed turns not done after {timeout}s") from None

    def wait_turn(self, turn_id: str, timeout: Optional[float] = None) -> dict:
        """Block until ``turn_id`` ends; returns ``{"status", "duration"}``."""
        try:
            return self.runner._submit(self.turns.wait(turn_id)).result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"turn {turn_id} not done after {timeout}s") from None

    async def await_turn(self, turn_id: str, timeout: Optional[float] = None) -> dict:
        """Async :meth:`wait_turn`."""
        fut = self.runner._submit(self.turns.wait(turn_id))
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout)

    def wait_idle(self, timeout: Optional[float] = None) -> None:
        """Block until this session has no queued renders or pending tasks."""
        fut = self.runner._submit(runtime.wait_idle(owner_filter=self.owns))
        try:
            fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"session {self.id!r} not idle after {timeout}s") from None

    async def await_idle(self, timeout: Optional[float] = None) -> None:
        """Async :meth:`wait_idle`."""
        fut = self.runner._submit(runtime.wait_idle(owner_filter=self.owns))
        await asyncio.wait_for(asyncio.wrap_future(fut), timeout)

    def nav(
        self, dest: str, query: Optional[dict] = None, fragment: str = ""
    ) -> None:
        """Navigate this session (via its Router if mounted). Non-blocking."""
        if self.closed or not dest:
            return
        self.touch()
        try:
            self.runner._submit(self._nav(dest, query, fragment))
        except RuntimeError:
            pass

    async def anav(
        self, dest: str, query: Optional[dict] = None, fragment: str = ""
    ) -> None:
        """Navigate and wait until the navigation has been applied."""
        if self.closed or not dest:
            return
        self.touch()
        fut = self.runner._submit(self._nav(dest, query, fragment))
        await asyncio.wrap_future(fut)

    def current_route(self) -> dict:
        nav = self.nav_service
        return {
            "path": nav.get_path(),
            "query": nav.get_query_params(),
            "fragment": nav.get_fragment(),
        }

    def print_vnode_tree(self) -> None:
        """Print this session's component tree (stdout attributed to it)."""
        if self.closed or self.root is None:
            return

        async def _task():
            token = runtime.current_owner.set(self.root)
            try:
                self.root.render_tree()
            finally:
                runtime.current_owner.reset(token)

        try:
            self.runner._submit(_task()).result(timeout=1.0)
        except Exception:
            pass

    def close(self) -> None:
        """Unmount this session (same as ``runner.close_session(id)``)."""
        self.runner.close_session(self.id)
