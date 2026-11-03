import asyncio
import concurrent.futures
import inspect
import threading
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional

from pyreact.core.hook import HookContext
from pyreact.core import runtime
from pyreact.core.events import EventBus
from pyreact.core.runtime import run_renders
from pyreact.core.turns import POLICIES, TurnManager
from pyreact.web.console import MessageBuffer
from pyreact.observability import install_session_error_events
from .session import DEFAULT_SESSION_ID, Session, _deprecated
from .stateful import (
    ManifestError,
    ManifestStore,
    MemoryManifestStore,
    SuspendError,
    check_manifest,
    load_manifest,
)

IDLE_POLICIES = ("close", "suspend")


def _accepts_arg(fn: Callable[..., Any]) -> bool:
    """True if ``fn`` can be called with one positional argument."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    for p in sig.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL):
            return True
    return False


class AppRunner:
    """Background runner that owns the render loop and hosts sessions.

    One runner = one loop thread = one (process-global) render queue, hosting
    any number of isolated :class:`Session` trees. Each session has its own
    services (InputBus, NavService, EventBus, TurnManager, Observer).

    Usage (single session, backward compatible)::

        app = AppRunner(Boot)          # or bootstrap(Boot)
        tid = app.invoke("hello")      # -> the default session
        app.usage, app.timeline        # default session's observability
        app.shutdown()

    Multi-session::

        app = AppRunner(Boot, default_session=False)
        a = app.open_session("a"); b = app.open_session("b", turn_policy="interrupt")
        a.invoke("hi", wait=True)      # only a's InputBus sees it
        app.close_session("a")

    ``app_component_fn`` is the root component every session mounts unless
    ``open_session(..., app=Other)`` overrides it.
    """

    def __init__(
        self,
        app_component_fn,
        *,
        fps: int = 20,
        turn_policy: str = "queue",
        session_id: str = DEFAULT_SESSION_ID,
        default_session: bool = True,
        participants: Iterable[Callable[..., Any]] = (),
        manifest_store: Optional[ManifestStore] = None,
        idle_policy: str = "close",
        auto_resume: Optional[bool] = None,
    ):
        """``turn_policy``: default ``"queue"`` or ``"interrupt"`` for every
        session (see ``pyreact.core.turns``); ``open_session`` may override it.

        ``session_id``: id of the default session. ``default_session=False`` starts the loop without mounting
        anything; use :meth:`open_session`.

        ``participants``: factories of checkpoint participants added to every
        session when it opens (see :meth:`add_participant`); registered
        before the default session opens.

        Stateful sessions (see ``pyreact.boot.stateful``):
        ``manifest_store`` is where :meth:`Session.suspend` writes manifests
        and :meth:`resume` reads them (default: in memory).
        ``idle_policy="suspend"`` makes idle reaping
        (:meth:`close_idle_sessions`) suspend sessions instead of closing
        them; ``auto_resume`` (default: on with that policy) makes
        ``open_session(id)`` resume a session that has a manifest.
        """
        if turn_policy not in POLICIES:
            raise ValueError(f"turn_policy must be 'queue' or 'interrupt', got {turn_policy!r}")
        if idle_policy not in IDLE_POLICIES:
            raise ValueError(f"idle_policy must be one of {IDLE_POLICIES}, got {idle_policy!r}")
        self._turn_policy = turn_policy
        self.manifest_store: ManifestStore = (
            manifest_store if manifest_store is not None else MemoryManifestStore())
        self.idle_policy = idle_policy
        self.auto_resume = (idle_policy == "suspend") if auto_resume is None else bool(auto_resume)
        self._app_component_fn = app_component_fn
        self._open_default = bool(default_session)
        self._default_id = session_id
        self._sessions: Dict[str, Session] = {}
        self._sessions_lock = threading.Lock()
        self._default: Optional[Session] = None
        self._session_hooks: List[Dict[str, Optional[Callable[[Session], Any]]]] = []
        self._hooks_lock = threading.Lock()
        participants = list(participants)
        if participants:
            _deprecated("AppRunner(participants=...)",
                        "create_root / hydrate_root (libraries dehydrate their own state)")
        for factory in participants:
            self._add_participant(factory)

        self._fps: int = max(1, int(fps))
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread: threading.Thread = threading.Thread(
            target=self._thread_main, name="pyreact-app-loop", daemon=True
        )
        self._stopping: bool = False
        self._ready: threading.Event = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._bus = None

        self._root_ctx: Optional[HookContext] = None
        self._uninstall_error_events: Optional[Callable[[], None]] = None

        # Web bridge callbacks (set by server): executed from runner thread
        self._on_nav: Optional[Callable[[str], None]] = None
        self._nav_listener = None
        self._on_console: Optional[Callable[[str], None]] = None
        self._console_listener = None
        self._did_enable_web_print: bool = False

        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError("AppRunner failed to start") from self._startup_error

    # -------------------------------
    # Default-session shortcuts (legacy single-session API)
    # -------------------------------
    @property
    def event_bus(self) -> Optional[EventBus]:
        return self._default.event_bus if self._default else None

    @property
    def observer(self):
        return self._default.observer if self._default else None

    @property
    def events(self):
        return self._default.events if self._default else None

    @property
    def usage(self):
        return self._default.usage if self._default else None

    @property
    def timeline(self):
        return self._default.timeline if self._default else None

    @property
    def turns(self) -> Optional[TurnManager]:
        """The default session's :class:`~pyreact.core.turns.TurnManager`."""
        return self._default.turns if self._default else None

    @property
    def is_running(self) -> bool:
        return not self._stopping and self._thread.is_alive()

    def _submit(self, coro):
        if not self.is_running:
            coro.close()
            raise RuntimeError("AppRunner is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _on_loop_thread(self) -> bool:
        return threading.current_thread() is self._thread

    # -------------------------------
    # Sessions
    # -------------------------------
    @property
    def default_session(self) -> Session:
        """The session behind the legacy single-session API."""
        if self._default is None:
            raise RuntimeError(
                "this AppRunner has no default session (default_session=False)"
            )
        return self._default

    @property
    def default_session_id(self) -> str:
        return self._default_id

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._sessions_lock:
            return self._sessions.get(session_id)

    def sessions(self) -> List[Session]:
        """Snapshot of the open sessions."""
        with self._sessions_lock:
            return list(self._sessions.values())

    # -------------------------------
    # Session lifecycle hooks
    # -------------------------------
    def on_session(
        self,
        open: Optional[Callable[[Session], Any]] = None,
        close: Optional[Callable[[Session], Any]] = None,
        *,
        replay: bool = False,
    ) -> Callable[[], None]:
        """Call ``open(session)`` whenever a session opens (or is resumed)
        and ``close(session)`` whenever one closes (or is suspended).
        Returns a function that removes the hooks.

        ``open`` runs on the runner loop *before the session's first render*
        and before ``session.opened`` is published: subscribe to its
        ``event_bus``, add participants, seed services.
        ``close`` runs after the tree unmounted (``session.close_reason`` is
        set). A failing hook is reported, never fatal. ``replay=True`` also
        calls ``open`` now for the sessions already open (from the calling
        thread).
        """
        entry = {"open": open, "close": close}
        with self._hooks_lock:
            self._session_hooks.append(entry)
        if replay and open is not None:
            for sess in self.sessions():
                self._call_hook(open, sess)

        def _remove() -> None:
            with self._hooks_lock:
                try:
                    self._session_hooks.remove(entry)
                except ValueError:
                    pass

        return _remove

    def add_participant(self, factory: Callable[..., Any]) -> Callable[[], None]:
        """Deprecated: libraries dehydrate their own state (``use_dehydrate``);
        suspend with ``root.suspend()``.

        Add a checkpoint participant to every session that opens from now
        on. ``factory(session)`` (or ``factory()``) returns the participant
        (a class works). Returns a function that stops adding it."""
        _deprecated("AppRunner.add_participant()",
                    "create_root / hydrate_root (libraries dehydrate their own state)")
        return self._add_participant(factory)

    def _add_participant(self, factory: Callable[..., Any]) -> Callable[[], None]:
        wants_session = _accepts_arg(factory)

        def _add(sess: Session) -> None:
            sess.participants.add(factory(sess) if wants_session else factory())

        return self.on_session(open=_add)

    def _call_hook(self, fn: Callable[[Session], Any], sess: Session) -> None:
        try:
            fn(sess)
        except Exception as exc:
            runtime.report_error(exc, sess.root, "session-hook")

    def _fire(self, which: str, sess: Session) -> None:
        with self._hooks_lock:
            hooks = [h[which] for h in self._session_hooks if h[which] is not None]
        for fn in hooks:
            self._call_hook(fn, sess)

    def _new_session(self, session_id, app, policy) -> Session:
        sess = Session(
            self,
            session_id,
            app or self._app_component_fn,
            turn_policy=policy,
            # Turn ids must be unique across sessions sharing the loop;
            # the default session keeps the plain "t1", "t2"... ids.
            turn_id_prefix="" if session_id == self._default_id else f"{session_id}/",
        )
        self._sessions[session_id] = sess
        return sess

    def _opened(self, sess: Session, **payload: Any) -> None:
        self._fire("open", sess)
        sess.event_bus.publish("session.opened", {"session_id": sess.id, **payload})

    def _open_on_loop(self, session_id, app, turn_policy=None) -> tuple:
        """Get-or-create (loop thread only). Returns (session, created)."""
        policy = turn_policy or self._turn_policy
        if policy not in POLICIES:
            raise ValueError(f"turn_policy must be one of {POLICIES}, got {policy!r}")
        with self._sessions_lock:
            sess = self._sessions.get(session_id)
            if sess is not None and not sess.closed:
                sess.touch()
                return sess, False
            sess = self._new_session(session_id, app, policy)
        self._opened(sess, resumed=False)
        sess._mount()
        return sess, True

    async def _aopen_on_loop(self, session_id, app, turn_policy=None) -> Session:
        if self.auto_resume:
            cur = self._sessions.get(session_id)
            if cur is None or cur.closed:
                manifest = self.manifest_store.load(session_id)
                if manifest is not None:
                    return await self._aresume_on_loop(
                        session_id, check_manifest(manifest), app, turn_policy,
                        resume_turns=True, wait=False, consume=True,
                        store=self.manifest_store)
        sess, created = self._open_on_loop(session_id, app, turn_policy)
        if created:
            # Commit the first render (and its effects, e.g. input
            # subscriptions) so input sent right after opening is not lost.
            await sess._drain()
        return sess

    def open_session(
        self,
        session_id: Optional[str] = None,
        *,
        app: Optional[Callable] = None,
        turn_policy: Optional[str] = None,
    ) -> Session:
        """Open (or resume, if ``session_id`` is already open) a session.

        ``session_id`` defaults to a random uuid4 hex. ``app`` overrides the root component and
        ``turn_policy`` the runner's default policy for a new session.
        Blocks until the first render is committed. Thread-safe; from inside
        a running event loop prefer :meth:`aopen_session`. Called on the
        runner's own loop thread (e.g. from an effect) it returns immediately
        without waiting for the first render (and never auto-resumes).

        With ``auto_resume``, a session that is not open but has a manifest
        in ``manifest_store`` is resumed (:meth:`resume`) instead.
        """
        sid = session_id or uuid.uuid4().hex
        if self._on_loop_thread():
            return self._open_on_loop(sid, app, turn_policy)[0]
        return self._submit(self._aopen_on_loop(sid, app, turn_policy)).result()

    async def aopen_session(
        self,
        session_id: Optional[str] = None,
        *,
        app: Optional[Callable] = None,
        turn_policy: Optional[str] = None,
    ) -> Session:
        """Async :meth:`open_session` for callers on another event loop."""
        sid = session_id or uuid.uuid4().hex
        return await asyncio.wrap_future(
            self._submit(self._aopen_on_loop(sid, app, turn_policy))
        )

    def _close_on_loop(self, session_id: str, reason: str = "closed") -> bool:
        with self._sessions_lock:
            sess = self._sessions.pop(session_id, None)
        if sess is None:
            return False
        if sess is self._default:
            self._default = None
        sess._unmount(reason)
        self._fire("close", sess)
        return True

    async def _aclose_on_loop(self, session_id: str, reason: str = "closed") -> bool:
        return self._close_on_loop(session_id, reason)

    def close_session(self, session_id: str, *, reason: str = "closed") -> bool:
        """Unmount a session, cancel its turns/tasks and flush its state.
        Returns False if unknown.

        ``reason`` is exposed as ``session.close_reason`` (readable by the
        tree's cleanups) and in the ``session.closed`` event: ``"closed"``
        (default), ``"idle"`` (TTL reaping) or ``"shutdown"`` (runner stop).
        """
        if self._on_loop_thread():
            return self._close_on_loop(session_id, reason)
        if not self.is_running:
            return False
        return self._submit(self._aclose_on_loop(session_id, reason)).result()

    async def aclose_session(self, session_id: str, *, reason: str = "closed") -> bool:
        """Async :meth:`close_session`."""
        if not self.is_running:
            return False
        return await asyncio.wrap_future(
            self._submit(self._aclose_on_loop(session_id, reason))
        )

    def close_idle_sessions(
        self, ttl: float, *, keep: Optional[Callable[[Session], bool]] = None
    ) -> List[str]:
        """Close sessions inactive for more than ``ttl`` seconds.

        The default session is never closed. ``keep(session)`` returning True
        also protects a session (e.g. one with a connected socket). Returns the
        ids closed. Blocking; see :meth:`aclose_idle_sessions`.

        With ``idle_policy="suspend"`` idle sessions are suspended
        (``reason="idle"``) instead; one that fails to suspend stays open
        (the error is reported) and is not in the result.
        """
        closed = []
        for sid in self._idle_session_ids(ttl, keep):
            if self.idle_policy == "suspend":
                sess = self.get_session(sid)
                try:
                    if sess is not None and not sess.closed:
                        self._submit(sess._suspend_on_loop("wait", None, None, "idle")).result()
                        closed.append(sid)
                except (SuspendError, RuntimeError) as exc:
                    runtime.report_error(exc, None, "idle-suspend")
            elif self.close_session(sid, reason="idle"):
                closed.append(sid)
        return closed

    async def aclose_idle_sessions(
        self, ttl: float, *, keep: Optional[Callable[[Session], bool]] = None
    ) -> List[str]:
        """Async :meth:`close_idle_sessions`."""
        closed = []
        for sid in self._idle_session_ids(ttl, keep):
            if self.idle_policy == "suspend":
                sess = self.get_session(sid)
                try:
                    if sess is not None and not sess.closed:
                        await asyncio.wrap_future(self._submit(
                            sess._suspend_on_loop("wait", None, None, "idle")))
                        closed.append(sid)
                except (SuspendError, RuntimeError) as exc:
                    runtime.report_error(exc, None, "idle-suspend")
            elif await self.aclose_session(sid, reason="idle"):
                closed.append(sid)
        return closed

    # -------------------------------
    # Suspend / resume (stateful sessions)
    # -------------------------------
    def suspend_session(self, session_id: str, **kw: Any) -> Dict[str, Any]:
        """:meth:`Session.suspend` by id (``KeyError`` if not open)."""
        sess = self.get_session(session_id)
        if sess is None or sess.closed:
            raise KeyError(f"no open session {session_id!r}")
        return sess.suspend(**kw)

    async def asuspend_session(self, session_id: str, **kw: Any) -> Dict[str, Any]:
        """Async :meth:`suspend_session`."""
        sess = self.get_session(session_id)
        if sess is None or sess.closed:
            raise KeyError(f"no open session {session_id!r}")
        return await sess.asuspend(**kw)

    def _manifest_for(self, session_id: str, manifest: Any, store: Optional[ManifestStore]):
        """-> (manifest dict, store it came from or None)."""
        if manifest is None:
            store = store if store is not None else self.manifest_store
            found = store.load(session_id)
            if found is None:
                raise KeyError(f"no manifest for session {session_id!r} in {store!r}")
            return check_manifest(found), store
        if isinstance(manifest, dict):
            return check_manifest(manifest), None
        return load_manifest(manifest), None  # a path

    async def _aresume_on_loop(self, session_id: str, manifest: Dict[str, Any], app,
                               turn_policy, *, resume_turns: bool, wait: bool,
                               consume: bool, store: Optional[ManifestStore] = None
                               ) -> Session:
        """The legacy manifest resume, on the root machinery
        (``pyreact.boot.root._hydrate_on_loop``): the manifest's slices and
        turns hydrate the tree; participants get their fragments before the
        first render and ``resumed()`` once hydration completed."""
        from pyreact.core.dehydrate import SNAPSHOT_FORMAT

        from .root import _hydrate_on_loop

        if manifest.get("session_id") != session_id:
            raise ManifestError(f"manifest is for session {manifest.get('session_id')!r}, "
                                f"not {session_id!r}")
        fragments = manifest.get("participants") or {}
        turns = dict(fragments.get("turns") or {})
        turns.setdefault("policy", manifest.get("turn_policy"))
        slices = manifest.get("slices")
        snapshot = {"format": SNAPSHOT_FORMAT, "session_id": session_id,
                    "created_at": manifest.get("created_at"),
                    "slices": slices if isinstance(slices, dict) else None,
                    "turns": turns, "versions": manifest.get("versions") or {}}
        runner = self

        class _Legacy:
            restored: List[Any] = []

            async def before_mount(self, sess: Session) -> None:
                sess.resumed_from = manifest
                if session_id == runner._default_id and runner._default is None:
                    runner._default = sess
                for name in fragments:
                    if name not in sess.participants:
                        sess.degrade(name, "no participant registered for this fragment")
                for p in sess.participants:
                    if p.name == "turns":  # the snapshot's turns (TurnManager.hydrate)
                        self.restored.append(p)
                        continue
                    if p.name not in fragments:
                        continue
                    try:
                        await p.resume(sess, fragments[p.name])
                        self.restored.append(p)
                    except Exception as exc:
                        sess.degrade(p.name, "resume failed",
                                     error=f"{type(exc).__name__}: {exc}")

            async def after_hydrate(self, sess: Session) -> None:
                for p in self.restored:
                    fn = getattr(p, "resumed", None)
                    if p.name == "turns" or not callable(fn):
                        continue
                    try:
                        await fn(sess)
                    except Exception as exc:
                        sess.degrade(p.name, "resumed() failed",
                                     error=f"{type(exc).__name__}: {exc}")

            def payload(self, sess: Session) -> Dict[str, Any]:
                bad = {d["participant"] for d in sess.degraded}
                return {"turn_seq": manifest.get("turn_seq"),
                        "suspended_at": manifest.get("suspended_at"),
                        "participants": [p.name for p in self.restored if p.name not in bad],
                        "degraded": [d["participant"] for d in sess.degraded]}

        legacy = _Legacy()
        legacy.restored = []
        return await _hydrate_on_loop(
            self, session_id, app, snapshot, turn_policy=turn_policy,
            resume_turns=resume_turns, consume_from=store if consume else None,
            wait=wait, legacy=legacy)

    def resume(
        self,
        session_id: str,
        manifest: Any = None,
        *,
        store: Optional[ManifestStore] = None,
        app: Optional[Callable] = None,
        turn_policy: Optional[str] = None,
        resume_turns: bool = True,
        consume: bool = True,
        wait: bool = False,
        timeout: Optional[float] = None,
    ) -> Session:
        """Deprecated (``DeprecationWarning``): use :func:`~pyreact.boot.root.hydrate_root`,
        which needs no participants. A wrapper over the same machinery
        (``pyreact.boot.root._hydrate_on_loop``), kept for one release; the
        manifest's ``slices`` hydrate the tree's ``use_dehydrate`` hooks.

        Re-open a suspended session (same id) from its manifest.

        ``manifest``: a manifest dict, a path to one, or ``None`` to load it
        from ``store`` (default: ``manifest_store``). Steps: the session is
        created; ``open`` hooks run (participant factories register their
        participants); ``session.opened`` (``resumed: true``); each
        participant's ``resume(session, fragment)`` -- **before the first
        render**; the tree mounts; optional ``resumed(session)`` hooks; then
        with ``resume_turns`` the turns the suspend carried and those a
        ``TurnStore`` saw cut short are queued again; ``session.resumed``.

        Failure semantics: the resume never fails because of one participant.
        A fragment nobody claims, or a participant that raises, is recorded
        in ``session.degraded`` and published as ``session.resume.degraded``;
        the session goes on with what was restored (e.g. a fresh sandbox).

        ``consume`` (default) deletes the manifest from the store it was
        loaded from once resumed (an old manifest must not be resumed twice).
        ``wait=True`` blocks until the re-run turns end.
        """
        _deprecated("AppRunner.resume()", "hydrate_root(App, session_id=..., store=...)")
        m, src = self._manifest_for(session_id, manifest, store)
        if self._on_loop_thread():
            raise RuntimeError("AppRunner.resume() blocks; use 'await app.aresume(...)' on the loop")
        fut = self._submit(self._aresume_on_loop(
            session_id, m, app, turn_policy, resume_turns=resume_turns, wait=wait,
            consume=consume, store=src))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"session {session_id!r} not resumed after {timeout}s") from None

    async def aresume(self, session_id: str, manifest: Any = None, *,
                      store: Optional[ManifestStore] = None, app: Optional[Callable] = None,
                      turn_policy: Optional[str] = None, resume_turns: bool = True,
                      consume: bool = True, wait: bool = False) -> Session:
        """Async :meth:`resume` (deprecated: ``ahydrate_root``)."""
        _deprecated("AppRunner.aresume()", "await ahydrate_root(App, session_id=..., store=...)")
        m, src = self._manifest_for(session_id, manifest, store)
        return await asyncio.wrap_future(self._submit(self._aresume_on_loop(
            session_id, m, app, turn_policy, resume_turns=resume_turns, wait=wait,
            consume=consume, store=src)))

    def _idle_session_ids(self, ttl, keep) -> List[str]:
        now = time.time()
        return [
            sess.id
            for sess in self.sessions()
            if sess is not self._default
            and not (keep is not None and keep(sess))
            and now - sess.last_active > ttl
        ]

    # -------------------------------
    # Legacy single-session API (delegates to the default session)
    # -------------------------------
    def invoke(
        self,
        text: str,
        *,
        wait: bool = False,
        timeout: Optional[float] = None,
        source: str = "term",
    ) -> str:
        """Submit a line of input as a new turn of the default session;
        returns the turn id.

        The turn is processed according to the session's turn policy (queued
        behind the current turn by default). If wait=True, blocks until the
        turn has ended: no queued renders and no pending background tasks
        spawned during the turn (e.g. LLM calls). Raises TimeoutError if that
        takes longer than ``timeout`` seconds. Do not call with wait=True
        from inside a running event loop; use :meth:`ainvoke` there.
        """
        if not self.is_running:
            raise RuntimeError("AppRunner is not running")
        return self.default_session.invoke(
            text, wait=wait, timeout=timeout, source=source
        )

    async def ainvoke(
        self,
        text: str,
        *,
        wait: bool = False,
        timeout: Optional[float] = None,
        source: str = "term",
    ) -> str:
        """Async version of :meth:`invoke` for callers on another event loop."""
        if not self.is_running:
            raise RuntimeError("AppRunner is not running")
        return await self.default_session.ainvoke(
            text, wait=wait, timeout=timeout, source=source
        )

    def wait_turn(self, turn_id: str, timeout: Optional[float] = None) -> dict:
        """Block until the default session's ``turn_id`` ends; returns
        ``{"status", "duration"}``."""
        return self.default_session.wait_turn(turn_id, timeout=timeout)

    def resume_pending(self, *, wait: bool = False,
                       timeout: Optional[float] = None) -> Dict[str, List[str]]:
        """Resume, in every open session, the turns that started but never
        ended (e.g. before a crash): ``{session_id: [turn ids]}``.

        Waits for the app to be idle first, so stores attached on mount
        (e.g. by a ``Durable`` provider) are in place. Sessions without a
        ``TurnStore`` resume nothing."""
        self.wait_idle(timeout=timeout)
        out: Dict[str, List[str]] = {}
        for sess in self.sessions():
            ids = sess.resume_pending(wait=wait, timeout=timeout)
            if ids:
                out[sess.id] = ids
        return out

    def resume_turn(self, turn_id: str, *, wait: bool = False,
                    timeout: Optional[float] = None) -> str:
        """:meth:`Session.resume_turn` on the default session."""
        return self.default_session.resume_turn(turn_id, wait=wait, timeout=timeout)

    def wait_idle(self, timeout: Optional[float] = None) -> None:
        """Block until no renders or component tasks are pending (all sessions)."""
        try:
            self._submit(runtime.wait_idle()).result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"app not idle after {timeout}s") from None

    def shutdown(self) -> None:
        """Stop render loop and background threads (closes every session)."""
        if self._stopping:
            return
        self._stopping = True

        # Remove console buffer listener
        try:
            if self._console_listener is not None:
                console = HookContext.get_service("console_buffer", MessageBuffer)
                try:
                    console.unsubscribe(self._console_listener)
                except Exception:
                    pass
                self._console_listener = None
        except Exception:
            pass

        # Remove nav listener
        try:
            if self._nav_listener is not None and self._default is not None:
                try:
                    self._default.nav_service.subs.remove(self._nav_listener)
                except Exception:
                    pass
                self._nav_listener = None
        except Exception:
            pass

        # Wake the loop so it notices _stopping promptly
        def _wake():
            runtime.get_render_signal().set()

        try:
            self._loop.call_soon_threadsafe(_wake)
        except Exception:
            pass
        # Wait loop thread to exit
        self._thread.join(timeout=2.0)
        if not self._thread.is_alive():
            try:
                self._loop.close()
            except Exception:
                pass

    # -------------------------------
    # Debug/Navigation helpers
    # -------------------------------
    def print_vnode_tree(self) -> None:
        """Print the default session's component tree."""
        if self._stopping or self._default is None:
            return
        self._default.print_vnode_tree()

    def print_render_trace(self) -> None:
        if self._stopping:
            return

        async def _task():
            try:
                from pyreact.core.debug import print_last_trace

                print_last_trace()
            except Exception:
                print("\x1b[90m[debug]\x1b[0m Render trace not available.")

        fut = asyncio.run_coroutine_threadsafe(_task(), self._loop)
        try:
            fut.result(timeout=1.0)
        except Exception:
            pass

    def nav(self, dest: str, query: Optional[dict] = None, fragment: str = "") -> None:
        """Navigate the default session."""
        if self._stopping or not dest or self._default is None:
            return
        self._default.nav(dest, query=query, fragment=fragment)

    def current_route(self) -> dict:
        """Route of the default session."""
        if self._stopping or self._default is None:
            return {"path": "", "query": {}, "fragment": ""}
        try:
            return self._default.current_route()
        except Exception:
            return {"path": "", "query": {}, "fragment": ""}

    def attach_web_bridge(
        self,
        *,
        on_nav: Optional[Callable[[str], object]] = None,
        on_console: Optional[Callable[[str], object]] = None,
        target_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Attach callbacks for HTML and navigation updates (default session).
        If ``target_loop`` is provided, callbacks are marshaled to that loop:
        - coroutine callbacks → scheduled with asyncio.run_coroutine_threadsafe
        - regular callbacks → scheduled with loop.call_soon_threadsafe
        Otherwise, callbacks are invoked directly on the runner thread.
        """

        def _wrap(cb):
            if cb is None:
                return None
            if target_loop is None:
                return cb
            if inspect.iscoroutinefunction(cb):

                def _call(*args, **kwargs):
                    try:
                        asyncio.run_coroutine_threadsafe(
                            cb(*args, **kwargs), target_loop
                        )
                    except Exception:
                        pass

                return _call
            else:

                def _call(*args, **kwargs):
                    try:
                        target_loop.call_soon_threadsafe(cb, *args, **kwargs)
                    except Exception:
                        pass

                return _call

        self._on_nav = _wrap(on_nav)
        self._on_console = _wrap(on_console)

    # -------------------------------
    # Internal: loop thread
    # -------------------------------
    def _thread_main(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._loop_main())
        except BaseException as exc:  # pragma: no cover - defensive
            if not self._ready.is_set():
                self._startup_error = exc
            else:
                runtime.report_error(exc, self._root_ctx, "runner")
        finally:
            self._stopping = True
            self._ready.set()

    async def _loop_main(self) -> None:
        from pyreact.core.debug import enable_tracing, clear_traces

        clear_traces()
        enable_tracing()
        runtime.bind_loop(asyncio.get_running_loop())
        # Component errors -> "error" events on the owning session's bus
        # (still printed to stderr).
        self._uninstall_error_events = install_session_error_events()

        try:
            # Open the default session within the loop thread
            if self._open_default:
                self._default = self._open_on_loop(self._default_id, None)[0]
                self._root_ctx = self._default.root
                self._bus = self._default.input_bus

                # Bridge navigation events to the legacy web bridge publisher
                def _nav_listener(path: str) -> None:
                    try:
                        if self._on_nav is not None:
                            self._on_nav(path)
                    except Exception:
                        pass

                self._default.nav_service.subs.append(_nav_listener)
                self._nav_listener = _nav_listener

            # Bridge console buffer appends to the legacy web bridge
            console = HookContext.get_service("console_buffer", MessageBuffer)

            def _console_listener(text: str) -> None:
                try:
                    if self._on_console is not None:
                        self._on_console(text)
                except Exception:
                    pass

            console.subscribe(_console_listener)
            self._console_listener = _console_listener

            # Commit the first render (and its effects, e.g. input
            # subscriptions) before reporting ready, so early invoke() calls
            # are not lost. schedule_rerender enqueues via call_soon; yield so
            # it lands, and keep going while the first commit schedules more.
            await asyncio.sleep(0)
            while not runtime.is_idle() or not runtime.rerender_queue.empty():
                if runtime.pending_tasks() and runtime.rerender_queue.empty():
                    break  # only background work left; don't block startup
                await run_renders()
                await asyncio.sleep(0)
        except BaseException as exc:
            self._startup_error = exc
            self._shutdown_on_loop()
            raise

        self._ready.set()

        # fps is now only an upper bound on how long we sleep between checks
        # of _stopping; renders run as soon as they are scheduled.
        interval = 1.0 / max(1, self._fps)
        signal = runtime.get_render_signal()
        try:
            while not self._stopping:
                try:
                    await asyncio.wait_for(signal.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    continue
                if self._stopping:
                    break
                try:
                    await run_renders()
                except Exception as exc:  # never let the loop die
                    runtime.report_error(exc, self._root_ctx, "render")
        finally:
            self._shutdown_on_loop()

    def _shutdown_on_loop(self) -> None:
        for sess in self.sessions():
            try:
                self._close_on_loop(sess.id, "shutdown")
            except Exception:
                pass
        for t in runtime.pending_tasks():
            t.cancel()
        runtime.bind_loop(None)
        if self._uninstall_error_events is not None:
            self._uninstall_error_events()
            self._uninstall_error_events = None
