import asyncio
import concurrent.futures
import inspect
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

from pyreact.core.hook import HookContext
from pyreact.core import runtime
from pyreact.core.events import EventBus
from pyreact.core.runtime import run_renders
from pyreact.core.turns import POLICIES, TurnManager
from pyreact.web.console import MessageBuffer
from pyreact.observability import install_session_error_events
from .session import DEFAULT_SESSION_ID, Session


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
    ):
        """``turn_policy``: default ``"queue"`` or ``"interrupt"`` for every
        session (see ``pyreact.core.turns``); ``open_session`` may override it.

        ``session_id``: id of the default session. ``default_session=False`` starts the loop without mounting
        anything; use :meth:`open_session`.
        """
        if turn_policy not in POLICIES:
            raise ValueError(f"turn_policy must be 'queue' or 'interrupt', got {turn_policy!r}")
        self._turn_policy = turn_policy
        self._app_component_fn = app_component_fn
        self._open_default = bool(default_session)
        self._default_id = session_id
        self._sessions: Dict[str, Session] = {}
        self._sessions_lock = threading.Lock()
        self._default: Optional[Session] = None

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
        sess._mount()
        return sess, True

    async def _aopen_on_loop(self, session_id, app, turn_policy=None) -> Session:
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
        without waiting for the first render.
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
        """
        closed = []
        for sid in self._idle_session_ids(ttl, keep):
            if self.close_session(sid, reason="idle"):
                closed.append(sid)
        return closed

    async def aclose_idle_sessions(
        self, ttl: float, *, keep: Optional[Callable[[Session], bool]] = None
    ) -> List[str]:
        """Async :meth:`close_idle_sessions`."""
        closed = []
        for sid in self._idle_session_ids(ttl, keep):
            if await self.aclose_session(sid, reason="idle"):
                closed.append(sid)
        return closed

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
