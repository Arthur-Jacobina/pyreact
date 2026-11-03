import asyncio
import concurrent.futures
import inspect
import threading
from typing import Optional, Callable

from pyreact.core.hook import HookContext
from pyreact.core import runtime
from pyreact.core.runtime import run_renders, schedule_rerender
from pyreact.core.events import EventBus
from pyreact.core.turns import TurnManager
from pyreact.input.bus import InputBus
from pyreact.router.nav_service import NavService
from pyreact.web.console import MessageBuffer
from pyreact.observability import Observer, install_error_events


class AppRunner:
    """Background runner that manages the render loop and input dispatching.

    Usage:
        app = AppRunner(Boot)
        app.invoke("hello")
        ...
        app.shutdown()
    """

    def __init__(self, app_component_fn, *, fps: int = 20, turn_policy: str = "queue"):
        if turn_policy not in ("queue", "interrupt"):
            raise ValueError(f"turn_policy must be 'queue' or 'interrupt', got {turn_policy!r}")
        self._turn_policy = turn_policy
        self._turns: Optional[TurnManager] = None
        self._app_component_fn = app_component_fn
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
        # Set in the loop thread before ready (see _loop_main)
        self._event_bus: Optional[EventBus] = None
        self.observer: Optional[Observer] = None
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

    # Observability accessors (EventLog / UsageAggregator / TurnTimeline)
    @property
    def event_bus(self) -> Optional[EventBus]:
        return self._event_bus

    @property
    def events(self):
        return self.observer.log if self.observer else None

    @property
    def usage(self):
        return self.observer.usage if self.observer else None

    @property
    def timeline(self):
        return self.observer.timeline if self.observer else None

    @property
    def is_running(self) -> bool:
        return not self._stopping and self._thread.is_alive()

    def _submit(self, coro):
        if not self.is_running:
            coro.close()
            raise RuntimeError("AppRunner is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    @property
    def turns(self) -> TurnManager:
        """The app's :class:`~pyreact.core.turns.TurnManager`."""
        return self._turns

    async def _submit_turn(self, turn, should_wait: bool):
        self._turns.enqueue(turn)
        if should_wait:
            await self._turns.wait(turn.id)

    def invoke(
        self,
        text: str,
        *,
        wait: bool = False,
        timeout: Optional[float] = None,
        source: str = "term",
    ) -> str:
        """Submit a line of input as a new turn; returns the turn id.

        The turn is processed according to the app's turn policy (queued
        behind the current turn by default). If wait=True, blocks until the
        turn has ended: no queued renders and no pending background tasks
        spawned during the turn (e.g. LLM calls). Raises TimeoutError if that
        takes longer than ``timeout`` seconds. Do not call with wait=True
        from inside a running event loop; use :meth:`ainvoke` there.
        """
        turn = self._turns.new_turn(text, source=source)
        fut = self._submit(self._submit_turn(turn, wait))
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
        """Async version of :meth:`invoke` for callers on another event loop."""
        turn = self._turns.new_turn(text, source=source)
        fut = self._submit(self._submit_turn(turn, wait))
        await asyncio.wait_for(asyncio.wrap_future(fut), timeout)
        return turn.id

    def wait_turn(self, turn_id: str, timeout: Optional[float] = None) -> dict:
        """Block until ``turn_id`` ends; returns ``{"status", "duration"}``."""
        try:
            return self._submit(self._turns.wait(turn_id)).result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"turn {turn_id} not done after {timeout}s") from None

    def wait_idle(self, timeout: Optional[float] = None) -> None:
        """Block until no renders or component tasks are pending."""
        try:
            self._submit(runtime.wait_idle()).result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"app not idle after {timeout}s") from None

    def shutdown(self) -> None:
        """Stop render loop and background threads."""
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
            if self._nav_listener is not None:
                navsvc = self._root_ctx.get_service("nav_service", NavService)
                try:
                    navsvc.subs.remove(self._nav_listener)
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
        if self._stopping or self._root_ctx is None:
            return

        async def _task():
            self._root_ctx.render_tree()

        fut = asyncio.run_coroutine_threadsafe(_task(), self._loop)
        try:
            fut.result(timeout=1.0)
        except Exception:
            pass

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
        if self._stopping or not dest:
            return

        navsvc = self._root_ctx.get_service("nav_service", NavService)
        go = getattr(navsvc, "navigate", None)
        if callable(go):
            if inspect.iscoroutinefunction(go):
                asyncio.run_coroutine_threadsafe(
                    go(dest, query=query, fragment=fragment), self._loop
                )
            else:

                def _call():
                    try:
                        go(dest, query=query, fragment=fragment)
                    except Exception:
                        pass

                try:
                    self._loop.call_soon_threadsafe(_call)
                except Exception:
                    pass

    def current_route(self) -> dict:
        if self._stopping:
            return {"path": "", "query": {}, "fragment": ""}

        async def _task():
            try:
                navsvc = self._root_ctx.get_service("nav_service", NavService)
                return {
                    "path": navsvc.get_path(),
                    "query": navsvc.get_query_params(),
                    "fragment": navsvc.get_fragment(),
                }
            except Exception:
                return {"path": "", "query": {}, "fragment": ""}

        fut = asyncio.run_coroutine_threadsafe(_task(), self._loop)
        try:
            return fut.result(timeout=1.0)
        except Exception:
            return {"path": "", "query": {}, "fragment": ""}

    def attach_web_bridge(
        self,
        *,
        on_nav: Optional[Callable[[str], object]] = None,
        on_console: Optional[Callable[[str], object]] = None,
        target_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Attach callbacks for HTML and navigation updates.
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

        # if self._on_console is not None:
        #     enable_web_print()

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

        # Initialize root context and services within the loop thread
        self._root_ctx = HookContext(
            self._app_component_fn.__name__, self._app_component_fn
        )
        # Observability: record events from the very first render and turn
        # component errors into "error" events (still printed to stderr).
        self._event_bus = self._root_ctx.get_service("event_bus", EventBus)
        self.observer = Observer().attach(self._event_bus)
        self._uninstall_error_events = install_error_events(self._event_bus)
        schedule_rerender(self._root_ctx, reason="app startup")
        self._bus = self._root_ctx.get_service("input_bus", InputBus)
        self._turns = self._root_ctx.get_service("turns", TurnManager).configure(
            policy=self._turn_policy,
            input_bus=self._bus,
            event_bus=self._root_ctx.get_service("event_bus", EventBus),
            loop=asyncio.get_running_loop(),
        )

        # Bridge navigation events to server publisher
        try:
            navsvc = self._root_ctx.get_service("nav_service", NavService)

            def _nav_listener(path: str) -> None:
                try:
                    if self._on_nav is not None:
                        self._on_nav(path)
                except Exception:
                    pass

            navsvc.subs.append(_nav_listener)
            self._nav_listener = _nav_listener
        except Exception:
            pass

        # Bridge console buffer appends to server publisher
        try:
            console = HookContext.get_service("console_buffer", MessageBuffer)

            def _console_listener(text: str) -> None:
                try:
                    if self._on_console is not None:
                        self._on_console(text)
                except Exception:
                    pass

            console.subscribe(_console_listener)
            self._console_listener = _console_listener
        except Exception:
            pass

        # Commit the first render (and its effects, e.g. input subscriptions)
        # before reporting ready, so early invoke() calls are not lost.
        try:
            # schedule_rerender enqueues via call_soon; yield so it lands,
            # and keep going while the first commit schedules more renders.
            await asyncio.sleep(0)
            while not runtime.is_idle() or not runtime.rerender_queue.empty():
                if runtime.pending_tasks() and runtime.rerender_queue.empty():
                    break  # only background work left; don't block startup
                await run_renders()
                await asyncio.sleep(0)
        except BaseException as exc:
            self._startup_error = exc
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
            try:
                if self._root_ctx is not None:
                    self._root_ctx.unmount()
            except Exception:
                pass
            pump = getattr(self._turns, "_pump", None)
            if pump is not None and not pump.done():
                pump.cancel()
            for t in runtime.pending_tasks():
                t.cancel()
            runtime.bind_loop(None)
            if self._uninstall_error_events is not None:
                self._uninstall_error_events()
