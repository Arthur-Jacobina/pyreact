# runtime.py -------------------------------------------------
import asyncio
import sys
import threading
import traceback
from typing import Any, Callable, Coroutine, Optional


rerender_queue: asyncio.Queue = asyncio.Queue()
_enqueued: set = set()
_enqueued_lock = threading.Lock()

# Loop that owns the render queue (bound by AppRunner). When unbound we fall
# back to whichever loop is running in the calling thread.
_loop: Optional[asyncio.AbstractEventLoop] = None

_render_idle: Optional[asyncio.Event] = None  # will be created on demand
_render_signal: Optional[asyncio.Event] = None  # set when a render is scheduled
_rendering: bool = False

# Background tasks spawned by components (async effects, LLM calls, ...).
# Tracked so callers can wait for a whole "turn" to settle.
_tasks: set = set()

ErrorHandler = Callable[[BaseException, Any, str], None]
_error_handler: Optional[ErrorHandler] = None


def bind_loop(loop: Optional[asyncio.AbstractEventLoop]) -> None:
    """Bind the render queue to ``loop`` (or unbind with ``None``).

    Once bound, ``schedule_rerender`` may be called from any thread.
    """
    global _loop, _render_idle, _render_signal, rerender_queue
    _loop = loop
    # asyncio primitives bind to the loop that first uses them; recreate them
    # so a new runner never inherits primitives tied to a dead loop.
    _render_idle = None
    _render_signal = None
    with _enqueued_lock:
        _enqueued.clear()
    rerender_queue = asyncio.Queue()
    _tasks.clear()


def _target_loop() -> asyncio.AbstractEventLoop:
    if _loop is not None and _loop.is_running():
        return _loop
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        raise RuntimeError(
            "No render loop available: schedule_rerender() was called outside an "
            "event loop and no AppRunner is running."
        ) from None


def get_render_idle() -> asyncio.Event:
    """Ensure the ``Event`` belongs to the currently running loop."""
    global _render_idle
    if _render_idle is None:
        _render_idle = asyncio.Event()
        _render_idle.set()  # start in the 'idle' state
    return _render_idle


def get_render_signal() -> asyncio.Event:
    """Event that is set whenever a rerender is scheduled, and cleared after run_renders drains the queue."""
    global _render_signal
    if _render_signal is None:
        _render_signal = asyncio.Event()
    return _render_signal


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------
def set_error_handler(handler: Optional[ErrorHandler]) -> None:
    """Install ``handler(exc, ctx, phase)`` for errors raised by components.

    ``phase`` is one of ``"render"``, ``"effect"``, ``"cleanup"``, ``"task"``.
    Without a handler, errors are printed to stderr and the app keeps running.
    """
    global _error_handler
    _error_handler = handler


def report_error(exc: BaseException, ctx: Any = None, phase: str = "render") -> None:
    if _error_handler is not None:
        try:
            _error_handler(exc, ctx, phase)
            return
        except Exception:
            pass
    name = getattr(ctx, "name", None) or "?"
    print(f"[pyreact] error during {phase} of <{name}>:", file=sys.stderr)
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------
def spawn(coro: Coroutine, *, owner: Any = None, name: Optional[str] = None) -> asyncio.Task:
    """Run ``coro`` as a tracked task on the render loop.

    Exceptions are routed to the error handler instead of being lost, and
    :func:`wait_idle` waits for tracked tasks to finish.
    """
    task = asyncio.get_running_loop().create_task(coro, name=name)
    _tasks.add(task)

    def _done(t: asyncio.Task) -> None:
        _tasks.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            report_error(exc, owner, "task")

    task.add_done_callback(_done)
    return task


def pending_tasks() -> list:
    return [t for t in list(_tasks) if not t.done()]


def is_idle() -> bool:
    with _enqueued_lock:
        queued = bool(_enqueued)
    return not queued and not _rendering and not pending_tasks()


async def wait_idle(poll: float = 0.01) -> None:
    """Wait until no renders are queued/running and no tracked task is pending."""
    while True:
        pending = pending_tasks()
        if pending:
            await asyncio.wait(pending)
            continue
        if not is_idle():
            await asyncio.sleep(poll)
            continue
        # Give call_soon callbacks (e.g. state set by a finishing task) a
        # chance to enqueue renders before declaring idle.
        await asyncio.sleep(0)
        if is_idle():
            return


async def run_until_idle() -> None:
    """Drive renders and tracked tasks on the *current* loop until quiescent.

    For headless use (tests, scripts, evaluation harnesses) where no
    AppRunner is draining the render queue. Do not use alongside a running
    AppRunner; use :func:`wait_idle` / ``AppRunner.wait_idle`` there.
    """
    while True:
        await asyncio.sleep(0)
        await run_renders()
        pending = pending_tasks()
        if pending:
            await asyncio.wait(pending)
            continue
        await asyncio.sleep(0)
        if is_idle() and rerender_queue.empty():
            return


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------
def schedule_rerender(ctx, reason: str = None):
    # Record schedule intent for debug tooling
    try:
        from .debug import record_schedule  # local import to avoid cycles

        record_schedule(ctx, reason)
    except Exception:
        pass
    loop = _target_loop()
    with _enqueued_lock:
        if ctx in _enqueued:
            return
        _enqueued.add(ctx)

    def _enqueue():
        try:
            rerender_queue.put_nowait(ctx)
        finally:
            get_render_idle().clear()
            get_render_signal().set()  # Set the signal only after the ctx is in the queue to avoid races

    # call_soon_threadsafe is safe from any thread, including the loop's own.
    loop.call_soon_threadsafe(_enqueue)


async def run_renders() -> None:
    global _rendering
    from .debug import start_trace, end_trace

    _rendering = True
    try:
        while True:
            try:
                ctx = rerender_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            with _enqueued_lock:
                _enqueued.discard(ctx)
            if not getattr(ctx, "_mounted", True):
                continue
            reasons = getattr(ctx, "_debug_reasons", [])
            try:
                start_trace(ctx, reasons)
            except Exception:
                pass
            # Clear reasons once consumed
            try:
                ctx._debug_reasons = []
            except Exception:
                pass
            try:
                try:
                    ctx.render()
                except Exception as exc:
                    # Component errors must not kill the render loop.
                    report_error(exc, ctx, "render")
                    continue
                await ctx.run_effects()
            finally:
                try:
                    end_trace()
                except Exception:
                    pass
    finally:
        _rendering = False

    # Mark idle and clear signal after draining current batch
    if rerender_queue.empty():
        get_render_idle().set()
        get_render_signal().clear()
