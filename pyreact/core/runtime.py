# runtime.py -------------------------------------------------
"""Process-global render runtime: one render queue bound to one loop.

Many sessions (see ``pyreact.boot.session``) can share the loop. Work is
attributed to them through *owners*: every tracked task records the
component (HookContext) that owns it, and a session owns every component of
its tree. Turns (see ``pyreact.core.turns``) are scoped per session too: each
session's TurnManager has its own active turn, and turn ids of non-default
sessions are prefixed with the session id, so waiting for / interrupting a
turn never touches another session's tasks.
"""

import asyncio
import contextvars
import sys
import threading
import traceback
from contextvars import ContextVar
from typing import Any, Callable, Coroutine, Dict, Optional


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

# Component (HookContext) on whose behalf code is currently running: set while
# a component's effects run and by session-scoped entry points (turn
# processing, ``Session.nav``...). Tasks inherit it, so ``spawn()`` without an
# explicit owner, and stdout written from effects/tasks, can be attributed to
# a session.
current_owner: ContextVar = ContextVar("pyreact_current_owner", default=None)

OwnerFilter = Callable[[Any], bool]

# Turn attribution (see pyreact.core.turns). Every tracked task is tagged with
# the turn that spawned it so a turn can end when *its* work is done, and be
# interrupted by cancelling only its tasks.
_UNSET: Any = object()
_turn_var: contextvars.ContextVar = contextvars.ContextVar("pyreact_turn", default=None)
# Legacy/unscoped active turn (``set_active_turn(tid)`` without a scope).
_active_turn: Optional[str] = None
# scope (a session's TurnManager) -> turn currently being processed by it.
_active_turns: Dict[Any, str] = {}
_task_turn: dict = {}  # task -> turn id (or None)
_task_owner: dict = {}  # task -> owner (HookContext or None)
_turn_errors: dict = {}  # turn id -> [(phase, exc)]

ErrorHandler = Callable[[BaseException, Any, str], None]
_error_handler: Optional[ErrorHandler] = None


def bind_loop(loop: Optional[asyncio.AbstractEventLoop]) -> None:
    """Bind the render queue to ``loop`` (or unbind with ``None``).

    Once bound, ``schedule_rerender`` may be called from any thread.
    """
    global _loop, _render_idle, _render_signal, rerender_queue, _active_turn
    _loop = loop
    # asyncio primitives bind to the loop that first uses them; recreate them
    # so a new runner never inherits primitives tied to a dead loop.
    _render_idle = None
    _render_signal = None
    with _enqueued_lock:
        _enqueued.clear()
    rerender_queue = asyncio.Queue()
    _tasks.clear()
    _task_turn.clear()
    _task_owner.clear()
    _turn_errors.clear()
    _active_turns.clear()
    _active_turn = None


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
# Owners / sessions
# ---------------------------------------------------------------------------
def current_component() -> Any:
    """The component the calling code runs for: the current owner (effects,
    tasks, session entry points) or the component being rendered."""
    owner = current_owner.get()
    if owner is not None:
        return owner
    from .core import _context_stack

    return _context_stack.get()


def services_of(ctx: Any) -> Optional[dict]:
    """Services dict of ``ctx``'s session root (``None`` if ``ctx`` is not a
    component)."""
    root = getattr(ctx, "_root", None)
    if root is None:
        return None
    try:
        return root.session_services
    except Exception:
        return None


def _scope_of(ctx: Any) -> Any:
    services = services_of(ctx)
    return services.get("turns") if services else None


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


# Phases an ErrorBoundary (pyreact.core.error_boundary) may catch.
BOUNDARY_PHASES = frozenset({"render", "effect", "task"})


def _to_error_boundary(exc: BaseException, ctx: Any, phase: str) -> bool:
    """Offer the error to the nearest mounted ancestor exposing
    ``_catch_error(exc, ctx, phase) -> bool`` (an ErrorBoundary); True if one
    handled it."""
    node = getattr(ctx, "parent", None)
    while node is not None:
        catch = getattr(node, "_catch_error", None)
        if catch is not None and getattr(node, "_mounted", True):
            try:
                if catch(exc, ctx, phase):
                    return True
            except Exception:
                pass  # a failing boundary must not hide the original error
        node = getattr(node, "parent", None)
    return False


def report_error(
    exc: BaseException, ctx: Any = None, phase: str = "render", *, turn: Any = _UNSET
) -> None:
    """Route a component error: to the nearest ``ErrorBoundary`` above
    ``ctx`` (render / effect / task errors), else record it on its turn
    (status ``"error"``) and pass it to the error handler (stderr by
    default)."""
    if ctx is not None and phase in BOUNDARY_PHASES and _to_error_boundary(exc, ctx, phase):
        return
    tid = current_turn_id(ctx) if turn is _UNSET else turn
    if tid is not None:
        _turn_errors.setdefault(tid, []).append((phase, exc))
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
# Turn attribution
# ---------------------------------------------------------------------------
def set_active_turn(turn_id: Optional[str], scope: Any = None) -> None:
    """Mark ``turn_id`` as the turn being processed (``None``: no turn).

    ``scope`` is the TurnManager processing it; each scope (session) has its
    own active turn. Without a scope the legacy process-wide value is set.
    """
    global _active_turn
    if scope is None:
        _active_turn = turn_id
    elif turn_id is None:
        _active_turns.pop(scope, None)
    else:
        _active_turns[scope] = turn_id


def active_turn(scope: Any = None) -> Optional[str]:
    """Active turn of ``scope`` (or the legacy unscoped one)."""
    if scope is None:
        return _active_turn
    return _active_turns.get(scope)


def current_turn_id(ctx: Any = None) -> Optional[str]:
    """Turn the calling code belongs to.

    Inside a tracked task this is the turn the task was spawned for (inherited
    through a contextvar, also by tasks it spawns); elsewhere (renders,
    effects) it is the turn currently being processed by the session of
    ``ctx`` (default: the current component), if any.
    """
    tid = _turn_var.get()
    if tid is not None:
        return tid
    if ctx is None:
        ctx = current_component()
    scope = _scope_of(ctx) if ctx is not None else None
    if scope is not None and scope in _active_turns:
        return _active_turns[scope]
    if _active_turn is not None:
        return _active_turn
    if scope is None and len(_active_turns) == 1:
        # Code not attributable to a session while exactly one session is
        # busy (e.g. single-session apps): that session's turn.
        return next(iter(_active_turns.values()))
    return None


def task_turn(task: asyncio.Task) -> Optional[str]:
    return _task_turn.get(task)


def task_owner(task: asyncio.Task) -> Any:
    """Owner recorded by :func:`spawn` for ``task`` (``None`` if unknown)."""
    return _task_owner.get(task)


def pending_turn_tasks(turn_id: Optional[str]) -> list:
    return [t for t in pending_tasks() if _task_turn.get(t) == turn_id]


def pop_turn_errors(turn_id: Optional[str]) -> list:
    return _turn_errors.pop(turn_id, [])


def renders_pending(owner_filter: Optional[OwnerFilter] = None) -> bool:
    """Renders queued or running (queued ones filtered by owner, if given)."""
    return has_queued(owner_filter) or _rendering


def is_turn_idle(
    turn_id: Optional[str], owner_filter: Optional[OwnerFilter] = None
) -> bool:
    """No queued/running renders and no pending tasks tagged with ``turn_id``."""
    return not renders_pending(owner_filter) and not pending_turn_tasks(turn_id)


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------
def spawn(
    coro: Coroutine,
    *,
    owner: Any = None,
    name: Optional[str] = None,
    turn: Any = _UNSET,
) -> asyncio.Task:
    """Run ``coro`` as a tracked task on the render loop.

    Exceptions are routed to the error handler instead of being lost, and
    :func:`wait_idle` waits for tracked tasks to finish.

    ``owner`` defaults to :data:`current_owner` (the component whose effect
    or session entry point is running); it attributes the task to a session.

    ``turn`` tags the task with a turn id; by default the current turn
    (:func:`current_turn_id`). A turn does not end while its tasks are
    pending, and interrupting a turn cancels them. Pass ``turn=None`` for
    long-lived background work that must not hold a turn open.
    """
    if owner is None:
        owner = current_owner.get()
    tid = current_turn_id(owner) if turn is _UNSET else turn
    ctx = contextvars.copy_context()
    ctx.run(_turn_var.set, tid)
    if owner is not None:
        ctx.run(current_owner.set, owner)
    task = asyncio.get_running_loop().create_task(coro, name=name, context=ctx)
    _tasks.add(task)
    _task_turn[task] = tid
    _task_owner[task] = owner

    def _done(t: asyncio.Task) -> None:
        _tasks.discard(t)
        _task_turn.pop(t, None)
        _task_owner.pop(t, None)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            report_error(exc, owner, "task", turn=tid)

    task.add_done_callback(_done)
    return task


def pending_tasks(owner_filter: Optional[OwnerFilter] = None) -> list:
    """Unfinished tracked tasks, optionally only those whose owner matches."""
    tasks = [t for t in list(_tasks) if not t.done()]
    if owner_filter is not None:
        tasks = [t for t in tasks if owner_filter(_task_owner.get(t))]
    return tasks


def has_queued(owner_filter: Optional[OwnerFilter] = None) -> bool:
    """True if a rerender is queued (for a matching component, if filtered)."""
    with _enqueued_lock:
        queued = list(_enqueued)
    if owner_filter is None:
        return bool(queued)
    return any(owner_filter(c) for c in queued)


def is_idle(owner_filter: Optional[OwnerFilter] = None) -> bool:
    return (
        not has_queued(owner_filter)
        and not _rendering
        and not pending_tasks(owner_filter)
    )


async def wait_idle(
    poll: float = 0.01, owner_filter: Optional[OwnerFilter] = None
) -> None:
    """Wait until no renders are queued/running and no tracked task is pending.

    With ``owner_filter`` only queued components / tasks whose owner matches
    are considered (used for per-session waits).
    """
    while True:
        pending = pending_tasks(owner_filter)
        if pending:
            await asyncio.wait(pending)
            continue
        if not is_idle(owner_filter):
            await asyncio.sleep(poll)
            continue
        # Give call_soon callbacks (e.g. state set by a finishing task) a
        # chance to enqueue renders before declaring idle.
        await asyncio.sleep(0)
        if is_idle(owner_filter):
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
