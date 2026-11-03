"""Turns: the unit of work of a long-running agent app.

Every submitted input (terminal line, web message, ``AppRunner.invoke``)
becomes a :class:`Turn`. The :class:`TurnManager` service (key ``"turns"``)
processes turns one at a time:

1. ``turn.start`` is published on the ``EventBus`` (``"event_bus"`` service),
   components using :func:`use_turn` rerender with the new turn, and the input
   is emitted on the ``InputBus`` as ``text``/``submit`` events carrying
   ``turn_id``.
2. Everything that happens next (renders, async effects, LLM calls spawned
   with :func:`pyreact.core.runtime.spawn`) is attributed to that turn.
3. The turn ends when the app is idle with respect to it: no queued renders
   and no pending tasks tagged with the turn. ``turn.end`` is published with
   ``{"status": "done" | "interrupted" | "error" | "blocked", "duration": seconds}``.

Steps 1-3 run as one ``"turn"`` harness boundary (``pyreact.core.harness``;
process-wide middleware, attrs ``turn_id``/``index``/``source``, then
``status``/``duration``). A middleware raising ``Blocked`` ends the turn as
``blocked`` (``turn.blocked`` event) before anything starts.

Policies (what a new input does while a turn is in flight):

- ``"queue"`` (default): the new turn waits until the current one ends.
- ``"interrupt"``: the current turn's in-flight tasks are cancelled, it ends
  as ``interrupted`` and the new turn starts right away.

The app-wide policy is set with ``bootstrap(App, turn_policy=...)`` /
``AppRunner(App, turn_policy=...)``. A component overrides it for the tasks
*it* spawns (and those of its descendants) with ``use_turn(policy=...)``:
e.g. under the ``queue`` app policy a ``use_turn(policy="interrupt")``
component has its in-flight work cancelled as soon as a new input arrives
(the new turn still waits for the remaining, non-interruptible work); under
the ``interrupt`` app policy a ``use_turn(policy="queue")`` component's work
is never cancelled and the new turn waits for it.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Deque, Dict, List, Literal, Optional, Protocol

from . import runtime
from .events import EventBus

TurnPolicy = Literal["queue", "interrupt"]
POLICIES = ("queue", "interrupt")


@dataclass(frozen=True)
class Turn:
    id: str
    index: int
    text: str
    source: str = "term"
    ts: float = field(default_factory=time.time)
    resumed: bool = False  # re-run of a turn that started but never ended


TurnListener = Callable[[Turn], None]


class TurnStore(Protocol):
    """Where a TurnManager remembers turns, so interrupted or crashed turns
    can be resumed (``TurnManager.resume`` / ``Session.resume_turn`` /
    ``AppRunner.resume_pending``). The core ships no implementation: a
    persistence layer provides one (the durable-memory package's ``DurableTurnStore``
    reads the ``turn.*`` events its ``Durable`` provider journals).

    - ``on_turn_start(turn)`` / ``on_turn_end(turn, result)``: called by the
      manager around every turn (``result`` = ``{"status", "duration"}``);
      a store fed by the event journal may ignore them;
    - ``pending()``: turns that started but never ended, oldest first;
    - ``get(turn_id)`` (optional): the last recorded start of any turn.
    """

    def on_turn_start(self, turn: Turn) -> None: ...

    def on_turn_end(self, turn: Turn, result: Dict[str, Any]) -> None: ...

    def pending(self) -> List[Turn]: ...


def _effective_policy(owner: Any, default: str) -> str:
    node = owner
    while node is not None:
        p = getattr(node, "_turn_policy", None)
        if p is not None:
            return p
        node = getattr(node, "parent", None)
    return default


class TurnManager:
    """Turn queue + lifecycle. Service key ``"turns"``.

    ``submit`` may be called from any thread; turn processing runs on the
    render loop (the AppRunner's, or the running loop in headless use).
    """

    max_history = 256

    def __init__(
        self,
        *,
        policy: TurnPolicy = "queue",
        input_bus: Any = None,
        event_bus: Optional[EventBus] = None,
        id_factory: Optional[Callable[[int], str]] = None,
    ) -> None:
        """``id_factory(index) -> str`` builds turn ids (default
        ``f"t{index}"``); the session's ``id_prefix`` is prepended either
        way. Use it (or :meth:`advance_to`) to keep ids unique across
        restarts, e.g. ``id_factory=lambda i: f"t-{run_id}-{i}"``."""
        self.policy: str = policy
        self._input_bus = input_bus
        self._event_bus = event_bus
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self._id_factory: Optional[Callable[[int], str]] = id_factory
        self._queue: Deque[Turn] = deque()
        self._active: Optional[Turn] = None
        self._latest: Optional[Turn] = None
        self._history: "OrderedDict[str, Turn]" = OrderedDict()
        self._results: Dict[str, Dict[str, Any]] = OrderedDict()
        self._waiters: Dict[str, List[asyncio.Future]] = {}
        self._interrupted: set = set()  # turns that had tasks cancelled
        self._interrupt_req: set = set()  # turns asked to give way
        self._abort_req: set = set()  # turns whose tasks are all cancelled
        self._held = False  # hold(): queued turns wait (session suspending)
        self._listeners: List[TurnListener] = []
        self.store: Optional[TurnStore] = None  # see TurnStore / resume()
        self._pump: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Session scoping (set by pyreact.boot.session.Session):
        self.id_prefix: str = ""  # turn ids are f"{id_prefix}t{n}"
        self._owner: Any = None  # session root: owner of turn-processing work
        self._owner_filter: Optional[Callable[[Any], bool]] = None

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    def configure(
        self,
        *,
        policy: Optional[TurnPolicy] = None,
        input_bus: Any = None,
        event_bus: Optional[EventBus] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        id_prefix: Optional[str] = None,
        owner: Any = None,
        owner_filter: Optional[Callable[[Any], bool]] = None,
        id_factory: Optional[Callable[[int], str]] = None,
        store: Optional[TurnStore] = None,
    ) -> "TurnManager":
        """Configure the manager. Session wiring:

        - ``id_prefix``: prefix of turn ids (makes them unique across the
          sessions sharing a loop, e.g. ``"s1/"`` -> ``"s1/t3"``);
        - ``owner``: component (session root) owning the work done while a
          turn's input is dispatched;
        - ``owner_filter``: restricts "renders pending" checks to this
          session's components, so busy sessions don't delay each other.
        - ``store``: a :class:`TurnStore` (enables :meth:`resume`).
        """
        if store is not None:
            self.store = store
        if id_prefix is not None:
            self.id_prefix = id_prefix
        if id_factory is not None:
            self._id_factory = id_factory
        if owner is not None:
            self._owner = owner
        if owner_filter is not None:
            self._owner_filter = owner_filter
        if policy is not None:
            if policy not in POLICIES:
                raise ValueError(f"turn policy must be one of {POLICIES}, got {policy!r}")
            self.policy = policy
        if input_bus is not None:
            self._input_bus = input_bus
        if event_bus is not None:
            self._event_bus = event_bus
        if loop is not None:
            self._loop = loop
        return self

    @property
    def input_bus(self):
        if self._input_bus is None:
            from pyreact.core.hook import HookContext
            from pyreact.input.bus import InputBus

            self._input_bus = HookContext.get_service("input_bus", InputBus)
        return self._input_bus

    @property
    def event_bus(self) -> EventBus:
        if self._event_bus is None:
            from pyreact.core.hook import HookContext

            self._event_bus = HookContext.get_service("event_bus", EventBus)
        return self._event_bus

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    @property
    def latest(self) -> Optional[Turn]:
        """Most recently *started* turn."""
        return self._latest

    @property
    def active(self) -> Optional[Turn]:
        return self._active

    def get(self, turn_id: Optional[str]) -> Optional[Turn]:
        if turn_id is None:
            return None
        return self._history.get(turn_id)

    def result(self, turn_id: str) -> Optional[Dict[str, Any]]:
        """``{"status", "duration"}`` of an ended turn, else ``None``."""
        return self._results.get(turn_id)

    def pending(self) -> List[Turn]:
        return list(self._queue)

    def subscribe(self, fn: TurnListener) -> Callable[[], None]:
        """Call ``fn(turn)`` whenever a turn starts."""
        self._listeners.append(fn)

        def _un() -> None:
            try:
                self._listeners.remove(fn)
            except ValueError:
                pass

        return _un

    # ------------------------------------------------------------------
    # submission
    # ------------------------------------------------------------------
    def advance_to(self, index: int) -> None:
        """Make the next allocated turn index at least ``index`` (never goes
        back). Lets a persistence layer continue numbering after a restart so
        turn ids journaled by an earlier run are not reused."""
        with self._lock:
            nxt = next(self._counter)
            self._counter = itertools.count(max(nxt, int(index)))

    def next_index(self) -> int:
        """Index the next allocated turn will get (does not allocate it)."""
        with self._lock:
            nxt = next(self._counter)
            self._counter = itertools.count(nxt)
        return nxt

    def new_turn(self, text: str, *, source: str = "term") -> Turn:
        """Allocate a Turn (thread-safe) without queueing it."""
        with self._lock:
            idx = next(self._counter)
        tid = self._id_factory(idx) if self._id_factory is not None else f"t{idx}"
        return Turn(id=f"{self.id_prefix}{tid}", index=idx, text=text, source=source)

    def submit(self, text: str, *, source: str = "term") -> Turn:
        """Queue ``text`` as a new turn. Callable from any thread."""
        turn = self.new_turn(text, source=source)
        self._call_on_loop(self.enqueue, turn)
        return turn

    def enqueue(self, turn: Turn) -> None:
        """Queue an allocated turn. Must run on the render loop."""
        self._remember(turn)
        self._queue.append(turn)
        self.interrupt()
        self._ensure_pump()

    # ------------------------------------------------------------------
    # resume
    # ------------------------------------------------------------------
    def pending_turns(self) -> List[Turn]:
        """Turns the :class:`TurnStore` saw start but never end (e.g. the
        process was killed mid-turn), minus those queued or running now."""
        if self.store is None:
            return []
        busy = {t.id for t in self._queue}
        if self._active is not None:
            busy.add(self._active.id)
        return [t for t in self.store.pending() if t.id not in busy]

    def resume(self, turn_id: str) -> Turn:
        """Run a recorded turn again, with the same id, index, text and
        source (``turn.resumed`` is True; ``turn.start`` carries
        ``"resumed": true``). Must run on the render loop.

        Resumable: a pending turn (started, never ended), or any turn the
        store can ``get`` (e.g. one that ended ``interrupted``). Raises
        ``KeyError`` for an unknown turn, ``ValueError`` for one that is
        queued or running, ``RuntimeError`` without a store.

        Re-running is at-least-once for the turn's side effects: make
        turns idempotent (commit results atomically at the end) and put
        LLM calls behind a record/replay LM (``durable_lm``) so the calls that
        already answered are replayed, not paid for again.
        """
        if self.store is None:
            raise RuntimeError("TurnManager.resume needs a TurnStore (configure(store=...))")
        if (self._active is not None and self._active.id == turn_id) or any(
                t.id == turn_id for t in self._queue):
            raise ValueError(f"turn {turn_id!r} is already queued or running")
        rec = next((t for t in self.store.pending() if t.id == turn_id), None)
        if rec is None:
            get = getattr(self.store, "get", None)
            rec = get(turn_id) if callable(get) else None
        if rec is None:
            raise KeyError(f"no recorded turn {turn_id!r} to resume")
        turn = replace(rec, resumed=True, ts=time.time())
        self.advance_to(turn.index + 1)
        self._results.pop(turn.id, None)
        self.enqueue(turn)
        return turn

    def interrupt(self) -> None:
        """Ask the active turn to give way: its interruptible tasks (those
        whose effective policy is ``"interrupt"``) are cancelled now and as
        they get spawned, until it ends."""
        if self._active is not None:
            self._interrupt_req.add(self._active.id)
            self._cancel_tasks(self._active, only_policy="interrupt")

    def abort_active(self) -> Optional[Turn]:
        """Cancel *every* task of the active turn, whatever its policy (it
        ends ``interrupted``). Used to suspend a session mid-turn. Returns the
        aborted turn, if any. Must run on the render loop."""
        turn = self._active
        if turn is not None:
            self._abort_req.add(turn.id)
            self._cancel_tasks(turn, only_policy=None)
        return turn

    def hold(self) -> None:
        """Stop starting queued turns (the active one runs on) until
        :meth:`release`. Must run on the render loop."""
        self._held = True

    def release(self) -> None:
        """Undo :meth:`hold`: queued turns start again."""
        self._held = False
        if self._queue:
            self._ensure_pump()

    @property
    def held(self) -> bool:
        return self._held

    def abandon(self, status: str = "abandoned") -> List[Turn]:
        """Drop the queued turns, resolving their waiters with ``status``
        (the session is going away). Returns the dropped turns."""
        dropped, self._queue = list(self._queue), deque()
        ids = {t.id for t in dropped}
        if self._active is not None:
            ids.add(self._active.id)
        for tid in ids:
            for fut in self._waiters.pop(tid, []):
                if not fut.done():
                    fut.set_result({"status": status, "duration": 0.0})
        return dropped

    async def wait(self, turn_id: str) -> Dict[str, Any]:
        """Wait until ``turn_id`` ends; returns ``{"status", "duration"}``."""
        res = self._results.get(turn_id)
        if res is not None:
            return res
        fut = asyncio.get_running_loop().create_future()
        self._waiters.setdefault(turn_id, []).append(fut)
        return await fut

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _call_on_loop(self, fn, *args) -> None:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        loop = self._loop or runtime._loop or running
        if loop is None:
            raise RuntimeError("TurnManager.submit() needs a running render loop")
        if loop is running:
            fn(*args)
        else:
            loop.call_soon_threadsafe(fn, *args)

    def _remember(self, turn: Turn) -> None:
        self._history[turn.id] = turn
        while len(self._history) > self.max_history:
            self._history.popitem(last=False)

    def _cancel_tasks(self, turn: Turn, *, only_policy: Optional[str]) -> None:
        cancelled = False
        for t in runtime.pending_turn_tasks(turn.id):
            if (only_policy is None
                    or _effective_policy(runtime.task_owner(t), self.policy) == only_policy):
                t.cancel()
                cancelled = True
        if cancelled:
            self._interrupted.add(turn.id)

    def _ensure_pump(self) -> None:
        if self._pump is None or self._pump.done():
            # Not a tracked task: wait_idle() must not wait on the pump itself.
            self._pump = asyncio.get_running_loop().create_task(
                self._run(), name="pyreact-turns"
            )

    async def _run(self) -> None:
        # The pump runs in its own task/context: attribute its work (and
        # errors) to the session that owns this manager.
        if self._owner is not None:
            runtime.current_owner.set(self._owner)
        while self._queue and not self._held:
            turn = self._queue.popleft()
            try:
                await self._process(turn)
            except asyncio.CancelledError:
                self._finish(turn, "interrupted", time.time())
                raise
            except Exception as exc:  # never let the pump die
                runtime.report_error(exc, None, "turn", turn=None)

    async def _process(self, turn: Turn) -> None:
        from .harness import Blocked, through

        t0 = time.time()
        self._active = turn
        self._latest = turn
        runtime.set_active_turn(turn.id, scope=self)
        status = None
        try:
            # The whole lifecycle (start -> settle) is a "turn" boundary.
            status = await through(
                "turn", "turn", {"text": turn.text}, lambda b: self._lifecycle(turn, b, t0),
                ctx=self._owner, turn_id=turn.id, index=turn.index, source=turn.source,
            )
        except Blocked as exc:
            status = "blocked"
            self.event_bus.publish(
                "turn.blocked",
                {"index": turn.index, "reason": exc.reason, "by": exc.by, "message": str(exc)},
                turn_id=turn.id, source=turn.source,
            )
        except Exception as exc:  # a failing middleware
            runtime.report_error(exc, None, "turn", turn=turn.id)
        if status not in ("done", "interrupted", "error", "blocked"):
            status = self._status(turn)  # short-circuited / failed: settle as usual
        self._finish(turn, status, t0)

    def _store_call(self, method: str, *args: Any) -> None:
        fn = getattr(self.store, method, None) if self.store is not None else None
        if fn is None:
            return
        try:
            fn(*args)
        except Exception as exc:  # a failing store must not break turns
            runtime.report_error(exc, None, "turn-store", turn=None)

    async def _lifecycle(self, turn: Turn, b: Any, t0: float) -> str:
        try:
            self._store_call("on_turn_start", turn)
            payload = {"index": turn.index, "text": turn.text, "source": turn.source}
            if turn.resumed:
                payload["resumed"] = True
            self.event_bus.publish(
                "turn.start",
                payload,
                turn_id=turn.id,
                source=turn.source,
            )
            for fn in list(self._listeners):
                try:
                    fn(turn)
                except Exception as exc:
                    runtime.report_error(exc, None, "turn-listener")
            token = runtime._turn_var.set(turn.id)
            owner_token = runtime.current_owner.set(self._owner)
            try:
                base = {"value": turn.text, "source": turn.source, "ts": turn.ts, "turn_id": turn.id}
                self.input_bus.emit({"type": "text", **base})
                self.input_bus.emit({"type": "submit", **base})
            finally:
                runtime.current_owner.reset(owner_token)
                runtime._turn_var.reset(token)
            await self._wait_settled(turn)
        except Exception as exc:
            runtime.report_error(exc, None, "turn", turn=turn.id)
        status = self._status(turn)
        b.attrs["status"] = status
        b.attrs["duration"] = round(time.time() - t0, 4)
        return status

    def _status(self, turn: Turn) -> str:
        if turn.id in self._interrupted:
            return "interrupted"
        if runtime.pop_turn_errors(turn.id):
            return "error"
        return "done"

    async def _wait_settled(self, turn: Turn, poll: float = 0.02) -> None:
        tid = turn.id
        while True:
            if tid in self._abort_req:
                self._cancel_tasks(turn, only_policy=None)
            elif tid in self._interrupt_req:
                self._cancel_tasks(turn, only_policy="interrupt")
            pending = runtime.pending_turn_tasks(tid)
            if pending:
                await asyncio.wait(pending, timeout=poll * 10)
                continue
            if runtime.renders_pending(self._owner_filter):
                await asyncio.sleep(poll / 4)
                continue
            # Let callbacks scheduled by finishing tasks enqueue renders.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            if runtime.is_turn_idle(tid, self._owner_filter):
                return

    def _finish(self, turn: Turn, status: str, t0: float) -> None:
        if self._active is turn:
            self._active = None
            runtime.set_active_turn(None, scope=self)
        self._interrupted.discard(turn.id)
        self._interrupt_req.discard(turn.id)
        self._abort_req.discard(turn.id)
        runtime.pop_turn_errors(turn.id)
        res = {"status": status, "duration": round(time.time() - t0, 4)}
        self._results[turn.id] = res
        while len(self._results) > self.max_history:
            self._results.popitem(last=False)
        self._store_call("on_turn_end", turn, res)
        self.event_bus.publish(
            "turn.end",
            {"index": turn.index, **res},
            turn_id=turn.id,
            source=turn.source,
        )
        for fut in self._waiters.pop(turn.id, []):
            if not fut.done():
                fut.set_result(res)


# ----------------------------------------------------------------------
# hooks
# ----------------------------------------------------------------------
def get_turn_manager() -> TurnManager:
    from .core import hooks

    return hooks.get_service("turns", TurnManager)


def use_turn(
    policy: Optional[TurnPolicy] = None, *, since_mount: bool = False
) -> Optional[Turn]:
    """Latest Turn of this app/session; rerenders on every new turn.

    ``policy`` (``"queue"``/``"interrupt"``) overrides the app's turn policy
    for tasks spawned by this component and its descendants.

    ``since_mount``: by default a newly mounted component immediately gets
    the session's latest turn -- usually the one *in flight* that caused it to
    mount (what a child mounted to answer this input wants). With
    ``since_mount=True`` it returns ``None`` until a turn starts after the
    component mounted, so a conditionally mounted component (a page shown by
    a navigation, a panel opened by a command) does not handle the input that
    mounted it again.
    """
    from .core import _context_stack, hooks

    if policy is not None:
        if policy not in POLICIES:
            raise ValueError(f"turn policy must be one of {POLICIES}, got {policy!r}")
        ctx = _context_stack.get()
        if ctx is not None:
            ctx._turn_policy = policy

    tm = get_turn_manager()
    at_mount = hooks.use_ref(tm.latest)  # latest turn when we mounted
    turn, set_turn = hooks.use_state(None if since_mount else tm.latest)

    def _subscribe():
        un = tm.subscribe(set_turn)
        latest = tm.latest
        # catch a turn that started before we subscribed
        if latest is not None and not (since_mount and latest is at_mount.current):
            set_turn(latest)
        return un

    hooks.use_effect(_subscribe, [id(tm)])
    return turn
