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
   ``{"status": "done" | "interrupted" | "error", "duration": seconds}``.

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
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Literal, Optional

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


TurnListener = Callable[[Turn], None]


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
    ) -> None:
        self.policy: str = policy
        self._input_bus = input_bus
        self._event_bus = event_bus
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self._queue: Deque[Turn] = deque()
        self._active: Optional[Turn] = None
        self._latest: Optional[Turn] = None
        self._history: "OrderedDict[str, Turn]" = OrderedDict()
        self._results: Dict[str, Dict[str, Any]] = OrderedDict()
        self._waiters: Dict[str, List[asyncio.Future]] = {}
        self._interrupted: set = set()  # turns that had tasks cancelled
        self._interrupt_req: set = set()  # turns asked to give way
        self._listeners: List[TurnListener] = []
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
    ) -> "TurnManager":
        """Configure the manager. Session wiring:

        - ``id_prefix``: prefix of turn ids (makes them unique across the
          sessions sharing a loop, e.g. ``"s1/"`` -> ``"s1/t3"``);
        - ``owner``: component (session root) owning the work done while a
          turn's input is dispatched;
        - ``owner_filter``: restricts "renders pending" checks to this
          session's components, so busy sessions don't delay each other.
        """
        if id_prefix is not None:
            self.id_prefix = id_prefix
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
    def new_turn(self, text: str, *, source: str = "term") -> Turn:
        """Allocate a Turn (thread-safe) without queueing it."""
        with self._lock:
            idx = next(self._counter)
        return Turn(id=f"{self.id_prefix}t{idx}", index=idx, text=text, source=source)

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

    def interrupt(self) -> None:
        """Ask the active turn to give way: its interruptible tasks (those
        whose effective policy is ``"interrupt"``) are cancelled now and as
        they get spawned, until it ends."""
        if self._active is not None:
            self._interrupt_req.add(self._active.id)
            self._cancel_tasks(self._active, only_policy="interrupt")

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

    def _cancel_tasks(self, turn: Turn, *, only_policy: str) -> None:
        cancelled = False
        for t in runtime.pending_turn_tasks(turn.id):
            if _effective_policy(runtime.task_owner(t), self.policy) == only_policy:
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
        while self._queue:
            turn = self._queue.popleft()
            try:
                await self._process(turn)
            except asyncio.CancelledError:
                self._finish(turn, "interrupted", time.time())
                raise
            except Exception as exc:  # never let the pump die
                runtime.report_error(exc, None, "turn", turn=None)

    async def _process(self, turn: Turn) -> None:
        t0 = time.time()
        self._active = turn
        self._latest = turn
        runtime.set_active_turn(turn.id, scope=self)
        status = "done"
        try:
            self.event_bus.publish(
                "turn.start",
                {"index": turn.index, "text": turn.text, "source": turn.source},
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
        if turn.id in self._interrupted:
            status = "interrupted"
        elif runtime.pop_turn_errors(turn.id):
            status = "error"
        self._finish(turn, status, t0)

    async def _wait_settled(self, turn: Turn, poll: float = 0.02) -> None:
        tid = turn.id
        while True:
            if tid in self._interrupt_req:
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
        runtime.pop_turn_errors(turn.id)
        res = {"status": status, "duration": round(time.time() - t0, 4)}
        self._results[turn.id] = res
        while len(self._results) > self.max_history:
            self._results.popitem(last=False)
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


def use_turn(policy: Optional[TurnPolicy] = None) -> Optional[Turn]:
    """Latest Turn of this app/session; rerenders on every new turn.

    ``policy`` (``"queue"``/``"interrupt"``) overrides the app's turn policy
    for tasks spawned by this component and its descendants.
    """
    from .core import _context_stack, hooks

    if policy is not None:
        if policy not in POLICIES:
            raise ValueError(f"turn policy must be one of {POLICIES}, got {policy!r}")
        ctx = _context_stack.get()
        if ctx is not None:
            ctx._turn_policy = policy

    tm = get_turn_manager()
    turn, set_turn = hooks.use_state(tm.latest)

    def _subscribe():
        un = tm.subscribe(set_turn)
        if tm.latest is not None:
            set_turn(tm.latest)  # catch a turn that started before we subscribed
        return un

    hooks.use_effect(_subscribe, [id(tm)])
    return turn
