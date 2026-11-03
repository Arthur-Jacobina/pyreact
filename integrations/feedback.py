"""Runs produce events; events become datasets.

Two event kinds feed the learning loop (published on the app ``EventBus``,
``hooks.get_service("event_bus", EventBus)``):

``module.call``  emitted by :class:`TracedModule` after every module call::

    {"call_id": str, "module_key": str, "inputs": {...}, "outputs": {...},
     "error": str | None, "latency": float}

``feedback``     emitted by the app/user/judge (see :func:`send_feedback`)::

    {"score": float, "text": str, "target": call_id | turn_id,
     "expected": {...} | None}      # optional corrected outputs (a label)

:class:`TraceRecorder` subscribes to both, joins feedback onto calls (by call
id, or onto every call of a turn when ``target`` is a turn id), persists the
raw events to JSONL and turns them into ``dspy.Example`` datasets for
:func:`integrations.optimize.optimize_with_gepa`.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional

import dspy

from pyreact.core.events import Event, EventBus
from pyreact.core.hook import HookContext

MODULE_CALL = "module.call"
FEEDBACK = "feedback"


def key_str(module_key: Any) -> str:
    """Stable string form of a module key (tuples -> ``a:b:c``)."""
    if isinstance(module_key, (tuple, list)):
        return ":".join("" if p is None else str(p) for p in module_key)
    return str(module_key)


def get_event_bus() -> EventBus:
    """The app-wide EventBus service (same instance components get)."""
    return HookContext.get_service("event_bus", EventBus)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    to_dict = getattr(value, "toDict", None) or getattr(value, "model_dump", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict())
        except Exception:
            pass
    return str(value)


def outputs_of(pred: Any) -> Dict[str, Any]:
    """Output fields of a dspy.Prediction (or any object) as plain data."""
    if pred is None:
        return {}
    if isinstance(pred, dict):
        return _jsonable(pred)
    keys = getattr(pred, "keys", None)
    if callable(keys):
        try:
            return {k: _jsonable(pred[k]) for k in keys()}
        except Exception:
            pass
    return {k: _jsonable(v) for k, v in vars(pred).items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Emitting side
# ---------------------------------------------------------------------------
class TracedModule:
    """Recorder-aware wrapper: calls ``module`` and publishes ``module.call``.

    Drop-in for ``use_dspy_call`` (it exposes ``acall`` and ``__call__``).
    ``last_call_id`` is the id to target with :func:`send_feedback`.
    """

    def __init__(
        self,
        module: Any,
        module_key: Any,
        *,
        bus: Optional[EventBus] = None,
        turn_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        self.module = module
        self.module_key = key_str(module_key)
        self.bus = bus or get_event_bus()
        self.turn_id = turn_id
        self.source = source
        self.last_call_id: Optional[str] = None

    def _publish(self, call_id, inputs, pred, error, t0, turn_id) -> None:
        self.bus.publish(
            MODULE_CALL,
            {
                "call_id": call_id,
                "module_key": self.module_key,
                "inputs": _jsonable(inputs),
                "outputs": outputs_of(pred) if error is None else {},
                "error": None if error is None else f"{type(error).__name__}: {error}",
                "latency": round(time.time() - t0, 4),
            },
            turn_id=turn_id or self.turn_id,
            source=self.source,
        )

    async def acall(self, *, _turn_id: Optional[str] = None, **inputs):
        call_id = uuid.uuid4().hex[:12]
        self.last_call_id = call_id
        t0 = time.time()
        try:
            acall = getattr(self.module, "acall", None)
            if acall is not None and inspect.iscoroutinefunction(acall):
                pred = await acall(**inputs)
            else:
                import asyncio

                pred = await asyncio.to_thread(self.module, **inputs)
        except BaseException as exc:
            if isinstance(exc, Exception):
                self._publish(call_id, inputs, None, exc, t0, _turn_id)
            raise
        self._publish(call_id, inputs, pred, None, t0, _turn_id)
        return pred

    def __call__(self, *, _turn_id: Optional[str] = None, **inputs):
        call_id = uuid.uuid4().hex[:12]
        self.last_call_id = call_id
        t0 = time.time()
        try:
            pred = self.module(**inputs)
        except Exception as exc:
            self._publish(call_id, inputs, None, exc, t0, _turn_id)
            raise
        self._publish(call_id, inputs, pred, None, t0, _turn_id)
        return pred


def send_feedback(
    target: str,
    score: float,
    text: str = "",
    *,
    expected: Optional[Dict[str, Any]] = None,
    bus: Optional[EventBus] = None,
    turn_id: Optional[str] = None,
) -> Event:
    """Publish a ``feedback`` event about a call id or a turn id."""
    payload = {"score": float(score), "text": text, "target": target}
    if expected is not None:
        payload["expected"] = _jsonable(expected)
    return (bus or get_event_bus()).publish(FEEDBACK, payload, turn_id=turn_id)


def use_traced(module: Any, module_key: Any, *, turn_id: Optional[str] = None) -> TracedModule:
    """Hook: memoized :class:`TracedModule` over ``module`` on the app bus."""
    from pyreact.core.core import hooks

    bus = hooks.get_service("event_bus", EventBus)
    return hooks.use_memo(
        lambda: TracedModule(module, module_key, bus=bus, turn_id=turn_id),
        deps=[id(module), key_str(module_key), turn_id],
    )


# ---------------------------------------------------------------------------
# Recording side
# ---------------------------------------------------------------------------
class TraceRecorder:
    """Subscribes to ``module.call``/``feedback`` and builds datasets.

    ``path``: optional JSONL file; every received event is appended to it and
    :meth:`load` replays such a file (so datasets survive restarts).
    """

    def __init__(self, bus: Optional[EventBus] = None, path: Optional[str] = None) -> None:
        self.bus = bus or get_event_bus()
        self.path = path
        self._lock = threading.RLock()
        self.calls: Dict[str, Dict[str, Any]] = {}  # call_id -> record (ordered)
        self._pending: List[Dict[str, Any]] = []  # feedback that arrived early
        self._unsub: Optional[Callable[[], None]] = self.bus.subscribe(
            self._on_event, kinds=[MODULE_CALL, FEEDBACK]
        )

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- ingestion --------------------------------------------------------
    def _on_event(self, ev: Event) -> None:
        self._ingest(ev.to_dict(), persist=True)

    def _ingest(self, ev: Dict[str, Any], *, persist: bool) -> None:
        with self._lock:
            if persist and self.path:
                d = os.path.dirname(os.path.abspath(self.path))
                os.makedirs(d, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(_jsonable(ev), ensure_ascii=False) + "\n")
            p = ev.get("payload") or {}
            if ev["kind"] == MODULE_CALL:
                cid = p.get("call_id") or str(ev.get("id"))
                self.calls[cid] = {
                    "call_id": cid,
                    "module_key": p.get("module_key"),
                    "inputs": p.get("inputs") or {},
                    "outputs": p.get("outputs") or {},
                    "error": p.get("error"),
                    "turn_id": ev.get("turn_id"),
                    "ts": ev.get("ts"),
                    "feedback": [],
                }
                still = []
                for fb in self._pending:
                    if not self._attach(fb):
                        still.append(fb)
                self._pending = still
            elif ev["kind"] == FEEDBACK:
                fb = {
                    "score": float(p.get("score", 0.0)),
                    "text": p.get("text", ""),
                    "target": p.get("target") or ev.get("turn_id"),
                    "expected": p.get("expected"),
                    "ts": ev.get("ts"),
                }
                if not self._attach(fb):
                    self._pending.append(fb)

    def _attach(self, fb: Dict[str, Any]) -> bool:
        target = fb["target"]
        rec = self.calls.get(target)
        if rec is not None:
            rec["feedback"].append(fb)
            return True
        hit = False
        for rec in self.calls.values():
            if target is not None and rec.get("turn_id") == target:
                rec["feedback"].append(fb)
                hit = True
        return hit

    def load(self, path: Optional[str] = None) -> "TraceRecorder":
        """Replay a JSONL file written by a recorder (without re-persisting)."""
        path = path or self.path
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._ingest(json.loads(line), persist=False)
        return self

    # -- queries ----------------------------------------------------------
    def records(self, module_key: Any = None) -> List[Dict[str, Any]]:
        k = None if module_key is None else key_str(module_key)
        with self._lock:
            return [
                dict(r) for r in self.calls.values() if k is None or r["module_key"] == k
            ]

    def dataset(
        self,
        module_key: Any,
        *,
        input_keys: Optional[Iterable[str]] = None,
        min_score: Optional[float] = None,
        require_feedback: bool = False,
        include_errors: bool = False,
    ) -> List[dspy.Example]:
        """Examples for ``module_key``: ``inputs`` + label fields.

        Label fields come from the latest feedback's ``expected`` (a
        correction) when present, else from the recorded outputs. Each example
        also carries ``feedback_score`` / ``feedback_text`` (latest feedback),
        and ``recorded_outputs`` (what the module actually said), which a
        GEPA metric can use to explain failures. ``min_score`` keeps only calls
        whose latest feedback score is >= it unless an ``expected`` label was
        given (corrected failures are the most useful examples).
        """
        out: List[dspy.Example] = []
        for r in self.records(module_key):
            if r["error"] and not include_errors:
                continue
            fb = r["feedback"][-1] if r["feedback"] else None
            if require_feedback and fb is None:
                continue
            expected = (fb or {}).get("expected")
            if min_score is not None and not expected:
                if fb is None or fb["score"] < min_score:
                    continue
            labels = dict(expected) if expected else dict(r["outputs"])
            keys = list(input_keys) if input_keys is not None else list(r["inputs"])
            ex = dspy.Example(
                **r["inputs"],
                **labels,
                call_id=r["call_id"],
                recorded_outputs=r["outputs"],
                feedback_score=None if fb is None else fb["score"],
                feedback_text=None if fb is None else fb["text"],
            ).with_inputs(*keys)
            out.append(ex)
        return out


def use_trace_recorder(path: Optional[str] = None) -> TraceRecorder:
    """Hook: the app-wide TraceRecorder service (created on first use)."""
    from pyreact.core.core import hooks

    bus = hooks.get_service("event_bus", EventBus)
    return hooks.get_service("trace_recorder", lambda: TraceRecorder(bus, path=path))
