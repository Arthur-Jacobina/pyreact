"""Runs -> events -> datasets.

:func:`dataset_from_events` accepts any event stream:

- ``Event`` objects or plain dicts with ``kind`` / ``payload`` (``turn_id``,
  ``ts`` optional) -- an ``EventBus`` log (``app.events``), a
  ``TraceRecorder`` / ``JSONLSink`` JSONL file (pass the path), a durable
  session log, a list you built yourself;
- an ``AppRunner`` / ``Session`` / ``EventLog`` (its recorded events).

Calls come from ``module.call`` events (``TracedModule``) or from the
``llm.start`` + ``llm.end`` pair every ``use_call`` / ``invoke_module`` call
publishes; ``feedback`` events (``send_feedback`` / ``:feedback``) are joined
onto them by call id or turn id. A ``feedback`` with ``expected={...}`` is a
correction: those fields become the labels.
"""

from __future__ import annotations

import json
import os
import random
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import dspy

from integrations.feedback import FEEDBACK, MODULE_CALL, TraceRecorder, key_str

from .metrics import META_FIELDS

__all__ = ["dataset_from_events", "split", "iter_events", "call_records"]

_NON_LABELS = {"reasoning", "rationale", "trajectory", "final_reasoning"}


def _as_dict(ev: Any) -> Optional[Dict[str, Any]]:
    if isinstance(ev, dict):
        if "kind" in ev:
            return ev
        inner = ev.get("event")  # {"event": {...}} envelopes
        return inner if isinstance(inner, dict) and "kind" in inner else None
    to_dict = getattr(ev, "to_dict", None)
    if callable(to_dict):
        d = to_dict()
        return d if isinstance(d, dict) and "kind" in d else None
    kind = getattr(ev, "kind", None)
    if kind is not None:
        return {"kind": kind, "payload": getattr(ev, "payload", {}) or {},
                "turn_id": getattr(ev, "turn_id", None), "ts": getattr(ev, "ts", None),
                "id": getattr(ev, "id", None)}
    return None


def iter_events(source: Any) -> Iterable[Dict[str, Any]]:
    """Normalize any supported event source into event dicts."""
    if source is None:
        return
    if isinstance(source, (str, os.PathLike)):
        with open(source, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = _as_dict(json.loads(line))
                    if d is not None:
                        yield d
        return
    log = getattr(source, "events", None)  # AppRunner / Session
    if log is not None and hasattr(log, "recent"):
        source = log
    if hasattr(source, "recent") and callable(source.recent):  # EventLog
        source = source.recent(None)
    for ev in source:
        d = _as_dict(ev)
        if d is not None:
            yield d


class _Collector(TraceRecorder):
    """TraceRecorder fed by hand (no bus) that also understands llm.* pairs."""

    def __init__(self) -> None:  # noqa: D401 - no bus subscription
        import threading

        self.bus, self.path = None, None
        self._lock = threading.RLock()
        self.calls, self._pending = {}, []
        self._unsub = None
        self._starts: Dict[str, Dict[str, Any]] = {}

    def feed(self, ev: Dict[str, Any]) -> None:
        kind = ev.get("kind")
        p = ev.get("payload") or {}
        if kind in (MODULE_CALL, FEEDBACK):
            self._ingest(ev, persist=False)
        elif kind == "llm.start":
            self._starts[p.get("call_id")] = p
        elif kind in ("llm.end", "llm.error"):
            start = self._starts.pop(p.get("call_id"), {})
            inputs = start.get("inputs") or {}
            self._ingest({
                "kind": MODULE_CALL,
                "turn_id": ev.get("turn_id"),
                "ts": ev.get("ts"),
                "payload": {
                    "call_id": p.get("call_id"),
                    "module_key": p.get("module_key") or p.get("module"),
                    "inputs": inputs,
                    "outputs": p.get("outputs") or {},
                    "error": p.get("error") if kind == "llm.error" else None,
                },
            }, persist=False)
            rec = self.calls.get(p.get("call_id"))
            if rec is not None:
                rec["label"] = p.get("module")
                rec["component"] = p.get("component")


def _matches(rec: Dict[str, Any], wanted: str) -> bool:
    k = rec.get("module_key") or ""
    if k == wanted or rec.get("label") == wanted:
        return True
    parts = k.split(":")
    scope = parts[-1] if parts else ""
    return scope == wanted or scope.endswith("#" + wanted) or scope.endswith("/" + wanted)


def call_records(events: Any, module_key: Any = None) -> List[Dict[str, Any]]:
    """Joined call records (inputs, outputs, feedback list) from any event source."""
    col = _Collector()
    for ev in iter_events(events):
        col.feed(ev)
    recs = list(col.calls.values())
    if module_key is None:
        return recs
    wanted = key_str(module_key)
    exact = [r for r in recs if r.get("module_key") == wanted]
    return exact or [r for r in recs if _matches(r, wanted)]


def dataset_from_events(
    events: Any,
    module_key: Any,
    *,
    input_keys: Optional[Iterable[str]] = None,
    min_score: Optional[float] = None,
    require_feedback: bool = False,
    include_errors: bool = False,
) -> List[dspy.Example]:
    """``dspy.Example`` list for one module from an event stream.

    ``module_key``: the module's key (tuple or ``"Cls:sig:scope"``), its
    label, or just its scope / component key (``"supervisor"``).
    Labels: the latest feedback's ``expected`` correction when present, else
    the recorded outputs (minus reasoning/trajectory). ``min_score`` keeps
    calls whose latest feedback score is >= it, plus corrected ones.
    Each example also carries ``call_id``, ``recorded_outputs``,
    ``feedback_score`` and ``feedback_text`` (used by metric feedback).
    """
    col_records = call_records(events, module_key)
    out: List[dspy.Example] = []
    truncated = 0
    for r in col_records:
        if r["error"] and not include_errors:
            continue
        fb = r["feedback"][-1] if r["feedback"] else None
        if require_feedback and fb is None:
            continue
        outputs = {k: v for k, v in (r["outputs"] or {}).items() if k not in _NON_LABELS}
        expected = (fb or {}).get("expected") or {}
        if expected and outputs:
            expected = {k: v for k, v in expected.items() if k in outputs}
        if min_score is not None and not expected:
            if fb is None or fb["score"] < min_score:
                continue
        labels = {**outputs, **expected} if expected else outputs
        keys = list(input_keys) if input_keys is not None else list(r["inputs"])
        if any(isinstance(v, str) and v.endswith("…") for v in r["inputs"].values()):
            truncated += 1
        ex = dspy.Example(
            **r["inputs"],
            **{k: v for k, v in labels.items() if k not in META_FIELDS},
            call_id=r["call_id"],
            recorded_outputs=r["outputs"],
            feedback_score=None if fb is None else fb["score"],
            feedback_text=None if fb is None else fb["text"],
        ).with_inputs(*keys)
        out.append(ex)
    if truncated:
        warnings.warn(
            f"{truncated} example(s) have inputs truncated to the llm.start preview "
            "(400 chars); wrap the module with TracedModule/use_traced for full inputs.",
            UserWarning, stacklevel=2,
        )
    return out


def split(dataset: Sequence[Any], val: float = 0.3, seed: int = 0) -> Tuple[List[Any], List[Any]]:
    """Shuffle deterministically and split into ``(train, val)``.

    ``val`` is a fraction (0..1) or an absolute count; both parts get at
    least one example when ``len(dataset) >= 2``.
    """
    items = list(dataset)
    random.Random(seed).shuffle(items)
    n = len(items)
    if n < 2:
        return items, []
    k = int(round(n * val)) if isinstance(val, float) and val < 1 else int(val)
    k = max(1, min(n - 1, k))
    return items[k:], items[:k]
