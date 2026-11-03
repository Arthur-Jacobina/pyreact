"""Observability plumbing for the DSPy hooks layer.

Every LM/module call and tool call is published on the app's
:class:`~pyreact.core.events.EventBus` service (``event_bus``):

    llm.start   {call_id, module, component, model, inputs}
    llm.chunk   {call_id, module, field, chunk, last}
    llm.status  {call_id, module, message}
    llm.end     {call_id, module, component, model, latency, usage, cost, outputs}
    llm.error   {call_id, module, component, model, latency, usage, cost, error}
    llm.cancel  {call_id, module, component, latency}
    tool.start  {call_id, tool, args}
    tool.end    {call_id, tool, args, duration, error, result}

Token usage / cost come from the LM history entries recorded *for that call
only* (a per-call recorder is injected into ``dspy.settings.caller_modules``,
which is a contextvar, so concurrent calls never mix their usage).
"""

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pyreact.core.events import EventBus
from pyreact.core.hook import HookContext

_call_ids = itertools.count(1)


def new_call_id(prefix: str = "llm") -> str:
    return f"{prefix}-{next(_call_ids)}"


def event_bus() -> EventBus:
    """The process-global EventBus service (``HookContext.get_service``)."""
    return HookContext.get_service("event_bus", EventBus)


def session_bus() -> Optional[EventBus]:
    """The calling component's *session* bus (``app.events`` / ``:events``
    observe it), or ``None`` outside a component/task."""
    try:
        from pyreact.core import runtime

        ctx = runtime.current_component()
        if ctx is not None and hasattr(ctx, "get_service"):
            return ctx.get_service("event_bus", EventBus)
    except Exception:
        pass
    return None


def publish(kind: str, payload: Dict[str, Any], *, turn_id=None, source=None) -> None:
    """Publish on the session bus of the calling component and on the global
    bus (once when they are the same object, e.g. headless trees/tests)."""
    try:
        glob = event_bus()
        sess = session_bus()
        if sess is not None and sess is not glob:
            sess.publish(kind, payload, turn_id=turn_id, source=source)
        glob.publish(kind, payload, turn_id=turn_id, source=source)
    except Exception:  # observability must never break a call
        pass


# ---------------------------------------------------------------------------
# Turn ids
# ---------------------------------------------------------------------------
def current_turn_id() -> Optional[str]:
    """Turn the calling code belongs to (see ``pyreact.core.turns``)."""
    from pyreact.core import runtime

    try:
        return runtime.current_turn_id()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Component path
# ---------------------------------------------------------------------------
def component_path(ctx) -> Optional[str]:
    if ctx is None:
        return None
    parts = []
    node = ctx
    while node is not None:
        label = node.name
        if node.key is not None:
            label += f"#{node.key}"
        parts.append(label)
        node = node.parent
    return "/".join(reversed(parts))


# ---------------------------------------------------------------------------
# Call scope: what the current task is doing (read by tool wrappers)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CallScope:
    call_id: str
    module: str
    component: Optional[str]
    turn_id: Optional[str]


_scope: ContextVar[Optional[CallScope]] = ContextVar("pyreact_dspy_call", default=None)


def current_scope() -> Optional[CallScope]:
    return _scope.get()


def enter_scope(scope: CallScope):
    return _scope.set(scope)


def exit_scope(token) -> None:
    try:
        _scope.reset(token)
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
class UsageRecorder:
    """Duck-typed 'module' placed in ``settings.caller_modules``.

    ``BaseLM.update_history`` appends each LM history entry (with ``usage``
    and ``cost``) to every caller module, so this collects exactly the LM
    calls made inside one hook call.
    """

    def __init__(self) -> None:
        self.history: List[dict] = []

    def summary(self) -> Dict[str, Any]:
        return summarize_usage(self.history)


_TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def summarize_usage(entries: List[dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {k: 0 for k in _TOKEN_KEYS}
    out["lm_calls"] = len(entries)
    out["cost"] = None
    models = []
    for e in entries:
        usage = e.get("usage") or {}
        for k in _TOKEN_KEYS:
            v = usage.get(k) if isinstance(usage, dict) else getattr(usage, k, None)
            if isinstance(v, (int, float)):
                out[k] += v
        cost = e.get("cost")
        if isinstance(cost, (int, float)):
            out["cost"] = (out["cost"] or 0.0) + cost
        m = e.get("model")
        if m and m not in models:
            models.append(m)
    out["models"] = models
    return out


# ---------------------------------------------------------------------------
# JSON-friendly previews for event payloads
# ---------------------------------------------------------------------------
def brief(value: Any, limit: int = 400) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, dict):
        return {str(k): brief(v, limit) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return [brief(v, limit) for v in list(value)[:20]]
    s = repr(value)
    return s if len(s) <= limit else s[:limit] + "…"


def outputs_of(result: Any) -> Any:
    """Prediction -> {field: value} (minus bulky trajectories); else a preview."""
    items = getattr(result, "items", None)
    if callable(items):
        try:
            return {k: brief(v) for k, v in items() if k != "trajectory"}
        except Exception:
            pass
    return brief(result)


def model_name(lm: Any) -> Optional[str]:
    return getattr(lm, "model", None) if lm is not None else None


# ---------------------------------------------------------------------------
# Debug: what exactly did the LM see? (``integrations.learn.explain``)
# ---------------------------------------------------------------------------
# The last CALL_BUFFER_SIZE calls keep their LM history entries (prompt
# messages + raw completions) in memory, keyed by call id. Event payloads stay
# small; with DEBUG on (``set_debug(True)`` or ``DSPyProvider(settings=
# {"debug": True})``) ``llm.end`` also carries them as ``lm_calls``.
CALL_BUFFER_SIZE = 64
DEBUG = False
_calls: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_calls_lock = threading.Lock()


def set_debug(on: bool = True) -> None:
    """Attach full prompts/completions (``lm_calls``) to every ``llm.end``."""
    global DEBUG
    DEBUG = bool(on)


def _content(c: Any) -> Any:
    if isinstance(c, list):  # multimodal / content blocks
        return "\n".join(str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in c)
    return c


def lm_calls_of(history: List[dict]) -> List[Dict[str, Any]]:
    """LM history entries -> ``[{model, messages, completion, usage}]`` (plain data)."""
    out = []
    for e in history:
        msgs = e.get("messages")
        if msgs is None and e.get("prompt") is not None:
            msgs = [{"role": "user", "content": e.get("prompt")}]
        outs = e.get("outputs") or []
        completion = outs[0] if outs else e.get("response")
        if isinstance(completion, dict):
            completion = completion.get("text", completion)
        usage = e.get("usage") or {}
        out.append({
            "model": e.get("model"),
            "messages": [{"role": m.get("role"), "content": _content(m.get("content"))}
                         for m in (msgs or []) if isinstance(m, dict)],
            "completion": completion if isinstance(completion, (str, type(None))) else str(completion),
            "usage": dict(usage) if isinstance(usage, dict) else {},
        })
    return out


def record_call(record: Dict[str, Any]) -> None:
    with _calls_lock:
        _calls[record["call_id"]] = record
        _calls.move_to_end(record["call_id"])
        while len(_calls) > CALL_BUFFER_SIZE:
            _calls.popitem(last=False)


def get_call(call_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """A recorded call by id, or the most recent one when ``call_id`` is None."""
    with _calls_lock:
        if call_id is None:
            return next(reversed(_calls.values()), None) if _calls else None
        return _calls.get(call_id)


def recorded_calls() -> List[Dict[str, Any]]:
    with _calls_lock:
        return list(_calls.values())
