"""Tracing middleware: one OpenTelemetry span per harness boundary.

    from pyreact.core.harness import install
    from pyreact_otel import otel_middleware

    uninstall = install(otel_middleware(tracer_provider))

Span shapes (see README for the full reference):

- ``turn``          -> ``pyreact.turn`` (root of a trace), ``gen_ai.operation.name=invoke_agent``
- ``llm``           -> ``chat <model>`` with GenAI attributes (model, provider, token usage)
- ``tool``          -> ``execute_tool <name>`` with ``gen_ai.tool.*``
- ``sandbox.exec``  -> ``sandbox.exec <name>``; any other kind -> ``<kind> <name>``

Parenting: the OTel context (contextvars) is the primary source, so nesting
works across ``asyncio`` tasks and ``asyncio.to_thread``. When the context
does not flow (the call runs in a task created before its parent started),
``Boundary.parent_id`` and then the live ``turn`` span of the same
session/turn are used as the parent.

Prompt/completion content is recorded only with ``capture_content=True``
(``gen_ai.input.messages`` / ``gen_ai.output.messages``, tool arguments and
results), each value truncated to ``max_content_length`` characters.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from pyreact.core.harness import Blocked, Boundary, Middleware

from . import attributes as A

__all__ = [
    "Enricher",
    "SpanInfo",
    "live_span",
    "otel_middleware",
    "provider_of",
    "span_name",
    "trace_name",
    "turn_span",
]


# ---------------------------------------------------------------------------
# Live span registry: fallback parents when the OTel context does not flow.
# ---------------------------------------------------------------------------
class _Registry:
    def __init__(self, maxlen: int = 4096) -> None:
        self._spans: "OrderedDict[Hashable, Span]" = OrderedDict()
        self._lock = threading.Lock()
        self._maxlen = maxlen

    def put(self, key: Hashable, span: Span) -> None:
        with self._lock:
            self._spans[key] = span
            self._spans.move_to_end(key)
            while len(self._spans) > self._maxlen:
                self._spans.popitem(last=False)

    def get(self, key: Hashable) -> Optional[Span]:
        with self._lock:
            return self._spans.get(key)

    def pop(self, key: Hashable, span: Span) -> None:
        with self._lock:
            if self._spans.get(key) is span:
                del self._spans[key]


_live = _Registry()
_trace_names = _Registry()  # trace_id -> name of the trace's root boundary


def trace_name(b: Boundary) -> str:
    """Display name of a trace rooted at ``b`` (``pyreact.turn <name>`` for turns)."""
    if b.kind == "turn":
        return f"pyreact.turn {b.name}" if b.name else "pyreact.turn"
    return span_name(b)


def live_span(boundary_id: str) -> Optional[Span]:
    """The open span of boundary ``boundary_id`` (None once it ended)."""
    return _live.get(("b", boundary_id))


def turn_span(turn_id: Optional[str], session_id: Optional[str] = None) -> Optional[Span]:
    """The open ``turn`` span for ``turn_id`` (optionally scoped to a session)."""
    if not turn_id:
        return None
    if session_id is not None:
        return _live.get(("turn", session_id, turn_id))
    return _live.get(("turn", turn_id))


def _register(b: Boundary, span: Span) -> List[Hashable]:
    keys: List[Hashable] = [("b", b.id)]
    if b.kind == "turn" and b.turn_id:
        keys += [("turn", b.session_id, b.turn_id), ("turn", b.turn_id)]
    for k in keys:
        _live.put(k, span)
    return keys


def _parent_context(b: Boundary):
    """Explicit parent > current OTel context > live turn span > new trace.

    Returns ``(context or None, is_root)``."""
    if b.parent_id:
        parent = live_span(b.parent_id)
        if parent is not None:
            return trace.set_span_in_context(parent), False
    if trace.get_current_span().get_span_context().is_valid:
        return None, False  # inherit the current context
    if b.kind != "turn":
        parent = turn_span(b.turn_id, b.session_id)
        if parent is not None:
            return trace.set_span_in_context(parent), False
    return None, True


# ---------------------------------------------------------------------------
# Naming / providers
# ---------------------------------------------------------------------------
_PROVIDERS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "claude-cli": "anthropic",
    "bedrock": "aws.bedrock",
    "azure": "azure.ai.openai",
    "azure_ai": "azure.ai.inference",
    "gemini": "gcp.gemini",
    "vertex_ai": "gcp.vertex_ai",
    "mistral": "mistral_ai",
    "groq": "groq",
    "deepseek": "deepseek",
    "xai": "x_ai",
    "cohere": "cohere",
    "perplexity": "perplexity",
    "ollama": "ollama",
}


def provider_of(model: Optional[str], explicit: Optional[str] = None) -> str:
    """``gen_ai.provider.name`` for a (LiteLLM-style) model string."""
    if explicit:
        return str(explicit)
    if not model:
        return "_OTHER"
    m = str(model).lower()
    prefix = m.split("/", 1)[0] if "/" in m else ""
    if prefix in _PROVIDERS:
        return _PROVIDERS[prefix]
    if m.startswith(("gpt", "o1", "o3", "o4", "text-embedding")):
        return "openai"
    if "claude" in m:
        return "anthropic"
    if m.startswith("gemini"):
        return "gcp.gemini"
    return prefix or "_OTHER"


def span_name(b: Boundary) -> str:
    if b.kind == "turn":
        return "pyreact.turn"
    if b.kind == "llm":
        model = b.attrs.get("model")
        return f"{A.OP_CHAT} {model}" if model else f"{A.OP_CHAT} {b.name}"
    if b.kind == "tool" or b.kind.startswith("tool."):
        return f"{A.OP_EXECUTE_TOOL} {b.name}"
    return f"{b.kind} {b.name}" if b.name else b.kind


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------
def _scalar(v: Any) -> Any:
    if isinstance(v, (str, bool, int, float)):
        return v
    return None


def _usage(attrs: Mapping[str, Any]) -> Dict[str, Any]:
    u = attrs.get("usage") or {}
    if not isinstance(u, Mapping):
        u = {k: getattr(u, k, None) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    out = {}
    for src, dst in (("prompt_tokens", "input"), ("input_tokens", "input"),
                     ("completion_tokens", "output"), ("output_tokens", "output"),
                     ("total_tokens", "total")):
        v = u.get(src)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and dst not in out:
            out[dst] = int(v)
    return out


def _base_attributes(b: Boundary) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        A.BOUNDARY_KIND: b.kind,
        A.BOUNDARY_NAME: b.name,
        A.BOUNDARY_ID: b.id,
        A.ATTEMPT: b.attempt,
    }
    if b.parent_id:
        out[A.BOUNDARY_PARENT_ID] = b.parent_id
    if b.component:
        out[A.COMPONENT] = b.component
    if b.session_id:
        out[A.SESSION] = b.session_id
        out[A.SESSION_ID] = b.session_id
    if b.turn_id:
        out[A.TURN] = b.turn_id
    user = b.attrs.get("user_id")
    if user is not None:
        out[A.USER_ID] = str(user)
    for key, attr in (("call_id", A.CALL_ID), ("module_key", A.MODULE_KEY), ("arch", A.ARCH)):
        v = _scalar(b.attrs.get(key))
        if v is not None:
            out[attr] = v
    return out


def _kind_attributes(b: Boundary) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if b.kind == "turn":
        out[A.GEN_AI_OPERATION_NAME] = A.OP_INVOKE_AGENT
        out[A.GEN_AI_AGENT_NAME] = b.component or b.name
    elif b.kind == "llm":
        model = _scalar(b.attrs.get("model"))
        provider = provider_of(model, b.attrs.get("provider"))
        out[A.GEN_AI_OPERATION_NAME] = A.OP_CHAT
        out[A.GEN_AI_PROVIDER_NAME] = provider
        out[A.GEN_AI_SYSTEM] = provider
        if model:
            out[A.GEN_AI_REQUEST_MODEL] = str(model)
        resp_model = _scalar(b.attrs.get("response_model"))
        if resp_model:
            out[A.GEN_AI_RESPONSE_MODEL] = str(resp_model)
        usage = _usage(b.attrs)
        if "input" in usage:
            out[A.GEN_AI_USAGE_INPUT_TOKENS] = usage["input"]
        if "output" in usage:
            out[A.GEN_AI_USAGE_OUTPUT_TOKENS] = usage["output"]
        if "total" in usage:
            out[A.TOTAL_TOKENS] = usage["total"]
        cost = b.attrs.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            out[A.COST] = float(cost)
        lm_calls = b.attrs.get("lm_calls")
        if isinstance(lm_calls, (list, tuple)):
            out[A.LM_CALLS] = len(lm_calls)
        elif isinstance(lm_calls, int) and not isinstance(lm_calls, bool):
            out[A.LM_CALLS] = lm_calls
    elif b.kind == "tool" or b.kind.startswith("tool."):
        out[A.GEN_AI_OPERATION_NAME] = A.OP_EXECUTE_TOOL
        out[A.GEN_AI_TOOL_NAME] = b.name
        out[A.GEN_AI_TOOL_TYPE] = "function"
        call_id = _scalar(b.attrs.get("call_id"))
        if call_id is not None:
            out[A.GEN_AI_TOOL_CALL_ID] = str(call_id)
    return out


# ---------------------------------------------------------------------------
# Content (opt-in)
# ---------------------------------------------------------------------------
def _jsonable(v: Any, depth: int = 0) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if depth > 6:
        return repr(v)
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x, depth + 1) for x in v]
    items = getattr(v, "items", None)  # dspy.Prediction & friends
    if callable(items):
        try:
            return {str(k): _jsonable(x, depth + 1) for k, x in items() if k != "trajectory"}
        except Exception:
            pass
    return repr(v)


def _dump(v: Any, limit: int) -> str:
    s = v if isinstance(v, str) else json.dumps(_jsonable(v), ensure_ascii=False, default=repr)
    if len(s) > limit:
        s = s[:limit] + f"...[truncated {len(s) - limit} chars]"
    return s


def _text(content: Any) -> str:
    if isinstance(content, list):  # content blocks
        return "\n".join(str(p.get("text", p)) if isinstance(p, Mapping) else str(p)
                         for p in content)
    return "" if content is None else str(content)


def _genai_messages(lm_calls: Sequence[Any]):
    """``attrs["lm_calls"]`` ([{messages, completion}]) -> semconv message lists."""
    inputs, outputs = [], []
    for call in lm_calls:
        if not isinstance(call, Mapping):
            continue
        for m in call.get("messages") or []:
            if isinstance(m, Mapping):
                inputs.append({"role": m.get("role") or "user",
                               "parts": [{"type": "text", "content": _text(m.get("content"))}]})
        completion = call.get("completion")
        if completion is not None:
            outputs.append({"role": "assistant", "finish_reason": "stop",
                            "parts": [{"type": "text", "content": _text(completion)}]})
    return inputs, outputs


def _content_attributes(b: Boundary, result: Any, has_result: bool, limit: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    outputs = b.attrs.get("outputs", result if has_result else None)
    if b.kind == "llm":
        lm_calls = b.attrs.get("lm_calls")
        if isinstance(lm_calls, (list, tuple)) and lm_calls:
            ins, outs = _genai_messages(lm_calls)
            if ins:
                out[A.GEN_AI_INPUT_MESSAGES] = _dump(ins, limit)
            if outs:
                out[A.GEN_AI_OUTPUT_MESSAGES] = _dump(outs, limit)
    elif b.kind == "tool" or b.kind.startswith("tool."):
        out[A.GEN_AI_TOOL_CALL_ARGUMENTS] = _dump(b.inputs, limit)
        if has_result:
            out[A.GEN_AI_TOOL_CALL_RESULT] = _dump(result, limit)
    if b.inputs:
        out[A.INPUTS] = _dump(b.inputs, limit)
    if outputs is not None:
        out[A.OUTPUTS] = _dump(outputs, limit)
    return out


# ---------------------------------------------------------------------------
# Enrichers: vendor-specific attributes (e.g. langfuse.*) computed at span end
# ---------------------------------------------------------------------------
@dataclass
class SpanInfo:
    """What an enricher sees when a boundary span ends."""

    boundary: Boundary
    attributes: Dict[str, Any]  # everything already set by the middleware
    error: Optional[BaseException] = None
    content: Dict[str, str] = field(default_factory=dict)  # empty unless capture_content
    trace_name: Optional[str] = None  # name of the trace's root boundary (when known)
    is_root: bool = False

    @property
    def blocked(self) -> bool:
        return isinstance(self.error, Blocked)


Enricher = Callable[[SpanInfo], Optional[Mapping[str, Any]]]


# ---------------------------------------------------------------------------
# The middleware
# ---------------------------------------------------------------------------
def _clean(attrs: Mapping[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (str, bool, int, float)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
            out[k] = list(v)
        else:
            out[k] = json.dumps(_jsonable(v), ensure_ascii=False, default=repr)
    return out


def otel_middleware(
    tracer_provider: Any = None,
    *,
    capture_content: bool = False,
    max_content_length: int = 8192,
    enrichers: Sequence[Enricher] = (),
    metrics: Any = None,
) -> Middleware:
    """Harness middleware that opens one span per boundary.

    ``tracer_provider`` defaults to the global one. ``enrichers`` add
    vendor-specific attributes when a span ends. ``metrics`` is an optional
    recorder with ``record(boundary, duration_s, error)`` (see
    :class:`pyreact_otel.metrics.BoundaryMetrics`).
    """
    from . import __version__

    tracer = trace.get_tracer("pyreact_otel", __version__, tracer_provider=tracer_provider)
    enrichers = tuple(enrichers)

    async def otel(b: Boundary, call_next):
        kind = SpanKind.CLIENT if b.kind == "llm" else SpanKind.INTERNAL
        attrs = {**_base_attributes(b), **_kind_attributes(b)}
        parent_ctx, is_root = _parent_context(b)
        with tracer.start_as_current_span(
            span_name(b), context=parent_ctx, kind=kind, attributes=_clean(attrs),
            record_exception=False, set_status_on_exception=False,
        ) as span:
            keys = _register(b, span)
            trace_id = span.get_span_context().trace_id
            if is_root:
                _trace_names.put(trace_id, trace_name(b))
            t0 = time.perf_counter()
            result: Any = None
            has_result = False
            error: Optional[BaseException] = None
            try:
                result = await call_next(b)
                has_result = True
                return result
            except BaseException as exc:
                error = exc
                raise
            finally:
                duration = time.perf_counter() - t0
                for k in keys:
                    _live.pop(k, span)
                try:
                    _finish(span, b, result, has_result, error, capture_content,
                            max_content_length, enrichers,
                            _trace_names.get(trace_id), is_root)
                except Exception:  # telemetry must never break a call
                    pass
                if metrics is not None:
                    try:
                        metrics.record(b, duration, error)
                    except Exception:
                        pass

    otel.harness_name = "otel"  # type: ignore[attr-defined]
    return otel


def _finish(span: Span, b: Boundary, result: Any, has_result: bool,
            error: Optional[BaseException], capture_content: bool, limit: int,
            enrichers: Sequence[Enricher], root_name: Optional[str] = None,
            is_root: bool = False) -> None:
    # The owner fills attrs (model, usage, cost...) during the call: re-read them.
    attrs = {**_base_attributes(b), **_kind_attributes(b)}
    span.update_name(span_name(b))
    if isinstance(error, asyncio.CancelledError):
        attrs[A.CANCELLED] = True
    elif isinstance(error, Blocked):
        attrs[A.ERROR_TYPE] = "Blocked"
        attrs[A.BLOCKED_REASON] = error.reason
        if error.by:
            attrs[A.BLOCKED_BY] = error.by
        span.set_status(Status(StatusCode.ERROR, f"blocked: {error.reason}"))
    elif error is not None:
        attrs[A.ERROR_TYPE] = type(error).__qualname__
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, f"{type(error).__name__}: {error}"))
    content = _content_attributes(b, result, has_result, limit) if capture_content else {}
    attrs.update(content)
    for enrich in enrichers:
        try:
            extra = enrich(SpanInfo(boundary=b, attributes=dict(attrs), error=error,
                                    content=dict(content), trace_name=root_name,
                                    is_root=is_root))
        except Exception:
            extra = None
        if extra:
            attrs.update(extra)
    span.set_attributes(_clean(attrs))
