"""Debug helpers for inspecting the Hook tree.

This module intentionally avoids importing from ``pyreact.core.hook`` to
prevent circular imports. Functions operate on any object that exposes the
expected attributes: ``name``, optional ``key``, optional ``props`` and
``children`` (iterable of similar nodes).
"""

import time
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

# ANSI constants (single source for this module)
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
FG_GRAY = "\x1b[90m"
FG_YELLOW = "\x1b[33m"
FG_MAGENTA = "\x1b[35m"
FG_CYAN = "\x1b[36m"
FG_BLUE = "\x1b[34m"
FG_GREEN = "\x1b[32m"

# Aliases used by some helpers below
GRAY = FG_GRAY
YELLOW = FG_YELLOW
CYAN = FG_CYAN


def render_tree(ctx, indent: int = 0) -> None:
    """Pretty-print the tree starting at ``ctx`` to stdout.

    Expects ``ctx`` to have ``name``, optional ``key`` and ``props``
    attributes, and ``children`` iterable.
    """

    pad = "  " * indent

    if indent == 0:
        print(f"{BOLD}{CYAN}=== VNode Tree ==={RESET}")

    def _fmt_val(v, depth: int = 0):
        # ANSI color helpers available via closure below
        if depth > 1:
            return f"{DIM}…{RESET}"
        if isinstance(v, (int, float)):
            return f"{FG_BLUE}{repr(v)}{RESET}"
        if isinstance(v, str):
            s = v.replace("\n", "\\n")
            text = s if len(s) <= 60 else s[:57] + "…"
            return f"{FG_YELLOW}{repr(text)}{RESET}"
        if v is None or isinstance(v, bool):
            return f"{FG_CYAN}{repr(v)}{RESET}"
        if isinstance(v, (list, tuple)):
            return f"{FG_CYAN}[{len(v)}]{RESET}"
        if isinstance(v, dict):
            items = []
            for i, (k, val) in enumerate(v.items()):
                if i >= 5:
                    items.append(f"{DIM}…{RESET}")
                    break
                if k == "children":
                    # Children can be very large; show only count
                    try:
                        clen = len(val)  # type: ignore[arg-type]
                    except Exception:
                        clen = "?"
                    items.append(f"{FG_CYAN}children{RESET}=[{FG_YELLOW}{clen}{RESET}]")
                else:
                    items.append(f"{FG_CYAN}{k}{RESET}={_fmt_val(val, depth + 1)}")
            body = ", ".join(items)
            return "{" + body + "}"
        if callable(v):
            name = getattr(v, "__name__", None) or getattr(
                type(v), "__name__", "callable"
            )
            return f"{FG_GREEN}<fn {name}>{RESET}"
        return f"{FG_GREEN}<{type(v).__name__}>{RESET}"  # ANSI helpers available from module scope

    name = getattr(ctx, "name", type(ctx).__name__)
    name_col = f"{FG_MAGENTA}{name}{RESET}"
    if getattr(ctx, "key", None) is not None:
        key_part = f" {FG_GRAY}key={RESET}{FG_YELLOW}{getattr(ctx, 'key')!r}{RESET}"
    else:
        key_part = ""
    props_val = _fmt_val(getattr(ctx, "props", {}))
    props_part = f" {FG_GRAY}props={RESET}{props_val}"
    print(f"{pad}{FG_GRAY}-{RESET} {name_col}{key_part}{props_part}")

    for ch in getattr(ctx, "children", []) or []:
        render_tree(ch, indent + 1)

    if indent == 0:
        print(f"{BOLD}{CYAN}==================={RESET}\n")


# ----------------------------------------------------------------------------
# Render trace instrumentation
# ----------------------------------------------------------------------------

_TRACE_CTX: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "_TRACE_CTX", default=None
)
_TRACE_DEPTH: ContextVar[int] = ContextVar("_TRACE_DEPTH", default=0)
_TRACE_ROOT: ContextVar[Any] = ContextVar("_TRACE_ROOT", default=None)
_TRACE_ENABLED: bool = False
# When true (and tracing is enabled), each finished render trace is also
# published as a ``trace.render`` event on the root's ``event_bus`` service.
_TRACE_EVENTS: bool = False

# Keep a log of recent traces (each trace is a dict with events)
_TRACE_LOG: List[Dict[str, Any]] = []
_TRACE_LOG_LIMIT = 50


def _push_trace_event(event: Dict[str, Any]) -> None:
    trace = _TRACE_CTX.get()
    if trace is None:
        return
    trace["events"].append(event)


def record_schedule(ctx: Any, reason: Optional[str] = None) -> None:
    if not _TRACE_ENABLED:
        return
    reasons: List[str] = getattr(ctx, "_debug_reasons", [])
    if reason:
        reasons.append(reason)
    setattr(ctx, "_debug_reasons", reasons)


def start_trace(root_ctx: Any, reasons: Optional[List[str]] = None) -> None:
    if not _TRACE_ENABLED:
        return
    trace = {
        "id": f"tr-{int(time.time() * 1000)}-{id(root_ctx)}",
        "root_id": id(root_ctx),
        "root_name": getattr(root_ctx, "name", type(root_ctx).__name__),
        "reasons": list(reasons or []),
        "ts": time.time(),
        "events": [],
    }
    _TRACE_LOG.append(trace)
    if len(_TRACE_LOG) > _TRACE_LOG_LIMIT:
        del _TRACE_LOG[:-_TRACE_LOG_LIMIT]
    _TRACE_CTX.set(trace)
    _TRACE_ROOT.set(root_ctx)
    _TRACE_DEPTH.set(0)


def end_trace() -> None:
    trace = _TRACE_CTX.get()
    root = _TRACE_ROOT.get()
    _TRACE_CTX.set(None)
    _TRACE_ROOT.set(None)
    _TRACE_DEPTH.set(0)
    if trace is not None:
        trace["duration"] = time.time() - trace["ts"]
        if _TRACE_EVENTS:
            _publish_trace(root, trace)


def trace_payload(trace: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-friendly view of a render trace (payload of ``trace.render``)."""
    return {
        "trace_id": trace["id"],
        "root": trace["root_name"],
        "reasons": [str(r) for r in trace["reasons"]],
        "duration": trace.get("duration"),
        "rendered": [
            {
                "name": e.get("name"),
                "key": e.get("key"),
                "depth": e.get("depth"),
                "kind": e.get("kind"),
            }
            for e in trace["events"]
        ],
    }


def _publish_trace(root: Any, trace: Dict[str, Any]) -> None:
    get_service = getattr(root, "get_service", None)
    if get_service is None:
        return
    try:
        from .events import EventBus

        bus = get_service("event_bus", EventBus)
        bus.publish("trace.render", trace_payload(trace), source="debug")
    except Exception:
        pass


def enter_render(ctx: Any) -> Any:
    if not _TRACE_ENABLED:
        return None
    depth = _TRACE_DEPTH.get()
    kind = "origin" if depth == 0 else "propagate"
    _push_trace_event(
        {
            "t": time.time(),
            "kind": kind,
            "depth": depth,
            "ctx_id": id(ctx),
            "name": getattr(ctx, "name", type(ctx).__name__),
            "key": getattr(ctx, "key", None),
        }
    )
    return _TRACE_DEPTH.set(depth + 1)


def exit_render(token: Any) -> None:
    try:
        if token is not None:
            _TRACE_DEPTH.reset(token)
    except Exception:
        pass


def print_last_trace() -> None:
    if not _TRACE_LOG:
        print("\x1b[90m[debug]\x1b[0m no render trace available yet.")
        return
    trace = _TRACE_LOG[-1]
    print(f"\n{BOLD}{CYAN}=== Render Trace ==={RESET}")
    print(f"{GRAY}root:{RESET} {YELLOW}{trace['root_name']}{RESET}")
    if trace["reasons"]:
        print(f"{GRAY}reasons:{RESET}")
        for i, r in enumerate(trace["reasons"]):
            try:
                text = str(r)
            except Exception:
                text = "<reason>"
            if "->" in text:
                left, right = text.split("->", 1)
                colored = f"{YELLOW}{left.strip()}{RESET} {FG_GREEN}->{RESET} {YELLOW}{right.strip()}{RESET}"
            else:
                colored = f"{YELLOW}{text}{RESET}"
            print(f"  - {CYAN}{i}{RESET}: {colored}")
    for ev in trace["events"]:
        pad = "  " * int(ev.get("depth", 0))
        kind = ev.get("kind", "?")
        name = ev.get("name", "?")
        key = ev.get("key")
        kind_col = (
            f"{FG_GREEN}{kind}{RESET}" if kind == "origin" else f"{CYAN}{kind}{RESET}"
        )
        name_col = f"{FG_MAGENTA}{name}{RESET}"
        key_part = f" key={YELLOW}{key!r}{RESET}" if key is not None else ""
        print(f"{pad}- {kind_col}: {name_col}{key_part}")
    print(f"{BOLD}{CYAN}===================={RESET}\n")


def enable_tracing(events: Optional[bool] = None) -> None:
    """Enable render tracing; ``events=True`` also publishes ``trace.render``."""
    global _TRACE_ENABLED, _TRACE_EVENTS
    _TRACE_ENABLED = True
    if events is not None:
        _TRACE_EVENTS = bool(events)


def set_trace_events(enabled: bool) -> None:
    """Toggle publishing of ``trace.render`` events (needs tracing enabled)."""
    global _TRACE_EVENTS
    _TRACE_EVENTS = bool(enabled)


def trace_events_enabled() -> bool:
    return _TRACE_EVENTS


def disable_tracing() -> None:
    global _TRACE_ENABLED
    _TRACE_ENABLED = False


def is_tracing_enabled() -> bool:
    return _TRACE_ENABLED


def clear_traces() -> None:
    del _TRACE_LOG[:]
