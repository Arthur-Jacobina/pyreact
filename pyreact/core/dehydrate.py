"""``use_dehydrate``: components own the serialization of their state.

The tree is the source of truth (ADR 0001). A component that has state worth
keeping across a suspend / a process restart declares it with a hook::

    @component
    def Counter():
        start = use_dehydrate("count", dehydrate=lambda: count, initial=0)
        count, set_count = hooks.use_state(start)
        ...

- The registration is addressed by ``(component path, key)`` and follows the
  component's lifecycle (it goes away when the component unmounts).
- When the tree is dehydrated (``root.suspend()`` / ``root.dehydrate()``),
  every registered ``dehydrate()`` is called; it returns a JSON value: the
  *slice* stored at ``snapshot["slices"][path][key]``.
- When a tree is hydrated from a snapshot (``hydrate_root``), the hook
  returns the hydrated value **during the component's first render**:
  ``hydrate(slice)`` if given (else the slice itself). Without a slice for it
  (or outside hydration) it returns ``initial``.
- ``hydrate`` may be async: the component then renders nothing (its effects
  do not run) until it resolves, and re-renders as a fresh mount with the
  hydrated value. A :func:`Hydrating` ancestor renders its ``fallback`` and
  holds its whole subtree (like a hidden ``Activity``) in the meantime.

The hook never fails a render: a ``hydrate`` that raises is reported as a
``hydration.mismatch`` (``kind: "error"``) and the hook starts from
``initial``.

Snapshot (``SNAPSHOT_FORMAT``), the dehydrated tree, JSON only::

    {"format": "pyreact.snapshot/1", "session_id": "coder-1",
     "created_at": 1793720877.92,
     "slices": {"App/Coder[coder]": {"history": [...], "program": {...}}},
     "turns": {"next_index": 3, "policy": "queue", "carry": [{"id": "coder-1/t2", ...}]},
     "versions": {"snapshot": 1, "pyreact": "0.1.0", "python": "3.13.0"}}

:func:`dehydrate_tree` builds one from a mounted tree; :func:`begin_hydration`
+ :func:`finish_hydration` hydrate a tree from its ``slices``. Hydration
mismatches (React's warning) are never fatal: a slice no hook claims by the
time hydration finishes (``unclaimed``), a hook with no slice (``missing``)
and a failing ``hydrate`` (``error``) are published as ``hydration.mismatch``
events ``{kind, path, key, message}`` on the session bus and warned
(:class:`HydrationWarning`).
"""

from __future__ import annotations

import inspect
import json
import platform
import time
import warnings
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, TypedDict, Union,
)

from .core import VNode, _context_stack, component

__all__ = [
    "SNAPSHOT_FORMAT",
    "SNAPSHOT_VERSION",
    "Hydrating",
    "HydrationState",
    "HydrationWarning",
    "Snapshot",
    "SnapshotError",
    "adehydrate_tree",
    "begin_hydration",
    "check_snapshot",
    "dehydrate_tree",
    "finish_hydration",
    "hydration_of",
    "use_dehydrate",
]

SNAPSHOT_VERSION = 1
SNAPSHOT_FORMAT = f"pyreact.snapshot/{SNAPSHOT_VERSION}"


class Snapshot(TypedDict):
    """The dehydrated tree (a plain JSON dict at runtime)."""

    format: str
    session_id: Optional[str]
    created_at: float
    slices: Dict[str, Dict[str, Any]]
    turns: Dict[str, Any]
    versions: Dict[str, Any]


class SnapshotError(ValueError):
    """A tree could not be dehydrated (a slice is not JSON, a dehydrate
    failed...), or a dict is not a snapshot this pyreact can hydrate."""


class HydrationWarning(RuntimeWarning):
    """A hydration mismatch (see ``hydration.mismatch`` events)."""


class DehydrateSlot:
    """Hook slot of one ``use_dehydrate`` call."""

    __slots__ = ("key", "dehydrate", "value", "raw", "hydrated", "pending", "task")

    def __init__(self, key: str, dehydrate: Callable[[], Any], initial: Any) -> None:
        self.key = key
        self.dehydrate = dehydrate
        self.value = initial
        self.raw: Any = None  # the slice it was hydrated from
        self.hydrated = False
        self.pending = False  # async hydrate not resolved yet
        self.task: Any = None

    def __repr__(self) -> str:
        state = "pending" if self.pending else ("hydrated" if self.hydrated else "fresh")
        return f"<use_dehydrate {self.key!r} {state}>"


class HydrationState:
    """Slices a tree hydrates from, and what its first render made of them.

    Set on a root before its first render (:func:`begin_hydration`); hooks
    *claim* their slice by ``(path, key)``. While :attr:`active`, a hook with
    no slice is recorded as ``missing``; when hydration finishes, slices
    nobody claimed are ``unclaimed`` (see :meth:`finish`).
    """

    def __init__(self, slices: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.slices: Dict[str, Dict[str, Any]] = {
            str(p): dict(v) for p, v in (slices or {}).items() if isinstance(v, dict)}
        self.claimed: Set[Tuple[str, str]] = set()
        self.active = True
        self.mismatches: List[Dict[str, Any]] = []
        self.pending: Set[Any] = set()  # tasks of async hydrates
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []

    def claim(self, path: str, key: str) -> Tuple[bool, Any]:
        part = self.slices.get(path)
        if part is None or key not in part or (path, key) in self.claimed:
            return False, None
        self.claimed.add((path, key))
        return True, part[key]

    def on_mismatch(self, fn: Callable[[Dict[str, Any]], None]) -> None:
        """Call ``fn(mismatch)`` for every mismatch from now on."""
        self._listeners.append(fn)

    def mismatch(self, kind: str, path: str, key: str, message: str, **info: Any) -> None:
        rec = {"kind": kind, "path": path, "key": key, "message": message, **info}
        self.mismatches.append(rec)
        warnings.warn(HydrationWarning(f"hydration mismatch: {message}"), stacklevel=3)
        for fn in list(self._listeners):
            try:
                fn(rec)
            except Exception:
                pass


    def unclaimed(self) -> List[Tuple[str, str]]:
        return [(p, k) for p, part in self.slices.items() for k in part
                if (p, k) not in self.claimed]


def begin_hydration(root: Any, slices: Optional[Dict[str, Dict[str, Any]]],
                    *, event_bus: Any = None) -> HydrationState:
    """Make ``root`` (a root HookContext, before its first render) hydrate
    from ``slices`` (``{path: {key: value}}``, or a whole snapshot).
    Mismatches are published as ``hydration.mismatch`` on ``event_bus``
    (default: the root's own services' ``event_bus``, if any)."""
    if isinstance(slices, dict) and "format" in slices and "slices" in slices:
        slices = slices["slices"]
    state = HydrationState(slices)
    if event_bus is None:
        services = getattr(root, "_session_services", None)
        event_bus = services.get("event_bus") if services else None
    if event_bus is not None:
        state.on_mismatch(lambda rec: event_bus.publish(
            "hydration.mismatch", rec, source=rec.get("path")))
    root._hydration = state
    return state


def finish_hydration(root: Any) -> List[Dict[str, Any]]:
    """End ``root``'s hydration (call it once the first render committed and
    the async hydrates resolved): every slice no hook claimed is reported as
    an ``unclaimed`` mismatch; hooks mounted from now on start fresh, with no
    ``missing`` report. Returns all the mismatches of this hydration."""
    state = hydration_of(root)
    if state is None:
        return []
    if state.active:
        state.active = False
        for path, key in state.unclaimed():
            state.mismatch("unclaimed", path, key,
                           f"slice {key!r} of {path} was not claimed by any "
                           "use_dehydrate (component gone, renamed or re-keyed?)")
    return list(state.mismatches)


def hydration_of(ctx: Any) -> Optional[HydrationState]:
    root = getattr(ctx, "_root", None)
    return getattr(root, "_hydration", None) if root is not None else None


# ---------------------------------------------------------------------------
# the hook
# ---------------------------------------------------------------------------
def use_dehydrate(
    key: str,
    *,
    dehydrate: Callable[[], Any],
    hydrate: Optional[Callable[[Any], Any]] = None,
    initial: Any = None,
) -> Any:
    """Register a dehydratable slice ``key`` of the calling component; returns
    the hydrated value during the first render (if the snapshot has a slice
    for it), else ``initial``. The value is stable across later renders
    (like ``use_state``'s initial value). See the module docs.

    - ``dehydrate()``: returns the slice (JSON; sync or async). The latest
      render's function is used, so it may close over render values.
    - ``hydrate(slice)``: turns a slice into the value (sync or async);
      default: the slice itself.
    """
    ctx = _context_stack.get()
    if ctx is None:
        raise RuntimeError("use_dehydrate() can only be used during render.")
    return _use_dehydrate(ctx, key, dehydrate, hydrate, initial)


def _use_dehydrate(ctx: Any, key: str, dehydrate, hydrate, initial) -> Any:
    if not isinstance(key, str) or not key:
        raise TypeError(f"use_dehydrate key must be a non-empty str, got {key!r}")
    if not callable(dehydrate):
        raise TypeError("use_dehydrate(dehydrate=...) must be callable")
    idx = ctx.hook_idx
    if idx < len(ctx.hooks):
        slot = ctx.hooks[idx]
        if not isinstance(slot, DehydrateSlot) or slot.key != key:
            raise RuntimeError(
                f"hook order changed in {ctx.path}: use_dehydrate({key!r}) is not "
                "called in the same order on every render")
        slot.dehydrate = dehydrate
        ctx.hook_idx += 1
        return slot.value

    regs = ctx.__dict__.setdefault("_dehydrators", {})
    if key in regs:
        raise ValueError(f"use_dehydrate key {key!r} is used twice in {ctx.path}")
    slot = ctx.__dict__.setdefault("_suspended_slots", {}).pop(key, None)
    if slot is None:
        slot = DehydrateSlot(key, dehydrate, initial)
        _hydrate_slot(ctx, slot, hydrate)
    slot.dehydrate = dehydrate
    regs[key] = slot
    ctx.hooks.append(slot)
    ctx.hook_idx += 1
    if slot.pending:
        ctx._render_suspended = True
    return slot.value


def _hydrate_slot(ctx: Any, slot: DehydrateSlot, hydrate) -> None:
    state = hydration_of(ctx)
    if state is None or not state.active:
        return
    path = ctx.path
    found, raw = state.claim(path, slot.key)
    if not found:
        state.mismatch("missing", path, slot.key,
                       f"no slice for use_dehydrate({slot.key!r}) of {path} "
                       "(it starts from its initial value)")
        return
    slot.raw, slot.hydrated = raw, True
    if hydrate is None:
        slot.value = raw
        return
    try:
        res = hydrate(raw)
    except Exception as exc:
        _hydrate_failed(state, path, slot, exc)
        return
    if not inspect.isawaitable(res):
        slot.value = res
        return
    from .runtime import spawn

    slot.pending = True

    async def _resolve() -> None:
        try:
            slot.value = await res
        except Exception as exc:
            _hydrate_failed(state, path, slot, exc)
        finally:
            slot.pending = False
            state.pending.discard(slot.task)
            _slot_resolved(ctx)

    slot.task = spawn(_resolve(), owner=ctx, name=f"{ctx.name}:hydrate[{slot.key}]", turn=None)
    state.pending.add(slot.task)


def _hydrate_failed(state: HydrationState, path: str, slot: DehydrateSlot, exc: BaseException) -> None:
    slot.hydrated = False
    state.mismatch("error", path, slot.key,
                   f"hydrate of use_dehydrate({slot.key!r}) of {path} failed "
                   f"({type(exc).__name__}: {exc}); it starts from its initial value",
                   error=f"{type(exc).__name__}: {exc}")


def _slot_resolved(ctx: Any) -> None:
    from .runtime import schedule_rerender

    if not getattr(ctx, "_mounted", False):
        return
    slots = ctx.__dict__.get("_suspended_slots") or {}
    if any(s.pending for s in slots.values()):
        return
    schedule_rerender(ctx, reason="hydrated")
    holder = _boundary_holder(ctx)
    if holder is not None:
        boundary = holder._hydration_boundary
        waiting = boundary.__dict__.setdefault("_hydration_waiting", set())
        waiting.discard(ctx)
        if not waiting:
            schedule_rerender(boundary, reason="hydrated")


def _boundary_holder(ctx: Any) -> Any:
    node = ctx.parent
    while node is not None:
        if getattr(node, "_hydration_boundary", None) is not None:
            return node
        node = node.parent
    return None


def discard_suspended_render(ctx: Any, old_children: Sequence[Any]) -> None:
    """Called by ``HookContext.render`` when a ``use_dehydrate`` of ``ctx`` is
    still hydrating: the render is thrown away (no children, no effects, no
    hook state but the pending slots), and the nearest :func:`Hydrating`
    boundary holds its subtree and shows its fallback."""
    from .runtime import schedule_rerender

    for orphan in old_children:
        orphan.unmount()
    ctx.children = []
    suspended = ctx.__dict__.setdefault("_suspended_slots", {})
    suspended.update(ctx.__dict__.get("_dehydrators") or {})
    ctx._dehydrators = {}
    ctx.hooks = []
    ctx.effects = []
    ctx._effect_slots.clear()
    ctx._effect_fns.clear()
    holder = _boundary_holder(ctx)
    if holder is not None:
        boundary = holder._hydration_boundary
        waiting = boundary.__dict__.setdefault("_hydration_waiting", set())
        holder._hides_children = True  # hold the subtree's effects right away
        if ctx not in waiting:
            waiting.add(ctx)
            schedule_rerender(boundary, reason="hydrating")


# ---------------------------------------------------------------------------
# dehydration
# ---------------------------------------------------------------------------
def _pyreact_version() -> str:
    try:
        from importlib.metadata import version

        return version("pyreact")
    except Exception:  # pragma: no cover - not installed as a dist
        return "unknown"


def snapshot_versions() -> Dict[str, Any]:
    return {"snapshot": SNAPSHOT_VERSION, "pyreact": _pyreact_version(),
            "python": platform.python_version()}


def _iter_slots(root: Any) -> Iterator[Tuple[str, DehydrateSlot]]:
    """(path, slot) of every live registration, tree order (root first)."""
    stack = [root]
    while stack:
        node = stack.pop()
        if not getattr(node, "_mounted", True):
            continue
        regs = dict(node.__dict__.get("_suspended_slots") or {})
        regs.update(node.__dict__.get("_dehydrators") or {})
        if regs:
            path = node.path
            for slot in regs.values():
                yield path, slot
        stack.extend(reversed(list(node.children)))


def _to_json(value: Any, path: str, key: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SnapshotError(f"use_dehydrate({key!r}) of {path} returned a value that is "
                            f"not JSON ({type(exc).__name__}: {exc})") from None


def _plan(root: Any) -> List[Tuple[str, DehydrateSlot]]:
    seen: Dict[Tuple[str, str], DehydrateSlot] = {}
    out = []
    for path, slot in _iter_slots(root):
        if (path, slot.key) in seen:
            raise SnapshotError(f"two components at {path} both dehydrate {slot.key!r}: "
                                "give them distinct keys")
        seen[(path, slot.key)] = slot
        out.append((path, slot))
    return out


def _call_dehydrate(path: str, slot: DehydrateSlot) -> Any:
    if slot.pending:  # never hydrated: keep the slice it would have used
        return slot.raw
    try:
        return slot.dehydrate()
    except Exception as exc:
        raise SnapshotError(f"use_dehydrate({slot.key!r}) of {path}: dehydrate() failed "
                            f"({type(exc).__name__}: {exc})") from exc


def _turns_of(root: Any, carry: Sequence[Any]) -> Dict[str, Any]:
    services = getattr(root, "_session_services", None) or {}
    tm = services.get("turns")
    fn = getattr(tm, "dehydrate", None)
    return fn(carry=carry) if callable(fn) else {}


def _session_id_of(root: Any) -> Optional[str]:
    services = getattr(root, "_session_services", None) or {}
    return getattr(services.get("session"), "id", None)


def _envelope(root: Any, slices, session_id, turns, carry) -> Snapshot:
    return {
        "format": SNAPSHOT_FORMAT,
        "session_id": session_id if session_id is not None else _session_id_of(root),
        "created_at": time.time(),
        "slices": slices,
        "turns": _to_json(turns if turns is not None else _turns_of(root, carry),
                          "<root>", "turns"),
        "versions": snapshot_versions(),
    }


def _add(slices: Dict[str, Dict[str, Any]], path: str, key: str, value: Any) -> None:
    slices.setdefault(path, {})[key] = _to_json(value, path, key)


def dehydrate_tree(root: Any, *, session_id: Optional[str] = None,
                   turns: Optional[Dict[str, Any]] = None,
                   carry: Sequence[Any] = ()) -> Snapshot:
    """Snapshot of the mounted tree under ``root``: calls every registered
    ``dehydrate()`` (see the module docs). Does not unmount anything.

    ``turns`` defaults to the session's ``TurnManager.dehydrate(carry=...)``
    (``carry``: turns cut short, re-run on hydrate); ``session_id`` to the
    session's. Raises :class:`SnapshotError` (naming the path and key) if a
    slice is not JSON or a ``dehydrate`` fails; an async ``dehydrate`` needs
    :func:`adehydrate_tree`. Must run on the render loop (or with no loop
    running the tree).
    """
    slices: Dict[str, Dict[str, Any]] = {}
    for path, slot in _plan(root):
        value = _call_dehydrate(path, slot)
        if inspect.isawaitable(value):
            close = getattr(value, "close", None)
            if callable(close):
                close()
            raise SnapshotError(f"use_dehydrate({slot.key!r}) of {path} has an async "
                                "dehydrate: use 'await adehydrate_tree(root)'")
        _add(slices, path, slot.key, value)
    return _envelope(root, slices, session_id, turns, carry)


async def adehydrate_tree(root: Any, *, session_id: Optional[str] = None,
                          turns: Optional[Dict[str, Any]] = None,
                          carry: Sequence[Any] = ()) -> Snapshot:
    """:func:`dehydrate_tree` that also awaits async ``dehydrate`` functions."""
    slices: Dict[str, Dict[str, Any]] = {}
    for path, slot in _plan(root):
        value = _call_dehydrate(path, slot)
        if inspect.isawaitable(value):
            try:
                value = await value
            except Exception as exc:
                raise SnapshotError(f"use_dehydrate({slot.key!r}) of {path}: dehydrate() "
                                    f"failed ({type(exc).__name__}: {exc})") from exc
        _add(slices, path, slot.key, value)
    return _envelope(root, slices, session_id, turns, carry)


def check_snapshot(snapshot: Any) -> Snapshot:
    """Validate a snapshot's envelope (format, version, ``slices`` shape);
    returns it. Raises :class:`SnapshotError`."""
    if not isinstance(snapshot, dict):
        raise SnapshotError(f"a snapshot is a dict, got {type(snapshot).__name__}")
    fmt = snapshot.get("format")
    if not isinstance(fmt, str) or not fmt.startswith("pyreact.snapshot/"):
        raise SnapshotError(f"not a pyreact snapshot (format={fmt!r})")
    try:
        ver = int(fmt.rsplit("/", 1)[1])
    except ValueError:
        raise SnapshotError(f"bad snapshot format {fmt!r}") from None
    if ver > SNAPSHOT_VERSION:
        raise SnapshotError(f"snapshot format {fmt!r} is newer than this pyreact "
                            f"({SNAPSHOT_FORMAT})")
    slices = snapshot.get("slices", {})
    if not isinstance(slices, dict):
        raise SnapshotError("snapshot['slices'] must be a dict {path: {key: value}}")
    for path, part in slices.items():
        if not isinstance(part, dict):
            raise SnapshotError(f"snapshot['slices'][{path!r}] must be a dict {{key: value}}")
    if not isinstance(snapshot.get("turns", {}), dict):
        raise SnapshotError("snapshot['turns'] must be a dict")
    return snapshot


def drop_registrations(ctx: Any) -> None:
    """Unmount: forget ``ctx``'s slices and cancel its pending hydrates."""
    for slot in (ctx.__dict__.get("_suspended_slots") or {}).values():
        task = slot.task
        if task is not None and not task.done():
            task.cancel()
    ctx.__dict__.pop("_suspended_slots", None)
    ctx.__dict__.pop("_dehydrators", None)


# ---------------------------------------------------------------------------
# Hydrating boundary
# ---------------------------------------------------------------------------
Fallback = Union[None, VNode, Sequence[VNode], Callable[[], Any]]


def _fallback_nodes(fallback: Fallback, tag: str) -> List[VNode]:
    out = fallback() if callable(fallback) and not isinstance(fallback, VNode) else fallback
    items = out if isinstance(out, (list, tuple)) else [out]
    nodes = [n for n in items if isinstance(n, VNode)]
    for i, n in enumerate(nodes):
        if n.key is None:
            n.key = f"{tag}.fallback{i}"
    return nodes


@component(transparent=True)
def _HydrationContent(boundary=None, children=None):
    ctx = _context_stack.get()
    ctx._hydration_boundary = boundary
    waiting = {c for c in boundary.__dict__.get("_hydration_waiting", ()) if c._mounted}
    boundary._hydration_waiting = waiting
    ctx._hides_children = bool(waiting)
    ctx._hide_reason = "hidden"
    return list(children or [])


@component(transparent=True)
def Hydrating(fallback: Fallback = None, children=None):
    """While a ``use_dehydrate`` below is hydrating asynchronously, render
    ``fallback`` (a VNode, a list, or a function returning them) and hold
    ``children`` (mounted, but their effects do not run, like a hidden
    ``Activity``); then show them. Transparent in paths (like providers):
    ``App/Chat[chat]`` with or without a ``Hydrating`` around it.
    """
    ctx = _context_stack.get()
    waiting = {c for c in ctx.__dict__.get("_hydration_waiting", ()) if c._mounted}
    ctx._hydration_waiting = waiting
    tag = str(ctx.key) if ctx.key is not None else f"#{ctx.index}"
    out = [_HydrationContent(key=f"{tag}.content", boundary=ctx, children=children)]
    if waiting and fallback is not None:
        out.extend(_fallback_nodes(fallback, tag))
    return out
