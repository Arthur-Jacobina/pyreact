"""Learned programs are props (ADR 0001): ``LearnedProgram`` + ``use_learned``.

    LearnedProgram(key="coder", src="artifacts/coder.json", children=[
        Coder(key="coder"),
    ])

    @component
    def Coder():
        agent = use_agent(CoderSig, tools=[...], program=use_learned("coder"))

- ``key`` names the learned program (and is the provider's reconciliation
  key); ``src`` is where it comes from: a result file (loaded with
  :func:`integrations.learn.load`), an :class:`OptimizationResult`, a
  ``dspy.Module`` or ``None`` (no learned program: components use their own,
  baseline module).
- :func:`use_learned` reads the nearest provider that declares the name
  (``None`` when none does). ``use_module`` / ``use_agent`` / ``use_call``
  apply the program's learned state (instructions, demos) onto the
  component's live module when given ``program=``; tools stay bound.
- **Hot swap = re-render.** When ``src`` changes the provider re-renders its
  subtree and its users apply the new program; back to ``None`` restores the
  baseline. No module cache is mutated from outside.
- ``watch=True`` polls file sources every ``interval`` seconds (off by
  default) and re-renders when a file changes on disk.

Many at once, from the root: ``LearnedPrograms(programs={name: src}, ...)``.
The provider *holds* its programs as state (``{name: path}``): re-rendering
it with another ``programs`` / ``src`` prop resets that state (declarative
swap from the root); :meth:`OptimizationResult.hot_swap` and ``:optimize``
update the state of the provider that owns the name (or that its users read
from) through :class:`LearnedRegistry`, and fall back to the legacy env
mutation only for apps that have no provider. In the tree,
:func:`use_learned_registry` returns the nearest provider's handle.

Providers are transparent (``@component(transparent=True)``): wrapping a
subtree in one does not change its components' paths.

Suspend / hydrate (ADR 0001): the provider's state is dehydrated with
``use_dehydrate("programs")`` (:func:`dehydrate_programs`: ``{name: path |
{"state": ...} | None}``), so ``root.suspend()`` carries the learned program
-- the one swapped in last, not only the declared prop -- and
``hydrate_root`` restores it during the provider's first render.
"""

from __future__ import annotations

import asyncio
import copy
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

from pyreact.core import report_hydration_mismatch, use_dehydrate
from pyreact.core.core import _context_stack, component, hooks
from pyreact.core.provider import create_context
from pyreact.core.runtime import schedule_rerender

__all__ = [
    "LearnedProgram",
    "LearnedPrograms",
    "LearnedRegistry",
    "use_learned",
    "use_learned_registry",
    "find_registries",
    "swap_learned",
    "LearnedContext",
    "SavedProgram",
    "LEARNED_SERVICE",
    "dehydrate_programs",
    "hydrate_programs",
]

# Session service: {provider path: LearnedRegistry}. Internal plumbing for
# OptimizationResult.hot_swap / :optimize (outside code is handed a root in
# a later step of ADR 0001: root.ref(...)); components use use_learned.
LEARNED_SERVICE = "learn.programs"


class SavedProgram:
    """The learned state of a result file whose program cannot be rebuilt on
    its own (e.g. a ReAct agent: its tools are bound by the component).
    ``program=`` only needs ``dump_state()``, so this is enough to apply it."""

    def __init__(self, state: Dict[str, Any], *, path: Optional[str] = None,
                 optimizer: Optional[str] = None) -> None:
        self.state = state
        self._pyreact_result_path = path
        self._pyreact_optimizer = optimizer

    def dump_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self.state)

    def __repr__(self) -> str:
        return f"<SavedProgram {self._pyreact_result_path or ''}>"


@dataclass(frozen=True)
class LearnedScope:
    """The context value: every learned program visible here, by name."""

    programs: Mapping[str, Any] = field(default_factory=dict)
    owners: Mapping[str, "LearnedRegistry"] = field(default_factory=dict)
    nearest: Optional["LearnedRegistry"] = None

    def get(self, name: str) -> Any:
        return self.programs.get(name)


class LearnedRegistry:
    """Handle on one provider's state (``{name: src}``).

    ``set(name, src)`` / ``update({...})`` / ``remove(name)`` change it and
    re-render the provider's subtree (thread-safe: from another thread the
    change runs on the app loop and the call waits for it to be applied).
    ``programs`` is the current state; ``consumers`` the component paths
    that read each name with :func:`use_learned` (the last render).
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.consumers: Dict[str, Set[str]] = {}
        # the same users as telemetry.component_path (the scope of their module keys)
        self._scopes: Dict[str, Set[str]] = {}
        self._ctx: Any = None
        self._state: Callable[[], Dict[str, Any]] = dict
        self._set: Callable[[Any], None] = lambda v: None
        self._cache: Dict[str, Any] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def programs(self) -> Dict[str, Any]:
        return dict(self._state())

    def names(self) -> Set[str]:
        return set(self._state()) | {n for n, c in self.consumers.items() if c}

    def _store(self, src: Any) -> Any:
        """What the state keeps for ``src``: a result's file path when it has
        one (JSON-able, and the program is primed in the cache so it is not
        re-read), the path again for a path (re-read: the file may be new)."""
        from .result import OptimizationResult

        if isinstance(src, OptimizationResult) and src.path:
            apath = os.path.abspath(src.path)
            self._cache[apath] = (_mtime(src.path), _of_result(src), None)
            return src.path
        if _is_path(src):
            self._cache.pop(os.path.abspath(os.fspath(src)), None)
            return os.fspath(src)
        return src

    def _apply(self, change: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        def run():
            new = change(dict(self._state()))
            self._set(lambda _old: new)
            ctx = self._ctx
            if ctx is not None and getattr(ctx, "_mounted", False):
                schedule_rerender(ctx, reason="learned program swapped")  # also when equal

        loop = self._loop
        if (loop is None or loop.is_closed() or not loop.is_running()
                or threading.current_thread() is self._thread):
            run()
            return
        done = threading.Event()
        err: List[BaseException] = []

        def on_loop():
            try:
                run()
            except BaseException as exc:  # pragma: no cover - surfaced to the caller
                err.append(exc)
            finally:
                loop.call_soon(done.set)  # after the rerender is enqueued

        loop.call_soon_threadsafe(on_loop)
        if not done.wait(10):
            raise TimeoutError(f"learned program swap on {self.path} not applied in 10s")
        if err:
            raise err[0]

    def set(self, name: str, src: Any) -> None:
        stored = self._store(src)
        self._apply(lambda cur: {**cur, str(name): stored})

    def update(self, programs: Mapping[str, Any]) -> None:
        stored = {str(k): self._store(v) for k, v in programs.items()}
        self._apply(lambda cur: {**cur, **stored})

    def dehydrate(self) -> Dict[str, Any]:
        """This provider's state as a JSON slice (:func:`dehydrate_programs`)."""
        return dehydrate_programs(self._state())

    def remove(self, name: str) -> None:
        self._apply(lambda cur: {k: v for k, v in cur.items() if k != str(name)})

    def __repr__(self) -> str:
        return f"<LearnedRegistry {self.path} {sorted(self._state())}>"


LearnedContext = create_context(default=None, name="Learned")


# ---------------------------------------------------------------------------
# Sources -> programs
# ---------------------------------------------------------------------------
def _tag(program: Any, path: Optional[str], optimizer: Optional[str]) -> Any:
    # a dehydrated slice can then point at the file instead of inlining it
    for attr, val in (("_pyreact_result_path", path), ("_pyreact_optimizer", optimizer)):
        if val:
            try:
                setattr(program, attr, val)
            except Exception:
                pass
    return program


def _of_result(res: Any) -> Any:
    prog = res.program
    if prog is None:
        return SavedProgram(dict(res.state), path=res.path, optimizer=res.optimizer)
    return _tag(prog, res.path, res.optimizer)


def _mtime(path: str) -> Optional[float]:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _is_path(src: Any) -> bool:
    return isinstance(src, (str, os.PathLike))


def _resolve(name: str, src: Any, cache: Dict[str, Any], watch: bool) -> Any:
    """``src`` -> program (``None`` when there is none or it can't be loaded;
    a load error is published once as an ``error`` event, phase ``learned``)."""
    from .result import OptimizationResult, load

    if src is None:
        return None
    if isinstance(src, OptimizationResult):
        prog = _of_result(src)
        if src.path:
            cache[os.path.abspath(src.path)] = (_mtime(src.path), prog, None)
        return prog
    if _is_path(src):
        path = os.fspath(src)
        apath = os.path.abspath(path)
        hit = cache.get(apath)
        mtime = _mtime(path) if (watch or hit is None) else None
        if hit is not None and (not watch or hit[0] == mtime):
            return hit[1]
        try:
            prog, err = _of_result(load(path)), None
        except Exception as exc:
            prog, err = None, exc
            from integrations import telemetry as tm

            tm.publish("error", {"phase": "learned", "name": name, "src": path,
                                 "error": repr(exc)})
        cache[apath] = (_mtime(path), prog, err)
        return prog
    if hasattr(src, "dump_state"):  # a dspy.Module (or anything with a learned state)
        return src
    raise TypeError(f"LearnedProgram src must be a path, an OptimizationResult, a program "
                    f"or None, got {type(src).__name__}")


def _src_sig(src: Any) -> Any:
    """Equality for prop changes: paths by value, objects by identity."""
    return ("path", os.fspath(src)) if _is_path(src) else ("obj", id(src))


def _declared_sig(declared: Mapping[str, Any]) -> Any:
    return tuple(sorted((str(k), _src_sig(v)) for k, v in declared.items()))


# ---------------------------------------------------------------------------
# Dehydration (the provider's use_dehydrate slice, ADR 0001)
# ---------------------------------------------------------------------------
def _plain(obj: Any) -> Any:
    import json

    return json.loads(json.dumps(obj, default=str))


def dehydrate_programs(programs: Mapping[str, Any]) -> Dict[str, Any]:
    """A provider's state -> its JSON slice: ``{name: path | {"state": ...} | None}``.

    Paths stay paths (the learned program is a *reference* to its artifact);
    a program/result with a file becomes that file's path; one without is
    inlined as its ``dump_state()``."""
    from .result import OptimizationResult

    out: Dict[str, Any] = {}
    for name, src in programs.items():
        if src is None or _is_path(src):
            out[str(name)] = None if src is None else os.fspath(src)
            continue
        path = (src.path if isinstance(src, OptimizationResult)
                else getattr(src, "_pyreact_result_path", None))
        if path and os.path.exists(path):
            out[str(name)] = str(path)
        elif isinstance(src, OptimizationResult):
            out[str(name)] = {"state": _plain(src.state), "optimizer": src.optimizer}
        else:
            out[str(name)] = {"state": _plain(src.dump_state()),
                              "optimizer": getattr(src, "_pyreact_optimizer", None)}
    return out


def hydrate_programs(slice_: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Inverse of :func:`dehydrate_programs`: paths stay paths (loaded on the
    first render), inline states become :class:`SavedProgram`."""
    out: Dict[str, Any] = {}
    for name, v in (slice_ or {}).items():
        if isinstance(v, dict) and "state" in v:
            out[str(name)] = SavedProgram(v["state"], optimizer=v.get("optimizer"))
        else:
            out[str(name)] = v
    return out


def _hydrate_checked(slice_: Any) -> Dict[str, Any]:
    """:func:`hydrate_programs` for the provider: a learned program whose
    result file is gone is a ``hydration.mismatch``; that name then uses the
    provider's declared ``src`` (dropped from the hydrated state)."""
    if not isinstance(slice_, Mapping):
        raise TypeError(f"a learned-programs slice is a dict {{name: src}}, "
                        f"got {type(slice_).__name__}")
    out = hydrate_programs(slice_)
    for name, src in list(out.items()):
        if _is_path(src) and not os.path.exists(src):
            del out[name]
            report_hydration_mismatch("programs", f"learned program {name!r}: result file "
                                      f"{os.fspath(src)!r} is gone (using the declared src)",
                                      name=name, src=os.fspath(src), reason="result missing")
    return out


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------
def _use_watch(ctx: Any, paths: tuple, watch: bool, interval: float) -> None:
    """Poll ``paths`` (loop timers, not a task: never holds ``wait_idle``)."""

    def start():
        if not watch or not paths:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        seen = {p: _mtime(p) for p in paths}
        handle: Dict[str, Any] = {}

        def tick():
            changed = False
            for p in paths:
                m = _mtime(p)
                if m != seen[p]:
                    seen[p], changed = m, True
            if changed and getattr(ctx, "_mounted", False):
                schedule_rerender(ctx, reason="learned program changed on disk")
            handle["h"] = loop.call_later(interval, tick)

        handle["h"] = loop.call_later(interval, tick)
        return lambda: handle["h"].cancel()

    hooks.use_effect(start, [watch, interval, paths])


def _use_learned_provider(declared: Mapping[str, Any], *, watch: bool, interval: float,
                          children: Any) -> list:
    ctx = _context_stack.get()
    # The provider's state: {name: src}, src a path whenever there is one, so
    # it is a single JSON-able value ({name: path}) held by ONE hook. It is
    # dehydrated with use_dehydrate (ADR 0001): a suspend carries the learned
    # program reference (a swap included) and hydrate_root gives it back
    # during the first render, before the children apply it.
    hydrated = use_dehydrate("programs", dehydrate=lambda: dehydrate_programs(reg._state()),
                             hydrate=_hydrate_checked)
    state_idx = ctx.hook_idx
    programs, set_programs = hooks.use_state(
        {**declared, **hydrated} if hydrated is not None else dict(declared))
    box = hooks.use_ref(None)
    if box.current is None:
        box.current = {"declared": _declared_sig(declared), "registry": LearnedRegistry(ctx.path)}
    st = box.current
    reg: LearnedRegistry = st["registry"]
    sig = _declared_sig(declared)
    if sig != st["declared"]:  # the prop changed: it wins (a re-render = the swap)
        st["declared"] = sig
        programs = dict(declared)
        set_programs(lambda _old: programs)
    outer = hooks.use_context(LearnedContext)

    # the handle outside code (hot_swap, :optimize) and use_learned_registry use
    reg._ctx, reg._set = ctx, set_programs
    reg._state = lambda: ctx.hooks[state_idx]  # the current value of the state above
    try:
        reg._loop, reg._thread = asyncio.get_running_loop(), threading.current_thread()
    except RuntimeError:
        reg._loop = reg._thread = None
    reg.consumers, reg._scopes = {}, {}  # refilled by use_learned while the subtree renders
    services = hooks.get_service(LEARNED_SERVICE, dict)
    services[ctx.path] = reg

    def _mount():
        def _unmount():
            if services.get(ctx.path) is reg:
                del services[ctx.path]

        return _unmount

    hooks.use_effect(_mount, [])

    resolved = dict(outer.programs) if outer is not None else {}
    owners = dict(outer.owners) if outer is not None else {}
    for name, src in programs.items():
        resolved[name] = _resolve(name, src, reg._cache, watch)
        owners[name] = reg
    paths = tuple(sorted(os.path.abspath(os.fspath(s)) for s in programs.values()
                         if _is_path(s)))
    _use_watch(ctx, paths, watch, interval)

    hooks.provide(LearnedContext, LearnedScope(programs=resolved, owners=owners, nearest=reg))
    return list(children or [])


@component(transparent=True)
def LearnedProgram(*, src: Any = None, watch: bool = False, interval: float = 1.0,
                   children: Any = None):
    """``LearnedProgram(key=name, src=path | OptimizationResult | program | None,
    watch=False, interval=1.0, children=[...])`` -- see the module docs."""
    ctx = _context_stack.get()
    name = getattr(ctx, "key", None)
    if name is None:
        raise TypeError("LearnedProgram needs key=<name> (the name use_learned(name) reads)")
    return _use_learned_provider({str(name): src}, watch=watch, interval=interval,
                                 children=children)


@component(transparent=True)
def LearnedPrograms(*, programs: Optional[Mapping[str, Any]] = None, watch: bool = False,
                    interval: float = 1.0, children: Any = None):
    """``LearnedPrograms(programs={name: src, ...}, watch=False, children=[...])``:
    many learned programs at once, typically at the root. Re-render it with a
    new ``programs`` prop to swap (each ``src`` as for :func:`LearnedProgram`)."""
    return _use_learned_provider(dict(programs or {}), watch=watch, interval=interval,
                                 children=children)


def _names_for(name_or_key: Any) -> tuple:
    if isinstance(name_or_key, tuple):
        from integrations.feedback import key_str

        return (key_str(name_or_key), str(name_or_key[-1]))
    return (str(name_or_key),)


def use_learned(name_or_key: Any) -> Any:
    """The learned program the nearest provider declares for ``name_or_key``
    (a name, or a module key tuple: its ``key_str`` or its scope), or ``None``.

    Pass it on: ``use_module(sig, program=use_learned("router"))``. The
    component re-renders with the new program whenever the provider's source
    changes (that is the hot swap)."""
    scope = hooks.use_context(LearnedContext)
    if scope is None:
        return None
    names = _names_for(name_or_key)
    name = next((n for n in names if n in scope.owners), names[-1])
    owner = scope.owners.get(name, scope.nearest)
    if owner is not None:  # so a swap from outside finds who reads the name
        from integrations import telemetry as tm

        owner.consumers.setdefault(name, set()).add(hooks.path)
        owner._scopes.setdefault(name, set()).add(tm.component_path(_context_stack.get()))
    for n in names:
        if n in scope.programs:
            return scope.programs[n]
    return None


def use_learned_registry() -> Optional[LearnedRegistry]:
    """The nearest ``LearnedProgram`` / ``LearnedPrograms`` provider's handle
    (``None`` without one): ``reg.set("router", "artifacts/new.json")`` from
    an effect re-renders that provider's subtree with the new program."""
    scope = hooks.use_context(LearnedContext)
    return scope.nearest if scope is not None else None


# ---------------------------------------------------------------------------
# Swapping from outside the tree (thin: it updates a provider's state)
# ---------------------------------------------------------------------------
def find_registries(target: Any = None) -> List[Tuple[str, LearnedRegistry]]:
    """``[(provider path, LearnedRegistry)]`` mounted in a Root (through its
    session), an AppRunner or a Session (``None``: headless trees)."""
    from .hmr import _services_of

    out: List[Tuple[str, LearnedRegistry]] = []
    for services in _services_of(target):
        for path, reg in list((services.get(LEARNED_SERVICE) or {}).items()):
            if all(reg is not r for _p, r in out):
                out.append((path, reg))
    return out


def _wanted(key: Any) -> Tuple[Set[str], Optional[str]]:
    from integrations.feedback import key_str

    if isinstance(key, tuple):
        return {key_str(key), str(key[-1])}, str(key[-1])
    return {str(key)}, None


def _match(reg: LearnedRegistry, key: Any) -> Optional[Tuple[int, str]]:
    from integrations.feedback import key_str

    from .hmr import _scope_match

    names, scope = _wanted(key)
    best = None
    for name in reg.names():
        users = reg.consumers.get(name) or set()
        hit = (name in names or _scope_match(key_str(key), name)
               or (scope is not None and scope in (reg._scopes.get(name) or ())))
        if hit:
            rank = (2 if users else 0) + (1 if name in reg._state() else 0)
            if best is None or rank > best[0]:
                best = (rank, name)
    return best


def swap_learned(target: Any, key: Any, src: Any) -> Optional[List[str]]:
    """Swap ``src`` in as the learned program for ``key`` by updating the
    state of the provider that owns it (preferring one whose users read it).

    ``key``: a learned name, or a module key (tuple / ``key_str`` / scope)
    matched against the names and their users. Returns the paths of the
    components that read it (they re-render), or ``None`` when no provider in
    ``target`` matches (the caller may fall back to the env mutation)."""
    best = None
    for _path, reg in find_registries(target):
        m = _match(reg, key)
        if m is not None and (best is None or m[0] > best[0][0]):
            best = (m, reg)
    if best is None:
        return None
    (_rank, name), reg = best
    reg.set(name, src)
    return sorted(reg.consumers.get(name) or ())
