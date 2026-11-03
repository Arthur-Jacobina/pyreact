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

Providers are transparent (``@component(transparent=True)``): wrapping a
subtree in one does not change its components' paths.
"""

from __future__ import annotations

import asyncio
import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from pyreact.core.core import _context_stack, component, hooks
from pyreact.core.provider import create_context
from pyreact.core.runtime import schedule_rerender

__all__ = ["LearnedProgram", "use_learned", "LearnedContext", "SavedProgram"]


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

    def get(self, name: str) -> Any:
        return self.programs.get(name)


LearnedContext = create_context(default=None, name="Learned")


# ---------------------------------------------------------------------------
# Sources -> programs
# ---------------------------------------------------------------------------
def _tag(program: Any, path: Optional[str], optimizer: Optional[str]) -> Any:
    # a suspend (CompiledModulesParticipant) can then point at the file
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
# The provider
# ---------------------------------------------------------------------------
def _use_watch(ctx: Any, box: Dict[str, Any], paths: tuple, watch: bool,
               interval: float) -> None:
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
    # it is a single JSON-able value ({name: path}) held by ONE hook.
    # TODO(use_dehydrate): in the migration PR this use_state becomes
    #   programs, set_programs = use_dehydrate("learned", initial=dict(declared),
    #       dehydrate=lambda: dehydrate_programs(programs), hydrate=hydrate_programs)
    # so the learned program reference survives suspend/resume as state of the
    # provider (replacing CompiledModulesParticipant).
    programs, set_programs = hooks.use_state(dict(declared))
    box = hooks.use_ref(None)
    if box.current is None:
        box.current = {"declared": _declared_sig(declared), "cache": {}}
    st = box.current
    sig = _declared_sig(declared)
    if sig != st["declared"]:  # the prop changed: it wins (a re-render = the swap)
        st["declared"] = sig
        programs = dict(declared)
        set_programs(programs)
    outer = hooks.use_context(LearnedContext)

    resolved = dict(outer.programs) if outer is not None else {}
    for name, src in programs.items():
        resolved[name] = _resolve(name, src, st["cache"], watch)
    paths = tuple(sorted(os.path.abspath(os.fspath(s)) for s in programs.values()
                         if _is_path(s)))
    _use_watch(ctx, st, paths, watch, interval)

    hooks.provide(LearnedContext, LearnedScope(programs=resolved))
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
    for n in _names_for(name_or_key):
        if n in scope.programs:
            return scope.programs[n]
    return None
