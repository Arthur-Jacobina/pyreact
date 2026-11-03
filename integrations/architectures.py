"""Architectures: *how* a component thinks, swappable like a prop.

The same signature can be solved by different DSPy programs. Pick one by
name and keep everything else (``use_call``, events, cancellation, hot swap)
identical::

    qa = use_module("context, question -> answer", arch="cot")      # or "rlm", ...
    call = use_call(qa)

    call = use_agent_call("context, question -> answer", arch="rlm",
                          max_iterations=6, sub_model="fast")      # one-liner

Built-ins (``list_architectures()``):

    predict     dspy.Predict                  one LM call
    cot         dspy.ChainOfThought           reasoning, then the answer
    react       dspy.ReAct                    tool-using loop (tools=[...])
    rlm         dspy.RLM                      REPL over huge inputs (Deno sandbox)
    pot         dspy.ProgramOfThought         writes + runs Python (Deno sandbox)
    refine      dspy.Refine(base)             retry with feedback until reward >= threshold
    best_of_n   dspy.BestOfN(base)            N samples, keep the best reward

Register your own with :func:`register_architecture` -- a factory
``(signature, **kwargs) -> dspy.Module``. Architectures whose module is
synchronous/blocking (like RLM's REPL loop) can also provide a ``runner``:
an ``async (module, inputs, *, call_id, publish) -> (prediction, extra)``
coroutine that :func:`integrations.hooks.invoke_module` awaits instead of the
default ``acall``; ``extra`` is merged into the ``llm.end`` payload.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import dspy

__all__ = [
    "Architecture",
    "register_architecture",
    "unregister_architecture",
    "get_architecture",
    "list_architectures",
    "build_module",
    "arch_of",
    "signature_of",
    "signature_text",
    "default_reward",
]

Runner = Callable[..., Any]


@dataclass(frozen=True)
class Architecture:
    """A registered way to turn a signature into a DSPy program."""

    name: str
    factory: Callable[..., Any]
    doc: str = ""
    key: str = ""  # module cache-key prefix (defaults to ``name``)
    runner: Optional[Runner] = None
    aliases: Tuple[str, ...] = field(default=())

    @property
    def cache_key(self) -> str:
        return self.key or self.name


_REGISTRY: Dict[str, Architecture] = {}
_ALIASES: Dict[str, str] = {}


def register_architecture(
    name: str,
    factory: Callable[..., Any],
    *,
    doc: str = "",
    key: Optional[str] = None,
    runner: Optional[Runner] = None,
    aliases: Tuple[str, ...] = (),
    replace: bool = True,
) -> Architecture:
    """Register ``factory(signature, **kwargs) -> dspy.Module`` under ``name``.

    ``key``: prefix of the module cache key (default ``name``); hot swap and
    datasets address modules by it. ``runner``: optional custom async
    invocation (see module docstring). ``replace=False`` refuses to overwrite.
    """
    if not name or not isinstance(name, str):
        raise ValueError("architecture name must be a non-empty string")
    if not callable(factory):
        raise TypeError(f"factory for architecture {name!r} must be callable")
    if not replace and name in _REGISTRY:
        raise ValueError(f"architecture {name!r} is already registered")
    arch = Architecture(name=name, factory=factory, doc=doc or (factory.__doc__ or "").strip(),
                        key=key or "", runner=runner, aliases=tuple(aliases))
    _REGISTRY[name] = arch
    for a in aliases:
        _ALIASES[a] = name
    return arch


def unregister_architecture(name: str) -> None:
    arch = _REGISTRY.pop(name, None)
    if arch is not None:
        for a in arch.aliases:
            _ALIASES.pop(a, None)


def get_architecture(name: str) -> Architecture:
    real = _ALIASES.get(name, name)
    try:
        return _REGISTRY[real]
    except KeyError:
        raise KeyError(
            f"unknown architecture {name!r}; registered: {sorted(_REGISTRY)} "
            "(add one with register_architecture(name, factory))"
        ) from None


def list_architectures() -> Dict[str, str]:
    """``{name: one-line doc}`` of every registered architecture."""
    return {n: (a.doc.splitlines()[0] if a.doc else "") for n, a in _REGISTRY.items()}


def _arch_for_class(cls) -> Optional[Architecture]:
    for a in _REGISTRY.values():
        if a.cache_key == getattr(cls, "__name__", None):
            return a
    return None


def arch_of(module: Any) -> str:
    """Architecture name of a module (``_pyreact_arch`` or inferred from its class)."""
    name = getattr(module, "_pyreact_arch", None)
    if name:
        return name
    a = _arch_for_class(type(module))
    if a is not None:
        return a.name
    for base in type(module).__mro__[1:]:
        a = _arch_for_class(base)
        if a is not None:
            return a.name
    return type(module).__name__


def runner_of(module: Any) -> Optional[Runner]:
    name = arch_of(module)
    arch = _REGISTRY.get(_ALIASES.get(name, name))
    return arch.runner if arch is not None else None


def signature_of(program: Any) -> Any:
    """The user-facing signature of a program (string or Signature class)."""
    sig = getattr(program, "_pyreact_signature", None)
    if sig is not None:
        return sig
    sig = getattr(program, "signature", None)
    if sig is not None:
        return sig
    named = getattr(program, "named_predictors", None)
    if callable(named):
        for _n, p in named():
            return p.signature
    return None


def signature_text(sig: Any) -> str:
    """``"a, b -> c"`` for a signature string or class (best effort)."""
    if sig is None:
        return ""
    if isinstance(sig, str):
        return sig
    text = getattr(sig, "signature", None)
    return text if isinstance(text, str) else getattr(sig, "__name__", str(sig))


def _tag(module: Any, arch: str, signature: Any, kwargs: Dict[str, Any]) -> Any:
    try:
        module._pyreact_arch = arch
        module._pyreact_signature = signature
        module._pyreact_arch_kwargs = {k: v for k, v in kwargs.items()
                                       if isinstance(v, (str, int, float, bool, type(None)))}
    except Exception:
        pass
    return module


def build_module(signature: Any, arch: Any = "predict", **kwargs: Any) -> Any:
    """Build a program outside components: ``build_module(sig, arch="cot")``.

    ``arch`` may be a registered name, a ``dspy.Module`` subclass or any
    ``factory(signature, **kwargs)`` callable.
    """
    if isinstance(arch, str):
        a = get_architecture(arch)
        return _tag(a.factory(signature, **kwargs), a.name, signature, kwargs)
    if isinstance(arch, type):
        a = _arch_for_class(arch)
        return _tag(arch(signature, **kwargs), a.name if a else arch.__name__, signature, kwargs)
    if callable(arch):
        name = getattr(arch, "__name__", "custom")
        return _tag(arch(signature, **kwargs), name, signature, kwargs)
    raise TypeError(f"arch must be a name, a dspy.Module class or a factory, got {arch!r}")


# ---------------------------------------------------------------------------
# Built-ins
# ---------------------------------------------------------------------------
_NON_LABEL = {"reasoning", "rationale", "trajectory"}


def default_reward(args: Dict[str, Any], pred: Any) -> float:
    """Reward for refine/best_of_n: 1.0 when every output field is non-empty."""
    try:
        values = {k: v for k, v in pred.items() if k not in _NON_LABEL}
    except Exception:
        return 0.0
    if not values:
        return 0.0
    return float(all(v not in (None, "") and str(v).strip() for v in values.values()))


def _predict(signature, **kw):
    """One LM call (dspy.Predict)."""
    return dspy.Predict(signature, **kw)


def _cot(signature, **kw):
    """Reason step by step, then answer (dspy.ChainOfThought)."""
    return dspy.ChainOfThought(signature, **kw)


def _react(signature, *, tools=(), max_iters: int = 10, **kw):
    """Tool-using agent loop (dspy.ReAct); tools emit tool.* events."""
    from .hooks import to_dspy_tool

    return dspy.ReAct(signature, tools=[to_dspy_tool(t) for t in (tools or [])],
                      max_iters=max_iters, **kw)


def _pot(signature, *, max_iters: int = 3, **kw):
    """Writes and runs Python to compute the answer (dspy.ProgramOfThought; Deno)."""
    return dspy.ProgramOfThought(signature, max_iters=max_iters, **kw)


def _wrapped(cls, doc):
    def factory(signature, *, N: int = 3, reward_fn=None, threshold: float = 1.0,  # noqa: N803
                base: str = "cot", fail_count=None, **base_kw):
        inner = build_module(signature, base, **base_kw)
        return cls(module=inner, N=N, reward_fn=reward_fn or default_reward,
                   threshold=threshold, fail_count=fail_count)

    factory.__doc__ = doc
    return factory


def _rlm(signature, *, tools=(), max_iterations: Optional[int] = None,
         max_llm_calls: Optional[int] = None, max_output_chars: int = 10_000,
         sub_lm=None, interpreter=None, adapter=None, verbose: bool = False, **kw):
    """Recursive LM: explores huge inputs from a sandboxed REPL (dspy.RLM; Deno)."""
    from . import rlm as R

    if R.PyreactRLM is None:  # pragma: no cover
        raise R.RLMSandboxError("This DSPy version has no dspy.RLM (needs dspy>=3.1).")
    mod = R.PyreactRLM(
        signature,
        max_iterations=max_iterations or R.DEFAULT_MAX_ITERATIONS,
        max_llm_calls=max_llm_calls or R.DEFAULT_MAX_LLM_CALLS,
        max_output_chars=max_output_chars,
        verbose=verbose,
        tools=[R._tool_to_dspy(t) for t in (tools or [])] or None,
        sub_lm=sub_lm,
        interpreter_factory=interpreter,
        **kw,
    )
    mod._pyreact_adapter = adapter
    return mod


async def _rlm_runner(module, inputs: dict, *, call_id: str, publish: Callable) -> Tuple[Any, dict]:
    """Run an RLM in a worker thread; cancellation stops it at the next checkpoint."""
    from . import rlm as R

    ctrl = R._RunControl(call_id)
    max_it = getattr(module, "max_iterations", None)

    def _on_iter(c, _result):
        publish("rlm.iteration", {"call_id": c.call_id, "iteration": c.iterations,
                                  "max_iterations": max_it, "sub_llm_calls": c.sub_llm_calls})

    ctrl.on_iteration = _on_iter
    explicit = getattr(module, "_pyreact_adapter", None)
    adapter = explicit or dspy.settings.adapter or R._DEFAULT_ADAPTER
    t0 = time.perf_counter()
    try:
        result = await asyncio.to_thread(R.run_rlm_sync, module, inputs, adapter=adapter, ctrl=ctrl)
    except asyncio.CancelledError:
        ctrl.cancelled.set()  # the worker stops at its next checkpoint
        raise
    except Exception as exc:
        if not isinstance(exc, R.RLMSandboxError) and "Deno" in str(exc):
            raise R.RLMSandboxError(f"{exc}\n{R.SANDBOX_HELP}") from exc
        try:
            exc.rlm_iterations = ctrl.iterations
        except Exception:
            pass
        raise
    traj = getattr(result, "trajectory", None) or []
    return result, {"iterations": ctrl.iterations or len(traj),
                    "sub_llm_calls": ctrl.sub_llm_calls, "trajectory_len": len(traj),
                    "worker_seconds": round(time.perf_counter() - t0, 3)}


def _install_builtins() -> None:
    register_architecture("predict", _predict, key="Predict", aliases=("Predict",))
    register_architecture("cot", _cot, key="ChainOfThought",
                          aliases=("chain_of_thought", "ChainOfThought"))
    register_architecture("react", _react, key="ReAct", aliases=("ReAct",))
    register_architecture("rlm", _rlm, key="RLM", runner=_rlm_runner, aliases=("RLM",))
    register_architecture("pot", _pot, key="ProgramOfThought",
                          aliases=("program_of_thought", "ProgramOfThought"))
    register_architecture("refine", _wrapped(dspy.Refine, "Run up to N times with feedback "
                          "until reward_fn >= threshold (dspy.Refine over base arch)."),
                          key="Refine", aliases=("Refine",))
    register_architecture("best_of_n", _wrapped(dspy.BestOfN, "Sample N times, keep the best "
                          "reward_fn (dspy.BestOfN over base arch)."),
                          key="BestOfN", aliases=("BestOfN",))


_install_builtins()


def builtin_names() -> List[str]:
    return ["predict", "cot", "react", "rlm", "pot", "refine", "best_of_n"]
