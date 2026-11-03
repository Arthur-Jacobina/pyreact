"""Prompt optimization (GEPA) and hot module replacement for DSPy modules.

The learning loop, in React terms:

    runs produce events  ->  events become datasets (integrations.feedback)
    GEPA turns feedback into better prompts       (optimize_with_gepa)
    the optimized module is hot-swapped through context (hot_swap)
    components using it re-render and the next call uses the new prompt

``hot_swap`` is the "HMR" step: it writes the compiled module into
``DSPyEnv.compiled``/``caches`` (which ``use_dspy_module`` already prefers),
bumps a per-module version in :data:`ModuleVersions` and schedules a rerender
of every mounted component that uses that module, without restarting the app.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import dspy

from pyreact.core.core import hooks
from pyreact.core.provider import create_context
from pyreact.core.runtime import schedule_rerender

# module_key -> version; bumped by hot_swap. Components that want to react to
# swaps explicitly (e.g. to show "prompt v3") read it with use_module_version.
ModuleVersions = create_context(default={}, name="ModuleVersions")


def module_key_for(module_cls, signature, name: Optional[str] = None) -> Tuple:
    """The key ``use_dspy_module`` caches a module under in ``DSPyEnv``."""
    try:
        from integrations.dspy_integration import _mk_module_key

        return _mk_module_key(module_cls, signature, name)
    except ImportError:  # pragma: no cover - keep working if the helper moves
        sig_key = signature if isinstance(signature, str) else signature.__name__
        return (module_cls.__name__, sig_key, name)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
def optimize_with_gepa(
    module: dspy.Module,
    trainset: List[dspy.Example],
    metric: Callable,
    *,
    reflection_lm: Any,
    valset: Optional[List[dspy.Example]] = None,
    budget: int = 40,
    lm: Any = None,
    num_threads: int = 4,
    reflection_minibatch_size: int = 3,
    log_dir: Optional[str] = None,
    seed: int = 0,
    **gepa_kwargs: Any,
) -> dspy.Module:
    """Run ``dspy.GEPA`` on ``module`` and return the compiled program.

    ``metric(gold, pred, trace, pred_name, pred_trace)`` may return a float or
    ``dspy.Prediction(score=..., feedback=...)``; the feedback text is what the
    reflection LM reads to propose better instructions. ``budget`` is GEPA's
    ``max_metric_calls`` (each one is a student LM call). ``lm`` is the student
    LM (defaults to the globally configured one).

    Note: GEPA *resumes* from the state saved in an existing ``log_dir``
    (including its budget and val scores); use a fresh directory per run.
    """
    gepa = dspy.GEPA(
        metric=metric,
        max_metric_calls=budget,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=reflection_minibatch_size,
        num_threads=num_threads,
        log_dir=log_dir,
        seed=seed,
        track_stats=True,
        **gepa_kwargs,
    )
    if lm is not None:
        with dspy.context(lm=lm):
            return gepa.compile(module, trainset=trainset, valset=valset)
    return gepa.compile(module, trainset=trainset, valset=valset)


def save_compiled(module: dspy.Module, path: str) -> str:
    """Persist a (compiled) module's state (instructions, demos) as JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    module.save(path)
    return path


def load_compiled(module_cls, signature, path: str) -> dspy.Module:
    """Rebuild ``module_cls(signature)`` and load a state saved by save_compiled."""
    mod = module_cls(signature)
    mod.load(path)
    return mod


def instructions_of(module: dspy.Module) -> Dict[str, str]:
    """``{predictor_name: instructions}`` -- handy for diffing prompts."""
    return {
        name: getattr(p.signature, "instructions", "")
        for name, p in module.named_predictors()
    }


# ---------------------------------------------------------------------------
# Hot module replacement
# ---------------------------------------------------------------------------
def _contains(obj: Any, needles: Tuple[Any, ...], depth: int = 0) -> bool:
    if depth > 3:
        return False
    for n in needles:
        if obj is n:
            return True
    if isinstance(obj, tuple):
        if obj in needles:
            return True
        return any(_contains(x, needles, depth + 1) for x in obj)
    return False


def _uses_module(hctx, needles: Tuple[Any, ...]) -> bool:
    """Does a component's hook state reference the module (or its cache key)?

    ``use_dspy_module`` memoizes ``(module, (module_key, deps))``, so either
    the old module object or the key shows up in the hook slots.
    """
    for slot in list(getattr(hctx, "hooks", [])):
        try:
            if _contains(slot, needles):
                return True
        except Exception:
            continue
    return False


def _dspy_subscribers() -> Iterable:
    try:
        from integrations.dspy_integration import DSPyContext
    except ImportError:  # pragma: no cover
        return []
    return list(getattr(DSPyContext, "_subs", []))


def hot_swap(
    env: Any,
    module_key: Tuple,
    compiled: dspy.Module,
    *,
    rerender_all: bool = False,
) -> List[Any]:
    """Replace the module cached under ``module_key`` and rerender its users.

    - ``env.compiled[module_key]`` / ``env.caches[module_key]`` <- ``compiled``
      (``use_dspy_module`` returns ``env.compiled`` first on its next render).
    - ``ModuleVersions[module_key]`` is bumped (subscribers of that context
      rerender, like a context value change).
    - every mounted component subscribed to the DSPy context whose hooks
      reference the old module / the key is scheduled for rerender
      (``rerender_all=True`` rerenders every DSPy-context subscriber).

    Thread-safe once an AppRunner is bound (``schedule_rerender`` is). Returns
    the list of contexts scheduled for rerender.
    """
    old = env.compiled.get(module_key) or env.caches.get(module_key)
    env.compiled[module_key] = compiled
    env.caches[module_key] = compiled

    versions = dict(ModuleVersions.get() or {})
    versions[module_key] = versions.get(module_key, 0) + 1
    needles = tuple(x for x in (module_key, old) if x is not None)

    scheduled = []
    for hctx in _dspy_subscribers():
        if not getattr(hctx, "_mounted", True):
            continue
        if rerender_all or _uses_module(hctx, needles):
            schedule_rerender(hctx, reason=f"hot_swap {module_key}")
            scheduled.append(hctx)
    # set() reschedules explicit ModuleVersions subscribers (deduped).
    try:
        ModuleVersions.set(versions)
    except RuntimeError:
        # No loop: nothing is mounted/subscribed; just record the version.
        ModuleVersions._ctx.value = versions
    return scheduled


def use_module_version(module_key: Tuple) -> int:
    """Hook: how many times ``module_key`` was hot-swapped (rerenders on swap)."""
    return (hooks.use_context(ModuleVersions) or {}).get(module_key, 0)
