# dspy_integration.py ---------------------------------------
"""DSPy environment as React-style context.

LMs, adapter and settings are *context*: ``DSPyProvider`` scopes them to a
subtree and hooks apply them per call with ``dspy.context(...)``. Nothing
here mutates global ``dspy.settings`` (``dspy.configure`` is owner-thread /
owner-task only, and per-render mutation leaked across subtrees).
"""

import json
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import dspy

from pyreact.core import report_hydration_mismatch, use_dehydrate
from pyreact.core.core import component, hooks, use_path
from pyreact.core.provider import create_context


# Environment to be injected into the tree
@dataclass
class DSPyEnv:
    # Optional registry of models to enable per-component selection.
    # The "default" key is always required and serves as the default LM.
    models: Dict[str, Any] = field(default_factory=dict)
    optimizer: Optional[Any] = None
    caches: Dict = field(default_factory=dict)  # (module_key -> module)
    compiled: Dict = field(default_factory=dict)  # (module_key -> compiled module)
    settings: Dict = field(default_factory=dict)  # free for flags (timeout, etc.)
    adapter: Optional[Any] = None  # e.g. dspy.JSONAdapter(); None = DSPy default
    # module keys with an optimizer compile in flight (dedupe across components)
    compiling: set = field(default_factory=set, repr=False)
    # hydration: {module key: learned state} applied when the module is built
    restore: Dict = field(default_factory=dict, repr=False)

    def lm_for(self, model: Optional[str] = None, lm: Optional[Any] = None) -> Any:
        """Resolve an LM: explicit ``lm`` > ``models[model]`` > ``models['default']``."""
        if lm is not None:
            return lm
        if model is not None and model in self.models:
            return self.models[model]
        return self.models.get("default")

    def context(self, model: Optional[str] = None, lm: Optional[Any] = None, **extra):
        """``dspy.context`` applying this env's LM (and adapter) for one call."""
        overrides = dict(extra)
        selected = self.lm_for(model, lm)
        if selected is not None:
            overrides["lm"] = selected
        if self.adapter is not None:
            overrides.setdefault("adapter", self.adapter)
        return dspy.context(**overrides) if overrides else nullcontext()


ENVS_SERVICE = "dspy_envs"  # session service: {provider path: DSPyEnv}


# ---------------------------------------------------------------------------
# Dehydration: compiled / hot-swapped programs are state of the provider
# ---------------------------------------------------------------------------
def _plain(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def dehydrate_compiled(env: Optional[DSPyEnv]) -> List[Dict[str, Any]]:
    """The provider's ``"compiled"`` slice: one entry per learned module in
    ``env.compiled`` (an optimizer compiled it, or a program was hot-swapped
    in by key)::

        {"key": ["ReAct", "Coder", "coder"], "key_str": "ReAct:Coder:coder",
         "arch": "react", "path": "artifacts/coder.json"}   # or "state": {...}

    ``path``: the ``integrations.learn`` result file the program came from,
    when it still exists; else its ``dump_state()`` inline. Modules that carry
    a ``LearnedProgram`` / ``program=`` are skipped (that provider dehydrates
    them); states hydrated but not applied in this run are carried over."""
    if env is None:
        return []
    from integrations.feedback import key_str

    from .architectures import arch_of
    from .hooks import _Applied

    out: List[Dict[str, Any]] = []
    seen = set()
    for key, program in list(env.compiled.items()):
        if not isinstance(key, tuple) or isinstance(
                getattr(program, "_pyreact_learned", None), _Applied):
            continue
        seen.add(key)
        entry: Dict[str, Any] = {"key": list(key), "key_str": key_str(key),
                                 "arch": arch_of(program)}
        path = getattr(program, "_pyreact_result_path", None)
        if path and os.path.exists(path):
            entry["path"] = str(path)
        else:
            entry["state"] = _plain(program.dump_state())
        out.append(entry)
    for key, state in list(env.restore.items()):
        if isinstance(key, tuple) and key not in seen:
            out.append({"key": list(key), "key_str": key_str(key), "state": _plain(state)})
    return out


def hydrate_compiled(slice_: Any, session: Any = None) -> Dict[Tuple, Any]:
    """Inverse of :func:`dehydrate_compiled`: ``{module key: state}``. A
    result file that is gone or unreadable is a ``hydration.mismatch`` (and
    ``session.resume.degraded``, ``participant: "dspy.compiled"``); that
    module starts from its signature."""
    if not isinstance(slice_, list):
        raise TypeError(f"a compiled-modules slice is a list, got {type(slice_).__name__}")
    out: Dict[Tuple, Any] = {}
    for e in slice_:
        if not isinstance(e, dict) or not e.get("key"):
            continue
        key = tuple(e["key"])
        state = e.get("state")
        if state is None and e.get("path"):
            try:
                with open(e["path"], encoding="utf-8") as f:
                    data = json.load(f)
                state = data["state"] if isinstance(data, dict) and "state" in data else data
            except Exception as exc:
                reason = ("result missing" if isinstance(exc, FileNotFoundError)
                          else "result unreadable")
                degrade = getattr(session, "degrade", None)
                if callable(degrade):
                    degrade("dspy.compiled", reason, key=e.get("key_str"), path=e["path"],
                            error=repr(exc))
                report_hydration_mismatch(
                    "compiled", f"learned module {e.get('key_str')}: {reason} "
                    f"({e['path']}); it starts from its signature",
                    reason=reason, module=e.get("key_str"), src=e["path"])
                continue
        if state is not None:
            out[key] = state
    return out

# Create DSPy context - will be automatically managed by create_context
DSPyContext = create_context(default=None, name="DSPy")


def use_dspy_env() -> DSPyEnv:
    env = hooks.use_context(DSPyContext)
    if env is None:
        raise RuntimeError(
            "DSPyProvider is not mounted. Please ensure DSPyProvider is configured."
        )
    return env


def _identity_key(mapping: Optional[Dict[str, Any]]) -> Tuple:
    # Inline dicts are rebuilt every render; key on their *items'* identity.
    if not mapping:
        return ()
    return tuple(sorted((str(k), id(v)) for k, v in mapping.items()))


@component(transparent=True)
def DSPyProvider(
    *,
    lm: Optional[Any] = None,
    models: Optional[Dict[str, Any]] = None,
    optimizer=None,
    settings=None,
    adapter=None,
    children=None,
):
    # Learned modules are state of the provider (ADR 0001): a suspend
    # dehydrates them, hydrate_root hands them back before the modules are
    # built (``use_module`` / ``use_agent`` apply them).
    session = hooks.get_service("session", lambda: None)
    current = hooks.use_ref(None)
    hydrated = use_dehydrate("compiled", dehydrate=lambda: dehydrate_compiled(current.current),
                             hydrate=lambda s: hydrate_compiled(s, session))
    # Module caches survive env rebuilds (modules are LM-agnostic: the LM is
    # applied per call). Compiled programs are tied to the optimizer.
    store = hooks.use_ref(
        {"caches": {}, "compiled": {}, "compiling": set(), "optimizer": None,
         "restore": dict(hydrated or {})}
    ).current

    def _factory():
        model_registry = dict(models or {})
        default_lm = lm if lm is not None else model_registry.get("default")
        if default_lm is None:
            raise ValueError(
                "DSPyProvider requires either 'lm' or models['default'] to be provided."
            )
        model_registry.setdefault("default", default_lm)
        if store["optimizer"] is not optimizer:
            store["optimizer"] = optimizer
            store["compiled"] = {}
        return DSPyEnv(
            models=model_registry,
            optimizer=optimizer,
            caches=store["caches"],
            compiled=store["compiled"],
            settings=dict(settings or {}),
            adapter=adapter,
            compiling=store["compiling"],
            restore=store["restore"],
        )

    env = hooks.use_memo(
        _factory,
        deps=[
            id(lm),
            _identity_key(models),
            id(optimizer),
            _identity_key(settings),
            id(adapter),
        ],
    )
    current.current = env
    # Make the env discoverable from outside the tree (dev tools, hot swap by
    # key): ``session.services["dspy_envs"][provider_path] = env``.
    try:
        hooks.get_service(ENVS_SERVICE, dict)[use_path()] = env
    except Exception:
        pass
    return [DSPyContext(value=env, children=children or [])]


def _mk_module_key(module_cls, signature, name: Optional[str]) -> Tuple:
    sig_key = signature if isinstance(signature, str) else signature.__name__
    return (module_cls.__name__, sig_key, name)


def use_dspy_module(
    signature,  # class dspy.Signature (or string "q -> a")
    module_cls,  # e.g.: dspy.Predict / dspy.ChainOfThought
    *,
    name: Optional[str] = None,  # optional, to distinguish instances
    compile_with_optimizer: bool = False,
    deps: Optional[list] = None,  # when changed, recreate (e.g., change few-shot)
) -> Any:
    """
    Creates (or reuses from cache) a DSPy Module parameterized by the Signature.
    Optionally compiles with the Provider's optimizer.

    Cache key is ``(module_cls.__name__, signature name, name)`` — shared by
    every component using the same triple. Prefer :func:`integrations.use_module`
    for per-component instances.
    """
    from .hooks import use_module

    return use_module(
        signature,
        module_cls,
        name=name,
        deps=deps,
        compile_with_optimizer=compile_with_optimizer,
        _key=_mk_module_key(module_cls, signature, name),
    )
