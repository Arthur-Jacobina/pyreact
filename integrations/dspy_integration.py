# dspy_integration.py ---------------------------------------
"""DSPy environment as React-style context.

LMs, adapter and settings are *context*: ``DSPyProvider`` scopes them to a
subtree and hooks apply them per call with ``dspy.context(...)``. Nothing
here mutates global ``dspy.settings`` (``dspy.configure`` is owner-thread /
owner-task only, and per-render mutation leaked across subtrees).
"""

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import dspy

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
    # Module caches survive env rebuilds (modules are LM-agnostic: the LM is
    # applied per call). Compiled programs are tied to the optimizer.
    store = hooks.use_ref(
        {"caches": {}, "compiled": {}, "compiling": set(), "optimizer": None}
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
