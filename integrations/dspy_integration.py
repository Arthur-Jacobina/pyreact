# dspy_integration.py ---------------------------------------
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional
import asyncio
import dspy
import os
from pyreact.core.core import component, hooks
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


# Create DSPy context - will be automatically managed by create_context
DSPyContext = create_context(default=None, name="DSPy")


def use_dspy_env() -> DSPyEnv:
    env = hooks.use_context(DSPyContext)
    if env is None:
        raise RuntimeError(
            "DSPyProvider is not mounted. Please ensure DSPyProvider is configured."
        )
    return env


@component
def DSPyProvider(
    *,
    lm: Optional[Any] = None,
    models: Optional[Dict[str, Any]] = None,
    optimizer=None,
    settings=None,
    children=None,
):
    def _factory():
        model_registry = dict(models or {})
        default_lm = lm if lm is not None else model_registry.get("default")
        if default_lm is None:
            raise ValueError(
                "DSPyProvider requires either 'lm' or models['default'] to be provided."
            )

        # Ensure registry has a 'default'
        model_registry.setdefault("default", default_lm)
        env = DSPyEnv(
            models=model_registry,
            optimizer=optimizer,
            settings=settings or {},
        )
        return env

    # Create environment directly without use_memo to avoid re-creation issues
    env = _factory()

    # Configure the global default LM once per env
    if env.models.get("default"):
        try:
            dspy.configure(lm=env.models["default"])
        except Exception:
            # If configuration fails, continue without global configuration
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
    """
    env = use_dspy_env()
    deps_key = tuple(deps) if deps is not None else None
    module_key = _mk_module_key(module_cls, signature, name)

    def factory():
        mod = module_cls(signature)
        env.caches[module_key] = mod
        return mod

    mod = hooks.use_memo(
        lambda: env.caches.get(module_key) or factory(), deps=[module_key, deps_key]
    )

    if compile_with_optimizer and env.optimizer is not None:
        compiled = env.compiled.get(module_key)

        if compiled is None:
            # trigger compilation outside the event-loop thread to avoid UI blocking
            def kick_compile():
                def work():
                    return env.optimizer.compile(mod)

                fut = asyncio.get_running_loop().run_in_executor(None, work)

                async def _await_and_store():
                    try:
                        cm = await fut
                        env.compiled[module_key] = cm
                        env.caches[module_key] = cm
                    except Exception:
                        pass

                asyncio.create_task(_await_and_store())

            hooks.use_effect(kick_compile, [module_key])  # run once

    return env.compiled.get(module_key, mod)
