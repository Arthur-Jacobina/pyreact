"""Hot module replacement by key: find modules in a running app and swap them.

    find_envs(app)              [(provider_path, DSPyEnv)] of a Root / AppRunner / Session
    list_modules(app)           every cached module: key, arch, compiled?, swaps
    resolve_module(app, key)    (env, key_tuple, module) for a key / scope / label
    hot_swap_into(app, key, p)  integrations.optimize.hot_swap on the app's loop
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple

from integrations.feedback import key_str

__all__ = ["find_envs", "list_modules", "resolve_module", "hot_swap_into"]


# ---------------------------------------------------------------------------
# Finding modules
# ---------------------------------------------------------------------------
def _session_of_root(target: Any) -> Any:
    """A ``pyreact.boot.Root`` is resolved to its session; anything else as is."""
    from pyreact.boot.root import Root

    return target.session if isinstance(target, Root) else target


def _services_of(target: Any) -> List[Dict[str, Any]]:
    from pyreact.core.hook import HookContext

    if target is None:
        return [HookContext._services]
    target = _session_of_root(target)
    sessions = getattr(target, "sessions", None)
    if callable(sessions):  # AppRunner
        return [s.services for s in sessions()]
    services = getattr(target, "services", None)
    if isinstance(services, dict):  # Session
        return [services]
    return []


def find_envs(target: Any = None) -> List[Tuple[str, Any]]:
    """``[(provider_path, DSPyEnv)]`` for a Root / AppRunner / Session / env / None
    (headless)."""
    from integrations.dspy_integration import ENVS_SERVICE, DSPyEnv

    if isinstance(target, DSPyEnv):
        return [("env", target)]
    out = []
    for services in _services_of(target):
        for path, env in (services.get(ENVS_SERVICE) or {}).items():
            if all(env is not e for _p, e in out):
                out.append((path, env))
    return out


def list_modules(target: Any = None) -> List[Dict[str, Any]]:
    from integrations.architectures import arch_of
    from integrations.optimize import ModuleVersions

    versions = ModuleVersions._ctx.value or {}
    rows = []
    for path, env in find_envs(target):
        for key, mod in list(env.caches.items()):
            rows.append({
                "key": key_str(key), "key_tuple": key, "arch": arch_of(mod),
                "label": getattr(mod, "_pyreact_name", type(mod).__name__),
                "compiled": key in env.compiled, "swaps": versions.get(key, 0),
                "module": env.compiled.get(key, mod), "env": env, "provider": path,
            })
    return rows


def _scope_match(k: str, wanted: str) -> bool:
    scope = k.split(":")[-1]
    return scope == wanted or scope.endswith("#" + wanted) or scope.endswith("/" + wanted)


def resolve_module(target: Any, key: Any) -> Tuple[Any, Any, Any]:
    """``(env, key_tuple, module)`` for a key/scope/label; clear errors otherwise."""
    rows = list_modules(target)
    if not rows:
        raise LookupError("no DSPy modules found (is a DSPyProvider mounted and rendered?)")
    if key is None:
        raise LookupError("which module? available: " + ", ".join(r["key"] for r in rows))
    wanted = key_str(key)
    for test in (lambda r: r["key_tuple"] == key or r["key"] == wanted,
                 lambda r: r["label"] == wanted,
                 lambda r: _scope_match(r["key"], wanted)):
        hits = [r for r in rows if test(r)]
        distinct = {r["key"] for r in hits}
        if len(distinct) == 1:
            r = hits[0]
            return r["env"], r["key_tuple"], r["module"]
        if len(distinct) > 1:
            raise LookupError(f"{wanted!r} is ambiguous: {sorted(distinct)}")
    raise LookupError(f"no module matches {wanted!r}; available: "
                      + ", ".join(r["key"] for r in rows))


def _loop_of(target: Any):
    runner = getattr(target, "runner", None) or target
    return getattr(runner, "_loop", None), getattr(runner, "_thread", None)


def hot_swap_into(target: Any, key: Any, program: Any) -> List[Any]:
    """Resolve ``key`` in ``target`` and hot-swap ``program`` in (on the app loop)."""
    from integrations.dspy_integration import DSPyEnv
    from integrations.optimize import hot_swap

    if isinstance(target, DSPyEnv) and isinstance(key, tuple):
        env, k = target, key
    else:
        env, k, _old = resolve_module(target, key)
    for attr, val in (("_pyreact_key", k),):
        try:
            setattr(program, attr, val)
        except Exception:
            pass
    loop, thread = _loop_of(target)
    if loop is not None and loop.is_running() and threading.current_thread() is not thread:
        done = threading.Event()
        out: List[Any] = []

        def _swap():
            try:
                out.extend(hot_swap(env, k, program))
            finally:
                done.set()

        loop.call_soon_threadsafe(_swap)
        done.wait(10)
        return out
    return hot_swap(env, k, program)
