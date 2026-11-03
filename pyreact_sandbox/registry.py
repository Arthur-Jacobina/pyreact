"""Driver registry: ``register_driver(name, factory)`` / ``get_driver(name)``.

Built-ins (imported lazily): ``"docker"`` (:class:`DockerDriver`) and
``"local"`` (:class:`LocalDriver`, NOT isolated). Vendors register theirs::

    register_driver("e2b", lambda **kw: E2BDriver(**kw))
    SandboxProvider(driver="e2b", spec=..., children=[...])
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from .base import SandboxDriver

__all__ = ["register_driver", "get_driver", "list_drivers", "unregister_driver", "gc"]

DriverFactory = Callable[..., SandboxDriver]

_factories: Dict[str, DriverFactory] = {}
_instances: Dict[str, SandboxDriver] = {}
_lock = threading.Lock()


def _local(**kw: Any) -> SandboxDriver:
    from .local import LocalDriver

    return LocalDriver(**kw)


_BUILTINS: Dict[str, DriverFactory] = {"local": _local}


def register_driver(name: str, factory: DriverFactory, *, replace: bool = False) -> None:
    """Register ``factory(**kwargs) -> SandboxDriver`` under ``name``."""
    with _lock:
        if name in _factories and not replace:
            raise ValueError(f"sandbox driver {name!r} already registered (replace=True)")
        _factories[name] = factory
        _instances.pop(name, None)


def unregister_driver(name: str) -> None:
    with _lock:
        _factories.pop(name, None)
        _instances.pop(name, None)


def list_drivers() -> List[str]:
    with _lock:
        return sorted({*_BUILTINS, *_factories})


def get_driver(driver: Union[str, SandboxDriver, None] = None, **kwargs: Any) -> SandboxDriver:
    """Resolve a driver: an instance is returned as is; a name is built with
    its factory (one shared instance per name when no ``kwargs``)."""
    if isinstance(driver, SandboxDriver):
        return driver
    name = driver or "docker"
    with _lock:
        factory = _factories.get(name) or _BUILTINS.get(name)
        if factory is None:
            raise KeyError(f"unknown sandbox driver {name!r} (known: {sorted({*_BUILTINS, *_factories})})")
        if kwargs:
            return factory(**kwargs)
        inst = _instances.get(name)
        if inst is None:
            inst = _instances[name] = factory()
        return inst


def gc(drivers: Optional[Iterable[Union[str, SandboxDriver]]] = None, **kwargs: Any
       ) -> Dict[str, Dict[str, List[str]]]:
    """Collect orphaned sandboxes (and, on request, snapshots / templates).

    ``drivers`` default: ``("docker", "local")`` (unavailable ones are
    skipped). Keyword arguments go to each driver's ``gc_sync``; the built-ins
    accept ``all=False`` (everything labeled as ours, not only orphans),
    ``snapshots=False``, ``templates=False``, ``labels=None`` (only matching
    resources), ``older_than=None`` (seconds) and ``dry_run=False``.

    Returns ``{driver: {"containers"|"sandboxes": [...], "images"|"snapshots": [...]}}``.
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    for d in drivers or tuple(_BUILTINS):
        drv = get_driver(d)
        if not drv.available():
            continue
        out[getattr(drv, "name", str(d))] = drv.gc_sync(**kwargs)
    return out
