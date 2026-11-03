"""Refs: imperative access to a component (cf. React ``useImperativeHandle``).

Data flows down through props and context; when code *outside* a component
needs to call into it (a test, a CLI, the root), the component exposes a
handle through a ref instead of a service lookup (ADR 0001, principle 4).

The ``ref=`` prop convention: a component that can be referenced takes a
``ref`` prop and hands it to :func:`use_imperative_handle`::

    @component
    def Coder(ref=None):
        sb = use_sandbox()
        use_imperative_handle(ref, lambda: CoderHandle(sb), [id(sb)])
        ...

    r = Ref()
    App = lambda: [Coder(key="coder", ref=r)]     # r.current: the handle
    App = lambda: [Coder(key="coder", ref="coder")]  # or by name, per session

A ``ref`` is one of:

- a :class:`~pyreact.core.hook.Ref` (``hooks.use_ref()`` in a parent, or
  ``Ref()`` outside the tree): ``ref.current`` is set to the handle;
- a callable (callback ref): called with the handle, then with ``None``;
- a ``str``: shorthand for ``name=`` (a "named ref");
- ``None``: nothing (the component works unreferenced).

Named refs are registered in the session's refs registry, the service
``"refs"``: a plain ``dict`` ``name -> handle`` (read from outside the tree
by the root, ``root.ref(name)``). Names are per session; keep them unique
within one.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Sequence, Union

from .core import hooks
from .hook import Ref

__all__ = ["REFS_SERVICE", "RefLike", "use_imperative_handle"]

REFS_SERVICE = "refs"

RefLike = Union[Ref, Callable[[Any], Any], str, None]


def _assign(ref: Any, value: Any) -> None:
    if ref is None:
        return
    if isinstance(ref, Ref) or hasattr(ref, "current"):
        ref.current = value
    elif callable(ref):
        ref(value)
    else:
        raise TypeError(f"ref must be a Ref, a callable, a str or None, got {ref!r}")


def use_imperative_handle(
    ref: RefLike,
    factory: Callable[[], Any],
    deps: Optional[Sequence[Any]] = None,
    *,
    name: Optional[str] = None,
) -> None:
    """Expose ``factory()`` as this component's handle through ``ref``.

    React semantics: after commit (mount, and whenever ``deps`` change;
    ``deps=None`` = every render) the handle is created and assigned to
    ``ref``; the previous one is cleared first, and on unmount ``ref`` is
    cleared (``ref.current = None`` / ``ref(None)``), only if it still holds
    this component's handle.

    ``name`` (or a ``str`` ``ref``) also registers the handle in the
    session's refs registry (``session.services["refs"][name]``, created
    lazily) and removes it on cleanup. A name already held by another
    handle is overwritten with a ``RuntimeWarning``.
    """
    if isinstance(ref, str):
        if name is not None and name != ref:
            raise ValueError(f"ref={ref!r} and name={name!r} disagree")
        ref, name = None, ref
    elif ref is not None and not (isinstance(ref, Ref) or hasattr(ref, "current")
                                  or callable(ref)):
        raise TypeError(f"ref must be a Ref, a callable, a str or None, got {ref!r}")
    registry = hooks.get_service(REFS_SERVICE, dict) if name is not None else None
    path = hooks.path

    def _attach():
        handle = factory()
        _assign(ref, handle)
        if registry is not None:
            held = registry.get(name)
            if held is not None and held is not handle:
                warnings.warn(f"ref name {name!r} already registered; {path} replaces it",
                              RuntimeWarning, stacklevel=2)
            registry[name] = handle

        def _detach():
            if ref is not None:
                if not hasattr(ref, "current") or ref.current is handle:
                    _assign(ref, None)
            if registry is not None and registry.get(name) is handle:
                del registry[name]

        return _detach

    hooks.use_effect(_attach, None if deps is None else [*deps, ref, name])
