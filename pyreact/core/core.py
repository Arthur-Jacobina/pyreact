# core.py ----------------------------------------------------
from contextvars import ContextVar
from functools import wraps

_context_stack = ContextVar("component_context", default=None)


class _HookProxy:
    def __getattr__(self, name):
        comp = _context_stack.get()
        if comp is None:
            raise RuntimeError(
                f"hook.{name}() can only be used during render or effect."
            )
        return getattr(comp, name)


hooks = _HookProxy()


def use_path() -> str:
    """The calling component's stable path (``HookContext.path``, e.g.
    ``App/Chat[chat]``; providers add no segment). Same as ``hooks.path``.

    Call it during render (or in an effect / cleanup, which also run "as" the
    component) and keep the string: closures and tasks that run later can use
    it without reaching into the render stack.
    """
    comp = _context_stack.get()
    if comp is None:
        raise RuntimeError("use_path() can only be used during render or effect.")
    return comp.path


class VNode:
    def __init__(self, component_fn, props=None, key=None):
        self.component_fn = component_fn
        self.props = props or {}
        self.key = key


def component(fn=None, *, transparent: bool = False):
    """Turn a function into a component (``@component``).

    ``@component(transparent=True)`` marks a *provider-like* component: it
    has its own path, but contributes no segment to its descendants' paths
    (see :attr:`HookContext.path`). Use it for components that only provide
    context / wrap ``children``, so inserting or removing them later does not
    change the address of the components below (durable state, traces).
    """
    if fn is None:
        return lambda f: component(f, transparent=transparent)

    @wraps(fn)
    def wrapper(*, key=None, __internal=False, **props):
        if __internal:
            return fn(**props)
        return VNode(wrapper, props=props, key=key)

    wrapper._pyreact_transparent = bool(transparent)
    return wrapper
