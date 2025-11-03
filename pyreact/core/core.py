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


class VNode:
    def __init__(self, component_fn, props=None, key=None):
        self.component_fn = component_fn
        self.props = props or {}
        self.key = key


def component(fn):
    @wraps(fn)
    def wrapper(*, key=None, __internal=False, **props):
        if __internal:
            return fn(**props)
        return VNode(wrapper, props=props, key=key)

    return wrapper
