# provider.py ----------------------------------------------------
from functools import wraps
from weakref import WeakSet
from pyreact.core.core import component, hooks
from pyreact.core.runtime import schedule_rerender

# Strong references to every created context (kept for introspection).
_CONTEXT_REGISTRY = []


class _ContextKey:
    """Identity key for a context; holds the global (provider-less) value."""

    __slots__ = ("name", "value")

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def get(self):
        return self.value

    def __repr__(self):
        return f"<ContextKey {self.name!r}>"


def provider(ctx_key, *, prop="value"):
    def decorator(body_fn):
        # Providers are transparent: they don't add a segment to the paths of
        # the components below them (see HookContext.path).
        @component(transparent=True)
        @wraps(body_fn)
        def wrapper(*, key=None, **props):
            try:
                value = props.pop(prop)
            except KeyError:
                raise TypeError(f"Provider missing required prop '{prop}'")

            # Scope the value to this provider's subtree (resolved by
            # descendants walking up their parent chain in use_context).
            hooks.provide(ctx_key, value)
            return body_fn(**props)

        return wrapper

    return decorator


def create_context(*, default=None, name="Context", prop="value"):
    """Create a new, independent context.

    - ``Ctx(value=..., children=[...])`` provides a value to a subtree.
    - ``hooks.use_context(Ctx)`` reads the nearest provided value, or the
      context's global value when no provider is mounted above.
    - ``Ctx.set(v)`` updates the global value and rerenders subscribers;
      it does not override values given by a mounted provider.
    """
    ctx_key = _ContextKey(name, default)
    subs_set = WeakSet()

    @provider(ctx_key, prop=prop)
    def _Provider(**props):
        return props.get("children", [])

    class _Context:
        # used by use_context / unmount
        _ctx = ctx_key
        _subs = subs_set

        @staticmethod
        def _subscribe(hctx):
            subs_set.add(hctx)

        @staticmethod
        def get():
            return ctx_key.value

        @staticmethod
        def set(value):
            ctx_key.value = value
            for hctx in list(subs_set):
                if getattr(hctx, "_mounted", True):
                    schedule_rerender(hctx, reason=f"context {name} set")

        def __call__(self, **props):
            return _Provider(**props)

        def __repr__(self):
            return f"<Context {name!r} default={default!r}>"

    context_instance = _Context()
    _CONTEXT_REGISTRY.append(context_instance)
    return context_instance
