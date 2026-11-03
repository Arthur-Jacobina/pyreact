# hook.py ----------------------------------------------------
from typing import Dict, Optional
from weakref import WeakSet
from . import core as _core
from .core import VNode
from .runtime import schedule_rerender, report_error, spawn, current_owner
import inspect
import warnings

_MISSING = object()


def _deps_key(deps):
    # None -> "run/recompute on every render" (React semantics); [] -> () once.
    return None if deps is None else tuple(deps)


def _deps_changed(old, new):
    return new is None or old != new


def _resolve_service(services: dict, key, factory):
    service = services.get(key)
    if service is None:
        service = factory()
        services[key] = service
    return service


class _ServiceAccessor:
    """Descriptor making ``get_service`` scope-aware.

    - ``ctx.get_service(key, factory)`` (instance call, which is what
      ``hooks.get_service`` resolves to) → services of ``ctx``'s *session
      root* (the top-most ancestor). Each session root owns its own dict, so
      two sessions never share an InputBus, NavService, EventBus, ...
      Roots created without a services dict (headless) use the global one.
    - ``HookContext.get_service(key, factory)`` (class call) → the
      process-global fallback dict ``HookContext._services`` (e.g. the
      stdout ``console_buffer``).
    """

    def __get__(self, obj, objtype=None):
        if obj is None:
            owner = objtype

            def _global_get_service(key, factory):
                return _resolve_service(owner._services, key, factory)

            return _global_get_service

        def _session_get_service(key, factory):
            return _resolve_service(obj.session_services, key, factory)

        return _session_get_service


class HookContext:
    # Process-global services (class-level ``HookContext.get_service`` calls).
    _services: Dict = {}

    get_service = _ServiceAccessor()

    def __init__(
        self,
        name,
        component_fn,
        *,
        props=None,
        key=None,
        parent=None,
        index=None,
        services: Optional[dict] = None,
    ) -> None:
        """``services``: only meaningful for a root (``parent is None``); the
        dict that backs ``hooks.get_service`` for the whole tree (each
        :class:`~pyreact.boot.session.Session` passes its own). A root created
        without one (headless use) resolves services from the process-global
        ``HookContext._services``."""
        self.name = name
        self.parent = parent
        # Position among the parent's rendered children (used in ``path`` for
        # unkeyed nodes; unkeyed children are reconciled by index too).
        self.index = index
        # Session root of this tree; parents never change after creation.
        self._root: "HookContext" = parent._root if parent is not None else self
        self._session_services: Optional[dict] = (
            services if parent is None else None
        )
        # Values provided to descendants by this node: {context_key: value}
        self._provided: dict = {}
        self.component_fn = component_fn
        self.props = props or {}
        self.key = key

        self.hooks: list = []
        self.effects: list = []
        self.children: list["HookContext"] = []
        self.hook_idx: int = 0
        self._ctx_subs: list[WeakSet] = []
        self._effect_slots: set[int] = set()
        self._mounted: bool = (
            True  # Track mount lifecycle to avoid rerenders after unmount
        )

    @property
    def root(self) -> "HookContext":
        """The session root (top-most ancestor) of this component."""
        return self._root

    @property
    def session_services(self) -> dict:
        """Services dict owned by this tree's session root (the process-global
        dict for a root created without ``services``)."""
        services = self._root._session_services
        return HookContext._services if services is None else services

    def use_state(self, initial):
        idx = self.hook_idx
        if idx >= len(self.hooks):
            self.hooks.append(initial)

        def set_state(val):
            nonlocal idx
            if not getattr(
                self, "_mounted", True
            ):  # Ignore state updates after unmount
                return
            if callable(val):
                val = val(self.hooks[idx])

            if val != self.hooks[idx]:
                self.hooks[idx] = val
                schedule_rerender(self, reason=f"use_state[{idx}] set -> {val}")

        self.hook_idx += 1
        return self.hooks[idx], set_state

    def use_reducer(self, reducer, initial, *, init_fn=None, deps=None):
        """
        Semantics similar to React.useReducer:
        - reducer(state, action) -> new_state
        - initial: initial state (used only on the first mount, unless 'deps' is provided)
        - init_fn(optional): lazy initializer init_fn(initial) -> state
        - deps(optional): if provided, when changed, the state is REINITIALIZED with init_fn(initial) or initial.
        """
        deps_key = None if deps is None else tuple(deps)  # [] → () (immutable object)
        idx = self.hook_idx

        if idx >= len(self.hooks):  # first mount
            state0 = init_fn(initial) if init_fn is not None else initial
            self.hooks.append((state0, reducer, deps_key))
        else:  # updates
            state, old_reducer, old_deps = self.hooks[idx]

            if (
                deps_key is not None and old_deps != deps_key
            ):  # Reinit due to deps change (optional)
                state = init_fn(initial) if init_fn is not None else initial
                self.hooks[idx] = (state, reducer, deps_key)
            else:
                if (
                    old_reducer is not reducer or old_deps != deps_key
                ):  # Update reducer reference (dispatch uses the current reducer)
                    self.hooks[idx] = (
                        state,
                        reducer,
                        deps_key if deps is not None else old_deps,
                    )

        def dispatch(action):
            nonlocal idx
            # Ignore dispatch after unmount
            if not getattr(self, "_mounted", True):
                return
            s, r, dkey = self.hooks[idx]
            new_state = r(s, action)
            if new_state != s:
                self.hooks[idx] = (new_state, r, dkey)

                schedule_rerender(
                    self, reason=f"use_reducer[{idx}] dispatch {action} -> {new_state}"
                )

        state, _r, _d = self.hooks[idx]
        self.hook_idx += 1
        return state, dispatch  # return the pair (state, dispatch)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def path(self) -> str:
        """Stable tree path, e.g. ``App/ToolProvider[tools]/StoreHome[store]``.

        Each segment is ``Name[key]`` (or ``Name[#index]`` when unkeyed); the
        root is just its name. A stable address for a component (e.g. for
        durable state or tracing).
        """
        cached = self.__dict__.get("_path")
        if cached is None:
            if self.key is not None:
                seg = f"{self.name}[{str(self.key).replace('/', '%2F')}]"
            elif self.index is not None:
                seg = f"{self.name}[#{self.index}]"
            else:
                seg = self.name
            cached = seg if self.parent is None else f"{self.parent.path}/{seg}"
            self._path = cached
        return cached

    def use_effect(self, effect_fn, deps):
        deps_key = _deps_key(deps)
        idx = self.hook_idx

        if idx >= len(self.hooks):  # first mount
            self.hooks.append((None, deps_key))
            self.effects.append((effect_fn, deps_key, idx))
            self._effect_slots.add(idx)
        else:  # updates
            old_cleanup, old_deps = self.hooks[idx]
            if _deps_changed(old_deps, deps_key):  # deps changed (or no deps)
                self.effects.append((effect_fn, deps_key, idx))
                self.hooks[idx] = (old_cleanup, deps_key)

        self.hook_idx += 1

    def use_callback(self, fn, deps=None):
        deps_key = _deps_key(deps)
        idx = self.hook_idx

        if idx >= len(self.hooks):  # first time
            self.hooks.append((fn, deps_key))
        else:
            cached_fn, old_deps = self.hooks[idx]
            if _deps_changed(old_deps, deps_key):
                self.hooks[idx] = (fn, deps_key)  # deps changed -> new fn
            else:
                fn = cached_fn  # use memo

        self.hook_idx += 1
        return fn

    def use_memo(self, factory, deps=None):
        deps_key = _deps_key(deps)
        idx = self.hook_idx

        if idx >= len(self.hooks):  # first time
            self.hooks.append((factory(), deps_key))
        else:
            value, old_key = self.hooks[idx]
            if _deps_changed(old_key, deps_key):
                value = factory()  # deps changed -> new value
                self.hooks[idx] = (value, deps_key)

        self.hook_idx += 1
        return self.hooks[idx][0]

    def provide(self, ctx_like, value):
        """Expose ``value`` for ``ctx_like`` to this node's descendants."""
        key = getattr(ctx_like, "_ctx", ctx_like)
        self._provided[key] = value

    def _lookup_context(self, ctx_like):
        key = getattr(ctx_like, "_ctx", ctx_like)
        node = self.parent
        while node is not None:
            value = node._provided.get(key, _MISSING)
            if value is not _MISSING:
                return value
            node = node.parent
        # No provider above us: fall back to the context's global value.
        getter = getattr(ctx_like, "get", None) or getattr(key, "get")
        return getter()

    def use_context(self, ctx_like):
        subscribe = getattr(ctx_like, "_subscribe", None)
        if subscribe is not None:
            subscribe(self)
            subs_set = getattr(ctx_like, "_subs")

            if subs_set not in self.__dict__.setdefault(
                "_ctx_subs", []
            ):  # keep reference to run on unmount
                self._ctx_subs.append(subs_set)

        idx = self.hook_idx
        # Resolve against the nearest provider among our ancestors. This is
        # independent of render order / task, so an isolated rerender of a
        # deep child sees the same value as a full rerender.
        value = self._lookup_context(ctx_like)

        if idx >= len(self.hooks):
            self.hooks.append(value)
        else:
            self.hooks[idx] = value

        self.hook_idx += 1
        return value

    def _call_cleanup(self, cleanup):
        if not cleanup:
            return
        # Like effects, cleanups run "as" this component (hooks.* usable).
        owner_token = current_owner.set(self)
        stack_token = _core._context_stack.set(self)
        try:
            res = cleanup()
            if inspect.isawaitable(res):
                spawn(res, owner=self, name=f"{self.name}:cleanup")
        except Exception as exc:
            report_error(exc, self, "cleanup")
        finally:
            _core._context_stack.reset(stack_token)
            current_owner.reset(owner_token)

    def _run_cleanup_slot(self, slot):
        cleanup = slot[0] if isinstance(slot, tuple) else None
        self._call_cleanup(cleanup)

    def unmount(self):
        for idx in list(getattr(self, "_effect_slots", set())):
            if idx < len(self.hooks):
                self._run_cleanup_slot(self.hooks[idx])

        # 2. Remove Context subscriptions
        for ws in getattr(self, "_ctx_subs", []):
            ws.discard(self)

        if hasattr(self, "_ctx_subs"):
            self._ctx_subs.clear()

        # 3. Unmount children recursively
        for child in self.children:
            child.unmount()

        # 4. GC
        self.children.clear()
        self.hooks.clear()
        self.effects.clear()
        if hasattr(self, "_effect_slots"):
            self._effect_slots.clear()
        # mark as unmounted to skip future rerenders
        self._mounted = False

    def render(self):
        import pyreact.core.core as core

        try:
            from .debug import enter_render, exit_render
        except Exception:

            def enter_render(*_args, **_kw):  # type: ignore
                return None

            def exit_render(*_args, **_kw):  # type: ignore
                return None

        token = core._context_stack.set(self)

        try:
            _depth_token = enter_render(self)
            self.hook_idx = 0
            self.effects = []

            # 1. store old children and start a new empty list
            old_children = self.children
            self.children = []

            # 2. execute component function. On failure keep the previous
            #    subtree mounted (last good render) instead of leaking it.
            try:
                output = self.component_fn(__internal=True, **self.props)
            except BaseException:
                self.children = old_children
                raise
            vnodes = output if isinstance(output, list) else [output]

            # 3. reconciliation – reuse or create child contexts
            for idx, vnode in enumerate(vnodes):
                if not isinstance(vnode, VNode):
                    continue

                vnode_key = vnode.key if vnode.key is not None else f"__idx_{idx}"

                # 4. search for match in old_children
                matched = next(
                    (
                        c
                        for c in old_children
                        if (
                            c.key
                            if c.key is not None
                            else f"__idx_{old_children.index(c)}"
                        )
                        == vnode_key
                        and c.component_fn is vnode.component_fn
                    ),
                    None,
                )

                # 5. warn if there are duplicate siblings without keys
                if vnode.key is None:
                    dup = any(
                        (c.component_fn is vnode.component_fn and c.key is None)
                        for c in self.children
                    )
                    if dup:
                        warnings.warn(
                            f"\n\n⚠️ [HookContext] Sibling <{vnode.component_fn.__name__}> with no explicit 'key'; it can cause extra re-render.",
                            RuntimeWarning,
                            stacklevel=2,
                        )

                if matched is None:
                    matched = HookContext(
                        vnode.component_fn.__name__,
                        vnode.component_fn,
                        props=vnode.props,
                        key=vnode.key,
                        parent=self,
                        index=idx,
                    )
                else:
                    matched.props = vnode.props

                self.children.append(matched)

            # 6. recursively unmount orphans
            for orphan in old_children:
                if orphan not in self.children:
                    orphan.unmount()

            # 7. recursively render current children; a failing child must
            #    not take down its siblings (per-component error boundary).
            for child in self.children:
                try:
                    child.render()
                except Exception as exc:
                    report_error(exc, child, "render")
        finally:
            try:
                exit_render(_depth_token)
            except Exception:
                pass
            # 8. restore previous component and original stack
            core._context_stack.reset(token)

    async def run_effects(self):
        effects, self.effects = self.effects, []
        for fx, deps, idx in effects:
            if not self._mounted or idx >= len(self.hooks):
                break
            cln, _ = self.hooks[idx]
            self.hooks[idx] = (None, deps)
            # Attribute work done by the effect (spawned tasks, stdout) to
            # this component and hence to its session, and make ``hooks.*``
            # (e.g. ``hooks.get_service``) resolve to this component inside
            # effects, their cleanups and async effect tasks.
            owner_token = current_owner.set(self)
            stack_token = _core._context_stack.set(self)
            try:
                self._call_cleanup(cln)
                try:
                    res = fx()
                except Exception as exc:
                    report_error(exc, self, "effect")
                    continue
                if inspect.isawaitable(res):
                    # Async effects run in the background so a slow effect (LLM
                    # call, tool, I/O) never blocks other renders. The stored
                    # cleanup cancels the task if it is still running when deps
                    # change or the component unmounts.
                    task = spawn(res, owner=self, name=f"{self.name}:effect[{idx}]")
                    res = self._make_task_cleanup(task)
            finally:
                _core._context_stack.reset(stack_token)
                current_owner.reset(owner_token)
            self.hooks[idx] = ((res if callable(res) else None), deps)

        for ch in list(self.children):
            try:
                await ch.run_effects()
            except Exception as exc:
                report_error(exc, ch, "effect")

    def _make_task_cleanup(self, task):
        def _cleanup():
            if not task.done():
                task.cancel()
                return None
            if task.cancelled() or task.exception() is not None:
                return None
            inner = task.result()
            return inner() if callable(inner) else None

        return _cleanup

    # FOR DEBUGGING
    def render_tree(self, indent=0):
        from .debug import render_tree as _render_tree

        _render_tree(self, indent)
