# hooks.py ----------------------------------------------------
"""The best of DSPy, the React way.

    Module   -> component memory      use_module / use_predict / use_chain_of_thought
    call     -> effect                use_call (async, cancellable, tracked by runtime.spawn)
    stream   -> effect + rerenders    use_stream (dspy.streamify + StreamListener)
    ReAct    -> capability-using agent use_agent (tools from ToolProvider)
    LM/adapter/settings -> context    DSPyProvider (applied per call via dspy.context)
    everything observable -> events   llm.* / tool.* on the EventBus service

Example::

    @component
    def QA(question):
        qa = use_chain_of_thought("question -> answer")
        call = use_call(qa, model="fast")
        hooks.use_effect(lambda: call.run(question=question), [question])
        if call.loading:
            return [Message(text="thinking…")]
        return [Message(text=call.result.answer)] if call.result else []
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import dspy

from pyreact.core.core import _context_stack, hooks
from pyreact.core.runtime import schedule_rerender, spawn
from pyreact.tools import Tool, resolve_tools, use_tools

from . import telemetry as tm
from .dspy_integration import DSPyEnv, _mk_module_key, use_dspy_env

__all__ = [
    "Call",
    "StreamCall",
    "use_module",
    "use_predict",
    "use_chain_of_thought",
    "use_call",
    "use_stream",
    "use_agent",
    "to_dspy_tool",
    "invoke_module",
]

POLICIES = ("replace", "parallel", "ignore")


# ---------------------------------------------------------------------------
# Modules = component memory
# ---------------------------------------------------------------------------
def _sig_key(signature) -> str:
    return signature if isinstance(signature, str) else signature.__name__


def use_module(
    signature_or_module,
    module_cls=dspy.Predict,
    *,
    name: Optional[str] = None,
    deps: Optional[list] = None,
    compile_with_optimizer: bool = False,
    _key: Optional[Tuple] = None,
    **kwargs,
) -> Any:
    """Memoized DSPy module (``useMemo`` for programs).

    - Passing a module instance returns it unchanged (already "memory").
    - Otherwise builds ``module_cls(signature, **kwargs)`` once and caches it
      in ``env.caches`` under ``(module_cls.__name__, signature, name)``, where
      ``name`` defaults to this component's path — so each component instance
      owns its module unless you share it explicitly by ``name``.
    - If ``env.compiled`` holds a compiled program for that key (optimizer,
      GEPA, ...) it is returned instead, transparently.
    - ``deps``: rebuild when changed (e.g. new demos / instructions).
    """
    if isinstance(signature_or_module, dspy.Module):
        return signature_or_module

    env = use_dspy_env()
    ctx = _context_stack.get()
    if _key is None:
        scope = name if name is not None else tm.component_path(ctx)
        key = _mk_module_key(module_cls, signature_or_module, scope)
        if kwargs:
            key = key + (repr(sorted(kwargs.items())),)
    else:
        key = _key
    deps_key = tuple(deps) if deps is not None else None

    def _factory():
        cached = env.caches.get(key)
        if cached is not None and deps is None:
            return cached
        mod = module_cls(signature_or_module, **kwargs)
        mod._pyreact_name = name or f"{module_cls.__name__}({_sig_key(signature_or_module)})"
        env.caches[key] = mod
        return mod

    mod = hooks.use_memo(_factory, deps=[key, deps_key, id(env.caches)])

    if compile_with_optimizer and env.optimizer is not None:
        _use_compile(env, key, mod, ctx)

    return env.compiled.get(key, mod)


def _use_compile(env: DSPyEnv, key, mod, ctx) -> None:
    def kick_compile():
        if key in env.compiled or key in env.compiling:
            return None
        env.compiling.add(key)

        async def _compile():
            try:
                def work():
                    # Worker thread: apply the env's LM/adapter explicitly
                    # (no global dspy.configure).
                    with env.context():
                        return env.optimizer.compile(mod)

                compiled = await asyncio.to_thread(work)
                env.compiled[key] = compiled
                env.caches[key] = compiled
            except Exception as exc:  # keep the uncompiled module
                tm.publish("error", {"phase": "compile", "key": repr(key), "error": repr(exc)})
                return
            finally:
                env.compiling.discard(key)
            if ctx is not None and getattr(ctx, "_mounted", False):
                schedule_rerender(ctx, reason="dspy module compiled")

        spawn(_compile(), owner=ctx, name=f"dspy_compile:{key}")
        return None

    hooks.use_effect(kick_compile, [key])


def use_predict(signature, *, name: Optional[str] = None, **kwargs):
    return use_module(signature, dspy.Predict, name=name, **kwargs)


def use_chain_of_thought(signature, *, name: Optional[str] = None, **kwargs):
    return use_module(signature, dspy.ChainOfThought, name=name, **kwargs)


# ---------------------------------------------------------------------------
# Calls = effects
# ---------------------------------------------------------------------------
def _module_label(module, name: Optional[str]) -> str:
    if name:
        return name
    label = getattr(module, "_pyreact_name", None)
    if label:
        return label
    sig = getattr(module, "signature", None)
    sig_name = getattr(sig, "__name__", None)
    cls = type(module).__name__
    return f"{cls}({sig_name})" if sig_name else cls


async def _plain_call(module, inputs: dict):
    acall = getattr(module, "acall", None)
    if acall is not None and inspect.iscoroutinefunction(acall):
        return await acall(**inputs)
    apredict = getattr(module, "apredict", None)
    if apredict is not None and inspect.iscoroutinefunction(apredict):
        return await apredict(**inputs)
    predict = getattr(module, "predict", None)
    if callable(predict):
        return await asyncio.to_thread(predict, **inputs)
    if callable(module):
        return await asyncio.to_thread(module, **inputs)
    raise AttributeError("DSPy module has no acall/apredict/predict/__call__ to invoke")


async def _stream_call(module, inputs: dict, fields: Sequence[str], on_chunk, on_status):
    """Run via dspy.streamify; returns the final Prediction.

    Falls back to a plain call when the module can't be streamed (not a
    dspy.Module, unknown field...). If the LM doesn't stream, no chunks
    arrive and only the final Prediction is returned — callers then fill
    the partial text from it.
    """
    from dspy.streaming import StatusMessage, StreamListener, StreamResponse

    program = None
    if isinstance(module, dspy.Module) and fields:
        try:
            listeners = [StreamListener(signature_field_name=f) for f in fields]
            program = dspy.streamify(module, stream_listeners=listeners, is_async_program=True)
        except Exception:
            program = None
    if program is None:
        return await _plain_call(module, inputs)

    final = None
    async with contextlib.aclosing(program(**inputs)) as agen:
        async for item in agen:
            if isinstance(item, StreamResponse):
                on_chunk(item.signature_field_name, item.chunk, item.is_last_chunk)
            elif isinstance(item, StatusMessage):
                on_status(item.message)
            elif isinstance(item, dspy.Prediction):
                final = item
    return final


async def invoke_module(
    module,
    inputs: dict,
    *,
    lm: Any = None,
    adapter: Any = None,
    name: Optional[str] = None,
    component: Optional[str] = None,
    turn_id: Optional[str] = None,
    stream_fields: Optional[Sequence[str]] = None,
    on_chunk: Optional[Callable[[str, str, bool], None]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Run one module call with per-call settings and publish llm.* events.

    Returns ``(result, usage)``. Usable outside components too (scripts,
    evaluation). On failure the exception gets a ``usage`` attribute.
    """
    if lm is None and getattr(dspy.settings, "lm", None) is None:
        raise RuntimeError(
            "No language model configured: mount DSPyProvider(lm=... or models={'default': ...}) "
            "or pass lm=... to the hook."
        )
    label = _module_label(module, name)
    call_id = tm.new_call_id()
    if turn_id is None:
        turn_id = tm.current_turn_id()
    model = tm.model_name(lm if lm is not None else dspy.settings.lm)
    base = {"call_id": call_id, "module": label, "component": component, "model": model}
    pub = functools.partial(tm.publish, turn_id=turn_id, source=component)

    recorder = tm.UsageRecorder()
    overrides: Dict[str, Any] = {
        "caller_modules": list(dspy.settings.caller_modules or []) + [recorder]
    }
    if lm is not None:
        overrides["lm"] = lm
    if adapter is not None:
        overrides["adapter"] = adapter

    def _chunk(field_name, chunk, last):
        pub("llm.chunk", {"call_id": call_id, "module": label, "field": field_name,
                          "chunk": chunk, "last": last})
        if on_chunk is not None:
            on_chunk(field_name, chunk, last)

    def _status(message):
        pub("llm.status", {"call_id": call_id, "module": label, "message": message})

    pub("llm.start", {**base, "inputs": tm.brief(inputs), "stream": bool(stream_fields)})
    token = tm.enter_scope(tm.CallScope(call_id, label, component, turn_id))
    t0 = time.perf_counter()
    try:
        with dspy.context(**overrides):
            if stream_fields:
                result = await _stream_call(module, inputs, stream_fields, _chunk, _status)
            else:
                result = await _plain_call(module, inputs)
    except asyncio.CancelledError:
        pub("llm.cancel", {**base, "latency": time.perf_counter() - t0})
        raise
    except Exception as exc:
        usage = recorder.summary()
        pub("llm.error", {**base, "latency": time.perf_counter() - t0, "usage": usage,
                          "cost": usage["cost"], "error": repr(exc)})
        try:
            exc.usage = usage
        except Exception:
            pass
        raise
    finally:
        tm.exit_scope(token)
    usage = recorder.summary()
    usage["latency"] = time.perf_counter() - t0
    pub("llm.end", {**base, "latency": usage["latency"], "usage": usage,
                    "cost": usage["cost"], "outputs": tm.outputs_of(result)})
    return result, usage


@dataclass(frozen=True)
class Call:
    """Snapshot of a call hook for this render.

    ``run(**inputs)`` fires the call (returns None — safe inside effects);
    ``submit(**inputs)`` does the same and returns the asyncio.Task.
    ``ver`` bumps only when a call settles, so ``use_effect(..., [call.ver])``
    fires exactly once per completed call.
    """

    run: Callable[..., None]
    submit: Callable[..., Optional[asyncio.Task]]
    cancel: Callable[[], None]
    result: Any = None
    ver: int = 0
    loading: bool = False
    error: Optional[BaseException] = None
    usage: Optional[Dict[str, Any]] = None

    def __iter__(self):  # legacy tuple-unpacking: run, (result, ver), loading, error
        return iter((self.run, (self.result, self.ver), self.loading, self.error))


@dataclass(frozen=True)
class StreamCall(Call):
    partial: Dict[str, str] = field(default_factory=dict)
    fields: Tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """Partial text of the first listened field (final value once settled)."""
        f = self.fields[0] if self.fields else None
        if f is None:
            return ""
        if f in self.partial:
            return self.partial[f]
        return str(getattr(self.result, f, "") or "") if self.result is not None else ""


def _reducer(state, action):
    typ = action["type"]
    if typ == "start":
        return {**state, "loading": True, "error": None, "partial": {}}
    if typ == "loading":
        return {**state, "loading": action["loading"]}
    if typ == "partial":
        return {**state, "partial": action["partial"]}
    if typ == "success":
        return {**state, "loading": action["loading"], "result": action["result"],
                "error": None, "ver": state["ver"] + 1, "usage": action["usage"],
                "partial": action.get("partial", state["partial"])}
    if typ == "error":
        return {**state, "loading": action["loading"], "result": None,
                "error": action["error"], "ver": state["ver"] + 1,
                "usage": action["usage"]}
    return state


_INITIAL = {"loading": False, "result": None, "error": None, "ver": 0, "usage": None,
            "partial": {}}


def _use_call_machine(
    module,
    *,
    model: Optional[str],
    lm: Optional[Any],
    policy: str,
    name: Optional[str],
    turn_id: Optional[str],
    stream_fields: Optional[Tuple[str, ...]] = None,
    throttle: float = 0.05,
):
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")

    state, dispatch = hooks.use_reducer(_reducer, _INITIAL)
    ref = hooks.use_memo(lambda: {"alive": True, "tasks": set(), "latest": None}, [])
    ctx = _context_stack.get()
    env = hooks.use_context(_dspy_ctx())  # optional: works without a provider if lm=...

    # Latest config read at run time (stable run/cancel, never stale).
    ref["cfg"] = {
        "module": module,
        "lm": env.lm_for(model, lm) if env is not None else lm,
        "adapter": env.adapter if env is not None else None,
        "env": env,
        "policy": policy,
        "name": name,
        "turn_id": turn_id,
        "component": tm.component_path(ctx),
        "stream_fields": stream_fields,
        "throttle": throttle,
    }

    def _mount():
        ref["alive"] = True

        def _unmount():
            ref["alive"] = False
            for t in list(ref["tasks"]):
                if not t.done():
                    t.cancel()

        return _unmount

    hooks.use_effect(_mount, [])

    def _current(task) -> bool:
        if not ref["alive"]:
            return False
        return ref["cfg"]["policy"] == "parallel" or ref["latest"] is task

    async def _do(inputs: dict, cfg: dict, turn: Optional[str]):
        me = asyncio.current_task()
        loop = asyncio.get_running_loop()
        partial: Dict[str, str] = {}
        flush = {"last": 0.0, "handle": None}

        def _flush():
            flush["handle"] = None
            flush["last"] = loop.time()
            if _current(me):
                dispatch({"type": "partial", "partial": dict(partial)})

        def _on_chunk(field_name, chunk, last):
            partial[field_name] = partial.get(field_name, "") + chunk
            wait = cfg["throttle"] - (loop.time() - flush["last"])
            if wait <= 0:
                _flush()
            elif flush["handle"] is None:
                flush["handle"] = loop.call_later(wait, _flush)

        if cfg["env"] is None and cfg["lm"] is None and getattr(dspy.settings, "lm", None) is None:
            err: Optional[Exception] = RuntimeError(
                "DSPy context not available: mount DSPyProvider or pass lm=... to the hook."
            )
            result, usage = None, None
            ref["tasks"].discard(me)
        else:
            err = None
            try:
                result, usage = await invoke_module(
                    cfg["module"], inputs, lm=cfg["lm"], adapter=cfg["adapter"],
                    name=cfg["name"], component=cfg["component"], turn_id=turn,
                    stream_fields=cfg["stream_fields"],
                    on_chunk=_on_chunk if cfg["stream_fields"] else None,
                )
            except Exception as exc:
                result, usage, err = None, getattr(exc, "usage", None), exc
            finally:
                if flush["handle"] is not None:
                    flush["handle"].cancel()
                ref["tasks"].discard(me)
        if not _current(me):
            return
        still = any(not t.done() for t in ref["tasks"])
        if err is not None:
            dispatch({"type": "error", "error": err, "usage": usage, "loading": still})
            return
        if cfg["stream_fields"]:
            # Non-streaming LM (or cache hit): show the final values.
            for f in cfg["stream_fields"]:
                val = getattr(result, f, None) if result is not None else None
                if val is not None:
                    partial[f] = str(val)
        dispatch({"type": "success", "result": result, "usage": usage, "loading": still,
                  "partial": dict(partial)})

    def _submit(**inputs) -> Optional[asyncio.Task]:
        cfg = ref["cfg"]
        pol = cfg["policy"]
        inflight = [t for t in ref["tasks"] if not t.done()]
        if pol == "ignore" and inflight:
            return None
        if pol == "replace":
            for t in inflight:
                t.cancel()  # superseded: don't keep paying for a stale LLM call
            ref["tasks"].clear()
        turn = cfg["turn_id"] if cfg["turn_id"] is not None else tm.current_turn_id()
        dispatch({"type": "start"})
        task = spawn(
            _do(inputs, cfg, turn), owner=ctx, turn=turn, name=f"dspy_call:{tm.component_path(ctx)}"
        )
        ref["latest"] = task
        ref["tasks"].add(task)
        return task

    def _run(**inputs) -> None:
        _submit(**inputs)

    def _cancel() -> None:
        for t in list(ref["tasks"]):
            if not t.done():
                t.cancel()
        ref["tasks"].clear()
        ref["latest"] = None
        dispatch({"type": "loading", "loading": False})

    submit = hooks.use_callback(_submit, deps=[])
    run = hooks.use_callback(_run, deps=[])
    cancel = hooks.use_callback(_cancel, deps=[])
    common = dict(
        run=run, submit=submit, cancel=cancel, result=state["result"], ver=state["ver"],
        loading=state["loading"], error=state["error"], usage=state["usage"],
    )
    if stream_fields is not None:
        return StreamCall(**common, partial=state["partial"], fields=tuple(stream_fields))
    return Call(**common)


def _dspy_ctx():
    from .dspy_integration import DSPyContext

    return DSPyContext


def use_call(
    module,
    *,
    model: Optional[str] = None,
    lm: Optional[Any] = None,
    policy: str = "replace",
    name: Optional[str] = None,
    turn_id: Optional[str] = None,
) -> Call:
    """Call a DSPy module as an effect.

    ``policy``: ``"replace"`` (new call cancels the in-flight one; default),
    ``"parallel"`` (all run; each settle bumps ``ver``), ``"ignore"`` (drop
    new calls while one is in flight).
    """
    return _use_call_machine(module, model=model, lm=lm, policy=policy, name=name,
                             turn_id=turn_id)


def use_stream(
    module,
    *,
    listen: Iterable[str] = ("answer",),
    model: Optional[str] = None,
    lm: Optional[Any] = None,
    policy: str = "replace",
    name: Optional[str] = None,
    turn_id: Optional[str] = None,
    throttle: float = 0.05,
) -> StreamCall:
    """Like :func:`use_call`, but rerenders with partial text while streaming.

    ``call.partial[field]`` / ``call.text`` grow as chunks arrive (rerenders
    throttled to one per ``throttle`` seconds); ``llm.chunk`` events are
    published for every chunk. LMs that don't stream simply produce the
    final text at once.
    """
    if isinstance(listen, str):
        listen = (listen,)
    return _use_call_machine(module, model=model, lm=lm, policy=policy, name=name,
                             turn_id=turn_id, stream_fields=tuple(listen), throttle=throttle)


# ---------------------------------------------------------------------------
# Tools = capabilities
# ---------------------------------------------------------------------------
def _instrumented(tool: Tool) -> Callable:
    fn = tool.fn
    is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)
    )

    @functools.wraps(fn)
    async def _tool(*args, **kwargs):
        scope = tm.current_scope()
        call_id = scope.call_id if scope else None
        turn_id = scope.turn_id if scope else tm.current_turn_id()
        source = scope.component if scope else None
        args_preview = tm.brief(kwargs if not args else {"args": args, **kwargs})
        tm.publish("tool.start", {"call_id": call_id, "tool": tool.name, "args": args_preview},
                   turn_id=turn_id, source=source)
        t0 = time.perf_counter()
        error = None
        result = None
        try:
            if is_async:
                result = await fn(*args, **kwargs)
            else:
                # Sync tools run in a worker thread: never block the render loop.
                result = await asyncio.to_thread(fn, *args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            return result
        except asyncio.CancelledError:
            error = "cancelled"
            raise
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            tm.publish("tool.end", {"call_id": call_id, "tool": tool.name, "args": args_preview,
                                    "duration": time.perf_counter() - t0, "error": error,
                                    "result": tm.brief(result) if error is None else None},
                       turn_id=turn_id, source=source)

    return _tool


def to_dspy_tool(tool: Union[Tool, Callable]) -> dspy.Tool:
    """pyreact Tool -> dspy.Tool (async, thread-offloaded, emits tool.* events)."""
    if not isinstance(tool, Tool):
        from pyreact.tools import make_tool

        tool = make_tool(tool)
    # Pass the core-derived JSON Schema as DSPy 'args' so DSPy uses our schema
    # instead of re-introspecting the callable.
    properties = tool.parameters.get("properties", {})
    arg_desc = {
        n: prop["description"]
        for n, prop in properties.items()
        if isinstance(prop, dict) and prop.get("description")
    }
    return dspy.Tool(
        _instrumented(tool),
        name=tool.name,
        desc=tool.description,
        args=properties or None,
        arg_desc=arg_desc or None,
    )


def use_agent(
    signature,
    *,
    tools: Optional[Iterable[Union[Tool, Callable, str]]] = None,
    max_iters: int = 20,
    model: Optional[str] = None,
    lm: Optional[Any] = None,
    name: Optional[str] = None,
    policy: str = "replace",
    turn_id: Optional[str] = None,
) -> Call:
    """ReAct agent over tools (Tool, callables, or ToolProvider names).

    Sync tools run in a worker thread; every tool call publishes
    ``tool.start`` / ``tool.end``. Returns a :class:`Call`.
    """
    registry = use_tools()
    resolved = resolve_tools(tools, registry)
    tools_key = tuple((t.name, id(t.fn)) for t in resolved)
    env = hooks.use_context(_dspy_ctx())
    ctx = _context_stack.get()
    key = ("ReAct", _sig_key(signature), name if name is not None else tm.component_path(ctx))

    def _build():
        react = dspy.ReAct(signature, tools=[to_dspy_tool(t) for t in resolved],
                           max_iters=max_iters)
        react._pyreact_name = name or f"ReAct({_sig_key(signature)})"
        if env is not None:
            env.caches[key] = react
        return react

    # Rebuild only when signature, tool set, limits, or name change.
    react = hooks.use_memo(_build, deps=[key, tools_key, max_iters])
    return use_call(react, model=model, lm=lm, policy=policy, name=name, turn_id=turn_id)
