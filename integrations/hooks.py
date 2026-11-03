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
from pyreact.core.harness import Blocked, Boundary, Harness, HarnessContext, resolve_harness, through
from pyreact.core.runtime import schedule_rerender, spawn
from pyreact.tools import Tool, resolve_tools, use_tools

from . import telemetry as tm
from .dspy_integration import DSPyEnv

__all__ = [
    "Call",
    "StreamCall",
    "use_module",
    "use_predict",
    "use_chain_of_thought",
    "use_call",
    "use_stream",
    "use_agent",
    "use_agent_call",
    "to_dspy_tool",
    "invoke_module",
    "call_tool",
    "call_tool_sync",
]

POLICIES = ("replace", "parallel", "ignore")
_NO_RESULT = object()


# ---------------------------------------------------------------------------
# Modules = component memory
# ---------------------------------------------------------------------------
def _sig_key(signature) -> str:
    return signature if isinstance(signature, str) else signature.__name__


def _stable(value: Any) -> Any:
    """Hashable, render-stable fingerprint of an arch kwarg (memo deps)."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Tool):
        return ("tool", value.name, id(value.fn))
    if isinstance(value, (list, tuple)):
        return tuple(_stable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _stable(v)) for k, v in value.items()))
    if callable(value) and getattr(value, "__qualname__", None):
        # inline lambdas are rebuilt every render: key on *what* they are
        return ("fn", getattr(value, "__module__", None), value.__qualname__)
    return ("obj", type(value).__name__, id(value))


def use_module(
    signature_or_module,
    module_cls=None,
    *,
    arch: Any = None,
    name: Optional[str] = None,
    deps: Optional[list] = None,
    compile_with_optimizer: bool = False,
    _key: Optional[Tuple] = None,
    **kwargs,
) -> Any:
    """Memoized DSPy module (``useMemo`` for programs).

    - Passing a module instance returns it unchanged (already "memory").
    - ``arch=`` picks *how* it thinks -- ``"predict"`` (default), ``"cot"``,
      ``"react"``, ``"rlm"``, ``"pot"``, ``"refine"``, ``"best_of_n"`` or any
      name from :func:`integrations.architectures.register_architecture`;
      ``**kwargs`` go to the architecture (``tools=``, ``max_iters=``,
      ``max_iterations=``, ``sub_model=``, ``N=`` ...). Changing ``arch``
      (e.g. from a prop) builds the other program on the next render.
      ``tools`` may name ToolProvider tools; ``sub_model`` names a
      ``DSPyProvider`` model (resolved to ``sub_lm``).
    - Legacy: ``use_module(sig, dspy.ChainOfThought)`` (a module class).
    - Built once and cached in ``env.caches`` under ``(cache_key, signature,
      scope)`` -- e.g. ``("ChainOfThought", "q -> a", "App/QA#qa")`` --
      where scope is ``name`` or this component's path, so each component
      instance owns its module unless you share it explicitly by ``name``.
    - If ``env.compiled`` holds a compiled/optimized program for that key
      (optimizer, GEPA, ``OptimizationResult.hot_swap``) it is returned
      instead, transparently.
    - ``deps``: rebuild when changed (e.g. new demos / instructions).
    """
    if isinstance(signature_or_module, dspy.Module):
        return signature_or_module

    from . import architectures as A

    # Works without a DSPyProvider too (lm=... on the call): modules then live
    # in a component-local store. Both hooks always run (stable hook order).
    provided = hooks.use_context(_dspy_ctx())
    local_env = hooks.use_memo(DSPyEnv, [])
    env = provided if provided is not None else local_env
    ctx = _context_stack.get()
    registry = use_tools()

    if arch is None and module_cls is None:
        arch = "predict"
    legacy = arch is None
    if legacy:
        factory, prefix, arch_name = module_cls, module_cls.__name__, None
        a = A._arch_for_class(module_cls)
        arch_name = a.name if a is not None else module_cls.__name__
    elif isinstance(arch, str):
        a = A.get_architecture(arch)
        factory, prefix, arch_name = a.factory, a.cache_key, a.name
    elif isinstance(arch, type):
        a = A._arch_for_class(arch)
        factory, prefix = arch, arch.__name__
        arch_name = a.name if a is not None else arch.__name__
    elif callable(arch):
        factory = arch
        prefix = arch_name = getattr(arch, "__name__", "custom")
    else:
        raise TypeError(f"arch must be a name, a dspy.Module class or a factory, got {arch!r}")

    build_kw = dict(kwargs)
    if not legacy:
        if "tools" in build_kw:
            build_kw["tools"] = resolve_tools(build_kw["tools"], registry)
        sub_model = build_kw.pop("sub_model", None)
        if sub_model is not None:
            if sub_model not in env.models:
                raise KeyError(
                    f"sub_model {sub_model!r} not in DSPyProvider models ({sorted(env.models)})"
                )
            build_kw["sub_lm"] = env.models[sub_model]

    if _key is None:
        scope = name if name is not None else tm.component_path(ctx)
        key = (prefix, _sig_key(signature_or_module), scope)
        if legacy and kwargs:
            key = key + (repr(sorted(kwargs.items())),)
    else:
        key = _key
    deps_key = tuple(deps) if deps is not None else None
    kw_sig = None if legacy else _stable(build_kw)

    def _factory():
        cached = env.caches.get(key)
        if (cached is not None and deps is None
                and (legacy or getattr(cached, "_pyreact_kw_sig", kw_sig) == kw_sig)):
            return cached
        mod = factory(signature_or_module, **build_kw)
        label = getattr(factory, "__name__", None) if legacy else arch_name
        try:
            mod._pyreact_name = name or f"{label}({_sig_key(signature_or_module)})"
            mod._pyreact_key = key
            mod._pyreact_arch = arch_name
            mod._pyreact_signature = signature_or_module
            mod._pyreact_kw_sig = kw_sig
        except Exception:
            pass
        env.caches[key] = mod
        return mod

    mod = hooks.use_memo(_factory, deps=[key, deps_key, id(env.caches), kw_sig])

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


def use_agent_call(
    signature,
    *,
    arch: Any = "predict",
    model: Optional[str] = None,
    lm: Optional[Any] = None,
    policy: str = "replace",
    name: Optional[str] = None,
    turn_id: Optional[str] = None,
    deps: Optional[list] = None,
    **arch_kwargs,
) -> "Call":
    """One-liner: ``use_call(use_module(signature, arch=arch, **arch_kwargs))``.

    Same :class:`Call` shape for every architecture (``run``/``submit``/
    ``cancel``, ``result``/``ver``/``loading``/``error``/``usage``), same
    ``llm.*`` events, same cancellation -- switch ``arch`` like a prop.
    """
    mod = use_module(signature, arch=arch, name=name, deps=deps, **arch_kwargs)
    return use_call(mod, model=model, lm=lm, policy=policy, name=name, turn_id=turn_id)


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
    call_id: Optional[str] = None,
    debug: Optional[bool] = None,
    harness: Optional[Harness] = None,
    models: Optional[Dict[str, Any]] = None,
    model_key: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Run one module call with per-call settings and publish llm.* events.

    Returns ``(result, usage)`` (``usage["call_id"]`` identifies the call for
    ``explain`` / feedback). Usable outside components too (scripts,
    evaluation). On failure the exception gets a ``usage`` attribute.

    The call is an ``"llm"`` boundary (:func:`pyreact.core.harness.through`):
    ``harness`` (default: process-wide + the providers of the calling
    component) wraps it. Middleware sees ``b.inputs`` and ``b.attrs``
    (``call_id``, ``module_key``, ``arch``, ``model``; after the call
    ``usage``, ``cost``, ``latency``, ``outputs``, ``lm_calls``); calling
    ``call_next`` again re-runs the module, not calling it short-circuits
    with the middleware's prediction, :class:`~pyreact.core.harness.Blocked`
    fails the call (``llm.blocked`` + ``llm.error``).

    Model switching (the hook point of
    :func:`integrations.recovery.fallback_model`): with a ``models``
    registry (``DSPyProvider(models=...)``; ``use_call`` passes it) the
    boundary also carries ``model_key`` (the registry key in use, ``None``
    for an explicit ``lm=``) and ``model_names`` (``{key: model name}``). A
    middleware that sets ``b.attrs["model_key"]`` to another key before
    ``call_next`` makes that run use ``models[key]``; ``b.attrs["model"]``
    and the ``llm.end`` payload then name the model that actually answered.

    Architectures with a custom runner (e.g. ``rlm``: worker thread +
    cooperative cancellation) are awaited through it; everything else via
    ``acall``. The LM history of the call (prompts + completions) is kept in
    a small in-memory buffer for :func:`integrations.learn.explain`; with
    ``debug=True`` (or ``telemetry.set_debug(True)``) it is also attached to
    ``llm.end`` as ``lm_calls``.
    """
    from .architectures import arch_of, runner_of

    if lm is None and getattr(dspy.settings, "lm", None) is None:
        raise RuntimeError(
            "No language model configured: mount DSPyProvider(lm=... or models={'default': ...}) "
            "or pass lm=... to the hook."
        )
    label = _module_label(module, name)
    call_id = call_id or tm.new_call_id()
    if turn_id is None:
        turn_id = tm.current_turn_id()
    model = tm.model_name(lm if lm is not None else dspy.settings.lm)
    models = dict(models or {})
    if model_key is not None and model_key not in models:
        model_key = None
    mkey = getattr(module, "_pyreact_key", None)
    arch = arch_of(module)
    base = {"call_id": call_id, "module": label, "component": component, "model": model}
    if mkey is not None:
        from .feedback import key_str

        base["module_key"] = key_str(mkey)
    pub = functools.partial(tm.publish, turn_id=turn_id, source=component)
    runner = runner_of(module)
    debug = tm.DEBUG if debug is None else debug

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

    def _record(result=None, error=None, latency=None, lm_calls=None):
        if lm_calls is None:
            lm_calls = tm.lm_calls_of(recorder.history)
        tm.record_call({
            "call_id": call_id, "module": label, "module_key": base.get("module_key"),
            "arch": arch, "component": component, "turn_id": turn_id, "model": used["model"],
            "inputs": tm.brief(inputs, 4000), "outputs": tm.outputs_of(result),
            "error": error, "latency": latency, "lm_calls": lm_calls, "ts": time.time(),
        })
        return lm_calls

    extra: Dict[str, Any] = {}
    used = {"model": model, "model_key": model_key}

    def _overrides_for(b: Boundary) -> Dict[str, Any]:
        # A middleware may switch the model by registry key (fallback_model).
        key = b.attrs.get("model_key")
        if not models or key is None or key == model_key:
            used.update(model=model, model_key=model_key)
            b.attrs["model"] = model
            return overrides
        if key not in models:
            raise ValueError(f"model_key {key!r} is not in the DSPyProvider models "
                             f"({sorted(models)})")
        chosen = models[key]
        used.update(model=tm.model_name(chosen), model_key=key)
        b.attrs["model"] = used["model"]
        return {**overrides, "lm": chosen}

    async def _attempt(b: Boundary):
        # One run of the module (a retrying middleware calls this again).
        nonlocal extra
        args = b.inputs
        settings = _overrides_for(b)
        t_run = time.perf_counter()
        try:
            with dspy.context(**settings):
                if runner is not None:
                    result, more = await runner(
                        module, args, call_id=call_id,
                        publish=lambda kind, payload: pub(kind, {"module": label, **payload}),
                    )
                    extra = dict(more or {})
                elif stream_fields:
                    result = await _stream_call(module, args, stream_fields, _chunk, _status)
                else:
                    result = await _plain_call(module, args)
        except BaseException:
            _annotate(b, recorder, t_run)
            raise
        _annotate(b, recorder, t_run, result)
        return result

    pub("llm.start", {**base, "arch": arch, "inputs": tm.brief(inputs),
                      "input_chars": _input_chars(inputs), "stream": bool(stream_fields)})
    token = tm.enter_scope(tm.CallScope(call_id, label, component, turn_id,
                                        loop=asyncio.get_running_loop()))
    t0 = time.perf_counter()
    attrs: Dict[str, Any] = {"call_id": call_id, "module_key": base.get("module_key"),
                             "arch": arch, "model": model}
    if models:
        attrs["model_key"] = model_key
        attrs["model_names"] = {k: tm.model_name(v) for k, v in models.items()}
    try:
        result = await through("llm", label, inputs, _attempt, harness=harness, **attrs)
    except asyncio.CancelledError:
        _record(error="cancelled", latency=time.perf_counter() - t0)
        pub("llm.cancel", {**base, "latency": time.perf_counter() - t0})
        raise
    except Exception as exc:
        usage = recorder.summary()
        usage["call_id"] = call_id
        lm_calls = _record(error=repr(exc), latency=time.perf_counter() - t0)
        payload = {**base, "latency": time.perf_counter() - t0, "usage": usage,
                   "cost": usage["cost"], "error": repr(exc)}
        if isinstance(exc, Blocked):
            info = {"reason": exc.reason, "by": exc.by, "message": str(exc)}
            pub("llm.blocked", {**base, **info})
            payload["blocked"] = info
        if debug:
            payload["lm_calls"] = lm_calls
        pub("llm.error", payload)
        try:
            exc.usage = usage
        except Exception:
            pass
        raise
    finally:
        tm.exit_scope(token)
    if used["model"] != model:  # a middleware switched models (fallback_model)
        base["model"] = used["model"]
    if models:
        base["model_key"] = used["model_key"]
    usage = recorder.summary()
    usage["latency"] = time.perf_counter() - t0
    usage["call_id"] = call_id
    lm_calls = _record(result=result, latency=usage["latency"])
    payload = {**base, **extra, "latency": usage["latency"], "usage": usage,
               "cost": usage["cost"], "outputs": tm.outputs_of(result)}
    if debug:
        payload["lm_calls"] = lm_calls
    pub("llm.end", payload)
    return result, usage


def _annotate(b: Boundary, recorder: "tm.UsageRecorder", t_run: float, result: Any = _NO_RESULT):
    """Fill the llm boundary's after-call attrs (usage so far, across attempts)."""
    usage = recorder.summary()
    b.attrs["usage"] = {k: usage[k] for k in ("prompt_tokens", "completion_tokens",
                                              "total_tokens")}
    b.attrs["cost"] = usage["cost"]
    b.attrs["latency"] = time.perf_counter() - t_run
    b.attrs["lm_calls"] = tm.lm_calls_of(recorder.history)
    if result is not _NO_RESULT:
        b.attrs["outputs"] = tm.outputs_of(result)


def _input_chars(inputs: dict) -> int:
    return sum(len(v) if isinstance(v, str) else len(str(v)) for v in inputs.values())


@dataclass(frozen=True)
class Call:
    """Snapshot of a call hook for this render.

    ``run(**inputs)`` fires the call (returns None — safe inside effects);
    ``submit(**inputs)`` does the same and returns the asyncio.Task.
    ``ver`` bumps only when a call settles, so ``use_effect(..., [call.ver])``
    fires exactly once per completed call.

    Call identity: ``inputs`` are the inputs of the in-flight call (the
    settled one once it settles) and ``tag`` the value passed as
    ``run(..., tag=...)`` with them -- e.g. the turn id, so a component can
    tell whether ``result`` answers *this* turn without extra state::

        check.run(comment=text, tag=turn.id)
        done = check.tag == turn.id and not check.loading

    ``tag`` is not sent to the module (unless its signature has a ``tag``
    input field).
    """

    run: Callable[..., None]
    submit: Callable[..., Optional[asyncio.Task]]
    cancel: Callable[[], None]
    result: Any = None
    ver: int = 0
    loading: bool = False
    error: Optional[BaseException] = None
    usage: Optional[Dict[str, Any]] = None
    inputs: Optional[Dict[str, Any]] = None
    tag: Any = None

    def __iter__(self):  # legacy tuple-unpacking: run, (result, ver), loading, error
        return iter((self.run, (self.result, self.ver), self.loading, self.error))

    @property
    def call_id(self) -> Optional[str]:
        """Id of the last settled call (for ``explain`` / ``send_feedback``)."""
        return (self.usage or {}).get("call_id")


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
        return {**state, "loading": True, "error": None, "partial": {},
                "inputs": action["inputs"], "tag": action["tag"]}
    if typ == "loading":
        return {**state, "loading": action["loading"]}
    if typ == "partial":
        return {**state, "partial": action["partial"]}
    if typ == "success":
        return {**state, "loading": action["loading"], "result": action["result"],
                "error": None, "ver": state["ver"] + 1, "usage": action["usage"],
                "partial": action.get("partial", state["partial"]),
                "inputs": action["inputs"], "tag": action["tag"]}
    if typ == "error":
        return {**state, "loading": action["loading"], "result": None,
                "error": action["error"], "ver": state["ver"] + 1,
                "usage": action["usage"], "inputs": action["inputs"], "tag": action["tag"]}
    return state


_INITIAL = {"loading": False, "result": None, "error": None, "ver": 0, "usage": None,
            "partial": {}, "inputs": None, "tag": None}


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
    scoped = hooks.use_context(HarnessContext)  # providers above us (see _do)

    # Latest config read at run time (stable run/cancel, never stale).
    ref["cfg"] = {
        "module": module,
        "lm": env.lm_for(model, lm) if env is not None else lm,
        "models": env.models if env is not None else None,
        "model_key": (None if lm is not None or env is None
                      else (model if model in env.models else "default")),
        "adapter": env.adapter if env is not None else None,
        "env": env,
        "policy": policy,
        "name": name,
        "turn_id": turn_id,
        "component": tm.component_path(ctx),
        "stream_fields": stream_fields,
        "throttle": throttle,
        "debug": (env.settings.get("debug") if env is not None else None) or None,
        "harness": scoped,
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

    async def _do(inputs: dict, cfg: dict, turn: Optional[str], call_id: Optional[str] = None,
                  tag: Any = None):
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
                    call_id=call_id, debug=cfg["debug"],
                    # scoped middleware captured at render + process-wide now
                    harness=resolve_harness(cfg["harness"]),
                    models=cfg["models"], model_key=cfg["model_key"],
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
        ident = {"inputs": dict(inputs), "tag": tag}
        if err is not None:
            dispatch({"type": "error", "error": err, "usage": usage, "loading": still, **ident})
            return
        if cfg["stream_fields"]:
            # Non-streaming LM (or cache hit): show the final values.
            for f in cfg["stream_fields"]:
                val = getattr(result, f, None) if result is not None else None
                if val is not None:
                    partial[f] = str(val)
        dispatch({"type": "success", "result": result, "usage": usage, "loading": still,
                  "partial": dict(partial), **ident})

    def _submit(**inputs) -> Optional[asyncio.Task]:
        call_id = inputs.pop("_call_id", None)  # internal: caller-chosen id
        cfg = ref["cfg"]
        tag = None
        if "tag" in inputs and not _has_input(cfg["module"], "tag"):
            tag = inputs.pop("tag")
        pol = cfg["policy"]
        inflight = [t for t in ref["tasks"] if not t.done()]
        if pol == "ignore" and inflight:
            return None
        if pol == "replace":
            for t in inflight:
                t.cancel()  # superseded: don't keep paying for a stale LLM call
            ref["tasks"].clear()
        turn = cfg["turn_id"] if cfg["turn_id"] is not None else tm.current_turn_id()
        dispatch({"type": "start", "inputs": dict(inputs), "tag": tag})
        task = spawn(
            _do(inputs, cfg, turn, call_id, tag), owner=ctx, turn=turn,
            name=f"dspy_call:{tm.component_path(ctx)}"
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
        inputs=state["inputs"], tag=state["tag"],
    )
    if stream_fields is not None:
        return StreamCall(**common, partial=state["partial"], fields=tuple(stream_fields))
    return Call(**common)


def _has_input(module, field_name: str) -> bool:
    """True if ``module``'s signature declares an input called ``field_name``."""
    from .architectures import signature_of

    try:
        sig = dspy.ensure_signature(signature_of(module))
        return field_name in getattr(sig, "input_fields", {})
    except Exception:
        return False


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
def _bind(fn: Callable, args: tuple, kwargs: dict) -> Tuple[Dict[str, Any], bool]:
    """Tool call args -> (named inputs, by_name). Positional args are bound to
    parameter names when possible so middleware sees one ``{name: value}``
    dict; otherwise they stay under ``"args"``."""
    if not args:
        return dict(kwargs), True
    try:
        sig = inspect.signature(fn)
        if not any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD, p.POSITIONAL_ONLY)
                   for p in sig.parameters.values()):
            return dict(sig.bind(*args, **kwargs).arguments), True
    except (TypeError, ValueError):
        pass
    return {"args": list(args), **kwargs}, False


async def call_tool(tool: Tool, args: tuple = (), kwargs: Optional[dict] = None, *,
                    harness: Optional[Harness] = None) -> Any:
    """Run one tool call as a ``"tool"`` boundary and publish ``tool.*`` events.

    Sync tools run in a worker thread (never blocking the render loop).
    ``b.inputs`` holds the named arguments (a middleware may rewrite them);
    attrs: ``call_id`` (the LLM call that asked for the tool), after the call
    ``result`` (preview) and ``latency``. A :class:`Blocked` tool call
    returns an ``"Error: ..."`` string to the agent (so it can adapt) and
    publishes ``tool.blocked``.
    """
    fn = tool.fn
    kwargs = kwargs or {}
    is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)
    )
    scope = tm.current_scope()
    call_id = scope.call_id if scope else None
    turn_id = scope.turn_id if scope else tm.current_turn_id()
    source = scope.component if scope else None
    args_preview = tm.brief(kwargs if not args else {"args": args, **kwargs})
    inputs, by_name = _bind(fn, args, kwargs)

    async def _run(b: Boundary):
        if by_name:
            a, kw = (), b.inputs
        else:
            kw = dict(b.inputs)
            a = tuple(kw.pop("args", ()))
        t_run = time.perf_counter()
        try:
            if is_async:
                out = await fn(*a, **kw)
            else:
                out = await asyncio.to_thread(fn, *a, **kw)
                if inspect.isawaitable(out):
                    out = await out
        finally:
            b.attrs["latency"] = time.perf_counter() - t_run
        b.attrs["result"] = tm.brief(out)
        return out

    tm.publish("tool.start", {"call_id": call_id, "tool": tool.name, "args": args_preview},
               turn_id=turn_id, source=source)
    t0 = time.perf_counter()
    error = None
    result = None
    try:
        result = await through("tool", tool.name, inputs, _run, harness=harness,
                               call_id=call_id)
        return result
    except Blocked as exc:
        error = f"blocked: {exc}"
        tm.publish("tool.blocked", {"call_id": call_id, "tool": tool.name, "args": args_preview,
                                    "reason": exc.reason, "by": exc.by, "message": str(exc)},
                   turn_id=turn_id, source=source)
        return f"Error: {exc}"
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


def call_tool_sync(tool: Tool, args: tuple = (), kwargs: Optional[dict] = None) -> Any:
    """:func:`call_tool` from synchronous code running in a worker thread of
    an LLM call (e.g. tools called from the RLM REPL): the boundary runs on
    the call's event loop while this thread waits for the result."""
    scope = tm.current_scope()
    loop = getattr(scope, "loop", None)
    try:
        asyncio.get_running_loop()
        on_loop_thread = True
    except RuntimeError:
        on_loop_thread = False
    if on_loop_thread:
        # Can't block a running loop waiting on itself: plain call.
        return tool.fn(*args, **(kwargs or {}))
    if loop is not None and loop.is_running():
        # The context (call scope, owner, turn, current boundary) travels
        # with the coroutine: run_coroutine_threadsafe copies this thread's.
        return asyncio.run_coroutine_threadsafe(call_tool(tool, args, kwargs), loop).result()
    return asyncio.run(call_tool(tool, args, kwargs))


def _instrumented(tool: Tool) -> Callable:
    @functools.wraps(tool.fn)
    async def _tool(*args, **kwargs):
        return await call_tool(tool, args, kwargs)

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
        react._pyreact_key, react._pyreact_arch = key, "react"
        react._pyreact_signature = signature
        if env is not None:
            env.caches[key] = react
        return react

    # Rebuild only when signature, tool set, limits, or name change.
    react = hooks.use_memo(_build, deps=[key, tools_key, max_iters])
    return use_call(react, model=model, lm=lm, policy=policy, name=name, turn_id=turn_id)
