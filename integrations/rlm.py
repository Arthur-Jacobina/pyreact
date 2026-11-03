"""``use_rlm``: a Recursive Language Model agent as a hook.

``dspy.RLM`` lets the LM explore a large input *programmatically*: inputs are
loaded as variables into a sandboxed Python REPL, the LM writes code to slice
/ search them and can call a (cheaper) sub-LM on pieces with
``llm_query(prompt)``, iterating until it calls ``SUBMIT(...)``. That makes
it the right tool for contexts that do not fit comfortably in one prompt.

In pyreact terms the RLM is a memoized module (``use_memo``) and each call is
a cancellable, tracked effect (``runtime.spawn``). RLM's REPL loop is
synchronous (its sandbox and ``llm_query`` callbacks block), so the call runs
in a worker thread; cancellation (a superseding call or unmount) is checked
between REPL iterations and before every sub-LM call.

It is the ``"rlm"`` architecture (``integrations.architectures``): use it
like any other program -- ``use_module(sig, arch="rlm", ...)`` +
``use_call``, or ``use_agent_call(sig, arch="rlm", ...)``. Events on the app
``EventBus`` (the usual ``llm.*`` lifecycle, plus progress)::

    llm.start      {call_id, module, arch: "rlm", inputs, input_chars, ...}
    rlm.iteration  {call_id, iteration, max_iterations, sub_llm_calls}
    llm.end        {call_id, ..., iterations, sub_llm_calls, trajectory_len, latency}
    llm.error      {call_id, error, latency}     llm.cancel {call_id, latency}

Sandbox: the default interpreter is DSPy's ``PythonInterpreter`` (Deno +
Pyodide/WASM). If Deno is missing, the call fails with
:class:`RLMSandboxError`; pass ``interpreter=`` a zero-arg factory returning
any ``dspy.primitives.code_interpreter.CodeInterpreter`` (e.g.
``lambda: PythonInterpreter(deno_command=[...])`` or an E2B/Modal-backed
interpreter) to use a different sandbox. A fresh interpreter is created per
call, so concurrent calls never share one.
"""

from __future__ import annotations

import contextvars
import functools
import re
import shutil
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Optional, Union

import dspy

from pyreact.core.core import hooks

try:  # dspy >= 3.1
    from dspy.predict.rlm import RLM as _RLM
except Exception:  # pragma: no cover
    _RLM = None

DEFAULT_MAX_ITERATIONS = 8
DEFAULT_MAX_LLM_CALLS = 10

SANDBOX_HELP = (
    "dspy.RLM needs a code sandbox. The default PythonInterpreter runs Python "
    "in Deno + Pyodide (WASM): install Deno (`brew install deno` or "
    "`curl -fsSL https://deno.land/install.sh | sh`) and make sure `deno` is on "
    "PATH. Alternatives: use_rlm(..., interpreter=lambda: "
    "PythonInterpreter(deno_command=['/path/to/deno', 'run', ...])) or any "
    "CodeInterpreter implementation (E2B, Modal, a local mock for tests)."
)


class RLMSandboxError(RuntimeError):
    """The RLM code sandbox is unavailable or crashed."""


class RLMCancelled(Exception):
    """Raised inside the worker thread when the call was superseded/unmounted."""


def check_sandbox() -> str:
    """Return the path to ``deno`` or raise :class:`RLMSandboxError`."""
    path = shutil.which("deno")
    if path is None:
        raise RLMSandboxError("Deno executable not found. " + SANDBOX_HELP)
    return path


class _RunControl:
    def __init__(self, call_id: str, on_iteration: Optional[Callable] = None) -> None:
        self.call_id = call_id
        self.cancelled = threading.Event()
        self.iterations = 0
        self.sub_llm_calls = 0
        self.on_iteration = on_iteration

    def check(self) -> None:
        if self.cancelled.is_set():
            raise RLMCancelled(self.call_id)


_CTRL: contextvars.ContextVar[Optional[_RunControl]] = contextvars.ContextVar(
    "pyreact_rlm_ctrl", default=None
)


if _RLM is not None:

    class PyreactRLM(_RLM):
        """``dspy.RLM`` + per-call interpreter factory, progress and cancellation."""

        def __init__(self, *args, interpreter_factory: Optional[Callable] = None, **kw):
            super().__init__(*args, **kw)
            self._interpreter_factory = interpreter_factory

        @contextmanager
        def _interpreter_context(self, execution_tools):
            if self._interpreter_factory is None:
                check_sandbox()
                with super()._interpreter_context(execution_tools) as repl:
                    yield repl
                return
            repl = self._interpreter_factory()
            self._inject_execution_context(repl, execution_tools)
            try:
                yield repl
            finally:
                try:
                    repl.shutdown()
                except Exception:
                    pass

        def _execute_iteration(self, repl, variables, history, iteration, *a, **kw):
            ctrl = _CTRL.get()
            if ctrl is not None:
                ctrl.check()
            result = super()._execute_iteration(repl, variables, history, iteration, *a, **kw)
            if ctrl is not None:
                ctrl.iterations = iteration + 1
                if ctrl.on_iteration is not None:
                    try:
                        ctrl.on_iteration(ctrl, result)
                    except Exception:
                        pass
            return result

        def _make_llm_tools(self, *args, **kw):
            tools = super()._make_llm_tools(*args, **kw)
            ctrl = _CTRL.get()
            if ctrl is None:
                return tools
            q, qb = tools["llm_query"], tools["llm_query_batched"]

            def llm_query(prompt: str) -> str:
                """Query the LLM with a prompt string."""
                ctrl.check()
                out = q(prompt)
                ctrl.sub_llm_calls += 1
                return out

            def llm_query_batched(prompts: list[str]) -> list[str]:
                """Query the LLM with multiple prompts concurrently."""
                ctrl.check()
                out = qb(prompts)
                ctrl.sub_llm_calls += len(prompts or [])
                return out

            return {"llm_query": llm_query, "llm_query_batched": llm_query_batched}

else:  # pragma: no cover
    PyreactRLM = None


_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


class LenientRLMAdapter(dspy.ChatAdapter):
    """ChatAdapter that tolerates the common "prose + ```python``` block" reply.

    Smaller models often answer the RLM action step (outputs ``reasoning``,
    ``code``) without the ``[[ ## field ## ]]`` markers. When strict parsing
    fails for that step we take the last fenced code block as ``code`` and the
    text before it as ``reasoning``; other signatures keep the normal
    ChatAdapter -> JSONAdapter fallback.
    """

    def parse(self, signature, completion):
        try:
            return super().parse(signature, completion)
        except Exception:
            if set(signature.output_fields) != {"reasoning", "code"}:
                raise
            blocks = list(_CODE_BLOCK.finditer(completion or ""))
            if not blocks:
                raise
            last = blocks[-1]
            return {
                "reasoning": completion[: last.start()].strip() or "(none)",
                "code": last.group(1).strip(),
            }


_DEFAULT_ADAPTER = LenientRLMAdapter()


def run_rlm_sync(
    module, inputs: dict, *, lm=None, adapter=None, ctrl: Optional[_RunControl] = None
):
    """Run an RLM synchronously with ``lm`` and a run control (worker thread)."""
    token = _CTRL.set(ctrl)
    try:
        overrides = {}
        if lm is not None:
            overrides["lm"] = lm
        if adapter is not None:
            overrides["adapter"] = adapter
        with dspy.context(**overrides):
            return module(**inputs)
    finally:
        _CTRL.reset(token)


def _tool_to_dspy(t: Any) -> Any:
    """Tool / callable -> dspy.Tool whose calls (from the REPL, in the worker
    thread) run as ``"tool"`` boundaries on the call's loop."""
    if isinstance(t, dspy.Tool) or not callable(t):
        return t
    from pyreact.tools import make_tool

    from .hooks import call_tool_sync

    tool = make_tool(t)

    @functools.wraps(tool.fn)
    def _call(*args, **kwargs):
        return call_tool_sync(tool, args, kwargs)

    return dspy.Tool(_call, name=tool.name, desc=tool.description or None)


def _input_chars(inputs: dict) -> int:
    return sum(len(v) if isinstance(v, str) else len(str(v)) for v in inputs.values())


def use_rlm(
    signature,
    *,
    tools: Optional[Iterable[Union[Any, Callable, str]]] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    sub_model: Optional[str] = None,
    model: Optional[str] = None,
    lm: Optional[Any] = None,
    interpreter: Optional[Callable[[], Any]] = None,
    max_output_chars: int = 10_000,
    adapter: Optional[Any] = None,
    name: Optional[str] = None,
    verbose: bool = False,
):
    """RLM agent hook. Returns ``(run, (result, ver), loading, error)``.

    Compatibility wrapper over the unified path::

        call = use_agent_call(signature, arch="rlm", max_iterations=..., sub_model=...)

    (same module cache, ``llm.*`` / ``rlm.iteration`` events, worker thread
    and cancellation as every other architecture). ``run(**inputs)`` returns
    the call id.

    - ``signature``: e.g. ``"context, question -> answer"`` or a Signature.
    - ``tools``: Tools / callables / ToolProvider names, callable from the REPL.
    - ``model`` / ``lm``: the root LM (``DSPyEnv.models[model]`` or default).
    - ``sub_model``: name in ``DSPyEnv.models`` for ``llm_query`` sub-calls.
    - ``interpreter``: zero-arg factory for a CodeInterpreter (default: Deno).
    - ``adapter``: DSPy adapter (default :class:`LenientRLMAdapter`).

    Policy: the last call wins, a superseded call is cancelled (``llm.cancel``)
    and its worker thread stops at the next REPL checkpoint.
    """
    if PyreactRLM is None:  # pragma: no cover
        raise RLMSandboxError("This DSPy version has no dspy.RLM (needs dspy>=3.1).")

    from integrations import telemetry as tm
    from integrations.hooks import use_call, use_module

    arch_kw: dict = dict(max_iterations=max_iterations, max_llm_calls=max_llm_calls,
                         max_output_chars=max_output_chars, verbose=verbose)
    if tools:
        arch_kw["tools"] = tools
    if sub_model is not None:
        arch_kw["sub_model"] = sub_model
    if interpreter is not None:
        arch_kw["interpreter"] = interpreter
    if adapter is not None:
        arch_kw["adapter"] = adapter
    module = use_module(signature, arch="rlm", name=name, **arch_kw)
    call = use_call(module, model=model, lm=lm, name=name)
    submit = call.submit

    def _run(**inputs):
        cid = tm.new_call_id("rlm")
        submit(_call_id=cid, **inputs)
        return cid

    run = hooks.use_callback(_run, deps=[])
    return run, (call.result, call.ver), call.loading, call.error
