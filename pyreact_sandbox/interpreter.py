"""``SandboxInterpreter``: a dspy ``CodeInterpreter`` that runs in a sandbox.

``dspy.RLM`` (``use_module(sig, arch="rlm")`` / ``use_rlm``) executes the
LM's code in a REPL; by default that is Deno + Pyodide on the host. Give it a
sandbox instead::

    sb = use_sandbox()                                    # Docker, network off
    rlm = use_module("context, question -> answer", arch="rlm",
                     interpreter=lambda: SandboxInterpreter(sb))

How it works: ``start()`` launches a small stdlib-only REPL server
(``_repl_server.py``) inside the sandbox through the driver's interactive
process capability (``docker exec -i python3 -u -c ...``). Each
``execute(code, variables)`` sends one JSON request; host-side tools
(``llm_query``, ``SUBMIT`` targets, your tools) are called back over the same
pipe, so the LM's code never gets credentials or network. The Python
namespace persists across ``execute`` calls; the sandbox *filesystem*
persists across interpreters (RLM creates one per call), so files written in
one call are there in the next.

With a :class:`~pyreact_sandbox.react.SandboxHandle` every ``execute`` is a
``sandbox.exec`` harness boundary (name ``python``, op ``repl``) and publishes
``sandbox.exec``; with a raw :class:`~pyreact_sandbox.base.Sandbox` it runs
directly. Needs ``python3`` in the image (``python=`` to change).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from .base import Sandbox, SandboxProcess

__all__ = ["SandboxInterpreter", "REPL_SERVER"]

REPL_SERVER = (Path(__file__).with_name("_repl_server.py")).read_text(encoding="utf-8")


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


class SandboxInterpreter:
    """dspy ``CodeInterpreter`` backed by a sandbox (handle or raw Sandbox).

    - ``tools``: host functions callable from the code (``RLM`` injects
      ``llm_query`` / ``llm_query_batched`` itself).
    - ``output_fields``: ``[{"name", "type"?}]`` for ``SUBMIT(...)`` (set by RLM).
    - ``timeout``: seconds per ``execute`` (the REPL is killed on timeout).
    - ``cwd``: working directory (default: the spec's workdir).
    """

    def __init__(self, target: Any, *, tools: Optional[Dict[str, Callable]] = None,
                 output_fields: Optional[List[dict]] = None, python: str = "python3",
                 cwd: Optional[str] = None, timeout: Optional[float] = 300,
                 start_timeout: float = 60) -> None:
        self.target = target
        self._tools: Dict[str, Callable] = dict(tools or {})
        self.output_fields = output_fields
        self.python = python
        self.cwd = cwd
        self.timeout = timeout
        self.start_timeout = start_timeout
        self._proc: Optional[SandboxProcess] = None
        self._lines: "queue.Queue[bytes]" = queue.Queue()
        self._lock = threading.Lock()

    @property
    def tools(self) -> Dict[str, Callable]:
        return self._tools

    @property
    def _handle(self):
        return None if isinstance(self.target, Sandbox) else self.target

    # ---------------------------------------------------------------- start
    def _sandbox(self) -> Sandbox:
        if isinstance(self.target, Sandbox):
            return self.target
        h = self.target
        return h.sandbox if h.sandbox is not None else h.run_sync(h.start)

    def start(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            sb = self._sandbox()
            proc = sb.open_process_sync([self.python, "-u", "-c", REPL_SERVER], cwd=self.cwd)
            self._lines = queue.Queue()
            threading.Thread(target=self._pump, args=(proc, self._lines), daemon=True,
                             name="sandbox-repl-reader").start()
            self._proc = proc
            msg = self._read(self.start_timeout)
            if msg.get("op") != "ready":
                raise CodeInterpreterError(f"sandbox REPL failed to start: {msg}")

    @staticmethod
    def _pump(proc: SandboxProcess, q: "queue.Queue[bytes]") -> None:
        while True:
            try:
                line = proc.readline()
            except Exception:
                line = b""
            q.put(line)
            if not line:
                return

    def _read(self, timeout: Optional[float]) -> dict:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            self.shutdown()
            raise CodeInterpreterError(f"sandbox REPL timed out after {timeout}s") from None
        if not line:
            err = self._proc.read_stderr() if self._proc is not None else ""
            self._proc = None
            raise CodeInterpreterError(f"sandbox REPL exited. {err[-2000:]}".strip())
        try:
            return json.loads(line)
        except ValueError:
            raise CodeInterpreterError(f"bad message from sandbox REPL: {line[:200]!r}") from None

    def _send(self, msg: dict) -> None:
        self._proc.write((json.dumps(msg, default=str) + "\n").encode("utf-8"))

    # -------------------------------------------------------------- execute
    def _call_tool(self, name: str, args: list, kwargs: dict) -> dict:
        fn = self._tools.get(name)
        if fn is None:
            return {"op": "tool_result", "ok": False, "error": f"unknown tool {name!r}"}
        try:
            value = fn(*args, **kwargs)
            if inspect.isawaitable(value):
                value = asyncio.run(value)
            return {"op": "tool_result", "ok": True, "value": _jsonable(value)}
        except Exception as exc:
            return {"op": "tool_result", "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _execute_raw(self, code: str, variables: Optional[dict]) -> Any:
        self.start()
        with self._lock:
            self._send({"op": "exec", "code": code, "variables": _jsonable(variables or {}),
                        "tools": sorted(self._tools), "outputs": self.output_fields or []})
            while True:
                msg = self._read(self.timeout)
                op = msg.get("op")
                if op == "tool_call":
                    self._send(self._call_tool(msg.get("name"), msg.get("args") or [],
                                               msg.get("kwargs") or {}))
                elif op == "result":
                    return msg.get("output")
                elif op == "final":
                    return FinalOutput(msg.get("output"))
                elif op == "error":
                    if msg.get("syntax"):
                        raise SyntaxError(f"Invalid Python syntax. message: {msg.get('message')}")
                    raise CodeInterpreterError(msg.get("message") or msg.get("type"))
                else:
                    raise CodeInterpreterError(f"unexpected message from sandbox REPL: {msg}")

    def execute(self, code: str, variables: Optional[Dict[str, Any]] = None) -> Any:
        h = self._handle
        if h is None:
            return self._execute_raw(code, variables)
        for name in variables or {}:
            if not str(name).isidentifier():
                raise CodeInterpreterError(f"Invalid variable name: {name!r}")

        async def run(i):
            h.dirty = True
            return await asyncio.to_thread(self._execute_raw, i["code"], variables)

        return h.run_sync(lambda: h._boundary("repl", "python", {"code": code}, run))

    def __call__(self, code: str, variables: Optional[Dict[str, Any]] = None) -> Any:
        return self.execute(code, variables)

    def shutdown(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.close()
            except Exception:
                pass

    def __enter__(self) -> "SandboxInterpreter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()
