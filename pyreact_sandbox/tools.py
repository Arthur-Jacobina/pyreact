"""``sandbox_tools(handle)``: sandbox capabilities as pyreact tools.

    sb = use_sandbox()
    tools = hooks.use_memo(lambda: sandbox_tools(sb), [id(sb)])
    call = use_agent("question -> answer", tools=tools)
    # or: ToolProvider(tools=sandbox_tools(sb), children=[...])

Four async tools — ``run_shell(command)``, ``run_python(code)``,
``read_file(path)``, ``write_file(path, content)`` — each one a call on the
handle, so it runs through the ``sandbox.exec`` harness boundary (nested
under the ``tool`` boundary of the agent's call) and publishes
``sandbox.exec``. Outputs are plain text (exit code + stdout/stderr),
truncated to ``max_output`` characters so a chatty command cannot flood the
LM's context.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from pyreact.tools import Tool, make_tool

from ._util import truncate
from .base import ExecResult

__all__ = ["sandbox_tools", "format_result"]

TOOL_NAMES = ("run_shell", "run_python", "read_file", "write_file")


def format_result(r: ExecResult, max_output: int = 4000) -> str:
    head = f"exit_code: {r.exit_code}" + (" (timed out)" if r.timed_out else "")
    parts = [head]
    if r.stdout:
        parts.append("stdout:\n" + truncate(r.stdout, max_output))
    if r.stderr:
        parts.append("stderr:\n" + truncate(r.stderr, max_output))
    if not r.stdout and not r.stderr:
        parts.append("(no output)")
    return "\n".join(parts)


def sandbox_tools(handle: Any, *, timeout: Optional[float] = 60, max_output: int = 4000,
                  prefix: str = "", include: Optional[Iterable[str]] = None) -> List[Tool]:
    """Tools over ``handle`` (a :class:`~pyreact_sandbox.react.SandboxHandle`).

    - ``timeout``: seconds per command (killed after it).
    - ``prefix``: name prefix (``"sb_"`` -> ``sb_run_shell``...) to avoid clashes.
    - ``include``: subset of ``run_shell``, ``run_python``, ``read_file``, ``write_file``.
    """
    workdir = handle.spec.workdir

    async def run_shell(command: str) -> str:
        return format_result(await handle.exec(command, timeout=timeout), max_output)

    async def run_python(code: str) -> str:
        return format_result(await handle.run_python(code, timeout=timeout), max_output)

    async def read_file(path: str) -> str:
        try:
            return truncate(await handle.read_text(path), max_output)
        except FileNotFoundError as exc:
            return f"error: file not found: {exc}"

    async def write_file(path: str, content: str) -> str:
        await handle.write_file(path, content)
        return f"wrote {len(content.encode('utf-8'))} bytes to {path}"

    funcs = {"run_shell": run_shell, "run_python": run_python, "read_file": read_file,
             "write_file": write_file}
    where = f" Relative paths are under {workdir}; files persist between calls."
    limit = f" Commands are killed after {timeout:g}s." if timeout else ""
    descs = {
        "run_shell": "Run a shell command (sh -c) in the sandbox; returns exit code, stdout "
                     "and stderr." + where + limit,
        "run_python": "Run a Python 3 program in the sandbox (use print() to show results); "
                      "returns exit code, stdout and stderr." + where + limit,
        "read_file": "Read a text file from the sandbox." + where,
        "write_file": "Create or overwrite a text file in the sandbox (parent directories "
                      "are created)." + where,
    }
    wanted = list(include) if include is not None else list(TOOL_NAMES)
    unknown = set(wanted) - set(TOOL_NAMES)
    if unknown:
        raise ValueError(f"unknown sandbox tools {sorted(unknown)} (known: {TOOL_NAMES})")
    return [make_tool(funcs[n], name=f"{prefix}{n}", description=descs[n]) for n in wanted]
