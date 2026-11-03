"""Devtools for the examples: trace a root's events, drive a root from stdin.

ADR 0001: output is components (``Message``) and outside code talks to the
tree only through its root. Events are for devtools -- so this module is the
*one* place the examples subscribe to a root's events:

    trace(root, ["tool.*", "llm.end"])         # print events as they happen
    ends = collect(root, ["llm.end"])          # a live list, for a metrics table
    show_messages(root)                        # the root-level renderer for Message
    run_repl(root, prompt="> ")                # stdin -> root.send(...); ":" commands

``trace`` / ``collect`` / ``show_messages`` subscribe to ``root.events`` (the
session's event log) with ``replay=True``, so what the first render published
before the call is not missed. ``trace`` / ``show_messages`` return their
unsubscribe; ``collect`` returns the list (it stops filling when the root
unmounts).
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Iterable, List, Optional, Sequence

from pyreact.boot import handle_command, parse_command
from pyreact.core.events import Event
from pyreact.observability import ConsoleSink, format_event

__all__ = ["trace", "collect", "show_messages", "command", "run_repl", "Format"]

# fmt(event) -> the line to print, or None to skip the event
Format = Callable[[Event], Optional[str]]

QUIT = ("q", "quit", "exit")


def _isatty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def trace(root: Any, kinds: Optional[Iterable[str]] = None, fmt: Optional[Format] = None, *,
          prefix: str = "", end: str = "\n", replay: bool = True) -> Callable[[], None]:
    """Print ``root``'s events of ``kinds`` (``"llm.*"``, exact kinds, ``None``
    = all) as they are published, formatted by ``fmt`` (default:
    :func:`pyreact.observability.format_event`); ``end=""`` for a token
    stream. Returns the unsubscribe."""
    color = _isatty()
    fmt = fmt or (lambda ev: format_event(ev, color=color))

    def show(ev: Event) -> None:
        line = fmt(ev)
        if line is not None:
            print(prefix + line if prefix else line, end=end, flush=True)

    return root.events.subscribe(show, list(kinds) if kinds is not None else None,
                                 replay=replay)


def collect(root: Any, kinds: Optional[Iterable[str]] = None, *,
            replay: bool = True) -> List[Event]:
    """A list that fills with ``root``'s events of ``kinds`` (for tables and
    assertions in demo scripts: latency, LM calls, routes...)."""
    out: List[Event] = []
    root.events.subscribe(out.append, list(kinds) if kinds is not None else None,
                          replay=replay)
    return out


def show_messages(root: Any, kinds: Sequence[str] = ConsoleSink.DEFAULT_KINDS, *,
                  replay: bool = True) -> Callable[[], None]:
    """Render ``Message`` components (and errors) to the terminal: a stand-in
    for a root-level renderer (``root.render_to(console)`` in the ADR).
    Returns the detach function."""
    sink = ConsoleSink(kinds=kinds)
    return sink.attach(root.events, replay=replay)


def command(root: Any, line: str) -> bool:
    """Run one ``:command`` line (``:events llm``, ``:turns``, ``:feedback +``...)
    against ``root``. Returns False if it is not a terminal command."""
    parsed = parse_command(line)
    if parsed is None:
        return False
    cmd, args = parsed
    try:
        # the built-in commands read the session's observability (events,
        # usage, timeline, route): the root's low-level side, devtools only.
        if cmd == "trace" and args.strip().lower() not in ("on", "off"):
            root.runner.print_render_trace()  # the last render trace (loop-wide)
        elif not handle_command(root.session, cmd, args):
            return False
    except Exception as exc:  # a dev command must never kill the prompt
        print(f"\x1b[31m[:{cmd}] {type(exc).__name__}: {exc}\x1b[0m", flush=True)
    try:  # let what it changed (e.g. a :nav) render before the next prompt
        root.wait_idle(timeout=5)
    except Exception:
        pass
    return True


def run_repl(root: Any, *, prompt: str = "> ",
             console_kinds: Optional[Sequence[str]] = ConsoleSink.DEFAULT_KINDS,
             timeout: Optional[float] = None) -> None:
    """Read lines from stdin; each line is a turn (``root.send(line,
    wait=True)``), lines starting with ``:`` are terminal commands (``:help``;
    ``:q`` quits). ``Message`` output (``console_kinds``) is printed as it is
    published. Returns on ``:q``, EOF or Ctrl+C."""
    detach = show_messages(root, console_kinds) if console_kinds else (lambda: None)
    try:
        while not root.closed:
            try:
                line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                break
            s = line.strip()
            if s == ":":
                continue
            parsed = parse_command(s)
            if parsed is not None:
                if parsed[0] in QUIT:
                    break
                if command(root, s):
                    continue
                print(f"(:{parsed[0]} is not a terminal command; sent to the app. "
                      ":help lists terminal commands)", flush=True)
            try:
                root.send(line, wait=True, timeout=timeout)
            except Exception as exc:  # keep the prompt alive, but say why
                print(f"\x1b[31m[pyreact]\x1b[0m input not delivered: {exc}", flush=True)
    finally:
        detach()
