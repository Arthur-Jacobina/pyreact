import asyncio
import sys
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from pyreact.observability import ConsoleSink, format_event

BOLD = "\x1b[1m"
CYAN = "\x1b[36m"
RESET = "\x1b[0m"
GRAY = "\x1b[90m"
YELLOW = "\x1b[33m"

HELP = """\
  :events [kind] [n]   recent events (kind: exact, 'llm.*', or '*'; default 20)
  :usage               tokens / cost / latency per session and turn
  :turns [n]           per-turn timeline (last n, default 5)
  :tree                component tree
  :trace [on|off]      last render trace; on/off toggles trace.render events
  :route               current route
  :nav <dest>          navigate
  :help                this help
  :q | :quit | :exit   quit"""


# Extension point: ``register_command("name", fn, "help")`` adds ``:name``;
# ``fn(target, args_str)`` gets the AppRunner or Session (a Root is passed as
# its session) and the raw arguments.
CommandFn = Callable[[Any, str], Any]
_COMMANDS: Dict[str, Tuple[CommandFn, str]] = {}


def register_command(name: str, fn: CommandFn, help: str = "") -> None:
    """Add (or replace) a ``:name`` terminal command. Built-ins win on clashes."""
    if not name or any(c.isspace() for c in name):
        raise ValueError(f"invalid command name {name!r}")
    _COMMANDS[name] = (fn, help)


def unregister_command(name: str) -> None:
    _COMMANDS.pop(name, None)


def registered_commands() -> Dict[str, str]:
    """``{name: help}`` of commands added with :func:`register_command`."""
    return {n: h for n, (_fn, h) in _COMMANDS.items()}


def help_text() -> str:
    extra = [f"  :{n:<19} {h}" for n, (_f, h) in sorted(_COMMANDS.items())]
    return HELP + ("\n" + "\n".join(extra) if extra else "")


def _header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}=== {title} ==={RESET}")


def _resolve(target: Any) -> Any:
    """The object terminal commands drive: a :class:`~pyreact.boot.root.Root`
    is driven through its session (its devtools side: events, usage,
    timeline, tree, route); an AppRunner / Session is used as is."""
    from .root import Root

    return target.session if isinstance(target, Root) else target


def handle_command(target: Any, cmd: str, args_str: str = "") -> bool:
    """Run a ``:command`` (built-in or registered) against ``target``: a
    ``Root``, an ``AppRunner`` or a ``Session``. Returns False for unknown
    commands."""
    app = _resolve(target)
    color = sys.stdout.isatty()
    if cmd == "help":
        print(help_text())
        return True
    if cmd == "tree":
        app.print_vnode_tree()
        return True
    if cmd == "trace":
        arg = args_str.strip().lower()
        if arg in ("on", "off"):
            from pyreact.core.debug import set_trace_events

            set_trace_events(arg == "on")
            print(f"{GRAY}[debug]{RESET} trace.render events {arg}")
        else:
            app.print_render_trace()
        return True
    if cmd == "events":
        log = app.events
        if log is None:
            print(f"{GRAY}[debug]{RESET} event log not available.")
            return True
        kind: Optional[str] = None
        n = 20
        for tok in args_str.split():
            if tok.isdigit():
                n = int(tok)
            else:
                kind = tok
        kinds: Optional[Iterable[str]] = None
        if kind:
            kinds = (kind, kind + ".*") if not kind.endswith("*") else (kind,)
        evs = log.recent(n, kinds)
        _header(f"Events ({len(evs)}{' ' + kind if kind else ''})")
        for ev in evs:
            tid = f" {GRAY}turn={ev.turn_id}{RESET}" if ev.turn_id else ""
            print(f"{GRAY}#{ev.id} {ev.kind}{RESET}{tid} {format_event(ev, color=color)}")
        print()
        return True
    if cmd == "usage":
        usage = app.usage
        _header("Usage")
        print(usage.render() if usage is not None else "usage not available.")
        print()
        return True
    if cmd == "turns":
        tl = app.timeline
        n = int(args_str) if args_str.strip().isdigit() else 5
        _header("Turns")
        print(tl.render(last=n, color=color) if tl is not None else "timeline not available.")
        print()
        return True
    if cmd == "route":
        try:
            info = app.current_route()
            _header("Route")
            print(f"{GRAY}path:{RESET} {YELLOW}{info.get('path', '')}{RESET}")
            q = info.get("query", {}) or {}
            if q:
                print(f"{GRAY}query:{RESET} {YELLOW}{q}{RESET}")
            frag = info.get("fragment", "") or ""
            if frag:
                print(f"{GRAY}fragment:{RESET} {YELLOW}{frag}{RESET}")
            print()
        except Exception:
            print(f"{GRAY}[debug]{RESET} Route not available.")
        return True
    if cmd == "nav":
        dest = (args_str or "").strip()
        if not dest:
            print(f"{GRAY}Usage:{RESET} {YELLOW}:nav /path[?query][#fragment]{RESET}")
        else:
            app.nav(dest)
        return True
    entry = _COMMANDS.get(cmd)
    if entry is not None:
        try:
            entry[0](app, args_str)
        except Exception as exc:  # a dev command must never kill the prompt
            print(f"\x1b[31m[:{cmd}] {type(exc).__name__}: {exc}\x1b[0m")
        return True
    return False


def parse_command(
    line: str, prefixes: Iterable[str] = (":",)
) -> Optional[Tuple[str, str]]:
    """``(cmd, args)`` if ``line`` is a terminal command line, else ``None``.

    Only ``prefixes`` (default ``":"``) mark terminal commands: ``/word`` is
    plain input that the app owns (e.g. its own slash commands).
    """
    s = (line or "").strip()
    for prefix in prefixes:
        if prefix and s.startswith(prefix):
            rest = s[len(prefix):].strip()
            if not rest:
                return None
            parts = rest.split(None, 1)
            return parts[0], (parts[1] if len(parts) > 1 else "")
    return None


async def read_terminal_and_invoke(
    target: Any,
    *,
    prompt: str = ">> ",
    wait: bool = True,
    console: bool = True,
    console_kinds: Optional[Iterable[str]] = ConsoleSink.DEFAULT_KINDS,
    command_prefixes: Iterable[str] = (":",),
):
    """Minimal async loop that reads lines from stdin and sends each one as a
    turn of ``target`` (a ``Root``, an ``AppRunner`` or a ``Session``).

    With ``console=True`` a :class:`ConsoleSink` renders ``console_kinds``
    events (default: chat messages and errors) as they are published,
    including those emitted before the prompt started.

    Terminal commands start with ``:`` (``command_prefixes``): see ``:help``
    -- ``:events``, ``:usage``, ``:turns``, ``:tree``, ``:trace``, ``:route``,
    ``:nav``, ``:q``. Everything else is a turn of the app, including
    ``/word`` lines (apps own their slash commands) and unknown ``:word``
    commands (sent to the app, with a hint). Pass
    ``command_prefixes=(":", "/")`` for the old behaviour where ``/word`` ran
    a terminal command.

    - Reading happens via run_in_executor to avoid blocking the event loop.
    - Stop the loop by sending :q / :quit / :exit (case-sensitive) or Ctrl+C.
    """
    app = _resolve(target)
    prefixes = tuple(command_prefixes)
    sink = None
    if console and app.events is not None:
        sink = ConsoleSink(kinds=console_kinds)
        sink.attach(app.events, replay=True)

    loop = asyncio.get_running_loop()
    try:
        while True:
            txt = await loop.run_in_executor(None, input, prompt)
            s = (txt or "").strip()
            parsed = parse_command(s, prefixes)
            if parsed is None and s in prefixes:
                continue  # a bare ":" is a no-op
            if parsed is not None:
                cmd, args_str = parsed
                if cmd in ("q", "quit", "exit"):
                    break
                if handle_command(app, cmd, args_str):
                    continue
                # Not a terminal command: the app may own it.
                print(f"{GRAY}(:{cmd} is not a terminal command; sent to the app. "
                      f":help lists terminal commands){RESET}")
            try:
                await app.ainvoke(txt, wait=wait)
            except Exception as exc:
                # Keep the loop resilient, but don't hide the problem
                print(f"\x1b[31m[pyreact]\x1b[0m input not delivered: {exc}")
                if not getattr(app, "is_running", not getattr(app, "closed", False)):
                    break
    except (KeyboardInterrupt, SystemExit, EOFError):
        pass
    finally:
        if sink is not None:
            sink.detach()
