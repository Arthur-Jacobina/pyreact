import asyncio
import sys
from typing import Iterable, Optional

from pyreact.observability import ConsoleSink, format_event

from .app_runner import AppRunner

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


def _header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}=== {title} ==={RESET}")


def handle_command(app: AppRunner, cmd: str, args_str: str = "") -> bool:
    """Run a built-in ``:command``. Returns False for unknown commands."""
    color = sys.stdout.isatty()
    if cmd == "help":
        print(HELP)
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
    return False


async def read_terminal_and_invoke(
    app: AppRunner,
    *,
    prompt: str = ">> ",
    wait: bool = True,
    console: bool = True,
    console_kinds: Optional[Iterable[str]] = ConsoleSink.DEFAULT_KINDS,
):
    """Minimal async loop that reads lines from stdin and forwards to app.invoke().

    With ``console=True`` a :class:`ConsoleSink` renders ``console_kinds``
    events (default: chat messages and errors) as they are published,
    including those emitted before the prompt started.

    Built-in commands (prefix with ':' or '/'): see ``:help`` -- ``:events``,
    ``:usage``, ``:turns``, ``:tree``, ``:trace``, ``:route``, ``:nav``, ``:q``.

    - Reading happens via run_in_executor to avoid blocking the event loop.
    - Stop the loop by sending :q / :quit / :exit (case-sensitive) or Ctrl+C.
    """
    sink = None
    if console and app.events is not None:
        sink = ConsoleSink(kinds=console_kinds)
        sink.attach(app.events, replay=True)

    loop = asyncio.get_running_loop()
    try:
        while True:
            txt = await loop.run_in_executor(None, input, prompt)
            s = (txt or "").strip()
            if s.startswith(":") or s.startswith("/"):
                rest = s[1:].strip()
                if not rest:
                    continue
                parts = rest.split(None, 1)
                cmd = parts[0]
                args_str = parts[1] if len(parts) > 1 else ""
                if cmd in ("q", "quit", "exit"):
                    break
                if not handle_command(app, cmd, args_str):
                    print(f"{GRAY}unknown command :{cmd} (try :help){RESET}")
                continue
            try:
                await app.ainvoke(txt, wait=wait)
            except Exception as exc:
                # Keep the loop resilient, but don't hide the problem
                print(f"\x1b[31m[pyreact]\x1b[0m input not delivered: {exc}")
                if not app.is_running:
                    break
    except (KeyboardInterrupt, SystemExit, EOFError):
        pass
    finally:
        if sink is not None:
            sink.detach()
