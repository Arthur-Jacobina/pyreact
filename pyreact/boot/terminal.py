import asyncio

from .app_runner import AppRunner


async def read_terminal_and_invoke(
    app: AppRunner, *, prompt: str = ">> ", wait: bool = True
):
    """Minimal async loop that reads lines from stdin and forwards to app.invoke().

    Built-in commands (prefix with ':' or '/'):
      - :tree            → app.print_vnode_tree()
      - :trace           → app.print_render_trace()
      - :route           → prints app.current_route()
      - :nav <dest>      → app.nav(dest)
      - :q|:quit|:exit   → quit

    - Reading happens via run_in_executor to avoid blocking the event loop.
    - Stop the loop by sending :q / :quit / :exit (case-sensitive) or Ctrl+C.
    """
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
                if cmd == "tree":
                    app.print_vnode_tree()
                    continue
                if cmd == "trace":
                    app.print_render_trace()
                    continue
                if cmd == "route":
                    try:
                        info = app.current_route()
                        BOLD = "\x1b[1m"
                        CYAN = "\x1b[36m"
                        RESET = "\x1b[0m"
                        GRAY = "\x1b[90m"
                        YELLOW = "\x1b[33m"
                        print(f"\n{BOLD}{CYAN}=== Route ==={RESET}")
                        print(
                            f"{GRAY}path:{RESET} {YELLOW}{info.get('path', '')}{RESET}"
                        )
                        q = info.get("query", {}) or {}
                        if q:
                            print(f"{GRAY}query:{RESET} {YELLOW}{q}{RESET}")
                        frag = info.get("fragment", "") or ""
                        if frag:
                            print(f"{GRAY}fragment:{RESET} {YELLOW}{frag}{RESET}")
                        print(f"{BOLD}{CYAN}=============== {RESET}\n")
                    except Exception:
                        print("\x1b[90m[debug]\x1b[0m Route not available.")
                    continue
                if cmd == "nav":
                    dest = (args_str or "").strip()
                    if not dest:
                        GRAY = "\x1b[90m"
                        RESET = "\x1b[0m"
                        YELLOW = "\x1b[33m"
                        print(
                            f"{GRAY}Usage:{RESET} {YELLOW}:nav /path[?query][#fragment]{RESET}"
                        )
                    else:
                        app.nav(dest)
                    continue
                # Unknown command → ignore
                continue
            if s in (":q", ":quit", ":exit"):
                break
            try:
                app.invoke(txt, wait=wait)
            except Exception:
                # Keep the loop resilient to user code errors
                pass
    except (KeyboardInterrupt, SystemExit):
        pass
