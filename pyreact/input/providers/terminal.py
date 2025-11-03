import asyncio
import time
from typing import Optional, Callable, Dict
from pyreact.core.runtime import get_render_idle
from pyreact.input.bus import InputBus


def _emit_text_submit(bus: InputBus, txt: str):
    now = time.time()
    bus.emit({"type": "text", "value": txt, "source": "term", "ts": now})
    bus.emit({"type": "submit", "value": txt, "source": "term", "ts": now})


class TerminalInput:
    """Simple terminal reader (line by line).
    For per-key input, raw mode can be implemented later.
    """

    def __init__(
        self,
        bus: InputBus,
        prompt: str = ">> ",
        commands: Optional[Dict[str, Callable[[str], None]]] = None,
    ):
        self.bus = bus
        self.prompt = prompt
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self._commands: Dict[str, Callable[[str], None]] = commands or {}

    async def _runner(self):
        loop = asyncio.get_running_loop()
        while not self._stopping:
            txt = await loop.run_in_executor(None, input, self.prompt)
            s = (txt or "").strip()

            # Local commands: use ":cmd" or "/cmd" (e.g. :tree)
            if s.startswith(":") or s.startswith("/"):
                rest = s[1:].strip()
                if rest:
                    parts = rest.split(None, 1)
                    cmd = parts[0]
                    args_str = parts[1] if len(parts) > 1 else ""
                else:
                    cmd = ""
                    args_str = ""
                # Built-in quit/exit commands
                if cmd in ("quit", "exit", "q"):
                    self._stopping = True
                    break
                fn = self._commands.get(cmd)
                if fn is not None:
                    try:
                        fn(args_str)
                    except Exception:
                        pass
                    # do not emit to app input bus for command lines
                    await asyncio.sleep(0)  # yield
                    continue
            _emit_text_submit(self.bus, txt)

            await asyncio.sleep(0.1)
            await get_render_idle().wait()

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._runner())

    def stop(self):
        self._stopping = True
