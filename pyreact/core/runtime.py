# runtime.py -------------------------------------------------
import asyncio
from typing import Optional


rerender_queue: asyncio.Queue = asyncio.Queue()
_enqueued: set = set()

_render_idle: Optional[asyncio.Event] = None  # will be created on demand
_render_signal: Optional[asyncio.Event] = None  # set when a render is scheduled


def get_render_idle() -> asyncio.Event:
    """Ensure the ``Event`` belongs to the currently running loop."""
    global _render_idle
    if _render_idle is None:
        _render_idle = asyncio.Event()
        _render_idle.set()  # start in the 'idle' state
    return _render_idle


def get_render_signal() -> asyncio.Event:
    """Event that is set whenever a rerender is scheduled, and cleared after run_renders drains the queue."""
    global _render_signal
    if _render_signal is None:
        _render_signal = asyncio.Event()
    return _render_signal


def schedule_rerender(ctx, reason: str = None):
    # Record schedule intent for debug tooling
    try:
        from .debug import record_schedule  # local import to avoid cycles

        record_schedule(ctx, reason)
    except Exception:
        pass
    loop = asyncio.get_running_loop()
    if ctx in _enqueued:
        return
    _enqueued.add(ctx)

    def _enqueue():
        try:
            rerender_queue.put_nowait(ctx)
        finally:
            get_render_signal().set()  # Set the signal only after the ctx is in the queue to avoid races

    loop.call_soon_threadsafe(_enqueue)


async def run_renders() -> None:
    from pyreact.core.hook import HookContext  # import here to avoid infinite loop
    from .debug import start_trace, end_trace

    while True:
        try:
            ctx: HookContext = rerender_queue.get_nowait()
        except Exception:
            break
        _enqueued.discard(ctx)
        if getattr(ctx, "_mounted", True):
            reasons = getattr(ctx, "_debug_reasons", [])
            try:
                start_trace(ctx, reasons)
            except Exception:
                pass
            # Clear reasons once consumed
            try:
                ctx._debug_reasons = []
            except Exception:
                pass
            ctx.render()
            await ctx.run_effects()
            try:
                end_trace()
            except Exception:
                pass

    # Mark idle and clear signal after draining current batch
    if rerender_queue.empty():
        get_render_idle().set()
        get_render_signal().clear()
