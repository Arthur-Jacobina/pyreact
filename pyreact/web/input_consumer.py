from __future__ import annotations

import json
from typing import Awaitable, Callable
from .broadcast import InMemoryBroadcast


class InputConsumer:
    """Simple input consumer that delegates message handling.

    This class subscribes to a broadcast input channel and forwards decoded
    JSON messages to a provided async handler. This enables decoupling of
    transport from business logic and simplifies testing.
    """

    def __init__(
        self,
        *,
        broadcast: InMemoryBroadcast,
        input_channel: str,
        handle_message: Callable[[dict], Awaitable[None]],
    ) -> None:
        self._broadcast = broadcast
        self._input_channel = input_channel
        self._handle_message = handle_message

    async def run(self) -> None:
        async with self._broadcast.subscribe(self._input_channel) as subscriber:
            async for event in subscriber:
                raw = event.message
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                try:
                    await self._handle_message(msg)
                except Exception:
                    pass
