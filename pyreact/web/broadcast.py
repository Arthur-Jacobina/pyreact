from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections import defaultdict
import json
from typing import AsyncIterator, Dict, Set


class BroadcastEvent:
    def __init__(self, message: str):
        self.message = message


class _Subscriber:
    def __init__(self, queue: asyncio.Queue[BroadcastEvent]):
        self._queue = queue

    def __aiter__(self) -> "_Subscriber":
        return self

    async def __anext__(self) -> BroadcastEvent:
        item = await self._queue.get()
        return item


class InMemoryBroadcast:
    def __init__(self) -> None:
        self._channels: Dict[str, Set[asyncio.Queue[BroadcastEvent]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def publish(self, channel: str, message: dict) -> None:
        async with self._lock:
            queues = list(self._channels.get(channel, set()))
        event = BroadcastEvent(json.dumps(message))
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                await q.put(event)

    @asynccontextmanager
    async def subscribe(self, channel: str) -> AsyncIterator[_Subscriber]:
        queue: asyncio.Queue[BroadcastEvent] = asyncio.Queue(maxsize=1024)
        async with self._lock:
            self._channels[channel].add(queue)
        try:
            yield _Subscriber(queue)
        finally:
            async with self._lock:
                try:
                    self._channels[channel].discard(queue)
                except Exception:
                    pass
