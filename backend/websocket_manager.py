"""Tracks WebSocket clients and broadcasts messages to all of them.

Works with any object exposing `async accept()`, `async send_text(str)` and `async close(code=..)`
(a Starlette WebSocket, or a fake in tests). A per-client lock keeps concurrent senders (the tick
broadcaster and a resync reply) from interleaving frames, and a send timeout drops stuck clients
so one slow browser cannot stall the stream for everyone else.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging

log = logging.getLogger("marketmind.ws")


class ConnectionManager:
    def __init__(self, send_timeout: float = 2.0) -> None:
        self._clients: dict = {}
        self.send_timeout = send_timeout

    @property
    def count(self) -> int:
        return len(self._clients)

    async def connect(self, ws) -> None:
        await ws.accept()
        self._clients[ws] = asyncio.Lock()
        log.info("client connected (%d total)", self.count)

    def disconnect(self, ws) -> None:
        if self._clients.pop(ws, None) is not None:
            log.info("client disconnected (%d total)", self.count)

    async def send(self, ws, message, timeout: float | None = None, skip_if_busy: bool = False) -> bool:
        """Send to one client; returns False (and forgets the client) on failure."""
        text = message if isinstance(message, str) else json.dumps(message, separators=(",", ":"))
        lock = self._clients.get(ws)
        if lock is None:
            return False
        if skip_if_busy and lock.locked():
            return True   # mid-way through another send (e.g. a snapshot): skip this tick; the client resyncs on the gap
        try:
            async with lock:
                await asyncio.wait_for(ws.send_text(text), timeout or self.send_timeout)
            return True
        except Exception as exc:  # closed socket, timeout, etc.
            log.info("dropping client: %s", type(exc).__name__)
            self.disconnect(ws)
            return False

    async def broadcast(self, message) -> int:
        text = message if isinstance(message, str) else json.dumps(message, separators=(",", ":"))
        clients = list(self._clients)
        if not clients:
            return 0
        results = await asyncio.gather(*(self.send(ws, text, skip_if_busy=True) for ws in clients))
        return sum(results)

    async def close_all(self, code: int = 1001) -> None:
        for ws in list(self._clients):
            with contextlib.suppress(Exception):
                await ws.close(code=code)
            self.disconnect(ws)
