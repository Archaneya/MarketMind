"""MarketMind backend: generates simulated ticks and streams them over WebSockets.

Run:  python websocket_server.py            (ws://localhost:8000/ws, dashboard at http://localhost:8000/)
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.market_state import MarketEngine
from backend.websocket_manager import ConnectionManager

FRONTEND = Path(__file__).parent / "frontend"
log = logging.getLogger("marketmind")


async def analytics_cycle(engine: MarketEngine, manager: ConnectionManager) -> None:
    try:
        payload = await engine.run_analytics()   # runs in a worker thread
        await manager.broadcast(payload)
    except Exception:
        log.exception("analytics failed")


async def producer(engine: MarketEngine, manager: ConnectionManager, interval: float) -> None:
    """Emit one tick every `interval` seconds on a fixed schedule (no drift, no catch-up bursts)."""
    loop = asyncio.get_running_loop()
    next_t = loop.time()
    pending: asyncio.Task | None = None
    while True:
        try:
            await manager.broadcast(engine.step())
            if engine.analytics_due() and (pending is None or pending.done()):
                pending = asyncio.create_task(analytics_cycle(engine, manager))
        except Exception:
            log.exception("tick failed")
        next_t += interval
        delay = next_t - loop.time()
        if delay < 0:
            next_t, delay = loop.time(), 0.0
        await asyncio.sleep(delay)


def create_app(engine: MarketEngine | None = None, tick_interval: float = 0.5) -> FastAPI:
    manager = ConnectionManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or MarketEngine()
        task = asyncio.create_task(producer(app.state.engine, manager, tick_interval))
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await manager.close_all(1001)   # tell clients we are going away

    app = FastAPI(title="MarketMind (simulated data)", lifespan=lifespan)
    app.state.manager = manager

    @app.get("/health")
    async def health():
        e = app.state.engine
        return {"status": "ok", "simulated": True, "clients": manager.count, "seq": e.seq, "regime": e.regime}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        e: MarketEngine = app.state.engine
        await manager.connect(ws)
        try:
            await manager.send(ws, e.history_message(), timeout=20)   # ~1 MB snapshot
            if e.analytics:
                await manager.send(ws, e.analytics)
            while True:
                try:
                    msg = json.loads(await ws.receive_text())
                except json.JSONDecodeError:
                    continue
                kind = msg.get("type") if isinstance(msg, dict) else None
                if kind == "ping":
                    await manager.send(ws, {"type": "pong"})
                elif kind == "resync":
                    await manager.send(ws, e.history_message(), timeout=20)
                    if e.analytics:
                        await manager.send(ws, e.analytics)
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("websocket error")
        finally:
            manager.disconnect(ws)

    @app.get("/")
    async def index():
        return FileResponse(FRONTEND / "chart.html")

    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="MarketMind simulated market server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=None, help="fix the random seed for reproducible runs")
    ap.add_argument("--tick-ms", type=int, default=500)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print("Preparing simulated history and training the regime model ...")
    app = create_app(MarketEngine(seed=a.seed), a.tick_ms / 1000)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info", timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
