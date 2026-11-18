"""End-to-end tests of the FastAPI app. Skipped automatically if fastapi/httpx are not installed."""
import importlib.util
import json
import unittest

HAVE = all(importlib.util.find_spec(m) for m in ("fastapi", "httpx", "uvicorn"))


@unittest.skipUnless(HAVE, "fastapi / httpx / uvicorn not installed")
class ServerTests(unittest.TestCase):
    def make_client(self):
        from fastapi.testclient import TestClient
        from backend.market_state import MarketEngine
        from websocket_server import create_app
        return TestClient(create_app(MarketEngine(seed=1, warmup_minutes=7000), tick_interval=0.02))

    def test_health_and_index(self):
        with self.make_client() as c:
            h = c.get("/health").json()
            self.assertTrue(h["simulated"])
            self.assertIn("SIMULATED DATA", c.get("/").text)

    def test_stream_history_then_ticks_with_increasing_seq(self):
        with self.make_client() as c, c.websocket_connect("/ws") as ws:
            first = json.loads(ws.receive_text())
            self.assertEqual(first["type"], "history")
            seqs = []
            while len(seqs) < 8:
                m = json.loads(ws.receive_text())
                if m["type"] == "tick":
                    for k in ("timestamp", "symbol", "ltp", "open", "high", "low", "close", "volume", "regime"):
                        self.assertIn(k, m)
                    seqs.append(m["seq"])
            self.assertEqual(seqs, sorted(seqs))
            self.assertGreater(seqs[0], first["seq"] - 1)

    def test_multiple_clients_and_resync_and_ping(self):
        with self.make_client() as c, c.websocket_connect("/ws") as a, c.websocket_connect("/ws") as b:
            a.send_text(json.dumps({"type": "ping"}))
            a.send_text("garbage")                       # must not crash the connection
            a.send_text(json.dumps({"type": "resync"}))
            types = set()
            for _ in range(60):
                types.add(json.loads(a.receive_text())["type"])
                if {"pong", "history", "tick"} <= types:
                    break
            self.assertTrue({"pong", "history", "tick"} <= types)
            self.assertEqual(json.loads(b.receive_text())["type"], "history")

    def test_disconnect_is_handled(self):
        with self.make_client() as c:
            with c.websocket_connect("/ws") as ws:
                ws.receive_text()
            self.assertEqual(c.app.state.manager.count, 0)


if __name__ == "__main__":
    unittest.main()
