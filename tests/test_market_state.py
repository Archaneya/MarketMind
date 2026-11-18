import asyncio
import json
import unittest

import pandas as pd

from backend.market_state import (CandleAggregator, MarketEngine, REFERENCE_PRICE, resample_candles, validate_tick)
from backend.simulator import MarketSimulator, Tick
from backend.websocket_manager import ConnectionManager

FIELDS = {"timestamp", "symbol", "ltp", "open", "high", "low", "close", "volume", "regime"}


class AggregationTests(unittest.TestCase):
    def test_ticks_become_correct_candle(self):
        agg = CandleAggregator()
        for i, (t, p, v) in enumerate([(0, 100, 1), (20, 103, 2), (40, 99, 3), (60, 101, 4)]):
            agg.update(Tick(t, p, v, "x", 0))
        first, second = agg.candles()
        self.assertEqual((first["open"], first["high"], first["low"], first["close"], first["volume"]), (100, 103, 99, 99, 6))
        self.assertEqual(second["open"], 99)          # opens at previous close
        self.assertEqual((second["high"], second["low"], second["close"]), (101, 99, 101))

    def test_out_of_order_tick_rejected(self):
        agg = CandleAggregator()
        agg.update(Tick(120, 100, 1, "x", 0))
        with self.assertRaises(ValueError):
            agg.update(Tick(10, 100, 1, "x", 0))

    def test_history_is_bounded(self):
        agg = CandleAggregator(maxlen=50)
        for i in range(5000):
            agg.update(Tick(i * 20, 100 + i % 7, 1, "x", 0))
        self.assertLessEqual(len(agg.candles()), 51)

    def test_resample_matches_manual_for_every_timeframe(self):
        sim, agg = MarketSimulator(seed=9, start_epoch=0), CandleAggregator()
        for _ in range(3 * 3000):
            agg.update(sim.next_tick())
        base = agg.candles()
        df = pd.DataFrame(base)
        for minutes in (1, 5, 15, 60, 240, 1440):
            out = resample_candles(base, minutes)
            sec = minutes * 60
            for t, row in list(out.iterrows())[:5] + list(out.iterrows())[-3:]:
                part = df[(df.time >= t) & (df.time < t + sec)]
                self.assertEqual(row["open"], part.open.iloc[0])
                self.assertEqual(row["close"], part.close.iloc[-1])
                self.assertEqual(row["high"], part.high.max())
                self.assertEqual(row["low"], part.low.min())
                self.assertAlmostEqual(row["volume"], part.volume.sum())
            self.assertTrue((out["high"] >= out[["open", "close"]].max(axis=1)).all())
            self.assertTrue((out["low"] <= out[["open", "close"]].min(axis=1)).all())
            self.assertTrue(out.index.is_monotonic_increasing and out.index.is_unique)


class ValidationTests(unittest.TestCase):
    def good(self):
        return {"timestamp": "t", "symbol": "S", "ltp": 10, "open": 10, "high": 11, "low": 9, "close": 10, "volume": 5, "regime": "r"}

    def test_accepts_good_rejects_bad(self):
        validate_tick(self.good())
        for k, v in (("ltp", -1), ("close", float("nan")), ("high", 5), ("low", 12), ("volume", -3)):
            bad = self.good(); bad[k] = v
            with self.assertRaises(ValueError, msg=k):
                validate_tick(bad)
        bad = self.good(); del bad["symbol"]
        with self.assertRaises(ValueError):
            validate_tick(bad)


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = MarketEngine(seed=21, warmup_minutes=7000)

    def test_starts_near_reference_price_and_trains_model(self):
        self.assertAlmostEqual(self.engine.sim.price, REFERENCE_PRICE, delta=500)
        self.assertGreaterEqual(self.engine.detector.fits, 1)

    def test_tick_message_has_required_fields_and_valid_ohlc(self):
        e, seqs = self.engine, []
        for _ in range(60):
            m = e.step()
            self.assertTrue(FIELDS <= set(m))
            self.assertEqual(m["symbol"], "NIFTY 50")
            self.assertTrue(m["low"] <= min(m["open"], m["close"]) and m["high"] >= max(m["open"], m["close"]))
            json.dumps(m, allow_nan=False)
            seqs.append(m["seq"])
        self.assertEqual(seqs, list(range(seqs[0], seqs[0] + 60)))

    def test_analytics_payload_is_strict_json_and_complete(self):
        a = self.engine.compute_analytics(self.engine.agg.candles(6600))
        json.dumps(a, allow_nan=False)
        self.assertEqual(len(a["strategies"]), 5)
        self.assertEqual([s["rank"] for s in a["strategies"]], [1, 2, 3, 4, 5])
        self.assertGreaterEqual(len(a["indicators"]), 10)
        self.assertIn("suitable", a["intelligence"])
        self.assertTrue(a["regime"]["reasons"])

    def test_strategy_ranking_changes_as_market_evolves(self):
        e, orders = MarketEngine(seed=5, warmup_minutes=7000), set()
        for _ in range(240):
            for _ in range(60):
                e.step()
            a = e.compute_analytics(e.agg.candles(6600))
            orders.add(tuple(s["name"] for s in a["strategies"]))
        self.assertGreater(len(orders), 1)

    def test_async_analytics_runs_off_the_event_loop(self):
        a = asyncio.run(self.engine.run_analytics())
        self.assertEqual(a["type"], "analytics")

    def test_history_message(self):
        h = self.engine.history_message(500)
        self.assertEqual(len(h["candles"]), 500)
        self.assertEqual(h["seq"], self.engine.seq)


class FakeWS:
    def __init__(self, fail=False, delay=0.0):
        self.sent, self.fail, self.delay, self.closed, self.accepted = [], fail, delay, None, False

    async def accept(self): self.accepted = True

    async def send_text(self, t):
        if self.fail:
            raise RuntimeError("closed")
        if self.delay:
            await asyncio.sleep(self.delay)
        self.sent.append(t)

    async def close(self, code=1000): self.closed = code


class ManagerTests(unittest.TestCase):
    def test_multiple_clients_receive_broadcast(self):
        async def go():
            m, a, b = ConnectionManager(), FakeWS(), FakeWS()
            await m.connect(a); await m.connect(b)
            self.assertEqual(await m.broadcast({"n": 1}), 2)
            self.assertEqual(json.loads(a.sent[0]), {"n": 1})
            self.assertEqual(a.sent, b.sent)
        asyncio.run(go())

    def test_dead_client_is_dropped_and_others_keep_receiving(self):
        async def go():
            m, good, bad = ConnectionManager(), FakeWS(), FakeWS(fail=True)
            await m.connect(good); await m.connect(bad)
            self.assertEqual(await m.broadcast({"n": 1}), 1)
            self.assertEqual(m.count, 1)
            await m.broadcast({"n": 2})
            self.assertEqual(len(good.sent), 2)
        asyncio.run(go())

    def test_slow_client_times_out_without_stalling_the_rest(self):
        async def go():
            m, fast, slow = ConnectionManager(send_timeout=0.05), FakeWS(), FakeWS(delay=1.0)
            await m.connect(fast); await m.connect(slow)
            await m.broadcast("x")
            self.assertEqual(m.count, 1)
            self.assertEqual(fast.sent, ["x"])
        asyncio.run(go())

    def test_client_busy_with_a_snapshot_does_not_stall_broadcasts(self):
        async def go():
            m, busy, other = ConnectionManager(), FakeWS(delay=0.3), FakeWS()
            await m.connect(busy); await m.connect(other)
            snapshot = asyncio.create_task(m.send(busy, "SNAPSHOT"))
            await asyncio.sleep(0.01)
            t0 = asyncio.get_running_loop().time()
            await m.broadcast("tick")
            self.assertLess(asyncio.get_running_loop().time() - t0, 0.1)
            await snapshot
            self.assertEqual(busy.sent, ["SNAPSHOT"])
            self.assertEqual(other.sent, ["tick"])
        asyncio.run(go())

    def test_disconnect_and_graceful_close_all(self):
        async def go():
            m, a, b = ConnectionManager(), FakeWS(), FakeWS()
            await m.connect(a); await m.connect(b)
            m.disconnect(a)
            self.assertEqual(await m.broadcast("x"), 1)
            await m.close_all(1001)
            self.assertEqual((b.closed, m.count), (1001, 0))
            self.assertEqual(await m.broadcast("x"), 0)    # no clients: no error
        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
