import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from backend.market_state import MarketEngine, resample_candles
from frontend.assemble import build_dashboard_html

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


class AssemblyTests(unittest.TestCase):
    def test_single_file_with_inlined_assets_and_ws_url(self):
        h = build_dashboard_html("ws://example:9/ws")
        self.assertNotIn("static/styles.css", h)
        self.assertNotIn("static/chart.js", h)
        self.assertNotIn("__WS_URL__", h)
        self.assertIn('"ws://example:9/ws"', h)
        self.assertIn("SIMULATED DATA", h)
        self.assertIn("lightweight-charts", h)


@unittest.skipUnless(NODE, "node is not installed")
class ChartJsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        e = MarketEngine(seed=31, warmup_minutes=7000)
        cls.engine = e
        cls.history = e.history_message()
        cls.analytics = e.analytics
        cls.ticks = [e.step() for _ in range(12)]

    def run_node(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "in.json"
            f.write_text(json.dumps({"history": self.history, "analytics": self.analytics, "ticks": self.ticks}))
            r = subprocess.run([NODE, str(ROOT / "tests" / "dom_harness.js"), str(f)], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_syntax(self):
        r = subprocess.run([NODE, "--check", str(ROOT / "frontend" / "chart.js")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_client_behaviour(self):
        o = self.run_node()
        self.assertEqual(o["url"], "ws://test/ws")
        self.assertEqual(o["ignoredBeforeHistory"], 0)                 # ticks before the history snapshot are ignored
        n5 = len(resample_candles(self.history["candles"], 5))
        self.assertEqual(o["csAfterHistory"], n5)
        self.assertEqual(o["vsAfterHistory"], n5)
        self.assertEqual(o["tickUpdates"], len(self.ticks))
        self.assertTrue(o["status"].startswith("conn on"))
        self.assertEqual(o["ltpText"], f"{o['lastLtp']:,.2f}")
        self.assertTrue(o["rankingHasAll"] and o["regimeCard"])
        self.assertGreaterEqual(o["indicatorsRows"], 10)
        self.assertEqual(o["compareRows"], 6)
        self.assertEqual(o["updatesFromBadMessages"], 0)               # duplicate + invalid are rejected
        self.assertTrue(o["resyncSent"])                               # a sequence gap triggers a resync request
        self.assertRegex(o["diag"], r"Missed ticks</td><td class=\"num\">4<")
        self.assertRegex(o["diag"], r"Duplicate / out-of-order</td><td class=\"num\">1<")

    def test_timeframe_aggregation_matches_python(self):
        o = self.run_node()
        merged = {c["time"]: c for c in self.history["candles"]}
        merged.update({t["candle"]["time"]: t["candle"] for t in self.ticks})   # live ticks extend/replace candles
        base = [merged[k] for k in sorted(merged)]
        # the client shifts times to IST (+05:30) before bucketing, so 1h/4h/1d bars align to IST clock boundaries
        for tf, minutes in (("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60), ("4h", 240), ("1d", 1440)):
            self.assertEqual(o["counts"][tf], len(resample_candles(base, minutes, offset=19800)), tf)

    def test_reconnect_and_watchdog(self):
        o = self.run_node()
        self.assertEqual(o["reconnectDelayMs"], 1000)
        self.assertEqual(o["socketsOpened"], 2)
        self.assertTrue(o["statusAfterReconnect"].startswith("conn on"))
        self.assertTrue(o["watchdogClosed"])


if __name__ == "__main__":
    unittest.main()
