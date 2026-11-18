import math
import unittest

import numpy as np

from backend.market_state import CandleAggregator
from backend.simulator import BEAR, BULL, HIGH_VOL, REGIMES, SIDEWAYS, MarketSimulator
import ml.model_utils as mu


class SimulatorTests(unittest.TestCase):
    def test_regime_names_match_ml_module(self):
        self.assertEqual(REGIMES, mu.REGIMES)

    def test_prices_positive_and_finite(self):
        sim = MarketSimulator(seed=1)
        for _ in range(20000):
            t = sim.next_tick()
            self.assertTrue(math.isfinite(t.price) and t.price > 0)
            self.assertGreater(t.volume, 0)

    def test_reproducible_with_seed(self):
        a = [MarketSimulator(seed=5).next_tick().price for _ in range(1)]
        s1, s2 = MarketSimulator(seed=5), MarketSimulator(seed=5)
        self.assertEqual([s1.next_tick().price for _ in range(200)], [s2.next_tick().price for _ in range(200)])

    def test_tick_clock_advances_by_fixed_step(self):
        sim = MarketSimulator(seed=2, seconds_per_tick=20)
        e = [sim.next_tick().epoch for _ in range(5)]
        self.assertEqual(np.diff(e).tolist(), [20.0] * 4)

    def test_regimes_change_and_never_repeat_consecutively(self):
        sim = MarketSimulator(seed=3)
        seq = []
        for _ in range(60000):
            r = sim.next_tick().regime
            if not seq or seq[-1] != r:
                seq.append(r)
        self.assertGreater(len(seq), 10)
        self.assertEqual(set(seq), set(REGIMES))

    def _returns(self, regime, n=30000):
        sim = MarketSimulator(seed=11, initial_regime=regime, regime_minutes=(10 ** 6, 10 ** 6))
        return np.array([sim.next_tick().log_return for _ in range(n)])

    def test_regime_drift_and_volatility_characteristics(self):
        bull, bear, side, hv = (self._returns(r) for r in (BULL, BEAR, SIDEWAYS, HIGH_VOL))
        self.assertGreater(bull.mean(), 0)
        self.assertLess(bear.mean(), 0)
        self.assertGreater(hv.std(), 2 * max(bull.std(), bear.std(), side.std()))

    def test_sideways_stays_in_a_range(self):
        sim = MarketSimulator(seed=4, initial_regime=SIDEWAYS, regime_minutes=(10 ** 6, 10 ** 6))
        p = np.array([sim.next_tick().price for _ in range(20000)])
        self.assertLess(abs(np.log(p / p[0])).max(), 0.02)

    def test_ohlc_consistency_after_aggregation(self):
        sim, agg = MarketSimulator(seed=6), CandleAggregator()
        for _ in range(9000):
            agg.update(sim.next_tick())
        cs = agg.candles()
        self.assertGreater(len(cs), 2900)
        for a, b in zip(cs, cs[1:]):
            self.assertLessEqual(max(a["open"], a["close"]), a["high"] + 1e-9)
            self.assertGreaterEqual(min(a["open"], a["close"]), a["low"] - 1e-9)
            self.assertGreater(a["low"], 0)
            self.assertAlmostEqual(b["open"], a["close"])   # continuity
            self.assertGreater(b["time"], a["time"])


if __name__ == "__main__":
    unittest.main()
