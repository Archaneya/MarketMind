import inspect
import unittest

import numpy as np

from indicators.technical_indicators import compute_all
from strategies import breakout, mean_reversion, momentum, strategy_engine, trend_following, volatility
from tests.helpers import make_ohlcv, ranging, trending

NAMES = {"Momentum Trading", "Mean Reversion", "Breakout Trading", "Trend Following", "Volatility-Based"}


def by_name(results):
    return {r.name: r for r in results}


class StrategyTests(unittest.TestCase):
    def test_all_five_return_valid_results(self):
        ind = compute_all(trending(300))
        for fn in (momentum.evaluate, mean_reversion.evaluate, breakout.evaluate, trend_following.evaluate, volatility.evaluate):
            r = fn(ind)
            self.assertTrue(0 <= r.score <= 100, r.name)
            self.assertIn(r.compatibility, {"High", "Moderate", "Low", "Poor"})
            self.assertTrue(r.reason and r.risks and r.supporting and r.components, r.name)

    def test_ranking_sorted_and_complete(self):
        res = strategy_engine.evaluate_all(compute_all(trending(300)))
        self.assertEqual({r.name for r in res}, NAMES)
        self.assertEqual([r.rank for r in res], [1, 2, 3, 4, 5])
        self.assertEqual([r.score for r in res], sorted((r.score for r in res), reverse=True))

    def test_trending_market_favours_trend_and_momentum(self):
        r = by_name(strategy_engine.evaluate_all(compute_all(trending(300, drift=0.0008))))
        self.assertGreater(r["Trend Following"].score, 70)
        self.assertGreater(r["Trend Following"].score, r["Mean Reversion"].score)
        self.assertGreater(r["Momentum Trading"].score, r["Mean Reversion"].score)

    def test_stretched_range_market_favours_mean_reversion_over_trend(self):
        wins = 0
        for seed in range(10):
            df = ranging(300, seed=seed)
            prior = df["close"].iloc[-21:-1]
            df.iloc[-1, df.columns.get_loc("close")] = prior.mean() + 2.4 * prior.std(ddof=0)
            df.iloc[-1, df.columns.get_loc("high")] = df["close"].iloc[-1] * 1.0002
            r = by_name(strategy_engine.evaluate_all(compute_all(df)))
            wins += r["Mean Reversion"].score > r["Trend Following"].score
        self.assertGreaterEqual(wins, 8)

    def test_noise_without_trend_does_not_score_as_trend(self):
        scores = [trend_following.evaluate(compute_all(ranging(300, seed=s))).score for s in range(10)]
        self.assertLess(np.mean(scores), 40)

    def test_breakout_detected_with_volume_expansion(self):
        df = ranging(200, seed=3)
        df.iloc[-1, df.columns.get_loc("close")] = df["high"].iloc[-30:-1].max() * 1.01
        df.iloc[-1, df.columns.get_loc("high")] = df["close"].iloc[-1] * 1.001
        df.iloc[-5:, df.columns.get_loc("volume")] *= 3
        quiet = breakout.evaluate(compute_all(ranging(200, seed=3)))
        loud = breakout.evaluate(compute_all(df))
        self.assertGreater(loud.score, quiet.score)
        self.assertIn("broke above resistance", loud.reason)

    def test_volatility_strategy_rewards_expansion(self):
        gaps = []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            calm = rng.normal(0, 0.0003, 200)
            wild = np.r_[calm, rng.normal(0, 0.003, 40)]
            s_calm = volatility.evaluate(compute_all(make_ohlcv(24500 * np.exp(np.cumsum(calm))))).score
            s_wild = volatility.evaluate(compute_all(make_ohlcv(24500 * np.exp(np.cumsum(wild)), spread=0.003))).score
            gaps.append(s_wild - s_calm)
        self.assertTrue(all(g > 0 for g in gaps), gaps)   # scores are relative to recent bars, so single draws are noisy
        self.assertGreater(np.mean(gaps), 15)

    def test_insufficient_history_is_flagged_not_guessed(self):
        res = strategy_engine.evaluate_all(compute_all(make_ohlcv(np.linspace(100, 101, 15))))
        for r in res:
            self.assertEqual(r.compatibility, "N/A")
            self.assertEqual(r.score, 0.0)

    def test_ranking_updates_when_market_changes(self):
        up = strategy_engine.evaluate_all(compute_all(trending(300, drift=0.0008)))
        flat = strategy_engine.evaluate_all(compute_all(ranging(300)))
        self.assertNotEqual([r.name for r in up], [r.name for r in flat])

    def test_strategies_do_not_depend_on_regime(self):
        for fn in (momentum.evaluate, mean_reversion.evaluate, breakout.evaluate, trend_following.evaluate, volatility.evaluate):
            self.assertEqual(list(inspect.signature(fn).parameters), ["ind"])


if __name__ == "__main__":
    unittest.main()
