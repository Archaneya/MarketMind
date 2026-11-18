import unittest

import numpy as np
import pandas as pd

from indicators import technical_indicators as ti
from tests.helpers import make_ohlcv, trending


class IndicatorTests(unittest.TestCase):
    def setUp(self):
        self.df = trending(200)
        self.c = self.df["close"]

    def test_sma_matches_manual_and_has_warmup_nans(self):
        s = ti.sma(self.c, 20)
        self.assertTrue(s.iloc[:19].isna().all())
        self.assertAlmostEqual(s.iloc[50], self.c.iloc[31:51].mean())

    def test_ema_matches_recursive_definition(self):
        e = ti.ema(self.c, 20)
        a = 2 / 21
        manual = self.c.iloc[0]
        for x in self.c.iloc[1:60]:
            manual = a * x + (1 - a) * manual
        self.assertAlmostEqual(e.iloc[59], manual, places=8)
        self.assertTrue(e.iloc[:19].isna().all())

    def test_rsi_bounds_and_extremes(self):
        r = ti.rsi(self.c).dropna()
        self.assertTrue(((r >= 0) & (r <= 100)).all())
        self.assertEqual(ti.rsi(pd.Series(np.arange(1, 60, dtype=float))).iloc[-1], 100.0)
        self.assertEqual(ti.rsi(pd.Series(np.full(60, 100.0))).iloc[-1], 50.0)
        self.assertLess(ti.rsi(pd.Series(np.arange(60, 1, -1, dtype=float))).iloc[-1], 1e-9)
        self.assertTrue(ti.rsi(self.c).iloc[:14].isna().all())

    def test_rsi_matches_manual_wilder(self):
        d = self.c.diff().to_numpy()
        up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
        ag, al = up[1], dn[1]
        for i in range(2, 80):
            ag, al = ag + (up[i] - ag) / 14, al + (dn[i] - al) / 14
        self.assertAlmostEqual(ti.rsi(self.c).iloc[79], 100 - 100 / (1 + ag / al), places=6)

    def test_macd_identities(self):
        m = ti.macd(self.c)
        ok = m.dropna()
        self.assertTrue(np.allclose(ok["macd_hist"], ok["macd"] - ok["macd_signal"]))
        self.assertTrue(m["macd"].iloc[:25].isna().all())
        self.assertTrue(m["macd_signal"].iloc[:33].isna().all())

    def test_bollinger_manual(self):
        b = ti.bollinger(self.c)
        w = self.c.iloc[100 - 19:101]
        self.assertAlmostEqual(b["bb_mid"].iloc[100], w.mean())
        self.assertAlmostEqual(b["bb_upper"].iloc[100], w.mean() + 2 * w.std(ddof=0))
        ok = b.dropna()
        self.assertTrue((ok["bb_upper"] >= ok["bb_mid"]).all() and (ok["bb_mid"] >= ok["bb_lower"]).all())

    def test_atr_manual_and_positive(self):
        df = self.df
        tr = ti.true_range(df)
        pc = df["close"].shift()
        self.assertAlmostEqual(tr.iloc[5], max(df.high.iloc[5] - df.low.iloc[5], abs(df.high.iloc[5] - pc.iloc[5]), abs(df.low.iloc[5] - pc.iloc[5])))
        a = ti.atr(df).dropna()
        self.assertTrue((a > 0).all())

    def test_adx_range_and_trend_sensitivity(self):
        a = ti.adx(trending(300))["adx14"].dropna()
        self.assertTrue(((a >= 0) & (a <= 100)).all())
        from tests.helpers import ranging
        self.assertGreater(a.iloc[-1], ti.adx(ranging(300))["adx14"].dropna().iloc[-1])

    def test_momentum_and_volatility(self):
        self.assertAlmostEqual(ti.momentum(self.c, 10).iloc[30], (self.c.iloc[30] / self.c.iloc[20] - 1) * 100)
        v = ti.rolling_volatility(self.c, 20)
        self.assertAlmostEqual(v.iloc[40], np.log(self.c).diff().iloc[21:41].std())
        self.assertTrue(v.iloc[:20].isna().all())

    def test_short_input_never_produces_misleading_values(self):
        out = ti.compute_all(make_ohlcv(np.linspace(100, 101, 10)))
        for col in ("sma20", "sma50", "ema20", "rsi14", "macd", "atr14", "adx14", "vol20", "bb_mid"):
            self.assertTrue(out[col].isna().all(), col)

    def test_compute_all_columns(self):
        out = ti.compute_all(self.df)
        for col in ("sma20", "sma50", "ema20", "rsi14", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower",
                    "atr14", "adx14", "momentum10", "vol20", "vol_ratio"):
            self.assertIn(col, out.columns)
        self.assertEqual(len(out), len(self.df))


if __name__ == "__main__":
    unittest.main()
