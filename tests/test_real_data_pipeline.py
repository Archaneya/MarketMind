import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.yahoo_data import resample_to_three_minutes
from ml.train_model import train
from ml.regime_detection import RegimeDetector


class YahooResamplingTests(unittest.TestCase):
    def test_resamples_session_aligned_three_minute_candles(self):
        idx = pd.date_range("2026-01-05 09:15", periods=6, freq="1min", tz="Asia/Kolkata")
        one = pd.DataFrame({
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "volume": [0] * 6,
        }, index=idx)
        out = resample_to_three_minutes(one)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.iloc[0]["open"], 100)
        self.assertEqual(out.iloc[0]["high"], 103)
        self.assertEqual(out.iloc[0]["low"], 99)
        self.assertEqual(out.iloc[0]["close"], 102.5)
        self.assertEqual(out.iloc[0]["source_1m_bars"], 3)

    def test_drops_incomplete_three_minute_buckets(self):
        idx = pd.date_range("2026-01-05 09:15", periods=5, freq="1min", tz="Asia/Kolkata")
        one = pd.DataFrame({"open": 100, "high": 101, "low": 99, "close": 100,
                            "volume": 0}, index=idx)
        self.assertEqual(len(resample_to_three_minutes(one)), 1)


class OfflineTrainingTests(unittest.TestCase):
    def test_train_saves_artifact_that_detector_can_load(self):
        rng = np.random.default_rng(7)
        n = 900
        # Synthetic fixture is only a software test, not the project's real training dataset.
        returns = rng.normal(0.0001, 0.001, n)
        close = 22000 * np.exp(np.cumsum(returns))
        open_ = np.r_[close[0], close[:-1]]
        spread = np.maximum(close * rng.uniform(0.0001, 0.001, n), 1)
        idx = pd.date_range("2026-01-01 09:15", periods=n, freq="3min", tz="Asia/Kolkata")
        candles = pd.DataFrame({"timestamp": idx, "open": open_, "high": np.maximum(open_, close) + spread,
                                "low": np.minimum(open_, close) - spread, "close": close, "volume": 0})
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "fixture.csv"
            model_path = Path(td) / "model.joblib"
            candles.to_csv(csv_path, index=False)
            train(csv_path, model_path, min_rows=250)
            detector = RegimeDetector.from_artifact(model_path)
            self.assertTrue(detector.offline_model)
            self.assertEqual(len(detector.feature_columns), 6)
            self.assertEqual(detector.gmm.n_components, 4)
            self.assertTrue(detector.cluster_notes)

            from indicators.technical_indicators import compute_all
            from ml.feature_engineering import build_features
            source = candles.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
            feats = build_features(compute_all(source), include_volume=True)
            result = detector.update(feats)
            self.assertTrue(result.ready)
            self.assertIn(result.regime, detector.mapping.values())
            self.assertEqual(detector.fits, 1)  # offline artifact is not refitted on incoming data


if __name__ == "__main__":
    unittest.main()
