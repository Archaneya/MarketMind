import unittest

import numpy as np
import pandas as pd

from backend.market_state import CandleAggregator, resample_candles
from backend.simulator import MarketSimulator
from indicators.technical_indicators import compute_all
from ml import model_utils as mu
from ml.feature_engineering import build_features
from ml.regime_detection import RegimeDetector


def simulated_features(seed, minutes=14000, tf=3):
    sim, agg, truth = MarketSimulator(seed=seed, start_epoch=0), CandleAggregator(), {}
    for _ in range(minutes * 3):
        t = sim.next_tick()
        agg.update(t)
        truth.setdefault(int(t.epoch // 60 * 60), []).append(t.regime)
    feats = build_features(compute_all(resample_candles(agg.candles(), tf))).dropna()
    true = pd.Series({k: max(set(v), key=v.count) for k, v in truth.items()}).reindex(feats.index).ffill()
    return feats, true


class ClusterInterpretationTests(unittest.TestCase):
    def test_hungarian_mapping_uses_each_regime_once_and_follows_centroids(self):
        z = lambda **k: [k.get(c, 0.0) for c in mu.FEATURE_COLUMNS]
        centroids = np.array([
            z(ma_diff=-1.5, momentum=-1.4, ret_5=-1.0),     # 0 -> bearish
            z(roll_vol=2.0, atr_pct=2.0, volume_change=1.0),  # 1 -> high vol
            z(ma_diff=0.05, momentum=0.0, roll_vol=-0.8, atr_pct=-0.8),  # 2 -> sideways
            z(ma_diff=1.4, momentum=1.5, ret_5=1.0),        # 3 -> bullish
        ])
        m = mu.assign_regime_labels(centroids)
        self.assertEqual(m, {0: mu.BEAR, 1: mu.HIGH_VOL, 2: mu.SIDEWAYS, 3: mu.BULL})
        # relabelling clusters (arbitrary KMeans numbering) must not change the meaning
        perm = [3, 1, 0, 2]
        m2 = mu.assign_regime_labels(centroids[perm])
        self.assertEqual([m2[i] for i in range(4)], [m[p] for p in perm])

    def test_wrong_cluster_count_rejected(self):
        with self.assertRaises(ValueError):
            mu.assign_regime_labels(np.zeros((3, 7)))

    def test_confidence_margin(self):
        self.assertAlmostEqual(mu.confidence_from_distances([1, 1, 5, 5]), 0.0)
        self.assertGreater(mu.confidence_from_distances([0.1, 3, 4, 5]), 0.9)


class DetectorTests(unittest.TestCase):
    def test_warming_up_until_enough_samples(self):
        feats, _ = simulated_features(1, minutes=3000)
        r = RegimeDetector(min_samples=10_000).update(feats)
        self.assertEqual(r.regime, mu.WARMING_UP)
        self.assertFalse(r.ready)

    def test_detector_beats_chance_by_a_wide_margin(self):
        accs = []
        for seed in range(6):
            feats, true = simulated_features(seed)
            det = RegimeDetector(window=2000, seed=seed)
            det.update(feats)
            X = feats.tail(2000)
            accs.append((det.classify_all(X) == true.loc[X.index]).mean())
        self.assertGreater(np.mean(accs), 0.6, accs)       # chance level is 0.25
        self.assertEqual(sorted(det.mapping.values()), sorted(mu.REGIMES))

    def test_online_update_refits_and_returns_explanations(self):
        feats, _ = simulated_features(3)
        det = RegimeDetector(refit_every=5, seed=3)
        for end in range(len(feats) - 12, len(feats)):
            r = det.update(feats.iloc[:end])
        self.assertGreaterEqual(det.fits, 2)
        self.assertIn(r.regime, mu.REGIMES)
        self.assertEqual(len(r.reasons), 3)
        self.assertEqual(len(det.cluster_notes), 4)

    def test_features_are_nan_until_history_exists(self):
        feats, _ = simulated_features(2, minutes=600)
        self.assertEqual(list(feats.columns), mu.FEATURE_COLUMNS)
        self.assertTrue(np.isfinite(feats.to_numpy()).all())


class GaussianMixtureProductionTests(unittest.TestCase):
    def test_production_detector_uses_gmm_and_returns_membership_score(self):
        feats, _ = simulated_features(17, minutes=5000)
        detector = RegimeDetector(min_samples=100, seed=17)
        result = detector.update(feats)
        self.assertIsNotNone(detector.gmm)
        self.assertTrue(result.ready)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertEqual(len(detector.mapping), 4)


if __name__ == "__main__":
    unittest.main()
