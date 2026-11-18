"""Helpers that turn anonymous clustering components into named market regimes.

A clustering algorithm knows nothing about "bullish" or "bearish": it only groups similar feature vectors and
numbers the groups 0..3 arbitrarily (and the numbering can change on every refit). We therefore
interpret each cluster AFTER fitting by looking at its centroid in standardised (z-score) space:

    trend  = z(ma_diff) + z(momentum) + 0.5 * z(ret_5)
    vol    = z(roll_vol) + z(atr_pct)

    Bullish   score =  trend
    Bearish   score = -trend
    High Vol  score =  vol + 0.25 * z(volume_change)
    Sideways  score = -(|z(ma_diff)| + |z(momentum)|) - 0.5 * vol

A one-to-one assignment (Hungarian algorithm) then picks the cluster->regime mapping that
maximises the total score, so each regime is used exactly once. These are transparent heuristics,
not learned labels.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

BULL = "Bullish Trend"
BEAR = "Bearish Trend"
SIDEWAYS = "Sideways / Range-bound"
HIGH_VOL = "High Volatility"
REGIMES = (BULL, BEAR, SIDEWAYS, HIGH_VOL)
WARMING_UP = "Warming Up"

FEATURE_COLUMNS = ["ret_5", "roll_vol", "rsi", "ma_diff", "momentum", "atr_pct", "volume_change"]


def regime_scores(z: dict) -> dict:
    trend = z["ma_diff"] + z["momentum"] + 0.5 * z["ret_5"]
    vol = z["roll_vol"] + z["atr_pct"]
    volume_change = z.get("volume_change", 0.0)
    return {
        BULL: trend,
        BEAR: -trend,
        HIGH_VOL: vol + 0.25 * volume_change,
        SIDEWAYS: -(abs(z["ma_diff"]) + abs(z["momentum"])) - 0.5 * vol,
    }


def assign_regime_labels(centroids_z: np.ndarray, feature_columns=FEATURE_COLUMNS) -> dict:
    """Map cluster index -> regime name with a one-to-one (Hungarian) assignment."""
    k = len(centroids_z)
    if k != len(REGIMES):
        raise ValueError(f"expected {len(REGIMES)} clusters, got {k}")
    scores = np.array([[regime_scores(dict(zip(feature_columns, c)))[r] for r in REGIMES] for c in centroids_z])
    rows, cols = linear_sum_assignment(-scores)
    return {int(i): REGIMES[int(j)] for i, j in zip(rows, cols)}


def describe_clusters(mapping: dict, centroids_raw: np.ndarray, feature_columns=FEATURE_COLUMNS) -> list:
    out = []
    for cluster, regime in sorted(mapping.items()):
        c = dict(zip(feature_columns, centroids_raw[cluster]))
        out.append(f"Cluster {cluster} -> {regime}: MA spread {c['ma_diff']:+.3f}%, momentum {c['momentum']:+.2f}%, "
                   f"volatility {c['roll_vol']:.3f}%/bar")
    return out


def confidence_from_distances(dists: np.ndarray) -> float:
    """Separation between the nearest and second-nearest centroid (0 = ambiguous, 1 = clear-cut).
    This is a geometric margin, NOT a probability."""
    d = np.sort(np.asarray(dists, dtype=float))
    if len(d) < 2 or d[1] <= 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - d[0] / d[1])))


def explain_observation(regime: str, raw: dict, z: dict) -> list:
    """Plain-language reasons for why the latest observation sits in `regime`."""
    r = []
    spread, mom, vol, atrp, rsi = raw["ma_diff"], raw["momentum"], raw["roll_vol"], raw["atr_pct"], raw["rsi"]
    if regime in (BULL, BEAR):
        side = "above" if spread >= 0 else "below"
        r.append(f"SMA20 is {abs(spread):.3f}% {side} SMA50 (z={z['ma_diff']:+.1f}), the short average leading the long one.")
        r.append(f"10-bar momentum is {mom:+.2f}% (z={z['momentum']:+.1f}).")
        r.append(f"RSI is {rsi:.0f}, {'above' if rsi >= 50 else 'below'} the neutral 50 line.")
    elif regime == HIGH_VOL:
        r.append(f"Rolling volatility is {vol:.3f}% per bar (z={z['roll_vol']:+.1f} vs recent history).")
        r.append(f"ATR is {atrp:.3f}% of price (z={z['atr_pct']:+.1f}).")
        if "volume_change" in z:
            r.append(f"Volume is {z['volume_change']:+.1f} standard deviations versus its recent average.")
    elif regime == SIDEWAYS:
        r.append(f"SMA20 and SMA50 are close together ({spread:+.3f}%), so there is no clear trend.")
        r.append(f"Momentum is small ({mom:+.2f}%) and volatility is subdued ({vol:.3f}% per bar).")
        r.append(f"RSI is {rsi:.0f}, near neutral." if 40 <= rsi <= 60 else f"RSI is {rsi:.0f}: price is stretching within the range.")
    return r
