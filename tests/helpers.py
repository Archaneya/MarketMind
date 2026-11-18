"""Shared synthetic-data builders for tests."""
import numpy as np
import pandas as pd


def make_ohlcv(closes, spread=0.0005, volume=1000.0, seed=0):
    rng = np.random.default_rng(seed)
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + spread * rng.random(len(c)))
    lo = np.minimum(o, c) * (1 - spread * rng.random(len(c)))
    v = volume * (1 + 0.1 * rng.standard_normal(len(c))).clip(0.5)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v},
                        index=pd.RangeIndex(len(c), name="time"))


def trending(n=300, drift=0.0006, vol=0.0004, seed=1):
    r = np.random.default_rng(seed).normal(drift, vol, n)
    return make_ohlcv(24500 * np.exp(np.cumsum(r)), seed=seed)


def ranging(n=300, vol=0.0005, seed=2, k=0.15):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] * (1 - k) + rng.normal(0, vol)
    return make_ohlcv(24500 * np.exp(x), seed=seed)
