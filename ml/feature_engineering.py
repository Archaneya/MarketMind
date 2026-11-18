"""Feature matrix for regime detection, built from the output of compute_all()."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.model_utils import FEATURE_COLUMNS


def build_features(ind: pd.DataFrame, include_volume: bool = True) -> pd.DataFrame:
    """Seven features per bar (all NaN until enough history exists):

    ret_5          5-bar log return, %          roll_vol   20-bar std of log returns, %
    rsi            RSI(14)                      ma_diff    (SMA20 - SMA50) / SMA50, %
    momentum       10-bar rate of change, %     atr_pct    ATR(14) / close, %
    volume_change  5-bar avg volume / 20-bar avg volume - 1
    """
    f = pd.DataFrame(index=ind.index)
    f["ret_5"] = np.log(ind["close"]).diff(5) * 100.0
    f["roll_vol"] = ind["vol20"] * 100.0
    f["rsi"] = ind["rsi14"]
    f["ma_diff"] = (ind["sma20"] - ind["sma50"]) / ind["sma50"] * 100.0
    f["momentum"] = ind["momentum10"]
    f["atr_pct"] = ind["atr14"] / ind["close"] * 100.0
    f["volume_change"] = ind["vol_ratio"] - 1.0
    columns = FEATURE_COLUMNS if include_volume else [c for c in FEATURE_COLUMNS if c != "volume_change"]
    return f[columns].replace([np.inf, -np.inf], np.nan)
