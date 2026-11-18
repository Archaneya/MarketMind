"""Technical indicators on OHLCV DataFrames (columns: open, high, low, close, volume).

Conventions
-----------
* Every indicator returns NaN until it has enough history (min_periods == window), so a
  half-warmed-up value is never shown as if it were real.
* RSI / ATR / ADX use Wilder smoothing: EMA with alpha = 1/n, seeded from the first value.
* Bollinger Bands use the population standard deviation (ddof = 0), as most charting tools do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)   # only gains -> 100
    out = out.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)   # flat -> 50
    return out.where(avg_gain.notna() & avg_loss.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower) / mid * 100.0
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
                         "bb_width": width, "pct_b": pct_b})


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_s = true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr_s
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return pd.DataFrame({"adx14": dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(),
                         "plus_di": plus_di, "minus_di": minus_di})


def momentum(close: pd.Series, n: int = 10) -> pd.Series:
    """Rate of change over n bars, in percent."""
    return (close / close.shift(n) - 1.0) * 100.0


def rolling_volatility(close: pd.Series, n: int = 20) -> pd.Series:
    """Std-dev of log returns over n bars (per-bar, not annualised)."""
    return np.log(close).diff().rolling(n, min_periods=n).std()


def pct_rank(s: pd.Series, window: int = 200) -> float:
    """Share (0-1) of the last `window` valid values that are <= the latest value."""
    v = s.dropna().tail(window)
    if len(v) < 20:
        return float("nan")
    return float((v <= v.iloc[-1]).mean())


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with every indicator column added."""
    out = df.copy()
    c = out["close"]
    out["ret"] = np.log(c).diff()
    out["sma20"], out["sma50"], out["ema20"] = sma(c, 20), sma(c, 50), ema(c, 20)
    out["rsi14"] = rsi(c, 14)
    out = out.join(macd(c)).join(bollinger(c))
    out["atr14"] = atr(out, 14)
    out = out.join(adx(out, 14))
    out["momentum10"] = momentum(c, 10)
    out["vol20"] = rolling_volatility(c, 20)
    out["vol_ratio"] = out["volume"].rolling(5, min_periods=5).mean() / out["volume"].rolling(20, min_periods=20).mean()
    return out
