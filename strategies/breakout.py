"""Strategy 3: Breakout trading - trade moves through recent support/resistance.

Support/resistance = lowest low / highest high of the previous 20 bars (excluding the current bar).
Components (weight): break or proximity to a level (0.40), volume expansion (0.25),
bar range vs ATR (0.15), prior range compression i.e. a Bollinger squeeze (0.20).
"""
import numpy as np

from indicators.technical_indicators import pct_rank
from strategies.base import finalize, insufficient, ready, scale

NAME = "Breakout Trading"
LOOKBACK = 20


def evaluate(ind):
    if not ready(ind, ["atr14", "vol_ratio", "bb_width", "close"], LOOKBACK + 2):
        return insufficient(NAME, "support/resistance and volume")
    r = ind.iloc[-1]
    prev = ind.iloc[-LOOKBACK - 1:-1]
    res, sup = float(prev["high"].max()), float(prev["low"].min())
    close, atr = float(r["close"]), float(r["atr14"])
    d_up, d_dn = (res - close) / atr, (close - sup) / atr
    broke = close > res or close < sup
    near = 1.0 if broke else 0.8 * (1.0 - scale(min(d_up, d_dn), 0.0, 3.0))
    squeeze = 1.0 - pct_rank(ind["bb_width"], 100)
    squeeze = 0.0 if np.isnan(squeeze) else squeeze
    rng = (float(r["high"]) - float(r["low"])) / atr
    parts = {
        "level break / proximity": (0.40, near),
        "volume expansion": (0.25, scale(float(r["vol_ratio"]), 1.0, 2.0)),
        "bar range vs ATR": (0.15, scale(rng, 1.0, 2.5)),
        "prior compression": (0.20, squeeze),
    }
    state = ("broke above resistance" if close > res else "broke below support" if close < sup
             else f"is {min(d_up, d_dn):.1f} ATR from the nearest level")
    risks = []
    if broke and float(r["vol_ratio"]) < 1.1:
        risks.append("Break on ordinary volume: higher risk of a false breakout.")
    if not broke:
        risks.append("No confirmed break yet; entering early risks a failed move.")
    return finalize(NAME, parts,
                    {"Resistance": f"{res:,.1f}", "Support": f"{sup:,.1f}", "Volume ratio": f"{float(r['vol_ratio']):.2f}x"},
                    f"Price {state}; volume is {float(r['vol_ratio']):.2f}x its 20-bar average.",
                    risks or ["False breaks reverse quickly; place stops back inside the range."])
