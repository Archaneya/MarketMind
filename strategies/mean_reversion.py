"""Strategy 2: Mean reversion - fade stretched prices back toward their averages.

Components (weight): Bollinger stretch (0.35), distance from SMA50 in ATRs (0.20),
RSI extreme in the same direction as the stretch (0.20), absence of a trend via ADX (0.25).
"""
import numpy as np

from strategies.base import finalize, insufficient, ready, scale

NAME = "Mean Reversion"


def evaluate(ind):
    cols = ["pct_b", "sma20", "sma50", "rsi14", "atr14", "adx14", "close"]
    if not ready(ind, cols):
        return insufficient(NAME, "Bollinger Bands, moving averages and ADX")
    r = ind.iloc[-1]
    z = (float(r["pct_b"]) - 0.5) * 4.0          # position in standard deviations (bands are +/-2 sd)
    s = 1 if z >= 0 else -1
    dist = (float(r["close"]) - float(r["sma50"])) / float(r["atr14"])
    rsi, adx = float(r["rsi14"]), float(r["adx14"])
    parts = {
        "Bollinger stretch": (0.35, scale(abs(z), 0.8, 2.5)),
        "distance from SMA50": (0.20, scale(abs(dist), 1.0, 4.0) * (1.0 if np.sign(dist) == s else 0.5)),
        "RSI extreme": (0.20, scale((rsi - 50) * s, 5, 25)),
        "no strong trend": (0.25, 1.0 - scale(adx, 15, 35)),
    }
    risks = []
    if adx > 30:
        risks.append(f"ADX {adx:.0f} signals a strong trend: fading it can mean repeated losses.")
    if abs(z) < 0.8:
        risks.append("Price is near its average: there is little stretch to revert from.")
    if abs(z) > 2.5:
        risks.append("Price is far outside the bands; a breakout may be under way, not a reversion.")
    where = "above" if z >= 0 else "below"
    return finalize(NAME, parts,
                    {"Bollinger %B": f"{float(r['pct_b']):.2f}", "Dist. from SMA50": f"{dist:+.1f} ATR", "ADX(14)": f"{adx:.0f}"},
                    f"Price is {abs(z):.1f} standard deviations {where} its 20-bar mean and {abs(dist):.1f} ATR from SMA50; ADX is {adx:.0f}.",
                    risks or ["Reversion fails if a trend starts; keep stops beyond recent extremes."])
