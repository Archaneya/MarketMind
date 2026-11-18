"""Strategy 4: Trend following - stay with an established trend.

Components (weight): alignment of close/EMA20/SMA20/SMA50 (0.35), ADX trend strength (0.30),
MACD agreeing with the trend direction (0.20), SMA20 slope in ATRs over 10 bars (0.15).
"""
from strategies.base import finalize, insufficient, ready, scale

NAME = "Trend Following"


def evaluate(ind):
    cols = ["ema20", "sma20", "sma50", "adx14", "macd", "macd_hist", "atr14", "close"]
    if not ready(ind, cols, 12):
        return insufficient(NAME, "moving-average alignment and ADX")
    r = ind.iloc[-1]
    c, e20, s20, s50 = (float(r[k]) for k in ("close", "ema20", "sma20", "sma50"))
    bull = (c > e20) + (e20 > s20) + (s20 > s50)
    bear = (c < e20) + (e20 < s20) + (s20 < s50)
    d, align = (1, bull / 3) if bull >= bear else (-1, bear / 3)
    adx = float(r["adx14"])
    macd_ok = 0.5 * (float(r["macd"]) * d > 0) + 0.5 * (float(r["macd_hist"]) * d > 0)
    slope = (s20 - float(ind["sma20"].iloc[-11])) * d / float(r["atr14"])
    parts = {
        "MA alignment": (0.35, scale(align, 2 / 3, 1.0)),   # 2 of 3 agreeing is what noise gives; only a full stack counts
        "trend strength (ADX)": (0.30, scale(adx, 15, 40)),
        "MACD agreement": (0.20, macd_ok),
        "SMA20 slope": (0.15, scale(slope, 0.2, 1.5)),
    }
    risks = []
    if adx < 20:
        risks.append(f"ADX {adx:.0f} is low: trends are weak and signals whipsaw.")
    if align < 1:
        risks.append("Averages are not fully stacked: the trend is unconfirmed or turning.")
    return finalize(NAME, parts,
                    {"Direction": "up" if d > 0 else "down", "ADX(14)": f"{adx:.0f}", "MACD hist": f"{float(r['macd_hist']):+.2f}"},
                    f"Averages are {align * 100:.0f}% aligned for a {'rising' if d > 0 else 'falling'} trend; ADX is {adx:.0f}.",
                    risks or ["Trend exits lag; late reversals give back open profit."])
