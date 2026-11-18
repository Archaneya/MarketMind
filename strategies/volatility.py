"""Strategy 5: Volatility-based trading - exploit large, fast moves (expansion).

Components (weight): ATR% percentile vs recent history (0.35), rolling-vol expansion vs its
50-bar average (0.35), ATR growth over 10 bars (0.15), Bollinger bandwidth percentile (0.15).
Contracting volatility scores low: there is little movement to exploit.
"""
from indicators.technical_indicators import pct_rank
from strategies.base import finalize, insufficient, ready, scale

NAME = "Volatility-Based"


def _nz(x):
    return 0.0 if x != x else x


def evaluate(ind):
    if not ready(ind, ["atr14", "vol20", "bb_width", "close"], 70):
        return insufficient(NAME, "ATR and rolling volatility")
    r = ind.iloc[-1]
    atr_pct = ind["atr14"] / ind["close"] * 100.0
    prank = _nz(pct_rank(atr_pct, 200))
    expansion = float(r["vol20"]) / float(ind["vol20"].iloc[-50:].mean())
    growth = float(r["atr14"]) / float(ind["atr14"].iloc[-11])
    bw = _nz(pct_rank(ind["bb_width"], 200))
    parts = {
        "ATR percentile": (0.35, prank),
        "volatility expansion": (0.35, scale(expansion, 0.9, 1.8)),
        "ATR growth": (0.15, scale(growth, 1.0, 1.6)),
        "band width": (0.15, bw),
    }
    state = "expanding" if expansion > 1.15 else "contracting" if expansion < 0.85 else "stable"
    risks = []
    if state == "contracting":
        risks.append("Volatility is shrinking: a volatility strategy has little movement to capture.")
    if prank > 0.9:
        risks.append("Volatility is in its top decile: gaps and slippage can be large; reduce position size.")
    return finalize(NAME, parts,
                    {"ATR%": f"{float(atr_pct.iloc[-1]):.3f}%", "Vol state": state, "Vol vs 50-bar avg": f"{expansion:.2f}x"},
                    f"Volatility is {state} ({expansion:.2f}x its 50-bar average); ATR is at the {prank * 100:.0f}th percentile of recent bars.",
                    risks or ["Volatility mean-reverts; spikes can fade as fast as they appear."])
