"""Turns regime + strategy + indicator output into short, readable explanations."""
from __future__ import annotations

import math

from indicators.technical_indicators import pct_rank
from strategies.base import SUITABLE_AT, UNSUITABLE_BELOW


def _f(x, nd=2, suffix=""):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{nd}f}{suffix}"


def describe_indicators(ind) -> list:
    """Latest indicator readings with a one-phrase interpretation each."""
    if ind.empty:
        return []
    r = ind.iloc[-1]
    c = float(r["close"])
    rsi = r["rsi14"]
    rsi_txt = ("insufficient data" if rsi != rsi else "overbought" if rsi >= 70 else "oversold" if rsi <= 30
               else "bullish bias" if rsi > 55 else "bearish bias" if rsi < 45 else "neutral")

    def vs(level):
        return "insufficient data" if level != level else ("price above" if c >= level else "price below")

    hist = r["macd_hist"]
    pb = r["pct_b"]
    return [
        {"name": "RSI (14)", "value": _f(rsi, 1), "reading": rsi_txt},
        {"name": "SMA (20)", "value": _f(r["sma20"]), "reading": vs(r["sma20"])},
        {"name": "SMA (50)", "value": _f(r["sma50"]), "reading": vs(r["sma50"])},
        {"name": "EMA (20)", "value": _f(r["ema20"]), "reading": vs(r["ema20"])},
        {"name": "MACD", "value": f"{_f(r['macd'])} / {_f(r['macd_signal'])}",
         "reading": "insufficient data" if hist != hist else "bullish (hist > 0)" if hist > 0 else "bearish (hist < 0)"},
        {"name": "Bollinger %B", "value": _f(pb, 2),
         "reading": "insufficient data" if pb != pb else "near upper band" if pb > 0.85 else "near lower band" if pb < 0.15 else "inside bands"},
        {"name": "Band width", "value": _f(r["bb_width"], 3, "%"), "reading": "squeeze" if pct_rank(ind["bb_width"], 200) < 0.2 else "normal/wide"},
        {"name": "ATR (14)", "value": _f(r["atr14"]), "reading": f"{_f(r['atr14'] / c * 100, 3)}% of price"},
        {"name": "ADX (14)", "value": _f(r["adx14"], 1),
         "reading": "insufficient data" if r["adx14"] != r["adx14"] else "strong trend" if r["adx14"] > 25 else "weak trend" if r["adx14"] < 20 else "developing trend"},
        {"name": "Momentum (10)", "value": _f(r["momentum10"], 2, "%"), "reading": "rising" if r["momentum10"] > 0 else "falling"},
        {"name": "Rolling vol (20)", "value": _f(r["vol20"] * 100, 3, "%"), "reading": "per bar, std of log returns"},
    ]


def build_intelligence(regime, strategies, ind) -> dict:
    suitable = [s.name for s in strategies if s.score >= SUITABLE_AT]
    unsuitable = [s.name for s in strategies if s.score < UNSUITABLE_BELOW]
    top = strategies[0]

    # Volatility conditions from ATR% percentile and vol expansion
    vol_text = "Volatility cannot be assessed yet."
    risks = []
    if len(ind) > 70 and ind["vol20"].iloc[-1] == ind["vol20"].iloc[-1]:
        prank = pct_rank(ind["atr14"] / ind["close"], 200)
        exp = float(ind["vol20"].iloc[-1] / ind["vol20"].iloc[-50:].mean())
        level = "elevated" if prank > 0.75 else "low" if prank < 0.25 else "moderate"
        trend = "expanding" if exp > 1.15 else "contracting" if exp < 0.85 else "steady"
        vol_text = f"Volatility is {level} (ATR at the {prank * 100:.0f}th percentile of recent bars) and {trend}."
        if level == "elevated":
            risks.append("Elevated volatility: wider swings, larger slippage; smaller position sizes are prudent.")
        if level == "low" and trend == "contracting":
            risks.append("Volatility compression often precedes a sharp move in either direction.")
    rsi = ind["rsi14"].iloc[-1] if len(ind) else float("nan")
    if rsi == rsi and rsi >= 70:
        risks.append(f"RSI {rsi:.0f} is overbought: upside moves may be stretched.")
    if rsi == rsi and rsi <= 30:
        risks.append(f"RSI {rsi:.0f} is oversold: downside moves may be stretched.")
    if regime.regime == "Sideways / Range-bound":
        risks.append("Range-bound markets produce false breakouts and trend signals that fail.")
    if regime.regime in ("Bullish Trend", "Bearish Trend"):
        risks.append("Trends can reverse abruptly; lagging indicators confirm turns late.")
    if regime.regime == "High Volatility":
        risks.append("Regime is volatile: indicator signals are noisier and stops are hit more often.")
    risks.append("Simulated data: regimes are generated randomly and cannot be predicted.")

    note = (f"The regime detector and the strategy scorer are independent. Top-ranked strategy: {top.name} "
            f"({top.score:.0f}/100). A disagreement between them is informative, not an error.")
    return {
        "headline": (f"{regime.regime} (cluster separation {regime.confidence:.2f})" if regime.ready else "Collecting data"),
        "why": regime.reasons,
        "suitable": suitable,
        "unsuitable": unsuitable,
        "volatility": vol_text,
        "risks": risks[:5],
        "note": note,
    }
