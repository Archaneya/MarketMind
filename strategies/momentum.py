"""Strategy 1: Momentum trading - ride recent strength/weakness.

Components (weight): momentum size vs noise (0.40), RSI confirmation (0.25),
5-bar return vs noise (0.25), agreement of momentum / RSI / recent return (0.10).
"""
import math

import numpy as np

from strategies.base import finalize, insufficient, ready, scale

NAME = "Momentum Trading"


def evaluate(ind):
    if not ready(ind, ["momentum10", "rsi14", "vol20", "close"], 12):
        return insufficient(NAME, "momentum and RSI")
    r = ind.iloc[-1]
    vol = max(float(r["vol20"]) * 100.0, 1e-9)
    mom, rsi = float(r["momentum10"]), float(r["rsi14"])
    d = 1 if mom >= 0 else -1
    mom_z = mom / (vol * math.sqrt(10))
    ret5 = (ind["close"].iloc[-1] / ind["close"].iloc[-6] - 1) * 100.0
    ret5_z = ret5 / (vol * math.sqrt(5))
    exhausted = rsi > 80 or rsi < 20
    parts = {
        "momentum size": (0.40, scale(abs(mom_z), 0.5, 2.5)),
        "RSI confirmation": (0.25, scale((rsi - 50) * d, 3, 20) * (0.5 if exhausted else 1.0)),
        "recent return": (0.25, scale(ret5_z * d, 0.3, 2.0)),
        "signal agreement": (0.10, 1.0 if np.sign(mom) == np.sign(ret5) == np.sign(rsi - 50) else 0.0),
    }
    risks = []
    if exhausted:
        risks.append(f"RSI {rsi:.0f} is at an extreme: the move may be exhausted and reverse.")
    if parts["signal agreement"][1] == 0:
        risks.append("Momentum, RSI and recent return disagree: whipsaw risk.")
    if parts["momentum size"][1] < 0.2:
        risks.append("Momentum is small compared with normal noise: signals are unreliable.")
    return finalize(NAME, parts,
                    {"Momentum(10)": f"{mom:+.2f}%", "RSI(14)": f"{rsi:.0f}", "5-bar return": f"{ret5:+.2f}%"},
                    f"{'Upward' if d > 0 else 'Downward'} momentum of {mom:+.2f}% is {abs(mom_z):.1f}x the typical noise level; RSI is {rsi:.0f}.",
                    risks or ["Momentum reverses sharply at turning points; use defined exits."])
