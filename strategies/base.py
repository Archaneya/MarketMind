"""Shared types and scoring helpers for the strategy evaluators.

Every evaluator is a pure function  evaluate(ind: DataFrame) -> StrategyResult  that reads ONLY
indicator values (never the ML regime), so strategy scoring stays independent of regime detection.

Scores are the weighted sum of components, each normalised to 0..1, times 100. They measure
RELATIVE SUITABILITY of the current conditions for that style. They are not win probabilities
and not expected returns. Where possible, inputs are divided by recent volatility or ATR so the
thresholds do not depend on the candle timeframe.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import pandas as pd

SUITABLE_AT = 60.0
UNSUITABLE_BELOW = 40.0


@dataclass
class StrategyResult:
    name: str
    score: float
    compatibility: str
    supporting: dict = field(default_factory=dict)
    reason: str = ""
    risks: list = field(default_factory=list)
    components: dict = field(default_factory=dict)
    rank: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def scale(x, lo: float, hi: float) -> float:
    """Linear map of x from [lo, hi] to [0, 1], clipped. NaN -> 0."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def compatibility(score: float) -> str:
    return "High" if score >= 70 else "Moderate" if score >= SUITABLE_AT - 15 else "Low" if score >= 25 else "Poor"


def insufficient(name: str, needs: str) -> StrategyResult:
    return StrategyResult(name, 0.0, "N/A", {}, f"Not enough history yet to evaluate {needs}.",
                          ["Insufficient data: score withheld rather than guessed."])


def ready(ind: pd.DataFrame, cols: list, min_rows: int = 2) -> bool:
    return len(ind) >= min_rows and not ind.iloc[-1][cols].isna().any()


def finalize(name: str, parts: dict, supporting: dict, detail: str, risks: list) -> StrategyResult:
    """parts: {label: (weight, value in 0..1)}."""
    total_w = sum(w for w, _ in parts.values())
    score = round(100.0 * sum(w * v for w, v in parts.values()) / total_w, 1)
    ranked = sorted(parts.items(), key=lambda kv: kv[1][0] * kv[1][1], reverse=True)
    drivers = ", ".join(k for k, (w, v) in ranked[:2] if v > 0.35)
    weak = ", ".join(k for k, (w, v) in ranked[::-1][:1] if v < 0.2)
    note = (f" Strongest support: {drivers}." if drivers else " No component gives strong support.")
    if weak:
        note += f" Weakest: {weak}."
    return StrategyResult(name, score, compatibility(score), supporting, detail + note, risks,
                          {k: round(v, 2) for k, (w, v) in parts.items()})
