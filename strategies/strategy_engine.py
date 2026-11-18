"""Runs all five evaluators on an indicator frame and ranks them by suitability score."""
from __future__ import annotations

from strategies import breakout, mean_reversion, momentum, trend_following, volatility

EVALUATORS = (momentum.evaluate, mean_reversion.evaluate, breakout.evaluate,
              trend_following.evaluate, volatility.evaluate)


def evaluate_all(ind) -> list:
    results = sorted((fn(ind) for fn in EVALUATORS), key=lambda r: r.score, reverse=True)
    for i, r in enumerate(results, 1):
        r.rank = i
    return results
