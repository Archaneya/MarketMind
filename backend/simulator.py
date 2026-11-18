"""Synthetic NIFTY 50 tick generator.

Prices follow a regime-switching geometric Brownian motion (GBM):

    log P[t+dt] = log P[t] + (mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z      Z ~ N(0, 1)

so P = exp(log P) is always strictly positive. Each regime has its own drift (mu) and
volatility (sigma). The sideways regime adds an Ornstein-Uhlenbeck pull towards the price
level where the regime began, and the high-volatility regime adds occasional jumps.
Regimes switch after random durations, to a random *different* regime.

This is a SIMULATION. Nothing here is real market data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Keep these strings identical to ml/model_utils.py (checked by tests).
BULL = "Bullish Trend"
BEAR = "Bearish Trend"
SIDEWAYS = "Sideways / Range-bound"
HIGH_VOL = "High Volatility"
REGIMES = (BULL, BEAR, SIDEWAYS, HIGH_VOL)


@dataclass(frozen=True)
class RegimeParams:
    drift: float            # log-drift per simulated minute
    sigma: float            # volatility per sqrt(simulated minute)
    reversion: float = 0.0  # OU mean-reversion speed per minute (sideways only)
    jump_prob: float = 0.0  # probability of a jump on any tick
    jump_sigma: float = 0.0  # std-dev of a jump (log-return)
    volume_mult: float = 1.0


PARAMS = {
    BULL: RegimeParams(drift=1.5e-4, sigma=4.0e-4, volume_mult=1.1),
    BEAR: RegimeParams(drift=-1.5e-4, sigma=4.5e-4, volume_mult=1.2),
    SIDEWAYS: RegimeParams(drift=0.0, sigma=3.0e-4, reversion=0.03, volume_mult=0.8),
    HIGH_VOL: RegimeParams(drift=0.0, sigma=1.2e-3, jump_prob=0.004, jump_sigma=2.5e-3, volume_mult=2.0),
}


@dataclass(frozen=True)
class Tick:
    epoch: float      # simulated unix time (seconds, UTC)
    price: float      # last traded price (LTP)
    volume: float     # volume traded in this tick
    regime: str       # the simulator's hidden true regime
    log_return: float


class MarketSimulator:
    def __init__(
        self,
        seed: int | None = None,
        start_price: float = 24500.0,
        start_epoch: float = 1_767_600_000.0,
        seconds_per_tick: float = 20.0,
        regime_minutes: tuple[int, int] = (120, 480),
        initial_regime: str | None = None,
        base_volume: float = 25_000.0,
    ) -> None:
        if start_price <= 0:
            raise ValueError("start_price must be positive")
        self.rng = np.random.default_rng(seed)
        self.price = float(start_price)
        self.epoch = float(start_epoch)
        self.spt = float(seconds_per_tick)
        self.dt = self.spt / 60.0  # tick length in minutes
        self.regime_minutes = regime_minutes
        self.base_volume = base_volume
        self.regime = initial_regime or str(self.rng.choice(REGIMES))
        self.tick_count = 0
        self._begin_regime()

    def _begin_regime(self) -> None:
        lo, hi = self.regime_minutes
        minutes = int(self.rng.integers(lo, hi + 1))
        self.remaining = max(1, int(minutes * 60 / self.spt))
        self.anchor = math.log(self.price)  # OU anchor for the sideways regime

    def _switch_regime(self) -> None:
        self.regime = str(self.rng.choice([r for r in REGIMES if r != self.regime]))
        self._begin_regime()

    def next_tick(self) -> Tick:
        p = PARAMS[self.regime]
        drift = p.drift - 0.5 * p.sigma ** 2
        if p.reversion:
            drift -= p.reversion * (math.log(self.price) - self.anchor)
        r = drift * self.dt + p.sigma * math.sqrt(self.dt) * float(self.rng.standard_normal())
        if p.jump_prob and self.rng.random() < p.jump_prob:
            r += float(self.rng.normal(0.0, p.jump_sigma))
        r = max(-0.05, min(0.05, r))  # safety clamp
        self.price *= math.exp(r)
        volume = (self.base_volume * p.volume_mult * math.exp(0.35 * float(self.rng.standard_normal()))
                  * (1.0 + 400.0 * abs(r)))
        tick = Tick(self.epoch, self.price, volume, self.regime, r)
        self.epoch += self.spt
        self.tick_count += 1
        self.remaining -= 1
        if self.remaining <= 0:
            self._switch_regime()
        return tick
