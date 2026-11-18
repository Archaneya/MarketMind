"""Market engine: simulator -> tick aggregation -> bounded history -> analytics (ML + strategies).

Framework-free (no FastAPI imports) so it can be unit-tested and reused. All message dicts it
returns are JSON-serialisable.
"""
from __future__ import annotations

import asyncio
import math
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from backend.intelligence import build_intelligence, describe_indicators
from backend.simulator import MarketSimulator, Tick
from indicators.technical_indicators import compute_all
from ml.feature_engineering import build_features
from ml.model_utils import WARMING_UP
from ml.regime_detection import RegimeDetector
from strategies.strategy_engine import evaluate_all

SYMBOL = "NIFTY 50"
SNAPSHOT_CANDLES = 6_600  # 2,200 three-minute bars: enough to train the regime model
REFERENCE_PRICE = 24_500.0
IST_OFFSET = 19_800  # +05:30, used only to decide where a "day" starts


def to_jsonable(obj):
    """Recursively convert numpy/pandas scalars to Python types; NaN/inf -> None (valid JSON)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, deque)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class CandleAggregator:
    """Builds 1-minute OHLCV candles from ticks. A new candle opens at the previous close, so
    there are no gaps, and high/low always bracket open and close."""

    def __init__(self, maxlen: int = 30_000, seconds: int = 60) -> None:
        self.sec = seconds
        self.closed: deque = deque(maxlen=maxlen)
        self.current: Optional[dict] = None

    def update(self, tick: Tick) -> dict:
        start = int(tick.epoch // self.sec * self.sec)
        cur = self.current
        if cur is not None and start < cur["time"]:
            raise ValueError("out-of-order tick")
        if cur is not None and start == cur["time"]:
            cur["high"] = max(cur["high"], tick.price)
            cur["low"] = min(cur["low"], tick.price)
            cur["close"] = tick.price
            cur["volume"] += tick.volume
        else:
            if cur is not None:
                self.closed.append(cur)
            o = cur["close"] if cur is not None else tick.price
            self.current = {"time": start, "open": o, "high": max(o, tick.price),
                            "low": min(o, tick.price), "close": tick.price, "volume": tick.volume}
        return self.current

    def candles(self, last: Optional[int] = None) -> list:
        out = list(self.closed) + ([dict(self.current)] if self.current else [])
        return out[-last:] if last else out


def resample_candles(candles: list, minutes: int, offset: int = 0) -> pd.DataFrame:
    """Aggregate base candles into `minutes`-minute candles (open=first, high=max, low=min,
    close=last, volume=sum). `offset` shifts bucket boundaries (e.g. IST day starts)."""
    df = pd.DataFrame(candles)
    sec = minutes * 60
    bucket = ((df["time"] + offset) // sec) * sec - offset
    g = df.groupby(bucket)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    out.index.name = "time"
    return out


def validate_tick(m: dict) -> None:
    for k in ("timestamp", "symbol", "ltp", "open", "high", "low", "close", "volume", "regime"):
        if k not in m:
            raise ValueError(f"missing field {k}")
    for k in ("ltp", "open", "high", "low", "close"):
        if not (isinstance(m[k], (int, float)) and math.isfinite(m[k]) and m[k] > 0):
            raise ValueError(f"invalid {k}: {m[k]!r}")
    if not (math.isfinite(m["volume"]) and m["volume"] >= 0):
        raise ValueError("invalid volume")
    if m["high"] < max(m["open"], m["close"]) - 1e-9 or m["low"] > min(m["open"], m["close"]) + 1e-9:
        raise ValueError("inconsistent OHLC")


class MarketEngine:
    def __init__(self, seed: Optional[int] = None, warmup_minutes: int = 10_000, analytics_every: int = 4,
                 seconds_per_tick: float = 20.0, analysis_minutes: int = 3, max_candles: int = 30_000) -> None:
        start = int(time.time() // 60 * 60) - warmup_minutes * 60
        self.sim = MarketSimulator(seed=seed, start_epoch=start, seconds_per_tick=seconds_per_tick)
        self.agg = CandleAggregator(maxlen=max_candles)
        self.detector = RegimeDetector(seed=seed or 0)
        model_path = os.environ.get("MARKETMIND_MODEL_PATH", "ml/models/nifty50_gmm_3m.joblib")
        if os.path.isfile(model_path):
            try:
                self.detector.load_artifact(model_path)
            except Exception as exc:
                # Keep the existing simulator fallback operational if an artifact is invalid.
                print(f"[MarketMind] Could not load offline ML model ({model_path}): {exc}")
        self.analytics_every = analytics_every
        self.analysis_minutes = analysis_minutes
        self.seq = 0
        self.regime = WARMING_UP
        self.confidence = 0.0
        self.analytics: Optional[dict] = None
        self.regime_history: deque = deque(maxlen=300)
        self.regime_changes: deque = deque(maxlen=20)
        self._day_key = None
        self._day_open = self.sim.price
        for _ in range(int(warmup_minutes * 60 / seconds_per_tick)):
            self._advance()
        self._rescale(REFERENCE_PRICE / self.sim.price)   # live market starts near Rs 24,500
        self.compute_analytics(self.agg.candles(SNAPSHOT_CANDLES))

    # -- ticks ---------------------------------------------------------------------------
    def _advance(self) -> tuple:
        tick = self.sim.next_tick()
        cur = self.agg.update(tick)
        day = int((tick.epoch + IST_OFFSET) // 86400)
        if day != self._day_key:
            self._day_key, self._day_open = day, tick.price
        self.seq += 1
        return tick, cur

    def _rescale(self, k: float) -> None:
        """Multiply every price by k. GBM is scale-invariant, so this changes the level, not the dynamics."""
        for c in (*self.agg.closed, self.agg.current):
            for f in ("open", "high", "low", "close"):
                c[f] *= k
        self.sim.price *= k
        self.sim.anchor += math.log(k)
        self._day_open *= k

    def step(self) -> dict:
        """Generate one tick and return the validated JSON-ready message."""
        tick, cur = self._advance()
        msg = {
            "type": "tick", "seq": self.seq,
            "timestamp": datetime.fromtimestamp(tick.epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "epoch": tick.epoch, "symbol": SYMBOL, "ltp": tick.price,
            "open": cur["open"], "high": cur["high"], "low": cur["low"], "close": cur["close"],
            "volume": cur["volume"], "tick_volume": tick.volume,
            "change": tick.price - self._day_open,
            "change_pct": (tick.price / self._day_open - 1.0) * 100.0,
            "regime": self.regime, "regime_confidence": self.confidence,
            "simulator_regime": tick.regime,
            "candle": dict(cur), "server_time_ms": int(time.time() * 1000),
        }
        validate_tick(msg)
        return msg

    def analytics_due(self) -> bool:
        return self.seq % self.analytics_every == 0

    # -- analytics -----------------------------------------------------------------------
    def compute_analytics(self, snapshot: list) -> dict:
        """CPU-heavy: resample, indicators, ML regime, strategies. Safe to run in a worker thread
        because it only reads `snapshot` and mutates detector/history that only this method touches."""
        df = resample_candles(snapshot, self.analysis_minutes)
        ind = compute_all(df)
        res = self.detector.update(build_features(ind))
        strategies = evaluate_all(ind)
        epoch = float(snapshot[-1]["time"])
        if res.regime != self.regime and res.ready:
            if self.regime != WARMING_UP:
                self.regime_changes.append({"epoch": epoch, "from": self.regime, "to": res.regime})
        self.regime, self.confidence = res.regime, res.confidence
        self.regime_history.append([epoch, res.regime])
        payload = {
            "type": "analytics", "epoch": epoch, "seq": self.seq,
            "analysis_timeframe": f"{self.analysis_minutes}m",
            "regime": {"name": res.regime, "raw": res.raw_regime, "cluster": res.cluster,
                       "confidence": res.confidence, "reasons": res.reasons, "features": res.features},
            "model": {"fits": self.detector.fits, "samples": self.detector.n_samples,
                      "cluster_map": self.detector.cluster_notes},
            "simulator_regime": self.sim.regime,
            "indicators": describe_indicators(ind),
            "strategies": [s.to_dict() for s in strategies],
            "intelligence": build_intelligence(res, strategies, ind),
            "regime_history": list(self.regime_history)[-120:],
            "regime_changes": list(self.regime_changes)[-8:],
            "server_time_ms": int(time.time() * 1000),
        }
        self.analytics = to_jsonable(payload)
        return self.analytics

    async def run_analytics(self) -> dict:
        snap = self.agg.candles(SNAPSHOT_CANDLES)  # copy, so the worker thread never sees live mutation
        return await asyncio.to_thread(self.compute_analytics, snap)

    # -- snapshots for new / resyncing clients --------------------------------------------
    def history_message(self, max_candles: int = 10_000) -> dict:
        return {"type": "history", "seq": self.seq, "symbol": SYMBOL,
                "candles": to_jsonable(self.agg.candles(max_candles)), "server_time_ms": int(time.time() * 1000)}
