"""Download Yahoo Finance NIFTY 50 1-minute bars and build session-aligned 3-minute OHLCV.

Yahoo's 1-minute endpoint has a short lookback window. The script records provider limits
rather than silently pretending this is a multi-year dataset.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SYMBOL = "^NSEI"
IST = "Asia/Kolkata"
REQUIRED_OHLC = ["open", "high", "low", "close"]
OUTPUT_COLUMNS = ["open", "high", "low", "close", "volume"]


def _flatten_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        # yfinance may return (Price, Ticker) or (Ticker, Price).
        if SYMBOL in frame.columns.get_level_values(-1):
            frame.columns = frame.columns.get_level_values(0)
        elif SYMBOL in frame.columns.get_level_values(0):
            frame.columns = frame.columns.get_level_values(1)
        else:
            frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    return frame


def download_one_minute(period: str = "7d") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Run: pip install -r requirements.txt") from exc

    raw = yf.download(SYMBOL, period=period, interval="1m", auto_adjust=False,
                      prepost=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no 1-minute NIFTY 50 data. Check connectivity or Yahoo availability.")
    raw = _flatten_yfinance_columns(raw)
    missing = [c for c in REQUIRED_OHLC if c not in raw.columns]
    if missing:
        raise RuntimeError(f"Yahoo response is missing required columns: {missing}; received {list(raw.columns)}")
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    raw.index = idx.tz_convert(IST)
    raw.index.name = "timestamp"
    if "volume" not in raw.columns:
        raw["volume"] = float("nan")
    raw = raw[OUTPUT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    raw = raw.dropna(subset=["open", "high", "low", "close"])
    # ^NSEI is an index, not a traded instrument. Yahoo volume is commonly zero/missing;
    # do not invent volume. We retain it only as source metadata and do not train on it.
    return raw


def resample_to_three_minutes(one_minute: pd.DataFrame, include_partial: bool = False) -> pd.DataFrame:
    if one_minute.empty:
        raise ValueError("Cannot resample an empty one-minute dataset")
    df = one_minute.copy().sort_index()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        raise ValueError("Input timestamps must be timezone-aware")
    local = idx.tz_convert(IST)
    df.index = local
    # Regular NSE cash-market session only: 09:15 inclusive to 15:30 exclusive.
    mins = local.hour * 60 + local.minute
    df = df[(mins >= 9 * 60 + 15) & (mins < 15 * 60 + 30)]
    if df.empty:
        raise ValueError("No observations fall inside NSE regular trading hours (09:15-15:30 IST)")

    # 09:15 IST is aligned to a 3-minute boundary. A bucket is retained only when all
    # three constituent minute bars are present, avoiding partial bars at session edges.
    grouped = df.resample("3min", origin="start_day", offset="15min", label="left", closed="left")
    out = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    counts = grouped["close"].count()
    out["source_1m_bars"] = counts
    if not include_partial:
        out = out[out["source_1m_bars"] == 3]
    out = out.dropna(subset=["open", "high", "low", "close"])
    out.index.name = "timestamp"
    if out.empty:
        raise ValueError("No complete 3-minute candles could be formed from the Yahoo 1-minute data")
    return out


def save_dataset(period: str = "7d", output_dir: str | Path = "data") -> tuple[Path, Path, dict]:
    root = Path(output_dir)
    raw_dir, processed_dir = root / "raw", root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw = download_one_minute(period)
    candles = resample_to_three_minutes(raw)
    raw_path = raw_dir / "nifty50_yahoo_1m.csv"
    processed_path = processed_dir / "nifty50_yahoo_3m.csv"
    raw.to_csv(raw_path, index_label="timestamp")
    candles.to_csv(processed_path, index_label="timestamp")
    metadata = {
        "provider": "Yahoo Finance via yfinance",
        "symbol": SYMBOL,
        "source_interval": "1m",
        "training_interval": "3m (resampled from 1m)",
        "period_requested": period,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone": IST,
        "session_filter": "09:15-15:30 Asia/Kolkata",
        "raw_rows": int(len(raw)),
        "three_minute_rows": int(len(candles)),
        "first_timestamp": candles.index.min().isoformat(),
        "last_timestamp": candles.index.max().isoformat(),
        "volume_note": "NIFTY 50 is an index; volume is not used as a model feature.",
        "limitations": "Yahoo 1-minute history is limited (typically up to 7 days); this is not a multi-year training set."
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return raw_path, processed_path, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Yahoo NIFTY 50 data and resample to 3-minute candles")
    parser.add_argument("--period", default="7d", help="Yahoo 1-minute period; typically limited to 7d")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    raw_path, processed_path, meta = save_dataset(args.period, args.output_dir)
    print(json.dumps(meta, indent=2))
    print(f"Saved raw 1m data: {raw_path}")
    print(f"Saved 3m training candles: {processed_path}")


if __name__ == "__main__":
    main()
