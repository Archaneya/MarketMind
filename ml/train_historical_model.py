"""Train MarketMind's production GMM on the bundled 2017-2026 NIFTY 1-minute history.

Usage from the project root:
    python -m ml.train_historical_model
    python -m ml.train_historical_model --data-dir data/raw/nifty50/1min

The resulting artifact is compatible with ml.regime_detection.RegimeDetector.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from indicators.technical_indicators import compute_all
from ml.feature_engineering import build_features
from ml.model_utils import assign_regime_labels, describe_clusters
from ml.regime_detection import LOG_COLUMNS, _prep

FEATURE_COLUMNS_REAL = ["ret_5", "roll_vol", "rsi", "ma_diff", "momentum", "atr_pct"]


def load_historical_1m(data_dir: Path) -> pd.DataFrame:
    paths = sorted(data_dir.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        cols = {str(c).strip().lower(): c for c in frame.columns}
        time_col = next((cols[k] for k in ("timestamp", "datetime", "date", "time") if k in cols), None)
        if time_col is None:
            continue
        needed = {}
        for key in ("open", "high", "low", "close"):
            if key not in cols:
                break
            needed[cols[key]] = key
        if len(needed) != 4:
            continue
        part = frame.rename(columns=needed).copy()
        part["timestamp"] = pd.to_datetime(frame[time_col], errors="coerce")
        part = part.dropna(subset=["timestamp", "open", "high", "low", "close"])
        for col in ("open", "high", "low", "close"):
            part[col] = pd.to_numeric(part[col], errors="coerce")
        part = part.dropna(subset=["open", "high", "low", "close"])
        part = part[(part["open"] > 0) & (part["high"] > 0) & (part["low"] > 0) & (part["close"] > 0)]
        part["date"] = part["timestamp"].dt.date
        frames.append(part[["timestamp", "date", "open", "high", "low", "close"]])

    if not frames:
        raise ValueError(f"No usable OHLC CSVs found under {data_dir}")

    one = pd.concat(frames, ignore_index=True)
    one = one.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    one = one.set_index("timestamp")
    return one[["open", "high", "low", "close"]]


def resample_3m(one_minute: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for _, session in one_minute.groupby(one_minute.index.date, sort=True):
        session = session.sort_index()
        bars = session.resample("3min", origin="start_day", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        )
        count = session["close"].resample(
            "3min", origin="start_day", label="left", closed="left"
        ).count()
        # Require complete 3-minute candles to avoid training on partial buckets.
        bars = bars[count == 3].dropna()
        chunks.append(bars)
    if not chunks:
        raise ValueError("No complete 3-minute bars could be created from the input data.")
    return pd.concat(chunks).sort_index()


def train(data_dir: Path, output_path: Path, test_fraction: float, seed: int) -> dict:
    one = load_historical_1m(data_dir)
    candles = resample_3m(one)
    candles["volume"] = 0.0  # NIFTY index volume is not used as a feature.
    indicators = compute_all(candles)
    features = build_features(indicators, include_volume=False)[FEATURE_COLUMNS_REAL]
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    if len(features) < 5000:
        raise ValueError(f"Only {len(features)} feature rows available; expected at least 5,000.")

    split = int(len(features) * (1.0 - test_fraction))
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    train_prepared = _prep(X_train, FEATURE_COLUMNS_REAL)
    test_prepared = _prep(X_test, FEATURE_COLUMNS_REAL)

    scaler = StandardScaler().fit(train_prepared)
    Z_train = scaler.transform(train_prepared)
    Z_test = scaler.transform(test_prepared)

    gmm = GaussianMixture(
        n_components=4, covariance_type="full", n_init=3,
        max_iter=200, random_state=seed, reg_covar=1e-5
    ).fit(Z_train)

    mapping = assign_regime_labels(gmm.means_, FEATURE_COLUMNS_REAL)
    centers_transformed = scaler.inverse_transform(gmm.means_)
    centers_raw = centers_transformed.copy()
    for col in LOG_COLUMNS:
        if col in FEATURE_COLUMNS_REAL:
            idx = FEATURE_COLUMNS_REAL.index(col)
            centers_raw[:, idx] = np.exp(centers_raw[:, idx])
    notes = describe_clusters(mapping, centers_raw, FEATURE_COLUMNS_REAL)

    test_clusters = gmm.predict(Z_test)
    unique = np.unique(test_clusters)
    metric_sample = Z_test
    metric_labels = test_clusters
    if len(Z_test) > 10000:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(Z_test), size=10000, replace=False))
        metric_sample = Z_test[idx]
        metric_labels = test_clusters[idx]

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_rows": int(len(features)),
        "n_clusters_present_in_test": int(len(unique)),
        "silhouette": float(silhouette_score(metric_sample, metric_labels))
            if 1 < len(np.unique(metric_labels)) < len(metric_labels) else None,
        "davies_bouldin": float(davies_bouldin_score(metric_sample, metric_labels))
            if len(np.unique(metric_labels)) > 1 else None,
        "calinski_harabasz": float(calinski_harabasz_score(metric_sample, metric_labels))
            if len(np.unique(metric_labels)) > 1 else None,
        "test_cluster_counts": {str(int(k)): int((test_clusters == k).sum()) for k in unique},
        "test_regime_counts": {mapping[int(k)]: int((test_clusters == k).sum()) for k in unique},
        "evaluation_note": "Unsupervised clustering metrics on a chronological holdout; not directional accuracy or profitability."
    }

    artifact = {
        "model_type": "GaussianMixture",
        "provider": "Bundled NIFTY 50 historical OHLC",
        "symbol": "NIFTY 50",
        "interval": "3m",
        "feature_columns": FEATURE_COLUMNS_REAL,
        "log_columns": [c for c in LOG_COLUMNS if c in FEATURE_COLUMNS_REAL],
        "scaler": scaler,
        "gmm": gmm,
        "mapping": mapping,
        "cluster_notes": notes,
        "training_start": X_train.index.min().isoformat(),
        "training_end": X_train.index.max().isoformat(),
        "test_start": X_test.index.min().isoformat(),
        "test_end": X_test.index.max().isoformat(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "limitations": [
            "Unsupervised clusters are interpreted using transparent centroid heuristics, not independent labels.",
            "NIFTY 50 index volume is excluded.",
            "Internal clustering metrics are not directional accuracy or profitability.",
            "Live dashboard candles are simulated; this artifact demonstrates inference integration, not live-market validation."
        ]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    report = {
        "model": artifact["model_type"], "source": artifact["provider"],
        "symbol": artifact["symbol"], "interval": artifact["interval"],
        "features": FEATURE_COLUMNS_REAL,
        "train_period": [artifact["training_start"], artifact["training_end"]],
        "test_period": [artifact["test_start"], artifact["test_end"]],
        "cluster_mapping": {str(k): v for k, v in mapping.items()},
        "cluster_notes": notes, "metrics": metrics,
        "limitations": artifact["limitations"], "artifact": str(output_path)
    }
    report_path = output_path.parent / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/nifty50/1min"))
    parser.add_argument("--output", type=Path, default=Path("ml/models/nifty50_gmm_3m.joblib"))
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train(args.data_dir, args.output, args.test_fraction, args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
