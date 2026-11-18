"""Offline training and chronological evaluation of MarketMind's real-data GMM model."""
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
from ml.model_utils import REGIMES, assign_regime_labels, describe_clusters
from ml.regime_detection import LOG_COLUMNS, _prep

REAL_FEATURE_COLUMNS = ["ret_5", "roll_vol", "rsi", "ma_diff", "momentum", "atr_pct"]


def load_candles(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing OHLC columns: {missing}")
    # Index volume is not a meaningful traded-volume measure; it is intentionally not used.
    if "volume" not in df:
        df["volume"] = 0.0
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df[~df.index.duplicated(keep="last")]
    return df


def train(input_csv: str | Path, output_path: str | Path, test_fraction: float = 0.25,
          min_rows: int = 250, seed: int = 42) -> dict:
    candles = load_candles(input_csv)
    indicators = compute_all(candles)
    features = build_features(indicators, include_volume=False)[REAL_FEATURE_COLUMNS]
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if len(features) < min_rows:
        raise ValueError(
            f"Only {len(features)} complete feature rows are available; need at least {min_rows}. "
            "Yahoo's short 1-minute history may be insufficient. Obtain more genuine 1-minute data or use a longer supported timeframe."
        )
    split = int(len(features) * (1.0 - test_fraction))
    if split < 150 or len(features) - split < 30:
        raise ValueError("Chronological split leaves too few training or test rows")
    X_train, X_test = features.iloc[:split], features.iloc[split:]

    # Volatility features are log-transformed before scaling, matching live inference.
    train_prep = _prep(X_train, REAL_FEATURE_COLUMNS)
    test_prep = _prep(X_test, REAL_FEATURE_COLUMNS)
    scaler = StandardScaler().fit(train_prep)
    Z_train, Z_test = scaler.transform(train_prep), scaler.transform(test_prep)
    gmm = GaussianMixture(n_components=4, covariance_type="full", n_init=5,
                          random_state=seed, reg_covar=1e-5).fit(Z_train)
    mapping = assign_regime_labels(gmm.means_, REAL_FEATURE_COLUMNS)
    centers_scaled = gmm.means_
    centers_transformed = scaler.inverse_transform(centers_scaled)
    centers_raw = centers_transformed.copy()
    for col in LOG_COLUMNS:
        if col in REAL_FEATURE_COLUMNS:
            centers_raw[:, REAL_FEATURE_COLUMNS.index(col)] = np.exp(centers_raw[:, REAL_FEATURE_COLUMNS.index(col)])
    notes = describe_clusters(mapping, centers_raw, REAL_FEATURE_COLUMNS)

    clusters = gmm.predict(Z_test)
    unique = np.unique(clusters)
    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_rows": int(len(features)),
        "n_clusters_present_in_test": int(len(unique)),
        "silhouette": float(silhouette_score(Z_test, clusters)) if 1 < len(unique) < len(Z_test) else None,
        "davies_bouldin": float(davies_bouldin_score(Z_test, clusters)) if 1 < len(unique) else None,
        "calinski_harabasz": float(calinski_harabasz_score(Z_test, clusters)) if 1 < len(unique) else None,
        "test_cluster_counts": {str(int(k)): int((clusters == k).sum()) for k in unique},
        "test_regime_counts": {mapping[int(k)]: int((clusters == k).sum()) for k in unique},
        "evaluation_note": "Unsupervised clustering metrics on a chronological holdout; these are not directional accuracy or profitability metrics."
    }
    artifact = {
        "model_type": "GaussianMixture",
        "provider": "Yahoo Finance",
        "symbol": "^NSEI",
        "interval": "3m",
        "feature_columns": REAL_FEATURE_COLUMNS,
        "log_columns": [c for c in LOG_COLUMNS if c in REAL_FEATURE_COLUMNS],
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
            "Yahoo 1-minute data has a short lookback, so this model is trained on a small recent sample.",
            "NIFTY 50 index volume is not used as a feature.",
            "Regime names are assigned from cluster-centroid heuristics; GMM itself is unsupervised.",
            "Clustering metrics do not establish trading profitability or predictive accuracy."
        ]
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)

    report = {
        "model": artifact["model_type"], "source": f"{artifact['provider']} {artifact['symbol']}",
        "interval": artifact["interval"], "features": REAL_FEATURE_COLUMNS,
        "train_period": [artifact["training_start"], artifact["training_end"]],
        "test_period": [artifact["test_start"], artifact["test_end"]],
        "cluster_mapping": {str(k): v for k, v in mapping.items()},
        "cluster_notes": notes, "metrics": metrics, "limitations": artifact["limitations"],
        "artifact": str(output)
    }
    report_path = output.parent / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MarketMind GMM offline on Yahoo NIFTY 50 3-minute data")
    parser.add_argument("--input", default="data/processed/nifty50_yahoo_3m.csv")
    parser.add_argument("--output", default="ml/models/nifty50_gmm_3m.joblib")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--min-rows", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.input, args.output, args.test_fraction, args.min_rows, args.seed)


if __name__ == "__main__":
    main()
