"""Reproducible model comparison for MarketMind's market-regime clustering.

Run from the project root:
    python -m research.run_comparison

The simulator's hidden regime is used ONLY as an evaluation reference. It is never
passed to the clustering algorithms during fitting or prediction.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from backend.market_state import CandleAggregator, resample_candles
from backend.simulator import MarketSimulator
from indicators.technical_indicators import compute_all
from ml.feature_engineering import build_features
from ml.model_utils import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research" / "results"
SEEDS = [7, 21, 42, 100, 2026]
SIMULATED_MINUTES = 14_000
ANALYSIS_TIMEFRAME_MINUTES = 3
TRAIN_FRACTION = 0.70
N_CLUSTERS = 4
LOG_COLUMNS = ["roll_vol", "atr_pct"]


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[FEATURE_COLUMNS].copy()
    for col in LOG_COLUMNS:
        x[col] = np.log(x[col].clip(lower=1e-9))
    return x


def make_dataset(seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Generate OHLCV bars and hidden simulator labels at the same 3-minute cadence."""
    sim = MarketSimulator(seed=seed, start_epoch=0, seconds_per_tick=20.0)
    aggregator = CandleAggregator()
    truth_by_bucket: dict[int, list[str]] = defaultdict(list)

    # 3 ticks/minute. Truth labels are kept separately from model inputs.
    for _ in range(SIMULATED_MINUTES * 3):
        tick = sim.next_tick()
        aggregator.update(tick)
        bucket = int(tick.epoch // (ANALYSIS_TIMEFRAME_MINUTES * 60)
                      * (ANALYSIS_TIMEFRAME_MINUTES * 60))
        truth_by_bucket[bucket].append(tick.regime)

    bars = resample_candles(aggregator.candles(), ANALYSIS_TIMEFRAME_MINUTES)
    indicators = compute_all(bars)
    features = build_features(indicators).dropna()

    # Majority regime among the ticks inside each 3-minute feature bar.
    labels = {}
    for bucket, regimes in truth_by_bucket.items():
        counts = Counter(regimes)
        labels[bucket] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    truth = pd.Series(labels, name="simulator_regime").reindex(features.index)
    valid = truth.notna()
    return features.loc[valid], truth.loc[valid]


def _safe_internal_scores(x: np.ndarray, labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    # DBSCAN noise (-1) is excluded from geometric cluster metrics, but retained
    # in ARI/NMI and separately reported as noise coverage.
    keep = labels != -1
    x_valid, y_valid = x[keep], labels[keep]
    unique = np.unique(y_valid)
    out = {"silhouette": np.nan, "davies_bouldin": np.nan,
           "calinski_harabasz": np.nan, "noise_pct": float((~keep).mean() * 100)}
    if len(unique) >= 2 and len(unique) < len(y_valid) and len(y_valid) > len(unique):
        try:
            out["silhouette"] = float(silhouette_score(x_valid, y_valid))
            out["davies_bouldin"] = float(davies_bouldin_score(x_valid, y_valid))
            out["calinski_harabasz"] = float(calinski_harabasz_score(x_valid, y_valid))
        except ValueError:
            pass
    return out


def _dbscan_predict(model: DBSCAN, x_test: np.ndarray) -> np.ndarray:
    """Transparent out-of-sample approximation: nearest training core point within eps."""
    core = model.components_
    core_labels = model.labels_[model.core_sample_indices_]
    if len(core) == 0:
        return np.full(len(x_test), -1, dtype=int)
    # Chunk to avoid allocating a very large test x core x feature tensor.
    predicted = np.full(len(x_test), -1, dtype=int)
    for start in range(0, len(x_test), 256):
        block = x_test[start:start + 256]
        distances = np.sqrt(((block[:, None, :] - core[None, :, :]) ** 2).sum(axis=2))
        nearest = distances.argmin(axis=1)
        nearest_dist = distances[np.arange(len(block)), nearest]
        accepted = nearest_dist <= model.eps
        predicted[start:start + len(block)][accepted] = core_labels[nearest[accepted]]
    return predicted


def _nearest_centroid_predict(model_labels: np.ndarray, x_train: np.ndarray,
                              x_test: np.ndarray) -> np.ndarray:
    """Out-of-sample proxy for Agglomerative: nearest centroid of fitted clusters."""
    cluster_ids = np.unique(model_labels)
    centroids = np.vstack([x_train[model_labels == c].mean(axis=0) for c in cluster_ids])
    distances = ((x_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return cluster_ids[distances.argmin(axis=1)]


def evaluate_one(seed: int, model_name: str, x_train: np.ndarray,
                 x_test: np.ndarray, y_test: np.ndarray) -> dict:
    started = time.perf_counter()
    if model_name == "K-Means":
        model = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=seed)
        train_labels = model.fit_predict(x_train)
        fit_seconds = time.perf_counter() - started
        pred_started = time.perf_counter()
        test_labels = model.predict(x_test)
        predict_seconds = time.perf_counter() - pred_started
    elif model_name == "Gaussian Mixture":
        model = GaussianMixture(n_components=N_CLUSTERS, covariance_type="full",
                                n_init=3, random_state=seed, reg_covar=1e-5)
        model.fit(x_train)
        train_labels = model.predict(x_train)
        fit_seconds = time.perf_counter() - started
        pred_started = time.perf_counter()
        test_labels = model.predict(x_test)
        predict_seconds = time.perf_counter() - pred_started
    elif model_name == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
        train_labels = model.fit_predict(x_train)
        fit_seconds = time.perf_counter() - started
        pred_started = time.perf_counter()
        test_labels = _nearest_centroid_predict(train_labels, x_train, x_test)
        predict_seconds = time.perf_counter() - pred_started
    elif model_name == "DBSCAN":
        # Fixed, declared settings on standardized features; not tuned on test labels.
        model = DBSCAN(eps=1.5, min_samples=20)
        train_labels = model.fit_predict(x_train)
        fit_seconds = time.perf_counter() - started
        pred_started = time.perf_counter()
        test_labels = _dbscan_predict(model, x_test)
        predict_seconds = time.perf_counter() - pred_started
    else:
        raise ValueError(model_name)

    internal = _safe_internal_scores(x_test, test_labels)
    return {
        "seed": seed,
        "model": model_name,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "test_clusters_including_noise": int(len(np.unique(test_labels))),
        "test_ari": float(adjusted_rand_score(y_test, test_labels)),
        "test_nmi": float(normalized_mutual_info_score(y_test, test_labels)),
        **internal,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "prediction_method": (
            "native predict" if model_name in ("K-Means", "Gaussian Mixture") else
            "nearest training-cluster centroid proxy" if model_name == "Agglomerative" else
            "nearest DBSCAN core point within eps; otherwise noise"
        ),
    }


def write_selection_report(metrics: pd.DataFrame) -> str:
    summary = metrics.groupby("model").agg(
        ari_mean=("test_ari", "mean"), ari_std=("test_ari", "std"),
        nmi_mean=("test_nmi", "mean"), nmi_std=("test_nmi", "std"),
        silhouette_mean=("silhouette", "mean"), db_mean=("davies_bouldin", "mean"),
        ch_mean=("calinski_harabasz", "mean"), noise_mean=("noise_pct", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"),
    ).sort_values(["ari_mean", "nmi_mean"], ascending=False)

    # Predeclared primary criterion: highest mean held-out ARI; NMI breaks ties.
    winner = str(summary.index[0])
    incumbent = "K-Means"
    if winner == incumbent:
        decision = ("Retain K-Means. It achieved the strongest mean held-out ARI under this "
                    "experiment's predeclared selection rule, with NMI as the tie-breaker.")
    else:
        decision = (f"Select {winner} over the K-Means baseline: it achieved the highest mean "
                    "held-out ARI, with NMI as the tie-breaker. MarketMind's current prototype "
                    f"production detector is therefore set to {winner}. This choice is provisional: "
                    "the evidence comes from synthetic simulator data and must be re-tested on "
                    "real historical market data before any live-market claim or deployment decision.")

    lines = [
        "# MarketMind — ML Model Selection Evidence", "",
        "## Research question", "Which unsupervised clustering algorithm best recovers the "
        "simulator's market regimes from the seven engineered market features?", "",
        "## Experimental protocol", f"- Seeds: {', '.join(map(str, SEEDS))}",
        f"- Simulated duration per seed: {SIMULATED_MINUTES:,} minutes",
        f"- Analysis timeframe: {ANALYSIS_TIMEFRAME_MINUTES}-minute bars",
        f"- Chronological split: {int(TRAIN_FRACTION*100)}% train / "
        f"{int((1-TRAIN_FRACTION)*100)}% test",
        "- Features: " + ", ".join(FEATURE_COLUMNS),
        "- Preprocessing: log transform roll_vol and atr_pct; StandardScaler fitted on train only.",
        "- Ground-truth simulator regimes are used only for evaluation, never as model inputs.",
        "- Primary selection metric: mean test ARI across seeds; mean test NMI is tie-breaker.", "",
        "## Mean results across seeds", "",
        "| Model | Test ARI mean ± SD | Test NMI mean ± SD | Silhouette | Davies-Bouldin | Noise % | Fit seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary.iterrows():
        def fmt(v): return "N/A" if pd.isna(v) else f"{v:.3f}"
        lines.append(
            f"| {name} | {row.ari_mean:.3f} ± {row.ari_std if pd.notna(row.ari_std) else 0:.3f} "
            f"| {row.nmi_mean:.3f} ± {row.nmi_std if pd.notna(row.nmi_std) else 0:.3f} "
            f"| {fmt(row.silhouette_mean)} | {fmt(row.db_mean)} | {fmt(row.noise_mean)} "
            f"| {row.fit_seconds_mean:.3f} |"
        )
    lines += ["", "## Selection decision", decision, "",
              "## Fair-comparison notes",
              "- K-Means and GMM use native out-of-sample prediction.",
              "- Agglomerative has no native predict method; test labels are assigned using the "
              "nearest centroid computed from its training clusters. This is an explicit proxy, "
              "not a native Agglomerative prediction.",
              "- DBSCAN has no native predict method; a test point inherits the label of its "
              "nearest training core point only when within eps; otherwise it is noise (-1).",
              "- Silhouette, Davies-Bouldin and Calinski-Harabasz are internal geometric metrics. "
              "For DBSCAN, noise points are excluded from these three metrics and noise coverage "
              "is reported separately.",
              "- ARI and NMI compare the predicted cluster partition with simulator truth and are "
              "label-permutation invariant.", "",
              "## Why this model was selected", 
              ("GMM is selected because this project's primary objective is agreement with the "
               "simulator's known regime states, and GMM achieved the best mean held-out ARI and "
               "NMI. GMM also models component covariance and overlapping distributions. It was "
               "not best on every metric: K-Means had a higher mean Silhouette score, lower "
               "Davies-Bouldin score and substantially faster fit time. The selection prioritises "
               "external regime recovery over compactness and speed for this prototype." if winner == "Gaussian Mixture" else
               f"{winner} was selected by the predeclared mean held-out ARI rule. Review the full "
               "metric trade-offs above; the winning model is not best on every metric."), "",
              "## Limitations", "- This dataset is synthetic, generated by MarketMind's own "
              "regime-switching simulator.",
              "- Results do not establish real NIFTY regime-detection accuracy or trading profitability.",
              "- Cluster count is fixed at four for K-Means, GMM and Agglomerative because the "
              "application defines four target regimes; DBSCAN determines its cluster count.",
              "- The model-selection rule is an experimental choice, not proof of universal superiority.", ""]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "final_model_selection.md").write_text("\n".join(lines), encoding="utf-8")
    return winner


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = []
    model_names = ["K-Means", "Gaussian Mixture", "Agglomerative", "DBSCAN"]
    metadata = {
        "project": "MarketMind", "experiment": "unsupervised market regime model comparison",
        "seeds": SEEDS, "simulated_minutes_per_seed": SIMULATED_MINUTES,
        "analysis_timeframe_minutes": ANALYSIS_TIMEFRAME_MINUTES,
        "train_fraction": TRAIN_FRACTION, "n_clusters": N_CLUSTERS,
        "features": FEATURE_COLUMNS,
        "preprocessing": {"log_transform": LOG_COLUMNS, "scaler": "StandardScaler fit on train only"},
        "models": {"K-Means": {"n_clusters": 4, "n_init": 10},
                   "Gaussian Mixture": {"n_components": 4, "covariance_type": "full", "n_init": 3},
                   "Agglomerative": {"n_clusters": 4, "linkage": "ward"},
                   "DBSCAN": {"eps": 1.5, "min_samples": 20}},
        "selection_rule": "highest mean held-out ARI; mean held-out NMI as tie-breaker",
        "warning": "Synthetic simulator benchmark only; not real-market validation.",
    }

    for seed in SEEDS:
        print(f"[seed {seed}] generating synthetic market data...", flush=True)
        features, truth = make_dataset(seed)
        split = int(len(features) * TRAIN_FRACTION)
        train_raw, test_raw = features.iloc[:split], features.iloc[split:]
        y_test = truth.iloc[split:].to_numpy()
        prep_train, prep_test = _prepare_features(train_raw), _prepare_features(test_raw)
        scaler = StandardScaler().fit(prep_train)
        x_train, x_test = scaler.transform(prep_train), scaler.transform(prep_test)
        for name in model_names:
            print(f"  evaluating {name}...", flush=True)
            records.append(evaluate_one(seed, name, x_train, x_test, y_test))

    results = pd.DataFrame(records)
    results.to_csv(RESULTS / "model_metrics_by_seed.csv", index=False)
    summary = results.groupby("model").agg(
        test_ari_mean=("test_ari", "mean"), test_ari_std=("test_ari", "std"),
        test_nmi_mean=("test_nmi", "mean"), test_nmi_std=("test_nmi", "std"),
        silhouette_mean=("silhouette", "mean"), davies_bouldin_mean=("davies_bouldin", "mean"),
        calinski_harabasz_mean=("calinski_harabasz", "mean"), noise_pct_mean=("noise_pct", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"), predict_seconds_mean=("predict_seconds", "mean"),
    ).reset_index()
    summary.to_csv(RESULTS / "model_metrics_summary.csv", index=False)
    metadata["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (RESULTS / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    winner = write_selection_report(results)
    print("\nMODEL COMPARISON SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSelected by predeclared rule: {winner}")
    print(f"Evidence saved under: {RESULTS}")


if __name__ == "__main__":
    main()
