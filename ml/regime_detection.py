"""Market-regime detector supporting offline-trained artifacts and legacy rolling-fit mode."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ml.model_utils import (FEATURE_COLUMNS, WARMING_UP, assign_regime_labels,
                            describe_clusters, explain_observation)


@dataclass
class RegimeResult:
    regime: str
    raw_regime: str = WARMING_UP
    cluster: int = -1
    confidence: float = 0.0
    reasons: list = field(default_factory=list)
    features: dict = field(default_factory=dict)
    z: dict = field(default_factory=dict)
    ready: bool = False


LOG_COLUMNS = ["roll_vol", "atr_pct"]


def _prep(X: pd.DataFrame, feature_columns=None, log_columns=None) -> pd.DataFrame:
    """Apply the exact feature transform used by the fitted scaler."""
    columns = list(feature_columns or FEATURE_COLUMNS)
    logs = list(LOG_COLUMNS if log_columns is None else log_columns)
    X = X[columns].copy()
    for c in logs:
        if c in X:
            X[c] = np.log(X[c].clip(lower=1e-9))
    return X


class RegimeDetector:
    """Use a saved offline model when loaded; otherwise retain the original rolling-fit mode.

    Offline mode never refits on incoming observations. Legacy mode is retained for simulator
    experiments and backwards-compatible tests.
    """

    def __init__(self, n_clusters: int = 4, window: int = 2000, refit_every: int = 15,
                 min_samples: int = 200, smooth: int = 5, seed: int = 0) -> None:
        self.k, self.window, self.refit_every = n_clusters, window, refit_every
        self.min_samples, self.seed = min_samples, seed
        self.scaler: StandardScaler | None = None
        self.gmm: GaussianMixture | None = None
        self.mapping: dict = {}
        self.cluster_notes: list = []
        self.feature_columns = list(FEATURE_COLUMNS)
        self.log_columns = list(LOG_COLUMNS)
        self.fits = 0
        self.n_samples = 0
        self._calls = 0
        self._recent: deque = deque(maxlen=smooth)
        self.offline_model = False
        self.model_metadata: dict = {}

    def load_artifact(self, path: str | Path) -> None:
        artifact = joblib.load(path)
        required = {"scaler", "gmm", "mapping", "feature_columns"}
        missing = required - set(artifact)
        if missing:
            raise ValueError(f"Invalid MarketMind model artifact; missing {sorted(missing)}")
        self.scaler = artifact["scaler"]
        self.gmm = artifact["gmm"]
        self.mapping = {int(k): v for k, v in artifact["mapping"].items()}
        self.feature_columns = list(artifact["feature_columns"])
        self.log_columns = list(artifact.get("log_columns", LOG_COLUMNS))
        self.cluster_notes = list(artifact.get("cluster_notes", []))
        self.n_samples = int(artifact.get("metrics", {}).get("train_rows", 0))
        self.k = int(getattr(self.gmm, "n_components", 4))
        self.fits = 1
        self.offline_model = True
        self.model_metadata = {k: v for k, v in artifact.items()
                               if k not in {"scaler", "gmm"}}

    @classmethod
    def from_artifact(cls, path: str | Path, smooth: int = 5) -> "RegimeDetector":
        detector = cls(smooth=smooth)
        detector.load_artifact(path)
        return detector

    def _fit(self, X: pd.DataFrame) -> None:
        prepared = _prep(X, self.feature_columns, self.log_columns)
        scaler = StandardScaler().fit(prepared)
        Z = scaler.transform(prepared)
        gmm = GaussianMixture(self.k, covariance_type="full", n_init=3,
                              random_state=self.seed, reg_covar=1e-5).fit(Z)
        self.scaler, self.gmm = scaler, gmm
        self.mapping = assign_regime_labels(gmm.means_, self.feature_columns)
        centers = scaler.inverse_transform(gmm.means_)
        centers_df = pd.DataFrame(centers, columns=self.feature_columns)
        for c in self.log_columns:
            if c in centers_df:
                centers_df[c] = np.exp(centers_df[c])
        self.cluster_notes = describe_clusters(self.mapping, centers_df.to_numpy(), self.feature_columns)
        self.fits += 1
        self.n_samples = len(X)

    def update(self, feats: pd.DataFrame) -> RegimeResult:
        if self.offline_model:
            missing = [c for c in self.feature_columns if c not in feats.columns]
            if missing:
                return RegimeResult(WARMING_UP, reasons=[f"Missing model features: {', '.join(missing)}"])
            X = feats[self.feature_columns].dropna()
            if X.empty:
                return RegimeResult(WARMING_UP)
            return self.predict(X.iloc[[-1]])

        X = feats.dropna()
        if len(X) < self.min_samples:
            return RegimeResult(WARMING_UP)
        self._calls += 1
        if self.gmm is None or self._calls % self.refit_every == 0:
            self._fit(X.tail(self.window))
        return self.predict(X.iloc[[-1]])

    def predict(self, row: pd.DataFrame) -> RegimeResult:
        if self.gmm is None or self.scaler is None:
            return RegimeResult(WARMING_UP)
        prepared = _prep(row, self.feature_columns, self.log_columns)
        z = self.scaler.transform(prepared)
        probabilities = self.gmm.predict_proba(z)[0]
        cluster = int(np.argmax(probabilities))
        raw_regime = self.mapping[cluster]
        self._recent.append(raw_regime)
        votes = Counter(self._recent)
        top = max(votes.values())
        regime = raw_regime if votes[raw_regime] == top else next(
            r for r in reversed(self._recent) if votes[r] == top)
        raw = {c: float(v) for c, v in zip(self.feature_columns, row[self.feature_columns].iloc[0])}
        zd = {c: float(v) for c, v in zip(self.feature_columns, z[0])}
        return RegimeResult(regime, raw_regime, cluster, round(float(probabilities[cluster]), 2),
                            explain_observation(regime, raw, zd), raw, zd, True)

    def classify_all(self, feats: pd.DataFrame) -> pd.Series:
        """Return unsmoothed labels for complete feature rows."""
        X = feats[self.feature_columns].dropna()
        z = self.scaler.transform(_prep(X, self.feature_columns, self.log_columns))
        return pd.Series([self.mapping[int(c)] for c in self.gmm.predict(z)], index=X.index)
