# Real NIFTY 50 experiment — execution summary

## Data and preprocessing
- Source: bundled `data/raw/nifty50/1min/` CSVs
- Cleaned one-minute rows: **890,607**
- 15-minute bars: **59,378**
- Coverage: **3 April 2017 – 17 November 2026**
- Feature rows after warm-up: **59,313**
- Features: `ret_5`, `roll_vol`, `rsi`, `ma_diff`, `momentum`, `atr_pct`
- Split: chronological 70% train / 30% test
- Deterministic fit sample: 10,000 training rows
- Held-out evaluation sample: 10,000 rows

## Held-out results

| Model | Clusters (excluding noise) | Noise % | Silhouette | Davies-Bouldin | Calinski-Harabasz | Adjacent-label persistence |
|---|---:|---:|---:|---:|---:|---:|
| K-Means | 4 | 0.00 | 0.236 | 1.300 | 2736.44 | 82.76% |
| Agglomerative | 4 | 0.00 | 0.222 | 1.307 | 2301.12 | 82.66% |
| GMM | 4 | 0.00 | 0.107 | 3.066 | 1051.84 | 81.90% |
| DBSCAN | 1 | 8.08 | N/A | N/A | N/A | 96.40% |

## Interpretation and limits
K-Means had the strongest internal cluster-separation metrics in this run, with Agglomerative close behind. DBSCAN did not identify multiple non-noise clusters with the selected `eps` and `min_samples`; its internal metrics are therefore not comparable. The high DBSCAN persistence largely reflects repeated assignments to its single cluster and should not be read as evidence of a superior regime model.

There are no independent ground-truth regime labels in the CSVs. Do not claim classification accuracy, ARI, NMI, or trading profitability from this experiment. The held-out clustering scores do not establish which model produces the best trading decisions. A proper next step is a walk-forward evaluation of strategy returns with realistic transaction costs and slippage, plus stability checks across different time windows and parameter settings.

The notebook excludes volume to keep one consistent feature set across the historical period; the existing production detector uses seven features including `volume_change`. This experiment does not create a production-compatible model artifact and does not change the production GMM.
