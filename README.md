# MarketMind: ML-powered simulated NIFTY 50 market intelligence terminal

An educational AIML demonstration. MarketMind trains a Gaussian Mixture Model (GMM) on bundled
historical NIFTY 50 OHLC data (2017–2026), saves the fitted preprocessing/model artifact, then runs
that fixed model against a simulated live candle stream. The dashboard displays regime predictions,
model membership scores, indicators, and strategy suitability scores.

**Historical data is used for offline model training. Dashboard prices are simulated.** No broker,
API key, live market feed, or order execution is used. This is a classroom demonstration, not a
trading system or investment advice.

## Quick start

Python 3.10 or newer.

```bash
pip install -r requirements.txt

# Train and save the production GMM from the bundled historical NIFTY data
python -m ml.train_historical_model

# Terminal 1: backend (simulator + WebSocket server)
python websocket_server.py

# Terminal 2: Streamlit frontend
streamlit run app.py
```

The dashboard also works without Streamlit at <http://localhost:8000/> (served by the backend).
Options: `python websocket_server.py --port 8000 --seed 42 --tick-ms 500` (a fixed seed gives a reproducible market).

The chart library (TradingView Lightweight Charts 4.1.3) loads from a CDN. For offline use, download
`lightweight-charts.standalone.production.js` into `frontend/vendor/` and it is inlined automatically.

## Architecture

```
 MarketSimulator ──ticks──> CandleAggregator ──1m candles──> bounded history (30k)
        │                                                          │ every 4th tick, in a worker thread
        │                                      resample ─> indicators ─> features ─> RegimeDetector (Gaussian Mixture Model)
        │                                                          └──────────────> five strategy evaluators
        ▼                                                                                   │
 FastAPI /ws  ── tick message every 500 ms ───────────────┐      analytics message (~2 s) ◄──┘
        │                                                  ▼
        └── history snapshot on connect / resync ──> browser (chart.js): aggregates candles per timeframe,
                                                      draws the chart and panels, reconnects automatically
 Streamlit (app.py) only embeds this page; updates never depend on Streamlit reruns.
```

| Folder | Role |
|---|---|
| `backend/simulator.py` | Regime-switching geometric Brownian motion tick generator |
| `backend/market_state.py` | Tick-to-candle aggregation, resampling, validation, `MarketEngine` |
| `backend/websocket_manager.py` | Client registry, broadcast, per-client locks, slow-client timeout |
| `backend/intelligence.py` | Plain-language explanations and indicator readings |
| `indicators/` | RSI, SMA, EMA, MACD, Bollinger, ATR, ADX, momentum, rolling volatility |
| `ml/` | Feature engineering, Gaussian Mixture regime detector, cluster interpretation |
| `strategies/` | Five evaluators and the ranking engine |
| `frontend/` | `chart.html`, `styles.css`, `chart.js`, and `assemble.py` (inlines them for Streamlit) |
| `websocket_server.py` / `app.py` | Backend entry point / Streamlit shell |

### Simulation details
* Per tick: `log P += (mu - sigma^2/2)dt + sigma*sqrt(dt)*Z`, so prices are always positive.
* Four regimes with their own drift and volatility. Sideways adds an Ornstein-Uhlenbeck pull to the level
  where the range began; High Volatility adds occasional jumps. Regimes last 2 to 8 simulated hours, then switch to a random *different* regime.
* Accelerated clock: one tick = 20 simulated seconds, so a 1-minute candle forms every 1.5 s and a daily candle every 36 min.
  The clock runs continuously (no market-hours gaps). Chart times are shown in IST.
* On start the server pre-generates about 7 simulated days of history, rescaled so the live price starts near ₹24,500.

### Message format (JSON)
Tick (every 500 ms): `type, seq, timestamp (UTC ISO), epoch, symbol, ltp, open, high, low, close, volume, tick_volume, change, change_pct, regime, regime_confidence, simulator_regime, candle, server_time_ms`.
`open/high/low/close/volume` describe the current 1-minute candle. `regime` is the ML-detected regime;
`simulator_regime` is the hidden ground truth, shown only so you can compare. Other types: `history` (snapshot), `analytics`, `pong`.
The client rejects invalid, duplicate and out-of-order ticks, and asks for a resync when it detects a sequence gap.

## Machine-learning pipeline

### Offline training
1. Read the bundled 1-minute OHLC files under `data/raw/nifty50/1min/` (2017–2026).
2. Combine yearly/monthly files, remove duplicate timestamps, and resample complete 3-minute candles.
3. Compute six OHLC-derived features: `ret_5`, `roll_vol`, `rsi`, `ma_diff`, `momentum`, and `atr_pct`.
4. Log-transform `roll_vol` and `atr_pct`; fit `StandardScaler` on the chronological training partition only.
5. Train a four-component `GaussianMixture` on the training partition.
6. Evaluate the held-out partition using internal clustering metrics and save:
   - `ml/models/nifty50_gmm_3m.joblib` (scaler + trained GMM + feature schema + regime mapping)
   - `ml/models/training_report.json` (date range, metrics, cluster counts, limitations)

Run training from the project root:

```bash
python -m ml.train_historical_model
```

### Live demonstration inference
`backend/market_state.py` loads `ml/models/nifty50_gmm_3m.joblib` at startup. In offline-artifact mode,
incoming simulated candles are transformed using the saved feature schema and scaler, then classified
with the saved GMM. The model is not retrained on each incoming candle. If the artifact is missing,
the code retains its legacy simulator fallback mode.

Cluster IDs have no inherent meaning. MarketMind assigns descriptive names using transparent centroid
heuristics (trend and volatility features); these names are interpretations, not ground-truth labels.
The maximum GMM component membership is displayed as a model membership score, not a calibrated
probability that a named regime is correct.

### Current training evidence
The bundled historical-data training run produced:
- 296,850 complete 3-minute feature rows
- 237,480 training rows and 59,370 chronological holdout rows
- Four GMM components present in the holdout
- Holdout Silhouette: 0.158
- Holdout Davies-Bouldin: 3.688
- Holdout Calinski-Harabasz: 1,004.94

These are internal clustering metrics, not supervised accuracy or trading performance. The cluster
sizes are imbalanced, and some centroid interpretations are weak; the model is suitable for
demonstrating the end-to-end ML pipeline, not for claims of objectively validated market regimes.

The repository also contains K-Means, HMM, wavelet, and multi-timeframe research notebooks/results.
Those are comparative experiments; the currently integrated dashboard inference model is GMM.

## Strategies
Each evaluator reads indicators only (never the ML regime) and returns a 0-100 suitability score, a compatibility label,
supporting indicators, a reason, and risk notes. Scores are **relative suitability, not win probability or expected return**.
Inputs are scaled by volatility or ATR so thresholds do not depend on the timeframe. Suitable means 60 or more; unsuitable means under 40.

1. **Momentum**: size of 10-bar momentum versus noise (40%), RSI confirmation with an exhaustion penalty (25%), 5-bar return (25%), agreement of signals (10%).
2. **Mean reversion**: Bollinger stretch (35%), distance from SMA50 in ATRs (20%), RSI extreme in the same direction (20%), absence of a trend via low ADX (25%).
3. **Breakout**: break of, or proximity to, the 20-bar high/low (40%), volume expansion (25%), bar range versus ATR (15%), prior Bollinger squeeze (20%).
4. **Trend following**: SMA/EMA stack alignment (35%), ADX strength (30%), MACD agreement (20%), SMA20 slope in ATRs (15%).
5. **Volatility-based**: ATR percentile (35%), volatility expansion versus its 50-bar average (35%), ATR growth (15%), band-width percentile (15%).

## Tests

```bash
python -m unittest discover -s tests -t .      # no extra packages needed
pytest                                          # also works
```

**What was run and passed (62 tests; run in a sandbox without network access):**
simulator (positivity, reproducibility, regime drift/volatility, range behaviour, OHLC consistency), tick aggregation and
timeframe resampling against manual calculation (1m to 1d), all indicators against manual formulas and warm-up behaviour,
regime detection (cluster interpretation, accuracy versus ground truth, online refits), all five strategies and ranking changes,
the engine and strict-JSON payloads, the connection manager (multiple clients, dead and slow clients, graceful close),
the real `producer` loop cadence, and the real `chart.js` run under Node with a stubbed DOM/WebSocket/chart library
(history load, live updates, duplicate/invalid/gap handling, all six timeframes, reconnect, watchdog).

**What was NOT run in the build sandbox** (FastAPI, uvicorn, Streamlit and a browser were unavailable offline):
the FastAPI app itself, a real WebSocket handshake, Streamlit embedding, and rendering with the actual TradingView library.
`tests/test_websocket_server.py` covers the FastAPI endpoints and is skipped automatically when those packages are missing;
after `pip install -r requirements.txt` it will run. Please run the suite once on your machine and open the dashboard to confirm.

## Limitations
* Simulated data cannot be predicted; strategy scores describe current conditions only, with no back-testing or P&L.
* The regime model lags and is imperfect. The heuristic mapping from clusters to names is a design choice, not ground truth.
* One shared market for all clients; the server keeps state in memory only.

## ML model selection research

See `research/README.md`. Reproduce the comparison with `python -m research.run_comparison`. The production regime detector currently uses Gaussian Mixture based on the highest mean held-out ARI and NMI in the synthetic-simulator benchmark. This does not establish real-market superiority.

## Real-data ML pipeline (Yahoo Finance, 3-minute)

The real-data pipeline is separate from the synthetic simulator used by the existing live dashboard.

1. Install dependencies: `pip install -r requirements.txt`
2. Download Yahoo NIFTY 50 1-minute data and resample to 3-minute bars: `python -m ml.yahoo_data --period 7d`
3. Train and chronologically evaluate the offline GMM: `python -m ml.train_model`
4. The model artifact is saved at `ml/models/nifty50_gmm_3m.joblib`; its metrics and date coverage are written to `ml/models/training_report.json`.
5. Start the existing backend and dashboard as before. `MarketEngine` automatically loads the artifact if it exists. Set `MARKETMIND_MODEL_PATH` to use a different artifact.

The offline detector uses the saved scaler, six price-derived features, GMM and cluster-to-regime mapping without refitting on incoming data. The current WebSocket market simulator remains synthetic; loading a Yahoo-trained model does not turn that simulated feed into live NSE data. The original rolling-fit detector remains as a fallback when no trained artifact is present, preserving existing simulator workflows.

Yahoo's 1-minute historical lookback is short (typically up to seven days), so the resulting training set is small. The pipeline fails with an explicit minimum-row error rather than silently claiming broad historical coverage. GMM is unsupervised: the four regime names are assigned using transparent cluster-centroid heuristics, and clustering metrics are not predictive accuracy or profitability.
