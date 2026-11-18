# MarketMind data

Run `python -m ml.yahoo_data --period 7d` from the project root to fetch Yahoo Finance `^NSEI` one-minute candles and resample complete regular-session bars into three-minute OHLC candles.

Outputs:

- `raw/nifty50_yahoo_1m.csv`: provider-returned one-minute observations.
- `processed/nifty50_yahoo_3m.csv`: complete 3-minute bars, aligned to the NSE 09:15 IST session open.
- `metadata.json`: provider, symbol, intervals, row counts, date coverage and limitations.

Yahoo Finance's one-minute lookback is short (typically up to seven days). This is a limited recent sample, not a multi-year history. The NIFTY 50 index itself is not a traded security; the pipeline deliberately excludes its volume field from ML features rather than treating zero/missing index volume as meaningful activity.
