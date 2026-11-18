# MarketMind demo deployment (Render)

This is an academic demonstration deployment. The dashboard uses a synthetic market simulator; it is not a live NSE feed and does not place trades.

## Deploy

1. Create a GitHub repository and upload the contents of this folder to its root.
2. Sign in to https://render.com and choose **New → Blueprint**.
3. Connect the GitHub repository containing `render.yaml`.
4. Review the `marketmind-demo` web service and click **Apply**.
5. Wait for the first build. Render will provide a public `https://...onrender.com` URL.
6. Open that URL. The FastAPI service serves the dashboard and its WebSocket endpoint from the same host, so HTTPS automatically uses secure WebSockets.

## Local run

```bash
pip install -r requirements.txt
uvicorn websocket_server:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. Health check: http://localhost:8000/health.

## Notes

- The Streamlit shell is not used for this hosted version. FastAPI serves `frontend/chart.html` and `/ws` from one service; this avoids a localhost WebSocket bridge that would fail for remote viewers.
- The saved GMM artifact is loaded from `ml/models/nifty50_gmm_3m.joblib`.
- The app intentionally labels the feed as simulated. No broker API keys are required.
- Render free services may sleep after inactivity and take time to wake on the first visit.
