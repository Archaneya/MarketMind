"""Builds a single self-contained HTML document for embedding inside Streamlit.

Streamlit renders components in a sandboxed iframe with no access to relative files, so the CSS
and JavaScript are inlined here and the WebSocket URL is injected.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
CDN_TAG = '<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>'
VENDOR = HERE / "vendor" / "lightweight-charts.standalone.production.js"


def build_dashboard_html(ws_url: str = "ws://localhost:8000/ws") -> str:
    html = (HERE / "chart.html").read_text(encoding="utf-8")
    css = (HERE / "styles.css").read_text(encoding="utf-8")
    js = (HERE / "chart.js").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="static/styles.css">', f"<style>{css}</style>")
    html = html.replace('<script src="static/chart.js"></script>', f"<script>{js}</script>")
    html = html.replace('"__WS_URL__"', json.dumps(ws_url))
    if VENDOR.exists():  # optional offline copy of the charting library
        html = html.replace(CDN_TAG, f"<script>{VENDOR.read_text(encoding='utf-8')}</script>")
    return html
