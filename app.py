"""MarketMind - Streamlit shell.

The live dashboard is a self-contained HTML/JS component that talks to the FastAPI WebSocket
server directly, so market updates never depend on Streamlit reruns.

Run:  streamlit run app.py      (start `python websocket_server.py` first)
"""
import json
import urllib.request
from urllib.parse import urlparse

import streamlit as st
import streamlit.components.v1 as components

from frontend.assemble import build_dashboard_html

st.set_page_config(page_title="MarketMind", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    "<style>header[data-testid='stHeader'],footer{display:none}"
    ".block-container{padding:0!important;max-width:100%!important}"
    "[data-testid='stAppViewContainer']{background:#0a0e13}</style>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Connection")
    ws_url = st.text_input("WebSocket URL", "ws://localhost:8000/ws")
    st.caption("Start the backend first: `python websocket_server.py`. All data is simulated.")


def backend_status(url: str) -> dict | None:
    u = urlparse(url)
    health = f"{'https' if u.scheme == 'wss' else 'http'}://{u.netloc}/health"
    try:
        with urllib.request.urlopen(health, timeout=1.5) as r:
            return json.load(r)
    except Exception:
        return None


status = backend_status(ws_url)
if status is None:
    st.warning("Backend not reachable. Start it with `python websocket_server.py`; the dashboard reconnects automatically.")

components.html(build_dashboard_html(ws_url), height=1560, scrolling=True)
