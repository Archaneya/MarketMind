/* MarketMind dashboard: WebSocket client, candle aggregation, chart and panels.
   Everything updates from the WebSocket stream; no page reruns are involved. */
(function () {
  'use strict';
  const IST = 19800;                    // display times in IST (+05:30)
  const MAX_BASE = 30000;               // bounded 1-minute history kept in the browser
  const TFS = { '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440 };
  const REG_COLOR = { 'Bullish Trend': 'var(--up)', 'Bearish Trend': 'var(--down)',
    'Sideways / Range-bound': 'var(--side)', 'High Volatility': 'var(--hv)' };
  const REG_HEX = { 'Bullish Trend': '#2fbf8f', 'Bearish Trend': '#ef5b6b',
    'Sideways / Range-bound': '#6aa7e0', 'High Volatility': '#e3a93b' };

  /* ---------- pure helpers (also exported for tests) ---------- */
  function aggregate(base, minutes) {
    const sec = minutes * 60, out = [];
    for (const c of base) {
      const t = Math.floor(c.time / sec) * sec, last = out[out.length - 1];
      if (last && last.time === t) {
        last.high = Math.max(last.high, c.high); last.low = Math.min(last.low, c.low);
        last.close = c.close; last.volume += c.volume;
      } else out.push({ time: t, open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume });
    }
    return out;
  }
  function bucketOf(base, minutes) {            // rebuild only the newest bucket
    const sec = minutes * 60, b = Math.floor(base[base.length - 1].time / sec) * sec;
    let i = base.length - 1, h = -Infinity, l = Infinity, v = 0, o = 0;
    while (i >= 0 && base[i].time >= b) { h = Math.max(h, base[i].high); l = Math.min(l, base[i].low); v += base[i].volume; o = base[i].open; i--; }
    return { time: b, open: o, high: h, low: l, close: base[base.length - 1].close, volume: v };
  }
  function validTick(m) {
    const ok = x => typeof x === 'number' && isFinite(x) && x > 0;
    return m && ok(m.ltp) && ok(m.open) && ok(m.high) && ok(m.low) && ok(m.close) && m.candle &&
      m.high >= Math.max(m.open, m.close) - 1e-9 && m.low <= Math.min(m.open, m.close) + 1e-9;
  }
  if (typeof module !== 'undefined') { module.exports = { aggregate, bucketOf, validTick }; }
  if (typeof document === 'undefined') return;

  /* ---------- state & DOM ---------- */
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const S = { base: [], agg: [], tf: '5m', ready: false, lastSeq: -1, resyncAt: 0, ws: null, retry: 0, lastMsg: 0,
    ov: { sma20: true, sma50: true, ema20: false, bb: false } };
  const D = { msgs: 0, rate: 0, latency: null, gaps: 0, dupes: 0, invalid: 0, bad: 0, reconnects: 0, resyncs: 0, analytics: 0 };
  const fmt = (x, d = 2) => Number(x).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d });
  const istStr = e => new Date((e + IST) * 1000).toISOString().replace('T', ' ').slice(0, 19);

  /* ---------- chart ---------- */
  let chart, cs, vs; const lines = {};
  if (typeof LightweightCharts === 'undefined') {
    $('chart').innerHTML = '<div class="block muted">The charting library could not load. Check your internet connection (or place lightweight-charts.standalone.production.js in frontend/vendor/).</div>';
  } else {
    chart = LightweightCharts.createChart($('chart'), {
      layout: { background: { type: 'solid', color: '#0e141b' }, textColor: '#8b99ab', fontFamily: 'IBM Plex Sans, system-ui, sans-serif' },
      grid: { vertLines: { color: '#17212b' }, horzLines: { color: '#17212b' } },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#243040', scaleMargins: { top: 0.06, bottom: 0.22 } },
      timeScale: { borderColor: '#243040', timeVisible: true, secondsVisible: false, rightOffset: 6 },
    });
    const box = $('chart'), fit = () => chart.applyOptions({ width: box.clientWidth, height: box.clientHeight });
    fit();
    if (typeof ResizeObserver !== 'undefined') new ResizeObserver(fit).observe(box);
    window.addEventListener('resize', fit);
    cs = chart.addCandlestickSeries({ upColor: '#2fbf8f', downColor: '#ef5b6b', borderVisible: false,
      wickUpColor: '#2fbf8f', wickDownColor: '#ef5b6b', priceLineVisible: true, lastValueVisible: true });
    vs = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol', lastValueVisible: false, priceLineVisible: false });
    chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    const L = (color, dash) => chart.addLineSeries({ color, lineWidth: 1, lineStyle: dash ? 2 : 0, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    lines.sma20 = L('#e3a93b'); lines.sma50 = L('#6aa7e0'); lines.ema20 = L('#c792ea');
    lines.bbU = L('#7e8c9d', true); lines.bbM = L('#4d5a6a', true); lines.bbL = L('#7e8c9d', true);
    chart.subscribeCrosshairMove(p => {
      const d = p && p.seriesData && p.seriesData.get(cs);
      const c = d || S.agg[S.agg.length - 1]; if (!c) return;
      const up = c.close >= c.open;
      $('legend').innerHTML = `<b>${S.tf}</b>O <b>${fmt(c.open)}</b>H <b>${fmt(c.high)}</b>L <b>${fmt(c.low)}</b>C <b class="${up ? 'up' : 'down'}">${fmt(c.close)}</b>`;
    });
  }

  /* ---------- overlays (computed on the aggregated candles) ---------- */
  const ind = { sma20: [], sma50: [], ema20: [], bbU: [], bbM: [], bbL: [] };
  function indAt(i) {
    const a = S.agg, n = a.length, c = a[i].close;
    const mean = (k) => { let s = 0; for (let j = i - k + 1; j <= i; j++) s += a[j].close; return s / k; };
    ind.sma20[i] = i >= 19 ? mean(20) : null;
    ind.sma50[i] = i >= 49 ? mean(50) : null;
    const al = 2 / 21;
    ind.ema20[i] = i < 19 ? null : (i === 19 ? mean(20) : ind.ema20[i - 1] * (1 - al) + c * al);
    if (i >= 19) {
      const m = ind.sma20[i]; let v = 0; for (let j = i - 19; j <= i; j++) v += (a[j].close - m) ** 2;
      const sd = Math.sqrt(v / 20); ind.bbM[i] = m; ind.bbU[i] = m + 2 * sd; ind.bbL[i] = m - 2 * sd;
    } else { ind.bbM[i] = ind.bbU[i] = ind.bbL[i] = null; }
  }
  const pt = (k, i) => ind[k][i] == null ? null : { time: S.agg[i].time, value: ind[k][i] };
  const keys = { sma20: 'sma20', sma50: 'sma50', ema20: 'ema20', bbU: 'bbU', bbM: 'bbM', bbL: 'bbL' };

  /* ---------- rendering the series ---------- */
  const volColor = c => c.close >= c.open ? 'rgba(47,191,143,.4)' : 'rgba(239,91,107,.4)';
  function rebuild(scroll) {
    S.agg = aggregate(S.base, TFS[S.tf]);
    if (!cs) return;
    cs.setData(S.agg.map(c => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close })));
    vs.setData(S.agg.map(c => ({ time: c.time, value: c.volume, color: volColor(c) })));
    for (const k in ind) ind[k] = [];
    for (let i = 0; i < S.agg.length; i++) indAt(i);
    for (const k in keys) lines[k].setData(S.agg.map((_, i) => pt(k, i)).filter(Boolean));
    applyOverlayVisibility();
    if (scroll) { const n = S.agg.length; chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - 120), to: n + 6 }); }
  }
  function liveUpdate(c1) {
    const last = S.base[S.base.length - 1];
    if (last && c1.time < last.time) { D.dupes++; return; }          // out-of-order candle
    if (last && c1.time === last.time) S.base[S.base.length - 1] = c1; else S.base.push(c1);
    if (S.base.length > MAX_BASE) S.base.splice(0, S.base.length - MAX_BASE);
    const b = bucketOf(S.base, TFS[S.tf]), a = S.agg, l = a[a.length - 1];
    if (l && b.time < l.time) return;
    if (l && b.time === l.time) a[a.length - 1] = b; else a.push(b);
    if (!cs) return;
    const i = a.length - 1; indAt(i);
    cs.update({ time: b.time, open: b.open, high: b.high, low: b.low, close: b.close });
    vs.update({ time: b.time, value: b.volume, color: volColor(b) });
    for (const k in keys) { const p = pt(k, i); if (p) lines[k].update(p); }
  }
  function applyOverlayVisibility() {
    if (!cs) return;
    lines.sma20.applyOptions({ visible: S.ov.sma20 }); lines.sma50.applyOptions({ visible: S.ov.sma50 });
    lines.ema20.applyOptions({ visible: S.ov.ema20 });
    ['bbU', 'bbM', 'bbL'].forEach(k => lines[k].applyOptions({ visible: S.ov.bb }));
  }

  /* ---------- toolbar ---------- */
  $('tf-buttons').innerHTML = Object.keys(TFS).map(k => `<button type="button" data-tf="${k}" class="${k === S.tf ? 'active' : ''}">${k}</button>`).join('');
  $('tf-buttons').onclick = e => {
    const k = e.target.dataset && e.target.dataset.tf; if (!k) return;
    S.tf = k; [...$('tf-buttons').children].forEach(b => b.classList.toggle('active', b.dataset.tf === k));
    if (S.base.length) rebuild(true);
  };
  const OVS = [['sma20', 'SMA 20'], ['sma50', 'SMA 50'], ['ema20', 'EMA 20'], ['bb', 'Bollinger']];
  $('overlay-buttons').innerHTML = OVS.map(([k, n]) => `<button type="button" data-ov="${k}" class="${S.ov[k] ? 'active' : ''}">${n}</button>`).join('');
  $('overlay-buttons').onclick = e => {
    const k = e.target.dataset && e.target.dataset.ov; if (!k) return;
    S.ov[k] = !S.ov[k]; e.target.classList.toggle('active', S.ov[k]); applyOverlayVisibility();
  };
  $('btn-latest').onclick = () => chart && chart.timeScale().scrollToRealTime();

  /* ---------- message handlers ---------- */
  function onHistory(m) {
    S.base = (m.candles || []).map(c => ({ ...c, time: c.time + IST }));
    S.lastSeq = m.seq; S.ready = true; S.resyncAt = 0;
    rebuild(true);
  }
  function onTick(m) {
    if (!S.ready) return;
    if (!validTick(m)) { D.invalid++; return; }
    if (m.seq <= S.lastSeq) { D.dupes++; return; }
    if (S.lastSeq >= 0 && m.seq > S.lastSeq + 1) {
      D.gaps += m.seq - S.lastSeq - 1;
      if (!S.resyncAt || Date.now() - S.resyncAt > 3000) { S.resyncAt = Date.now(); D.resyncs++; send({ type: 'resync' }); }
    }
    S.lastSeq = m.seq;
    D.latency = Math.max(0, Date.now() - m.server_time_ms);
    liveUpdate({ ...m.candle, time: m.candle.time + IST });
    const up = m.change >= 0, cls = up ? 'up' : 'down';
    $('ltp').textContent = fmt(m.ltp); $('ltp').className = 'big ' + cls;
    $('chg').textContent = (up ? '+' : '') + fmt(m.change); $('chg').className = 'mid ' + cls;
    $('chgp').textContent = (up ? '+' : '') + fmt(m.change_pct) + '%'; $('chgp').className = 'mid ' + cls;
    $('vol').textContent = Math.round(m.volume).toLocaleString('en-IN');
    $('sim-time').textContent = istStr(m.epoch) + ' IST';
    setRegimeChip(m.regime);
  }
  function setRegimeChip(name) {
    const c = $('regime-chip'); c.textContent = name;
    c.style.color = REG_COLOR[name] || 'var(--muted)'; c.style.borderColor = REG_COLOR[name] || 'var(--line)';
  }
  const bullets = a => '<ul>' + a.map(x => `<li>${esc(x)}</li>`).join('') + '</ul>';
  function onAnalytics(a) {
    D.analytics++;
    const r = a.regime, I = a.intelligence;
    $('tf-note').textContent = `(${a.analysis_timeframe} bars)`;
    $('regime-card').className = 'block';
    $('regime-card').innerHTML =
      `<div class="regime-name"><span class="dot" style="background:${REG_COLOR[r.name] || '#777'}"></span>${esc(r.name)}</div>
       <div class="muted">Cluster separation ${r.confidence.toFixed(2)} (0 = ambiguous, 1 = clear; not a probability)</div>
       <div class="kv"><b>Why this regime</b>${bullets(r.reasons || [])}</div>
       <details class="kv"><summary class="muted">How clusters become regimes</summary>${bullets(a.model.cluster_map || [])}
       <div class="muted">KMeans only groups similar bars. Names are assigned afterwards from each cluster's average trend and volatility. Fits: ${a.model.fits}, samples: ${a.model.samples}.</div></details>`;
    $('ranking').className = 'block';
    $('ranking').innerHTML = a.strategies.map(s => `
      <div class="rank"><span class="n">${s.rank}</span>
        <span class="nm">${esc(s.name)}<span class="tag ${s.compatibility}">${esc(s.compatibility)}</span></span>
        <span class="sc">${s.score.toFixed(0)}</span>
        <div class="bar"><i style="width:${s.score}%"></i></div>
        <details><summary>Why this score</summary>${esc(s.reason)}${bullets(s.risks)}</details></div>`).join('');
    $('intel').className = 'block';
    $('intel').innerHTML =
      `<div><b>Suitable now:</b> ${I.suitable.length ? esc(I.suitable.join(', ')) : 'none stand out'}</div>
       <div class="kv"><b>Unsuitable now:</b> ${I.unsuitable.length ? esc(I.unsuitable.join(', ')) : 'none'}</div>
       <div class="kv">${esc(I.volatility)}</div>
       <div class="kv"><b>Main risks</b>${bullets(I.risks)}</div>
       <div class="kv muted">${esc(I.note)}</div>`;
    $('indicators').innerHTML = '<table><tbody>' + a.indicators.map(i =>
      `<tr><td>${esc(i.name)}</td><td class="num">${esc(i.value)}</td><td class="muted">${esc(i.reading)}</td></tr>`).join('') + '</tbody></table>';
    const strip = (a.regime_history || []).map(h => `<i style="background:${REG_HEX[h[1]] || '#444'}" title="${istStr(h[0])} ${esc(h[1])}"></i>`).join('');
    const ch = (a.regime_changes || []).slice().reverse().map(c =>
      `<tr><td>${istStr(c.epoch).slice(5, 16)}</td><td class="muted">${esc(c.from)}</td><td>${esc(c.to)}</td></tr>`).join('');
    $('regime-history').innerHTML = `<div class="hist-strip" title="Last 120 analysis steps, oldest to newest">${strip}</div>
      <table><thead><tr><th>Sim time</th><th>From</th><th>To</th></tr></thead><tbody>${ch || '<tr><td colspan="3" class="muted">No regime change observed yet.</td></tr>'}</tbody></table>
      <div class="kv muted">Hidden simulator state (ground truth, for comparison): <b style="color:var(--text)">${esc(a.simulator_regime)}</b></div>`;
    $('compare').innerHTML = '<div style="overflow-x:auto"><table><thead><tr><th>#</th><th>Strategy</th><th class="num">Score</th><th>Compatibility</th><th>Supporting indicators</th><th>Reason</th><th>Risk observations</th></tr></thead><tbody>' +
      a.strategies.map(s => `<tr><td>${s.rank}</td><td>${esc(s.name)}</td><td class="num">${s.score.toFixed(1)}</td><td>${esc(s.compatibility)}</td>
        <td>${Object.entries(s.supporting).map(([k, v]) => esc(k) + ': ' + esc(v)).join('<br>')}</td><td>${esc(s.reason)}</td><td>${s.risks.map(esc).join('<br>')}</td></tr>`).join('') + '</tbody></table></div>';
  }
  function renderDiag() {
    D.rate = D.msgs; D.msgs = 0;
    const age = S.lastMsg ? ((Date.now() - S.lastMsg) / 1000).toFixed(1) + ' s' : '--';
    const rows = [['Messages/sec', D.rate], ['Latency', D.latency == null ? '--' : D.latency + ' ms'], ['Last message', age + ' ago'],
      ['Last tick seq', S.lastSeq], ['Missed ticks', D.gaps], ['Duplicate / out-of-order', D.dupes], ['Invalid messages', D.invalid + D.bad],
      ['Resyncs requested', D.resyncs], ['Reconnects', D.reconnects], ['Analytics updates', D.analytics], ['1m candles in memory', S.base.length]];
    $('diag').innerHTML = '<table><tbody>' + rows.map(r => `<tr><td>${r[0]}</td><td class="num">${r[1]}</td></tr>`).join('') + '</tbody></table>';
  }
  setInterval(renderDiag, 1000); renderDiag();

  /* ---------- WebSocket with automatic reconnection ---------- */
  function wsUrl() {
    const c = window.MARKETMIND_WS_URL;
    if (c && !c.startsWith('__')) return c;
    return (location.protocol === 'https:' ? 'wss://' : 'ws://') + (location.host || 'localhost:8000') + '/ws';
  }
  function setStatus(kind, text) { const e = $('ws-status'); e.className = 'conn ' + kind; e.querySelector('b').textContent = text; }
  function send(o) { if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify(o)); }
  function connect() {
    setStatus('wait', 'Connecting');
    let ws;
    try { ws = new WebSocket(wsUrl()); } catch (e) { return scheduleReconnect(); }
    S.ws = ws;
    ws.onopen = () => { S.retry = 0; S.ready = false; S.lastSeq = -1; S.lastMsg = Date.now(); setStatus('on', 'Connected'); };
    ws.onmessage = ev => {
      S.lastMsg = Date.now(); D.msgs++;
      let m; try { m = JSON.parse(ev.data); } catch (e) { D.bad++; return; }
      if (m.type === 'history') onHistory(m); else if (m.type === 'tick') onTick(m); else if (m.type === 'analytics') onAnalytics(m);
    };
    ws.onclose = () => { if (S.ws === ws) { S.ws = null; scheduleReconnect(); } };
    ws.onerror = () => { try { ws.close(); } catch (e) { /* already closed */ } };
  }
  function scheduleReconnect() {
    D.reconnects++;
    const delay = Math.min(10000, 1000 * Math.pow(2, S.retry++));
    setStatus('off', `Disconnected, retrying in ${Math.round(delay / 1000)} s`);
    setTimeout(connect, delay);
  }
  setInterval(() => {                       // watchdog: a silent socket is treated as dead
    if (S.ws && S.ws.readyState === 1) {
      if (Date.now() - S.lastMsg > 5000) { try { S.ws.close(); } catch (e) { /* ignore */ } }
      else if (Date.now() - S.lastMsg > 2500) send({ type: 'ping' });
    }
  }, 1000);
  connect();
})();
