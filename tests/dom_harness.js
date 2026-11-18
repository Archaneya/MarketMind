// Runs frontend/chart.js with a stubbed DOM, WebSocket and charting library (no browser needed).
const fs = require('fs'), vm = require('vm'), path = require('path');
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const els = {};
function mk(id) {
  return { id, innerHTML: '', textContent: '', className: '', style: {}, dataset: {}, children: [],
    classList: { toggle() {}, add() {}, remove() {} }, querySelector() { return el(id + '>b'); }, applyOptions() {} };
}
function el(id) { return els[id] || (els[id] = mk(id)); }
const stats = { cs: { setData: 0, update: 0, n: 0 }, vs: { setData: 0, update: 0, n: 0 }, lines: { setData: 0, update: 0 } };
const series = name => ({ setData(d) { stats[name].setData++; stats[name].n = d.length; stats[name].first = d[0]; stats[name].last = d[d.length - 1]; },
  update(p) { stats[name].update++; stats[name].lastUpdate = p; }, applyOptions() {} });
const timers = [];
global.window = global; global.addEventListener = () => {};
global.document = { getElementById: el };
els['chart'] = Object.assign(mk('chart'), { clientWidth: 900, clientHeight: 500 });
global.location = { protocol: 'http:', host: 'localhost:8000' };
global.setInterval = (f, ms) => { timers.push({ kind: 'i', f, ms }); return timers.length; };
global.setTimeout = (f, ms) => { timers.push({ kind: 't', f, ms }); return timers.length; };
global.LightweightCharts = { CrosshairMode: { Normal: 0 }, createChart() { return {
  addCandlestickSeries: () => series('cs'), addHistogramSeries: () => series('vs'), addLineSeries: () => series('lines'),
  applyOptions() {}, priceScale: () => ({ applyOptions() {} }),
  timeScale: () => ({ setVisibleLogicalRange() {}, scrollToRealTime() {} }), subscribeCrosshairMove() {} }; } };
const sockets = [];
global.WebSocket = class { constructor(url) { this.url = url; this.readyState = 1; this.sent = []; sockets.push(this); }
  send(s) { this.sent.push(JSON.parse(s)); } close() { this.readyState = 3; if (this.onclose) this.onclose(); } };
global.MARKETMIND_WS_URL = 'ws://test/ws';
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'frontend', 'chart.js'), 'utf8'));

const out = {};
const ws = sockets[0];
const msg = m => ws.onmessage({ data: JSON.stringify(m) });
ws.onopen();
out.url = ws.url;
msg(input.ticks[0]);                         // tick before history must be ignored
out.ignoredBeforeHistory = stats.cs.update;
msg(input.history); msg(input.analytics);
out.csAfterHistory = stats.cs.n; out.vsAfterHistory = stats.vs.n;
out.status = els['ws-status'].className;
for (const t of input.ticks) msg(t);
out.tickUpdates = stats.cs.update;
out.ltpText = els['ltp'].textContent;
out.lastLtp = input.ticks[input.ticks.length - 1].ltp;
out.rankingHasAll = input.analytics.strategies.every(s => els['ranking'].innerHTML.includes(s.name));
out.regimeCard = els['regime-card'].innerHTML.includes(input.analytics.regime.name);
out.indicatorsRows = (els['indicators'].innerHTML.match(/<tr>/g) || []).length;
out.compareRows = (els['compare'].innerHTML.match(/<tr>/g) || []).length;
out.simTime = els['sim-time'].textContent;
// duplicate, invalid and gap handling
const before = stats.cs.update;
msg(input.ticks[2]);                                         // duplicate seq
const bad = JSON.parse(JSON.stringify(input.ticks[input.ticks.length - 1])); bad.seq += 1; bad.high = bad.low - 5;
msg(bad);                                                    // invalid OHLC
out.updatesFromBadMessages = stats.cs.update - before;
const gap = JSON.parse(JSON.stringify(input.ticks[input.ticks.length - 1])); gap.seq += 5;
msg(gap);
out.resyncSent = ws.sent.some(s => s.type === 'resync');
ws.onmessage({ data: 'not json{' });
// timeframe switching
const counts = {};
for (const tf of ['1m', '5m', '15m', '1h', '4h', '1d']) { els['tf-buttons'].onclick({ target: { dataset: { tf } } }); counts[tf] = stats.cs.n; }
out.counts = counts;
els['overlay-buttons'].onclick({ target: { dataset: { ov: 'bb' }, classList: { toggle() {} } } });
timers.filter(x => x.kind === 'i')[0].f();   // fire the 1-second diagnostics timer
out.diag = els['diag'].innerHTML;
// reconnection after an interruption
ws.onclose();
const t = timers.filter(x => x.kind === 't').pop();
out.reconnectDelayMs = t.ms; t.f();
out.socketsOpened = sockets.length;
sockets[1].onopen();
out.statusAfterReconnect = els['ws-status'].className;
sockets[1].onmessage({ data: JSON.stringify(input.history) });
out.readyAgain = stats.cs.setData;
// watchdog closes a silent socket
const dog = timers.filter(x => x.kind === 'i' && x.ms === 1000).pop();
const realNow = Date.now; Date.now = () => realNow() + 9000; dog.f(); Date.now = realNow;
out.watchdogClosed = sockets[1].readyState === 3;
console.log(JSON.stringify(out));
