import { LambdaTables } from './lambda.js';
import { computeGrid, DEFAULTS, pointModel, REGION, epsOfSt } from './model.js';
import { MapPlot, makeLUT, drawColorbar, contourSegments, categoryEdges, drawHistogram, fmt, el } from './plot.js';

// ------------------------------------------------------------ parameters
const PARAMS = {
  vfrag: { label: 'v<sub>frag</sub>', unit: 'm/s', min: 0.5, max: 30, log: true, digits: 2 },
  epsBkg: { label: 'ε<sub>bkg</sub>', unit: '', min: 1e-3, max: 0.1, log: true, digits: 2 },
  epsMax: { label: 'ε<sub>max</sub>', unit: '', min: 0.1, max: 3, log: true, digits: 2 },
  sigEps: { label: 'σ<sub>ε</sub>', unit: 'dex', min: 0.2, max: 2, log: false, digits: 2 },
  mstar: { label: 'M<sub>★</sub>', unit: 'M⊙', min: 0.1, max: 3, log: true, digits: 2 },
  a0: { label: 'a<sub>0</sub>', unit: 'μm', min: 0.1, max: 100, log: true, digits: 2 },
  rhoS: { label: 'ρ<sub>s</sub>', unit: 'g/cm³', min: 0.5, max: 3.5, log: false, digits: 2 },
};

const QUANT = {
  alpha: { label: 'α<sub>GI</sub>', text: 'alpha_GI', lo: -4, hi: -1.2, scope: 'gas' },
  sigma: { label: 'Σ<sub>g</sub> [g cm⁻²]', text: 'Sigma_g', lo: -0.5, hi: 3.5, scope: 'gas' },
  T: { label: 'T [K]', text: 'T', lo: 0.8, hi: 3.3, scope: 'gas' },
  hr: { label: 'H<sub>g</sub>/r', text: 'H/r', lo: -1.7, hi: -0.3, scope: 'gas' },
  beta: { label: 'β = t<sub>cool</sub>Ω', text: 'beta', lo: 0.8, hi: 3.7, scope: 'gas' },
  cs: { label: 'c<sub>s</sub> [m/s]', text: 'c_s', lo: 2.2, hi: 3.4, scope: 'gas', scale: 0.01 },
  stMax: { label: 'St<sub>max</sub>', text: 'St_max', lo: -4, hi: 1, scope: 'self' },
  eps: { label: 'ε (spiral arms)', text: 'eps', lo: -2, hi: 0, scope: 'self' },
  cdcs: { label: 'c<sub>d</sub>/c<sub>s</sub>', text: 'c_d/c_s', lo: -2.5, hi: 0, scope: 'self' },
  ddgiRatio: { label: 'c<sub>d</sub> / (c<sub>s</sub> ε)', text: 'c_d/(c_s eps)', lo: -1, hi: 1, scope: 'self', cmap: 'PuOr', diverging: true },
  lamFac: { label: 'Λ<sub>J</sub>', text: 'Lambda_J', lo: -1.5, hi: 0.2, scope: 'self' },
  mJ: { label: 'M<sub>J</sub> [M⊕]', text: 'M_J', lo: -0.5, hi: 2, scope: 'ddgi' },
  tGrowth: { label: 't<sub>growth</sub> [yr]', text: 't_growth', lo: 2, hi: 6, scope: 'self' },
  tDDGI: { label: 't<sub>DDGI</sub> [yr]', text: 't_DDGI', lo: 0.5, hi: 4, scope: 'ddgi' },
};

const REGION_INFO = [
  { id: REGION.SELF, name: 'Self-regulated GI', color: '#fc8d62' },
  { id: REGION.DDGI, name: 'DDGI', color: '#66c2a5' },
  { id: REGION.FRAG, name: 'Gas fragmentation', color: '#8da0cb' },
  { id: REGION.MRI, name: 'MRI', color: '#7b7b7b' },
  { id: REGION.NOGI, name: 'No GI', color: '#cfcfcf' },
];
const REGION_RGB = [];
for (const r of REGION_INFO) REGION_RGB[r.id] = [1, 3, 5].map((i) => parseInt(r.color.slice(i, i + 2), 16));

const GRID = { rMin: 1, rMax: 1000, mMin: 1e-8, mMax: 1e-3 };
const HIST_BINS = Array.from({ length: 41 }, (_, i) => 10 ** (-1 + i * 0.1)); // 0.1 - 1000 M_earth
const PIN_COLORS = ['#e08a2c', '#7a5bc4', '#c4425b'];

// ----------------------------------------------------------------- state
const state = {
  p: { ...DEFAULTS },
  res: 200,
  cmap: 'RdBu_r',
  gasAll: false,
  edges: true,
  qa: 'stMax', qb: 'mJ', ra: 'default', rb: 'default',
  pins: [],
  pinned: null,
  hover: null,
};
let T = null; // Lambda tables
let G = null; // last computed grid

// --------------------------------------------------------- URL hash state
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  for (const k of Object.keys(PARAMS)) if (h.has(k) && Number.isFinite(+h.get(k))) state.p[k] = +h.get(k);
  if (h.has('qa') && QUANT[h.get('qa')]) state.qa = h.get('qa');
  if (h.has('qb') && QUANT[h.get('qb')]) state.qb = h.get('qb');
  if (h.get('b') === 'cdcs' || h.get('b') === 'cdcs2') state.p.bConv = h.get('b');
}
function writeHash() {
  const h = new URLSearchParams();
  for (const k of Object.keys(PARAMS)) h.set(k, +state.p[k].toPrecision(4));
  h.set('qa', state.qa); h.set('qb', state.qb);
  if (state.p.bConv !== DEFAULTS.bConv) h.set('b', state.p.bConv);
  history.replaceState(null, '', `#${h}`);
}

// -------------------------------------------------------------- controls
function buildControls() {
  for (const div of document.querySelectorAll('.ctl')) {
    const key = div.dataset.key, P = PARAMS[key];
    const id = `in-${key}`;
    div.innerHTML = `
      <div class="ctl-head"><label for="${id}">${P.label}</label>
        <span><input type="number" class="num" step="any" aria-label="${key} value"> <span class="unit">${P.unit}</span></span></div>
      <input type="range" id="${id}" min="0" max="1000" step="1">`;
    const range = div.querySelector('input[type=range]');
    const num = div.querySelector('.num');
    const toU = (v) => (P.log ? Math.log(v / P.min) / Math.log(P.max / P.min) : (v - P.min) / (P.max - P.min)) * 1000;
    const fromU = (u) => (P.log ? P.min * (P.max / P.min) ** (u / 1000) : P.min + (u / 1000) * (P.max - P.min));
    const sync = () => { range.value = toU(state.p[key]); num.value = +state.p[key].toPrecision(3); };
    sync();
    div.sync = sync;
    range.addEventListener('input', () => {
      state.p[key] = +fromU(+range.value).toPrecision(3);
      num.value = state.p[key];
      schedule(true);
    });
    range.addEventListener('change', () => schedule(false));
    num.addEventListener('change', () => {
      const v = +num.value;
      if (!(v > 0)) { sync(); return; }
      state.p[key] = v;
      range.value = toU(Math.min(P.max, Math.max(P.min, v)));
      schedule(false);
    });
  }

  for (const [selId, key] of [['q-a', 'qa'], ['q-b', 'qb']]) {
    const sel = document.getElementById(selId);
    const groups = { Gas: ['alpha', 'sigma', 'T', 'hr', 'beta', 'cs'],
      Dust: ['stMax', 'eps', 'cdcs', 'ddgiRatio', 'lamFac', 'mJ', 'tGrowth', 'tDDGI'] };
    for (const [gname, keys] of Object.entries(groups)) {
      const og = document.createElement('optgroup');
      og.label = gname;
      for (const k of keys) {
        const o = document.createElement('option');
        o.value = k;
        o.textContent = QUANT[k].label.replace(/<[^>]+>/g, '');
        og.appendChild(o);
      }
      sel.appendChild(og);
    }
    sel.value = state[key];
    sel.addEventListener('change', () => { state[key] = sel.value; writeHash(); render(); });
  }
  for (const [selId, key] of [['r-a', 'ra'], ['r-b', 'rb']]) {
    const sel = document.getElementById(selId);
    sel.addEventListener('change', () => { state[key] = sel.value; render(); });
  }

  const on = (id, ev, f) => document.getElementById(id).addEventListener(ev, f);
  on('sel-res', 'change', (e) => { state.res = +e.target.value; schedule(false); });
  on('sel-cmap', 'change', (e) => { state.cmap = e.target.value; render(); });
  on('sel-bconv', 'change', (e) => { state.p.bConv = e.target.value; schedule(false); });
  document.getElementById('sel-bconv').value = state.p.bConv;
  on('chk-gas-all', 'change', (e) => { state.gasAll = e.target.checked; render(); });
  on('chk-edges', 'change', (e) => { state.edges = e.target.checked; render(); });
  on('btn-reset', 'click', () => {
    state.p = { ...DEFAULTS };
    document.getElementById('sel-bconv').value = state.p.bConv;
    for (const d of document.querySelectorAll('.ctl')) d.sync();
    schedule(false);
  });
  on('btn-pin', 'click', pinCurrent);
  on('btn-theme', 'click', () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    render();
  });
}

// -------------------------------------------------------------- compute
let pending = null;
let lastMs = 0;
function schedule(dragging) {
  writeHash();
  drawEpsPreview();
  if (pending) cancelAnimationFrame(pending);
  pending = requestAnimationFrame(() => {
    pending = null;
    // while dragging, drop resolution if a full update is slow
    const res = dragging && lastMs > 80 ? Math.max(60, Math.round(state.res / 2)) : state.res;
    const t0 = performance.now();
    G = computeGrid(T, { nx: res, ny: res, ...GRID }, state.p);
    lastMs = performance.now() * 1 - t0;
    if (res !== state.res) lastMs *= 4;
    analyse();
    render();
    document.getElementById('status').textContent = `${res}×${res} grid computed in ${Math.round(performance.now() - t0)} ms`;
  });
}

// derived: DDGI contour, labels, histogram, stats
function analyse() {
  const { nx, ny, region } = G;
  const field = new Float32Array(nx * ny);
  for (let k = 0; k < field.length; k++) field[k] = Math.log10(G.ddgiRatio[k]);
  G.ddgiSeg = contourSegments(field, nx, ny, 0);
  G.edgeSeg = categoryEdges(region, nx, ny, (a, b) => !((a === REGION.SELF && b === REGION.DDGI) || (a === REGION.DDGI && b === REGION.SELF)));
  G.labels = regionLabels();

  const counts = new Array(HIST_BINS.length - 1).fill(0);
  const mj = [], tg = [];
  let nSelf = 0;
  for (let k = 0; k < region.length; k++) {
    if (region[k] === REGION.SELF || region[k] === REGION.DDGI) nSelf++;
    if (region[k] !== REGION.DDGI || !(G.mJ[k] > 0)) continue;
    mj.push(G.mJ[k]); tg.push(G.tGrowth[k]);
    const b = Math.floor((Math.log10(G.mJ[k]) + 1) / 0.1);
    counts[Math.min(counts.length - 1, Math.max(0, b))]++;
  }
  const n = mj.length;
  G.hist = { counts, n, frac: counts.map((c) => (n ? c / n : 0)) };
  const q = (a, f) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.floor(f * (s.length - 1))]; };
  G.stats = { n, nSelf, mjMed: q(mj, 0.5), mj10: q(mj, 0.1), mj90: q(mj, 0.9), tgMed: q(tg, 0.5), tg10: q(tg, 0.1), tg90: q(tg, 0.9) };
}

// one label per connected component of each region (largest ones only)
function regionLabels() {
  const { nx, ny, region } = G;
  const seen = new Uint8Array(nx * ny);
  const out = [];
  const lr0 = Math.log10(GRID.rMin), dlr = Math.log10(GRID.rMax / GRID.rMin) / nx;
  const lm0 = Math.log10(GRID.mMin), dlm = Math.log10(GRID.mMax / GRID.mMin) / ny;
  for (let s = 0; s < nx * ny; s++) {
    if (seen[s]) continue;
    const c = region[s], cells = [s];
    seen[s] = 1;
    for (let h = 0; h < cells.length; h++) {
      const k = cells[h], i = k % nx, j = (k / nx) | 0;
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const ii = i + di, jj = j + dj;
        if (ii < 0 || jj < 0 || ii >= nx || jj >= ny) continue;
        const kk = jj * nx + ii;
        if (!seen[kk] && region[kk] === c) { seen[kk] = 1; cells.push(kk); }
      }
    }
    if (cells.length < 0.03 * nx * ny) continue;
    // label at the cell farthest from the component's border (approx.: largest inscribed square)
    let best = cells[0], bestD = -1;
    const inC = new Set(cells);
    const step = Math.max(1, Math.floor(cells.length / 4000));
    for (let h = 0; h < cells.length; h += step) {
      const k = cells[h], i = k % nx, j = (k / nx) | 0;
      let d = 1;
      while (d < nx && [[d, 0], [-d, 0], [0, d], [0, -d]].every(([a, b]) => inC.has((j + b) * nx + i + a) && i + a >= 0 && i + a < nx)) d++;
      if (d > bestD) { bestD = d; best = k; }
    }
    const i = best % nx, j = (best / nx) | 0;
    out.push({ region: c, r: 10 ** (lr0 + (i + 0.5) * dlr), mdot: 10 ** (lm0 + (j + 0.5) * dlm) });
  }
  return out;
}

// --------------------------------------------------------------- render
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
};

let maps;
function rgbaFor(key, rangeMode) {
  const { nx, ny, region } = G;
  const Q = QUANT[key];
  const data = G[key];
  const scale = Q.scale || 1;
  const lut = makeLUT(Q.cmap || state.cmap);
  const out = new Uint8ClampedArray(nx * ny * 4);
  const mri = isDark() ? [92, 92, 92] : [150, 150, 150];
  let lo = Q.lo, hi = Q.hi;
  const show = (c) => {
    if (Q.scope === 'gas') return state.gasAll || c === REGION.SELF || c === REGION.DDGI;
    if (Q.scope === 'ddgi') return c === REGION.DDGI;
    return c === REGION.SELF || c === REGION.DDGI;
  };
  if (rangeMode === 'auto') {
    const v = [];
    for (let k = 0; k < data.length; k++) if (show(region[k]) && data[k] > 0) v.push(Math.log10(data[k] * scale));
    if (v.length > 10) {
      v.sort((a, b) => a - b);
      lo = v[Math.floor(0.02 * (v.length - 1))]; hi = v[Math.floor(0.98 * (v.length - 1))];
      if (Q.diverging) { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; }
      if (hi - lo < 1e-3) { lo -= 0.5; hi += 0.5; }
    }
  }
  for (let k = 0; k < data.length; k++) {
    const c = region[k];
    if (c === REGION.MRI && !(Q.scope === 'gas' && state.gasAll)) {
      out.set(mri, k * 4); out[k * 4 + 3] = 255; continue;
    }
    if (!show(c) || !(data[k] > 0)) continue; // transparent
    let u = (Math.log10(data[k] * scale) - lo) / (hi - lo);
    u = Math.min(1, Math.max(0, u));
    const li = Math.round(u * 255) * 3;
    out[k * 4] = lut[li]; out[k * 4 + 1] = lut[li + 1]; out[k * 4 + 2] = lut[li + 2]; out[k * 4 + 3] = 255;
  }
  return { rgba: out, lo, hi };
}

function regimeRGBA() {
  const out = new Uint8ClampedArray(G.nx * G.ny * 4);
  const dark = isDark();
  for (let k = 0; k < G.region.length; k++) {
    let c = REGION_RGB[G.region[k]];
    if (dark && G.region[k] === REGION.NOGI) c = [70, 70, 70];
    if (dark && G.region[k] === REGION.MRI) c = [40, 40, 40];
    out[k * 4] = c[0]; out[k * 4 + 1] = c[1]; out[k * 4 + 2] = c[2]; out[k * 4 + 3] = 255;
  }
  return out;
}

// pinned DDGI boundaries are stored in data coordinates; map them to this grid
function pinSegments(pin) {
  const sx = G.nx / Math.log10(GRID.rMax / GRID.rMin), sy = G.ny / Math.log10(GRID.mMax / GRID.mMin);
  return pin.seg.map(([a, b, c, d]) => [(a - Math.log10(GRID.rMin)) * sx, (b - Math.log10(GRID.mMin)) * sy,
    (c - Math.log10(GRID.rMin)) * sx, (d - Math.log10(GRID.mMin)) * sy]);
}

function render() {
  if (!G) return;
  const ddgi = { segments: G.ddgiSeg, cls: 'ddgi-line' };
  const edges = state.edges ? [{ segments: G.edgeSeg, cls: 'edge-line' }] : [];
  const pins = state.pins.map((p, n) => ({ segments: pinSegments(p), cls: `pin-line pin-${n}` }));

  const names = Object.fromEntries(REGION_INFO.map((r) => [r.id, r.name]));
  maps.reg.draw(G, regimeRGBA(), {
    contours: [{ segments: G.edgeSeg, cls: 'edge-line strong' }, ...pins, ddgi],
    labels: G.labels.map((l) => ({ ...l, text: names[l.region] })),
  });
  for (const [m, qk, rk, cb] of [[maps.a, 'qa', 'ra', 'cbar-a'], [maps.b, 'qb', 'rb', 'cbar-b']]) {
    const key = state[qk], Q = QUANT[key];
    const { rgba, lo, hi } = rgbaFor(key, state[rk]);
    m.draw(G, rgba, { contours: [...edges, ddgi] });
    drawColorbar(document.getElementById(cb), { cmap: Q.cmap || state.cmap, vmin: lo, vmax: hi, label: Q.label, log: true, diverging: Q.diverging });
  }
  drawHist();
  drawStats();
  updateCursor();
  drawInspector();
}

function drawHist() {
  const tip = document.getElementById('tooltip');
  drawHistogram(document.getElementById('hist'), {
    bins: HIST_BINS, current: G.hist.n ? G.hist : null, pinned: state.pins, xLabel: 'M_J [M⊕]',
    onHover: (i, e) => {
      if (i === null) { tip.hidden = true; return; }
      const lines = [`<b>${fmt(HIST_BINS[i], 2)}–${fmt(HIST_BINS[i + 1], 2)} M⊕</b>`,
        `current: ${G.hist.counts[i]} cells (${(G.hist.frac[i] * 100).toFixed(1)}%)`,
        ...state.pins.map((p) => `${p.label}: ${(p.hist.frac[i] * 100).toFixed(1)}%`)];
      showTip(lines.join('<br>'), e);
    },
  });
}

function drawStats() {
  const s = G.stats;
  const st = document.getElementById('stats');
  if (!s.n) { st.innerHTML = '<p>No DDGI-unstable cells for these parameters.</p>'; return; }
  st.innerHTML = `
    <div class="stat"><span class="stat-v">${(100 * s.n / s.nSelf).toFixed(0)}%</span><span class="stat-l">of self-regulated area is DDGI</span></div>
    <div class="stat"><span class="stat-v">${fmt(s.mjMed, 2)} M⊕</span><span class="stat-l">median M<sub>J</sub> (10–90%: ${fmt(s.mj10, 2)}–${fmt(s.mj90, 2)})</span></div>
    <div class="stat"><span class="stat-v">${fmt(s.tgMed, 2)} yr</span><span class="stat-l">median t<sub>growth</sub> (10–90%: ${fmt(s.tg10, 2)}–${fmt(s.tg90, 2)})</span></div>`;
}

function pinCurrent() {
  if (!G) return;
  if (state.pins.length >= PIN_COLORS.length) state.pins.shift();
  const sx = Math.log10(GRID.rMax / GRID.rMin) / G.nx, sy = Math.log10(GRID.mMax / GRID.mMin) / G.ny;
  const seg = G.ddgiSeg.map(([a, b, c, d]) => [Math.log10(GRID.rMin) + a * sx, Math.log10(GRID.mMin) + b * sy,
    Math.log10(GRID.rMin) + c * sx, Math.log10(GRID.mMin) + d * sy]);
  const p = state.p;
  const label = `v_f=${fmt(p.vfrag, 2)}, ε_max=${fmt(p.epsMax, 2)}, σ=${fmt(p.sigEps, 2)}, ε_bkg=${fmt(p.epsBkg, 2)}, M★=${fmt(p.mstar, 2)}`;
  state.pins.push({ label, seg, hist: { ...G.hist } });
  drawPins();
  render();
}

function drawPins() {
  const box = document.getElementById('pins');
  box.replaceChildren();
  state.pins.forEach((p, n) => {
    const row = document.createElement('div');
    row.className = 'pin-row';
    row.innerHTML = `<span class="swatch pin-${n}"></span><span class="pin-label">${p.label}</span>`;
    const x = document.createElement('button');
    x.textContent = '×'; x.title = 'Remove'; x.type = 'button';
    x.addEventListener('click', () => { state.pins.splice(n, 1); drawPins(); render(); });
    row.appendChild(x);
    box.appendChild(row);
  });
}

// ------------------------------------------------------- hover / inspector
function showTip(html, e) {
  const tip = document.getElementById('tooltip');
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > innerWidth - 8) x = e.clientX - w - pad;
  if (y + h > innerHeight - 8) y = e.clientY - h - pad;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}

function onMapHover(which) {
  return (p, e) => {
    state.hover = p;
    updateCursor();
    drawInspector();
    const tip = document.getElementById('tooltip');
    if (!p) { tip.hidden = true; return; }
    const reg = REGION_INFO.find((r) => r.id === G.region[p.k]).name;
    let line = '';
    if (which !== 'reg') {
      const key = state[which === 'a' ? 'qa' : 'qb'];
      const v = G[key][p.k] * (QUANT[key].scale || 1);
      line = `<br>${QUANT[key].label} = ${fmt(v)}`;
    }
    showTip(`Ṁ = ${fmt(p.mdot, 2)} M⊙/yr, r = ${fmt(p.r, 3)} au<br>${reg}${line}`, e);
  };
}

function updateCursor() {
  for (const m of Object.values(maps)) m.showCursor(state.hover, state.pinned);
}

function drawInspector() {
  const p = state.hover || state.pinned;
  const box = document.getElementById('inspector');
  document.getElementById('insp-mode').textContent = state.hover ? 'hovered point' : state.pinned ? 'pinned point (click a map to move)' : 'hover a map, click to pin';
  if (!p || !T) { box.innerHTML = ''; return; }
  const q = pointModel(T, p.mdot, p.r, state.p);
  const reg = REGION_INFO.find((r) => r.id === q.region).name;
  const rows = [
    ['Ṁ', `${fmt(p.mdot, 3)} M⊙/yr`], ['r', `${fmt(p.r, 3)} au`], ['Regime', reg],
    ['Σ<sub>g</sub>', `${fmt(q.sigma)} g/cm²`], ['T', `${fmt(q.T)} K`], ['α<sub>GI</sub>', fmt(q.alpha)],
    ['β', fmt(q.beta)], ['H<sub>g</sub>/r', fmt(q.hr)], ['c<sub>s</sub>', `${fmt(q.cs / 100)} m/s`],
  ];
  if (q.stMax !== undefined) {
    rows.push(
      ['St<sub>max</sub>', fmt(q.stMax)], ['ε', fmt(q.eps)], ['c<sub>d</sub>/c<sub>s</sub>', fmt(q.cdcs)],
      ['velocity regime', q.coupled ? '(i) λ<sub>J</sub> > Δr<sub>St</sub>, eq. 9' : '(ii) λ<sub>J</sub> < Δr<sub>St</sub>, eq. 10'],
      ['c<sub>d</sub>/(c<sub>s</sub>ε)', `${fmt(q.ddgiRatio)} ${q.ddgi ? '< 1 → DDGI' : '> 1'}`],
      ['Λ<sub>J</sub>', fmt(q.lamFac)], ['M<sub>J</sub>', `${fmt(q.mJ)} M⊕`],
      ['St<sub>0</sub>', fmt(q.st0)], ['t<sub>growth</sub>', `${fmt(q.tGrowth)} yr`], ['t<sub>DDGI</sub>', `${fmt(q.tDDGI)} yr`],
    );
  }
  box.innerHTML = rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join('');
}

// ε(St) preview in the sidebar
function drawEpsPreview() {
  const box = document.getElementById('eps-preview');
  box.replaceChildren();
  const W = box.clientWidth || 260, H = 120, m = { l: 34, r: 8, t: 8, b: 24 };
  const svg = el('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, box);
  const lx = [-5, 2], ly = [-3, Math.max(0.5, Math.log10(state.p.epsMax) + 0.2)];
  const xp = (v) => m.l + ((v - lx[0]) / (lx[1] - lx[0])) * (W - m.l - m.r);
  const yp = (v) => H - m.b - ((v - ly[0]) / (ly[1] - ly[0])) * (H - m.t - m.b);
  const ax = el('g', { class: 'axis' }, svg);
  for (let e = lx[0]; e <= lx[1]; e++) {
    el('line', { x1: xp(e), x2: xp(e), y1: H - m.b, y2: H - m.b + 3 }, ax);
    if (e % 2 === 1 || e === lx[1]) { const t = el('text', { x: xp(e), y: H - m.b + 14, 'text-anchor': 'middle' }, ax); t.textContent = `${e}`; }
  }
  for (let e = ly[0]; e <= ly[1]; e++) {
    el('line', { x1: m.l, x2: W - m.r, y1: yp(e), y2: yp(e), class: 'grid' }, ax);
    const t = el('text', { x: m.l - 5, y: yp(e) + 3, 'text-anchor': 'end' }, ax); t.textContent = `${e}`;
  }
  const t1 = el('text', { x: W - m.r, y: H - 2, 'text-anchor': 'end', class: 'axis-title' }, svg); t1.textContent = 'log St';
  const t2 = el('text', { x: 2, y: m.t + 2, class: 'axis-title' }, svg); t2.textContent = 'log ε';
  let d = '';
  for (let i = 0; i <= 140; i++) {
    const x = lx[0] + ((lx[1] - lx[0]) * i) / 140;
    d += `${i ? 'L' : 'M'}${xp(x).toFixed(1)} ${yp(Math.log10(epsOfSt(10 ** x, state.p))).toFixed(1)}`;
  }
  el('path', { d, class: 'eps-curve' }, svg);
}

// ------------------------------------------------------------------ boot
async function main() {
  readHash();
  buildControls();
  maps = {
    reg: new MapPlot(document.getElementById('map-regimes'), {}),
    a: new MapPlot(document.getElementById('map-a'), {}),
    b: new MapPlot(document.getElementById('map-b'), {}),
  };
  for (const [k, m] of Object.entries(maps)) {
    m.opts.onHover = onMapHover(k);
    m.opts.onClick = (p) => { state.pinned = p; updateCursor(); drawInspector(); };
  }
  const legend = document.getElementById('legend-regimes');
  legend.innerHTML = REGION_INFO.map((r) => `<span class="lg"><span class="swatch" style="background:${r.color}"></span><span>${r.name}</span></span>`).join('')
    + '<span class="lg"><span class="dash"></span><span>c<sub>d</sub>/c<sub>s</sub> = ε</span></span>';
  drawEpsPreview();
  try {
    T = await LambdaTables.load('data');
  } catch (err) {
    document.getElementById('status').textContent = `Could not load Λ tables: ${err}`;
    return;
  }
  schedule(false);
  let rt;
  addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { render(); drawEpsPreview(); }, 150); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
}

main();
