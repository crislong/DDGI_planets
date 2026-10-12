// Minimal plotting helpers: log-log raster maps with SVG axes, colorbars,
// marching-squares contours and a histogram. No dependencies.

const SVGNS = 'http://www.w3.org/2000/svg';

export function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

// ---------------------------------------------------------------- colormaps
const ANCHORS = {
  RdBu_r: ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
           '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'],
  viridis: ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89',
            '#35b779', '#6ece58', '#b5de2b', '#fde725'],
  magma: ['#000004', '#180f3d', '#440f76', '#721f81', '#9e2f7f', '#cd4071',
          '#f1605d', '#fd9668', '#feca8d', '#fcfdbf'],
  PuOr: ['#2d004b', '#542788', '#8073ac', '#b2abd2', '#d8daeb', '#f7f7f7',
         '#fee0b6', '#fdb863', '#e08214', '#b35806', '#7f3b08'],
};

const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

export function makeLUT(name, n = 256) {
  const a = ANCHORS[name].map(hex);
  const lut = new Uint8ClampedArray(n * 3);
  for (let i = 0; i < n; i++) {
    const u = (i / (n - 1)) * (a.length - 1);
    const k = Math.min(Math.floor(u), a.length - 2), w = u - k;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = a[k][c] + w * (a[k + 1][c] - a[k][c]);
  }
  return lut;
}

export function cssGradient(name) {
  const a = ANCHORS[name];
  return a.map((c, i) => `${c} ${((i / (a.length - 1)) * 100).toFixed(1)}%`).join(', ');
}

// -------------------------------------------------------------- formatting
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
export const pow10 = (e) => (e === 0 ? '1' : e === 1 ? '10' : `10${String(e).split('').map((c) => SUP[c]).join('')}`);

export function fmt(v, digits = 3) {
  if (v === undefined || v === null || Number.isNaN(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e-2 && a < 1e4) return Number(v.toPrecision(digits)).toString();
  const e = Math.floor(Math.log10(a));
  const m = v / 10 ** e;
  return `${Number(m.toPrecision(digits))}×${pow10(e)}`;
}

// ------------------------------------------------------------------ map plot
export class MapPlot {
  constructor(container, opts = {}) {
    this.c = container;
    this.opts = opts;
    this.margin = { l: 58, r: 12, t: 10, b: 42 };
    this.wrap = document.createElement('div');
    this.wrap.className = 'map-wrap';
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'map-canvas';
    this.svg = el('svg', { class: 'map-svg' });
    this.wrap.append(this.canvas, this.svg);
    container.appendChild(this.wrap);
    this.off = document.createElement('canvas');
    this.svg.addEventListener('pointermove', (e) => this._hover(e));
    this.svg.addEventListener('pointerleave', () => opts.onHover && opts.onHover(null));
    this.svg.addEventListener('click', (e) => {
      const p = this._toData(e);
      if (p && opts.onClick) opts.onClick(p);
    });
  }

  _geom() {
    const W = this.wrap.clientWidth;
    const H = Math.round(W * 0.82);
    const m = this.margin;
    return { W, H, x0: m.l, x1: W - m.r, y0: m.t, y1: H - m.b };
  }

  _toData(e) {
    if (!this.grid) return null;
    const r = this.svg.getBoundingClientRect();
    const g = this.g;
    const px = e.clientX - r.left, py = e.clientY - r.top;
    if (px < g.x0 || px > g.x1 || py < g.y0 || py > g.y1) return null;
    const G = this.grid;
    const lr = Math.log10(G.rMin) + ((px - g.x0) / (g.x1 - g.x0)) * Math.log10(G.rMax / G.rMin);
    const lm = Math.log10(G.mMax) - ((py - g.y0) / (g.y1 - g.y0)) * Math.log10(G.mMax / G.mMin);
    const i = Math.min(G.nx - 1, Math.floor(((lr - Math.log10(G.rMin)) / Math.log10(G.rMax / G.rMin)) * G.nx));
    const j = Math.min(G.ny - 1, Math.floor(((lm - Math.log10(G.mMin)) / Math.log10(G.mMax / G.mMin)) * G.ny));
    return { r: 10 ** lr, mdot: 10 ** lm, i, j, k: j * G.nx + i };
  }

  _hover(e) {
    if (this.opts.onHover) this.opts.onHover(this._toData(e), e);
  }

  xPix(r) {
    const G = this.grid, g = this.g;
    return g.x0 + (Math.log10(r / G.rMin) / Math.log10(G.rMax / G.rMin)) * (g.x1 - g.x0);
  }

  yPix(m) {
    const G = this.grid, g = this.g;
    return g.y1 - (Math.log10(m / G.mMin) / Math.log10(G.mMax / G.mMin)) * (g.y1 - g.y0);
  }

  // rgba: Uint8ClampedArray(nx*ny*4), row 0 = lowest Mdot
  draw(grid, rgba, overlays = {}) {
    this.grid = grid;
    const g = (this.g = this._geom());
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(g.W * dpr);
    this.canvas.height = Math.round(g.H * dpr);
    this.canvas.style.width = `${g.W}px`;
    this.canvas.style.height = `${g.H}px`;
    this.svg.setAttribute('width', g.W);
    this.svg.setAttribute('height', g.H);
    this.svg.setAttribute('viewBox', `0 0 ${g.W} ${g.H}`);

    const { nx, ny } = grid;
    this.off.width = nx;
    this.off.height = ny;
    const octx = this.off.getContext('2d');
    const img = octx.createImageData(nx, ny);
    for (let j = 0; j < ny; j++) {
      const src = j * nx * 4, dst = (ny - 1 - j) * nx * 4; // flip: top row = high Mdot
      img.data.set(rgba.subarray(src, src + nx * 4), dst);
    }
    octx.putImageData(img, 0, 0);
    const ctx = this.canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, g.W, g.H);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.off, g.x0, g.y0, g.x1 - g.x0, g.y1 - g.y0);

    this.svg.replaceChildren();
    this._axes();
    for (const c of overlays.contours || []) this._contour(c);
    for (const lab of overlays.labels || []) {
      const t = el('text', { x: this.xPix(lab.r), y: this.yPix(lab.mdot), class: 'region-label', 'text-anchor': 'middle' }, this.svg);
      t.textContent = lab.text;
    }
    this.cross = el('g', { class: 'crosshair' }, this.svg);
    this.marker = el('g', { class: 'pin-marker' }, this.svg);
  }

  _axes() {
    const g = this.g, G = this.grid, s = this.svg;
    el('rect', { x: g.x0, y: g.y0, width: g.x1 - g.x0, height: g.y1 - g.y0, class: 'frame' }, s);
    const ax = el('g', { class: 'axis' }, s);
    for (let e = Math.floor(Math.log10(G.rMin)); e <= Math.ceil(Math.log10(G.rMax)); e++) {
      for (let k = 1; k < 10; k++) {
        const v = k * 10 ** e;
        if (v < G.rMin * 0.999 || v > G.rMax * 1.001) continue;
        const x = this.xPix(v);
        el('line', { x1: x, x2: x, y1: g.y1, y2: g.y1 + (k === 1 ? 6 : 3) }, ax);
        if (k === 1) {
          const t = el('text', { x, y: g.y1 + 19, 'text-anchor': 'middle' }, ax);
          t.textContent = pow10(e);
        }
      }
    }
    for (let e = Math.floor(Math.log10(G.mMin)); e <= Math.ceil(Math.log10(G.mMax)); e++) {
      for (let k = 1; k < 10; k++) {
        const v = k * 10 ** e;
        if (v < G.mMin * 0.999 || v > G.mMax * 1.001) continue;
        const y = this.yPix(v);
        el('line', { x1: g.x0 - (k === 1 ? 6 : 3), x2: g.x0, y1: y, y2: y }, ax);
        if (k === 1) {
          const t = el('text', { x: g.x0 - 9, y: y + 4, 'text-anchor': 'end' }, ax);
          t.textContent = pow10(e);
        }
      }
    }
    const xl = el('text', { x: (g.x0 + g.x1) / 2, y: g.y1 + 36, 'text-anchor': 'middle', class: 'axis-title' }, s);
    xl.textContent = 'r [au]';
    const yl = el('text', { x: 0, y: 0, 'text-anchor': 'middle', class: 'axis-title',
      transform: `translate(14 ${(g.y0 + g.y1) / 2}) rotate(-90)` }, s);
    yl.textContent = 'Ṁ [M⊙/yr]';
  }

  // segments in grid index coordinates (cell centres at i+0.5)
  _contour(c) {
    const G = this.grid, g = this.g;
    const sx = (g.x1 - g.x0) / G.nx, sy = (g.y1 - g.y0) / G.ny;
    let d = '';
    for (const line of chainSegments(c.segments)) {
      line.forEach(([a, b], n) => { d += `${n ? 'L' : 'M'}${(g.x0 + a * sx).toFixed(1)} ${(g.y1 - b * sy).toFixed(1)}`; });
    }
    el('path', { d, class: `contour ${c.cls || ''}` }, this.svg);
  }

  showCursor(p, pinned) {
    if (!this.cross) return;
    this.cross.replaceChildren();
    this.marker.replaceChildren();
    const g = this.g;
    if (p) {
      const x = this.xPix(p.r), y = this.yPix(p.mdot);
      el('line', { x1: x, x2: x, y1: g.y0, y2: g.y1 }, this.cross);
      el('line', { x1: g.x0, x2: g.x1, y1: y, y2: y }, this.cross);
    }
    if (pinned) {
      el('circle', { cx: this.xPix(pinned.r), cy: this.yPix(pinned.mdot), r: 5 }, this.marker);
    }
  }
}

// --------------------------------------------------------------- colorbar
export function drawColorbar(container, { cmap, vmin, vmax, label, log, diverging }) {
  container.replaceChildren();
  const bar = document.createElement('div');
  bar.className = 'cbar-bar';
  bar.style.background = `linear-gradient(to right, ${cssGradient(cmap)})`;
  const ticks = document.createElement('div');
  ticks.className = 'cbar-ticks';
  const nT = 5;
  for (let i = 0; i < nT; i++) {
    const v = vmin + ((vmax - vmin) * i) / (nT - 1);
    const s = document.createElement('span');
    s.style.left = `${(i / (nT - 1)) * 100}%`;
    s.textContent = log ? fmt(10 ** v, 2) : fmt(v, 2);
    ticks.appendChild(s);
  }
  const lab = document.createElement('div');
  lab.className = 'cbar-label';
  lab.innerHTML = label + (diverging ? ' <span class="muted">(centre = 1)</span>' : '');
  container.append(lab, bar, ticks);
}

// ------------------------------------------------------- marching squares
// field: Float32Array (nx*ny), contour at `level`; NaN cells are skipped.
export function contourSegments(field, nx, ny, level) {
  const seg = [];
  const v = (i, j) => field[j * nx + i] - level;
  const lerp = (a, b) => a / (a - b);
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = v(i, j), b = v(i + 1, j), c = v(i + 1, j + 1), d = v(i, j + 1);
      if (Number.isNaN(a) || Number.isNaN(b) || Number.isNaN(c) || Number.isNaN(d)) continue;
      const pts = [];
      const x = i + 0.5, y = j + 0.5;
      if ((a < 0) !== (b < 0)) pts.push([x + lerp(a, b), y]);
      if ((b < 0) !== (c < 0)) pts.push([x + 1, y + lerp(b, c)]);
      if ((c < 0) !== (d < 0)) pts.push([x + 1 - lerp(c, d), y + 1]);
      if ((d < 0) !== (a < 0)) pts.push([x, y + 1 - lerp(d, a)]);
      if (pts.length === 2) seg.push([...pts[0], ...pts[1]]);
      else if (pts.length === 4) seg.push([...pts[0], ...pts[1]], [...pts[2], ...pts[3]]);
    }
  }
  return seg;
}

// join segments sharing endpoints into polylines (so that dashes run on)
export function chainSegments(segs) {
  const key = (x, y) => `${Math.round(x * 1e4)},${Math.round(y * 1e4)}`;
  const ends = new Map();
  const used = new Uint8Array(segs.length);
  segs.forEach(([a, b, c, d], n) => {
    for (const k of [key(a, b), key(c, d)]) {
      if (!ends.has(k)) ends.set(k, []);
      ends.get(k).push(n);
    }
  });
  const next = (pt) => {
    for (const n of ends.get(key(...pt)) || []) {
      if (used[n]) continue;
      used[n] = 1;
      const [a, b, c, d] = segs[n];
      return key(a, b) === key(...pt) ? [c, d] : [a, b];
    }
    return null;
  };
  const lines = [];
  for (let n = 0; n < segs.length; n++) {
    if (used[n]) continue;
    used[n] = 1;
    const [a, b, c, d] = segs[n];
    const fwd = [[a, b], [c, d]];
    for (let p = next([c, d]); p; p = next(p)) fwd.push(p);
    const back = [];
    for (let p = next([a, b]); p; p = next(p)) back.push(p);
    lines.push([...back.reverse(), ...fwd]);
  }
  return lines;
}

// boundaries between cells of different category (staircase along cell edges)
export function categoryEdges(cat, nx, ny, filter = () => true) {
  const seg = [];
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      if (i < nx - 1 && cat[k] !== cat[k + 1] && filter(cat[k], cat[k + 1])) seg.push([i + 1, j, i + 1, j + 1]);
      if (j < ny - 1 && cat[k] !== cat[k + nx] && filter(cat[k], cat[k + nx])) seg.push([i, j + 1, i + 1, j + 1]);
    }
  }
  return seg;
}

// -------------------------------------------------------------- histogram
export function drawHistogram(container, { bins, current, pinned, xLabel, onHover }) {
  container.replaceChildren();
  const W = container.clientWidth, H = Math.round(W * 0.62);
  const m = { l: 50, r: 12, t: 10, b: 42 };
  const svg = el('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: 'hist-svg' }, container);
  const x0 = m.l, x1 = W - m.r, y0 = m.t, y1 = H - m.b;
  const lx0 = Math.log10(bins[0]), lx1 = Math.log10(bins[bins.length - 1]);
  const xp = (v) => x0 + ((Math.log10(v) - lx0) / (lx1 - lx0)) * (x1 - x0);
  const all = [current, ...pinned.map((p) => p.hist)].filter(Boolean);
  const ymax = Math.max(0.02, ...all.flatMap((h) => h.frac)) * 1.1;
  const yp = (v) => y1 - (v / ymax) * (y1 - y0);

  const ax = el('g', { class: 'axis' }, svg);
  const yStep = ymax > 0.3 ? 0.1 : ymax > 0.12 ? 0.05 : 0.02;
  for (let v = 0; v <= ymax + 1e-9; v += yStep) {
    el('line', { x1: x0, x2: x1, y1: yp(v), y2: yp(v), class: 'grid' }, ax);
    const t = el('text', { x: x0 - 7, y: yp(v) + 4, 'text-anchor': 'end' }, ax);
    t.textContent = v.toFixed(2);
  }
  const xt = W < 520 ? [0.1, 1, 10, 100, 1000] : [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
  for (const v of xt) {
    if (v < bins[0] || v > bins[bins.length - 1]) continue;
    el('line', { x1: xp(v), x2: xp(v), y1: y1, y2: y1 + 5 }, ax);
    const t = el('text', { x: xp(v), y: y1 + 19, 'text-anchor': 'middle' }, ax);
    t.textContent = v;
  }
  el('line', { x1: x0, x2: x1, y1, y2: y1, class: 'baseline' }, svg);
  const xl = el('text', { x: (x0 + x1) / 2, y: H - 6, 'text-anchor': 'middle', class: 'axis-title' }, svg);
  xl.textContent = xLabel;
  const yl = el('text', { x: 0, y: 0, 'text-anchor': 'middle', class: 'axis-title',
    transform: `translate(12 ${(y0 + y1) / 2}) rotate(-90)` }, svg);
  yl.textContent = 'N / N_tot';

  if (current) {
    const g = el('g', { class: 'hist-bars' }, svg);
    current.frac.forEach((f, i) => {
      if (f <= 0) return;
      const xa = xp(bins[i]) + 1, xb = xp(bins[i + 1]) - 1;
      const h = y1 - yp(f);
      const r = Math.min(3, (xb - xa) / 2, h);
      el('path', { d: `M${xa} ${y1}V${yp(f) + r}Q${xa} ${yp(f)} ${xa + r} ${yp(f)}H${xb - r}Q${xb} ${yp(f)} ${xb} ${yp(f) + r}V${y1}Z` }, g);
    });
  }
  pinned.forEach((p, n) => {
    let d = '';
    p.hist.frac.forEach((f, i) => {
      const xa = xp(bins[i]), xb = xp(bins[i + 1]);
      d += `${i === 0 ? 'M' : 'L'}${xa} ${yp(f)}L${xb} ${yp(f)}`;
    });
    el('path', { d, class: `hist-pin pin-${n}` }, svg);
  });

  // hover: one hit-target per bin
  const hit = el('g', { class: 'hit' }, svg);
  for (let i = 0; i < bins.length - 1; i++) {
    const r = el('rect', { x: xp(bins[i]), y: y0, width: xp(bins[i + 1]) - xp(bins[i]), height: y1 - y0 }, hit);
    r.addEventListener('pointerenter', (e) => onHover && onHover(i, e));
    r.addEventListener('pointerleave', () => onHover && onHover(null));
  }
}
