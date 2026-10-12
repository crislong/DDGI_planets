// Precomputed Lambda tables (see python/make_lambda_tables.py) and
// multilinear interpolation in log space.

export class LambdaTables {
  constructor(meta, buf3d, buf2d) {
    const enc = meta.encoding;
    this.decode = (q) =>
      q === 0 ? NaN : enc.log10_lo + ((q - 1) / 65534) * (enc.log10_hi - enc.log10_lo);
    // store log10(Lambda) as Float32 for fast lookups
    this.t3 = Float32Array.from(new Uint16Array(buf3d), this.decode);
    this.t2 = Float32Array.from(new Uint16Array(buf2d), this.decode);
    this.ax3 = { b: meta.lambda3d.b, eps: meta.lambda3d.eps, St: meta.lambda3d.St };
    this.ax2 = { b: meta.lambda2d.b, eps: meta.lambda2d.eps };
  }

  static async load(base) {
    const meta = await (await fetch(`${base}/lambda_tables.json`)).json();
    const [b3, b2] = await Promise.all([
      fetch(`${base}/${meta.lambda3d.file}`).then((r) => r.arrayBuffer()),
      fetch(`${base}/${meta.lambda2d.file}`).then((r) => r.arrayBuffer()),
    ]);
    return new LambdaTables(meta, b3, b2);
  }
}

// fractional index along a uniform log10 axis, clamped to the table
function locate(ax, logx) {
  let u = ((logx - ax.lo) / (ax.hi - ax.lo)) * (ax.n - 1);
  if (!(u > 0)) u = 0; // also catches NaN
  if (u > ax.n - 1) u = ax.n - 1;
  let i = Math.floor(u);
  if (i > ax.n - 2) i = ax.n - 2;
  return [i, u - i];
}

// Lambda jumps where the two peaks of the marginal stability curve swap.
// If the corners of a cell straddle such a jump, use the nearest corner
// instead of blending the two branches.
const JUMP_DEX = 0.3;
const V8 = new Float64Array(8);

// Lambda(b, eps, St) with drag + back-reaction (marg_stab Q=2)
export function lambda3(T, b, eps, st) {
  const ax = T.ax3;
  const [ib, wb] = locate(ax.b, Math.log10(b));
  const [ie, we] = locate(ax.eps, Math.log10(eps));
  const [is, ws] = locate(ax.St, Math.log10(st));
  const nb = ax.b.n, ne = ax.eps.n, t = T.t3;
  const o00 = (is * ne + ie) * nb + ib;
  const o01 = (is * ne + ie + 1) * nb + ib;
  const o10 = ((is + 1) * ne + ie) * nb + ib;
  const o11 = ((is + 1) * ne + ie + 1) * nb + ib;
  const v = V8;
  v[0] = t[o00]; v[1] = t[o00 + 1]; v[2] = t[o01]; v[3] = t[o01 + 1];
  v[4] = t[o10]; v[5] = t[o10 + 1]; v[6] = t[o11]; v[7] = t[o11 + 1];
  let mn = v[0], mx = v[0];
  for (let k = 1; k < 8; k++) { if (v[k] < mn) mn = v[k]; else if (v[k] > mx) mx = v[k]; }
  if (mx - mn > JUMP_DEX) {
    return Math.pow(10, v[(wb > 0.5 ? 1 : 0) + (we > 0.5 ? 2 : 0) + (ws > 0.5 ? 4 : 0)]);
  }
  const c00 = v[0] + wb * (v[1] - v[0]);
  const c01 = v[2] + wb * (v[3] - v[2]);
  const c10 = v[4] + wb * (v[5] - v[4]);
  const c11 = v[6] + wb * (v[7] - v[6]);
  const c0 = c00 + we * (c01 - c00);
  const c1 = c10 + we * (c11 - c10);
  return Math.pow(10, c0 + ws * (c1 - c0));
}

// Lambda(b) at fixed (eps, St): the eps/St weights are computed once, for
// the bisection on c_d/c_s
export function lambda3At(T, eps, st) {
  const ax = T.ax3;
  const [ie, we] = locate(ax.eps, Math.log10(eps));
  const [is, ws] = locate(ax.St, Math.log10(st));
  const nb = ax.b.n, ne = ax.eps.n, t = T.t3;
  const r00 = (is * ne + ie) * nb, r01 = (is * ne + ie + 1) * nb;
  const r10 = ((is + 1) * ne + ie) * nb, r11 = ((is + 1) * ne + ie + 1) * nb;
  const kSel = (we > 0.5 ? 2 : 0) + (ws > 0.5 ? 4 : 0);
  const v = new Float64Array(8);
  return (b) => {
    const [ib, wb] = locate(ax.b, Math.log10(b));
    v[0] = t[r00 + ib]; v[1] = t[r00 + ib + 1]; v[2] = t[r01 + ib]; v[3] = t[r01 + ib + 1];
    v[4] = t[r10 + ib]; v[5] = t[r10 + ib + 1]; v[6] = t[r11 + ib]; v[7] = t[r11 + ib + 1];
    let mn = v[0], mx = v[0];
    for (let k = 1; k < 8; k++) { if (v[k] < mn) mn = v[k]; else if (v[k] > mx) mx = v[k]; }
    if (mx - mn > JUMP_DEX) return Math.pow(10, v[(wb > 0.5 ? 1 : 0) + kSel]);
    const c00 = v[0] + wb * (v[1] - v[0]);
    const c01 = v[2] + wb * (v[3] - v[2]);
    const c10 = v[4] + wb * (v[5] - v[4]);
    const c11 = v[6] + wb * (v[7] - v[6]);
    const c0 = c00 + we * (c01 - c00);
    const c1 = c10 + we * (c11 - c10);
    return Math.pow(10, c0 + ws * (c1 - c0));
  };
}

// Lambda(b, eps) for St -> infinity (used for the Jeans mass)
export function lambda2(T, b, eps) {
  const ax = T.ax2;
  const [ib, wb] = locate(ax.b, Math.log10(b));
  const [ie, we] = locate(ax.eps, Math.log10(eps));
  const nb = ax.b.n, t = T.t2;
  const o0 = ie * nb + ib, o1 = (ie + 1) * nb + ib;
  const v0 = t[o0], v1 = t[o0 + 1], v2 = t[o1], v3 = t[o1 + 1];
  if (Math.max(v0, v1, v2, v3) - Math.min(v0, v1, v2, v3) > JUMP_DEX) {
    return Math.pow(10, [v0, v1, v2, v3][(wb > 0.5 ? 1 : 0) + (we > 0.5 ? 2 : 0)]);
  }
  const c0 = v0 + wb * (v1 - v0);
  const c1 = v2 + wb * (v3 - v2);
  return Math.pow(10, c0 + we * (c1 - c0));
}
