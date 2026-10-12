// DDGI model of Longarini & Clarke (2026): self-regulated gas disc
// (Clarke 2009) + fragmentation-limited dust. Port of the reference
// Python code in python/ddgi_reference.py.

import { lambda2, lambda3At } from './lambda.js';

export const CGS = {
  kb: 1.380649e-16,
  mp: 1.6726e-24,
  G: 6.674e-8,
  Msun: 1.989e33,
  Mearth: 5.972e27,
  au: 1.496e13,
  yr: 3.156e7,
};

// fixed thresholds of the paper
export const ALPHA_FRAG = 0.06;
export const ALPHA_NOGI = 1e-4;
export const ALPHA_MRI = 0.01;
export const T_MRI = 1000;
export const R_DEAD = 100; // au

export const REGION = { SELF: 0, DDGI: 1, FRAG: 2, MRI: 3, NOGI: 4 };

// Table A1 (M* = 1 Msun) with the M*^(1/3) radial rescaling.
export function gasDisc(mdot, r, mstar) {
  const m = mdot / 1e-6;
  const s13 = Math.cbrt(mstar);
  const x = r / s13 / 100; // R_100 of the equivalent solar-mass disc
  const mc = 1 / s13;
  const b0 = 5.5 * m ** 0.24 * s13;
  const b1 = 17 * m ** (2 / 9) * s13;
  const b2 = 27 * m ** (2 / 9) * s13;
  const b3 = 70 * m ** (2 / 9) * s13;
  let sigma, T, alpha, hr, zone;
  if (r <= b0) {
    zone = 0;
    sigma = 170 * m ** 0.018 * x ** -1.64;
    T = 500 * m ** 0.036 * x ** -0.27;
    alpha = 2.1e-4 * m ** 0.945 * x ** 0.41;
    hr = 0.68 * m ** 0.018 * x ** 0.36 * mc;
  } else if (r <= b1) {
    zone = 1;
    sigma = 29 * m ** (1 / 6) * x ** (-9 / 4);
    T = 14 * m ** (1 / 3) * x ** -1.5;
    alpha = 4.4e-2 * m ** 0.5 * x ** (9 / 4);
    hr = 0.11 * m ** (1 / 6) * x ** -0.25 * mc;
  } else if (r <= b2) {
    zone = 2;
    sigma = 75 * m ** (1 / 21) * x ** (-12 / 7);
    T = 95 * m ** (2 / 21) * x ** (-3 / 7);
    alpha = 2.5e-3 * m ** (6 / 7) * x ** (9 / 14);
    hr = 0.3 * m ** (1 / 21) * x ** (2 / 7) * mc;
  } else if (r <= b3) {
    zone = 3;
    sigma = 14 * m ** (1 / 3) * x ** -3;
    T = 3.2 * m ** (2 / 3) * x ** -3;
    alpha = 0.4 * x ** 4.5;
    hr = 0.055 * m ** (1 / 3) * x ** -1 * mc;
  } else {
    zone = 4;
    sigma = 25 * x ** -1.5;
    T = 10;
    alpha = 0.06 * m;
    hr = 0.1 * x ** 0.5 * mc;
  }
  const cs = Math.sqrt((CGS.kb * T) / (2.3 * CGS.mp));
  const beta = 4 / (9 * alpha);
  const omega = Math.sqrt((CGS.G * mstar * CGS.Msun) / (r * CGS.au) ** 3);
  return { zone, sigma, T, alpha, hr, cs, beta, omega };
}

export function region(g, r) {
  if (g.alpha < ALPHA_MRI && (g.T > T_MRI || r > R_DEAD)) return REGION.MRI;
  if (g.alpha > ALPHA_FRAG) return REGION.FRAG;
  if (g.alpha < ALPHA_NOGI) return REGION.NOGI;
  return REGION.SELF;
}

// eq. (7)
export function epsOfSt(st, p) {
  const x = Math.log10(st);
  return p.epsBkg + (p.epsMax - p.epsBkg) * Math.exp(-(x * x) / (2 * p.sigEps * p.sigEps));
}

// Implicit equation (C1): c_d/c_s = [Lambda((c_d/c_s)^2, eps, St) / beta]^(1/3).
// It can have three roots where St ~ 0.2-1: the middle one is unstable,
// and of the two stable ones we take the largest, i.e. the branch reached
// by grains growing from small St (the only root at small St). The residual
// is scanned downwards from 10 to 1e-4 (0.025 dex steps) to bracket the
// largest root, then refined by geometric bisection.
// Returns NaN if no root is found.
const SCAN_HI = 10, SCAN_LO = 1e-4, SCAN_STEP = 10 ** -0.025;
export function solveCdcs(T, eps, st, beta, p) {
  const sq = p.bConv === 'cdcs2';
  const lam = lambda3At(T, eps, st);
  const res = (x) => x - Math.cbrt(lam(sq ? x * x : x) / beta);
  let hi = SCAN_HI, fhi = res(hi), lo = NaN, flo = NaN;
  for (let x = hi * SCAN_STEP; x >= SCAN_LO * 0.999; x *= SCAN_STEP) {
    const f = res(x);
    if (Math.sign(f) !== Math.sign(fhi)) { lo = x; flo = f; break; }
    hi = x; fhi = f;
  }
  if (Number.isNaN(lo)) { // nothing in the scanned range: try below it
    lo = 1e-8; flo = res(lo); hi = SCAN_LO; fhi = res(hi);
    if (!(Math.sign(flo) !== Math.sign(fhi))) return NaN;
  }
  // stop once the bracket is narrower than 1e-7 in ratio (well below the
  // table accuracy), or after nIter iterations
  for (let i = 0; i < p.nIter && hi > lo * (1 + 1e-7); i++) {
    const mid = Math.sqrt(lo * hi);
    const fm = res(mid);
    if (Math.sign(fm) === Math.sign(flo)) { lo = mid; flo = fm; } else hi = mid;
  }
  return Math.sqrt(lo * hi);
}

// eq. (14): t_growth = (1/Omega) int_{St0}^{Stmax} dlnSt / eps(St)
// (Simpson's rule in log10 St)
export function growthTime(st0, stMax, omega, p, n = 32) {
  if (!(stMax > st0)) return 0;
  const a = Math.log10(st0), b = Math.log10(stMax), h = (b - a) / n;
  const k = 1 / (2 * p.sigEps * p.sigEps);
  let s = 0;
  for (let i = 0; i <= n; i++) {
    const x = a + i * h;
    const w = i === 0 || i === n ? 1 : i % 2 ? 4 : 2;
    s += w / (p.epsBkg + (p.epsMax - p.epsBkg) * Math.exp(-x * x * k));
  }
  return ((s * h) / 3) * Math.LN10 / omega;
}

// All quantities at one point of the (Mdot, r) plane.
export function pointModel(T, mdot, r, p) {
  const g = gasDisc(mdot, r, p.mstar);
  const out = { ...g, region: region(g, r) };
  if (out.region !== REGION.SELF) return out;

  const stMax = g.beta * ((p.vfrag * 100) / g.cs) ** 2; // eq. (6)
  const eps = epsOfSt(stMax, p);
  const xImpl = solveCdcs(T, eps, stMax, g.beta, p); // regime (i), eq. (9)
  const xColl = Math.sqrt(stMax / g.beta); // regime (ii), eq. (10)
  // lambda_J > Delta r_St  <=>  xImpl > xColl  (equivalent to the St_star test)
  const coupled = xImpl > xColl;
  const cdcs = Number.isNaN(xImpl) ? NaN : coupled ? xImpl : xColl;

  // Jeans mass as in dust_quantities_stokes2
  const lamJ = lambda2(T, cdcs * cdcs, eps);
  const lamFac = lamJ / (1 + eps);
  const lj = (g.cs * g.cs) / (Math.PI * CGS.G * g.sigma);
  const mJ = (lj * lj * lamFac * lamFac * g.sigma) / CGS.Mearth;

  const st0 = (Math.PI * p.rhoS * p.a0 * 1e-4) / (2 * g.sigma);
  const tGrowth = growthTime(st0, stMax, g.omega, p) / CGS.yr;
  const tDDGI = Math.sqrt(lamFac / (3 * (1 + eps))) / g.omega / CGS.yr; // eq. (16)

  const ddgi = cdcs < eps; // eq. (11)
  if (ddgi) out.region = REGION.DDGI;
  return {
    ...out, stMax, eps, xImpl, xColl, cdcs, coupled, ddgiRatio: cdcs / eps,
    lamFac, mJ, st0, tGrowth, tDDGI, ddgi,
  };
}

export const FIELDS = [
  'sigma', 'T', 'alpha', 'hr', 'beta', 'cs',
  'stMax', 'eps', 'cdcs', 'ddgiRatio', 'lamFac', 'mJ', 'tGrowth', 'tDDGI',
];

// Evaluate the model on a log grid. Returns typed arrays (row-major, row = Mdot).
export function computeGrid(T, grid, p) {
  const { nx, ny, rMin, rMax, mMin, mMax } = grid;
  const N = nx * ny;
  const res = { nx, ny, rMin, rMax, mMin, mMax, region: new Uint8Array(N), coupled: new Uint8Array(N) };
  for (const f of FIELDS) res[f] = new Float32Array(N).fill(NaN);
  const lr0 = Math.log10(rMin), dlr = Math.log10(rMax / rMin) / nx;
  const lm0 = Math.log10(mMin), dlm = Math.log10(mMax / mMin) / ny;
  for (let j = 0; j < ny; j++) {
    const mdot = 10 ** (lm0 + (j + 0.5) * dlm);
    for (let i = 0; i < nx; i++) {
      const r = 10 ** (lr0 + (i + 0.5) * dlr);
      const k = j * nx + i;
      const q = pointModel(T, mdot, r, p);
      res.region[k] = q.region;
      res.coupled[k] = q.coupled ? 1 : 0;
      for (const f of FIELDS) if (q[f] !== undefined) res[f][k] = q[f];
    }
  }
  return res;
}

export const DEFAULTS = {
  vfrag: 10, // m/s
  epsBkg: 1e-2,
  epsMax: 1,
  sigEps: 0.5, // dex (value in dust_quantities_stokes2; reproduces Figs. 3-4)
  mstar: 1, // Msun
  a0: 1, // micron
  rhoS: 1, // g/cm^3
  bConv: 'cdcs2', // argument of Lambda in eq. (C1): (c_d/c_s)^2 (temperature ratio), or c_d/c_s (original code)
  nIter: 50,
};
