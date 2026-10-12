"""
Precompute the Lambda tables used by the website.

Lambda depends only on dimensionless quantities, so it is tabulated once
and interpolated (trilinearly in log space) in the browser:

  lambda3d : Lambda(b, eps, St) from the two-fluid marginal stability curve
             with drag and back-reaction (marg_stab with Q=2). This is the
             Lambda entering the implicit equation for c_d/c_s (eq. C1).
             `b` is the argument called `beta` in the dispersion relation.
  lambda2d : Lambda(b, eps) for St -> infinity (Q=1 form with St=1e5), used
             for the Jeans mass in dust_quantities_stokes2.

The location of the maximum of the marginal stability curve is found on a
dense log-spaced grid in lambda and refined with a parabolic fit, so the tables are
smoother than a plain argmax.

Output (in ../web/data):
  lambda3d.bin, lambda2d.bin : uint16 little-endian, log10(Lambda) mapped
                               linearly onto [1, 65535]; 0 means NaN.
  lambda_tables.json         : axes and encoding.

Usage: python make_lambda_tables.py [--check]
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from ddgi_reference import A, B_st_BR, C_st_BR, get_Lambda_vec

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'web', 'data')

# log10 axes (uniform)
AX_B = dict(lo=-5.0, hi=3.0, n=321)       # b
AX_EPS = dict(lo=-3.0, hi=0.625, n=59)    # eps (up to ~4.2)
AX_ST = dict(lo=-4.0, hi=3.0, n=113)      # St
AX_B2 = dict(lo=-8.0, hi=4.0, n=961)      # b for the Jeans-mass table
AX_EPS2 = dict(lo=-3.0, hi=0.625, n=363)  # eps for the Jeans-mass table

LOG_LAM_LO, LOG_LAM_HI = -7.0, 1.5
N_LAMB = 3000
LAMB_MIN = 1e-8


def axis(ax):
    return np.linspace(ax['lo'], ax['hi'], ax['n'])


def lambda_peak(b, eps, st, n_lamb=N_LAMB):
    """Lambda = 2 * argmax of the marginal stability curve (refined)."""
    b = b[:, None]; eps = eps[:, None]; st = st[:, None]
    # log-spaced in lambda, to resolve the short-wavelength (dust) peak
    lmax = 1.05 * (1 + eps)
    t = np.linspace(0, 1, n_lamb)[None, :]
    lamb = LAMB_MIN * (lmax / LAMB_MIN)**t
    Bb = B_st_BR(lamb, eps, b, st)
    Cb = C_st_BR(lamb, eps, st)
    disc = Bb**2 - 4 * A(b) * Cb
    with np.errstate(invalid='ignore'):
        ms = (-Bb + np.sqrt(np.where(disc < 0, np.nan, disc))) / (2 * A(b))
    all_nan = np.all(np.isnan(ms), axis=1)
    ms = np.where(np.isnan(ms), -np.inf, ms)
    i = np.clip(np.argmax(ms, axis=1), 1, n_lamb - 2)
    rows = np.arange(len(i))
    y0, y1, y2 = ms[rows, i - 1], ms[rows, i], ms[rows, i + 1]
    den = y0 - 2 * y1 + y2
    with np.errstate(invalid='ignore', divide='ignore'):
        shift = np.where(np.isfinite(den) & (den < 0), 0.5 * (y0 - y2) / den, 0.0)
    shift = np.clip(np.nan_to_num(shift), -1, 1)
    lam = 2 * LAMB_MIN * (lmax[:, 0] / LAMB_MIN)**((i + shift) / (n_lamb - 1))
    return np.where(all_nan, np.nan, lam)


def encode(lam):
    ll = np.log10(lam)
    q = np.round(1 + (ll - LOG_LAM_LO) / (LOG_LAM_HI - LOG_LAM_LO) * 65534)
    q = np.where(np.isfinite(ll), np.clip(q, 1, 65535), 0)
    return q.astype('<u2')


def decode(q):
    q = q.astype(float)
    ll = LOG_LAM_LO + (q - 1) / 65534 * (LOG_LAM_HI - LOG_LAM_LO)
    return np.where(q == 0, np.nan, 10**ll)


def _slab(k):
    """All (b, eps) for one St value."""
    lb, le = np.meshgrid(axis(AX_B), axis(AX_EPS), indexing='ij')
    st = np.full(lb.size, 10**axis(AX_ST)[k])
    out = np.empty(lb.size)
    chunk = 4000
    for s in range(0, lb.size, chunk):
        sl = slice(s, s + chunk)
        out[sl] = lambda_peak(10**lb.ravel()[sl], 10**le.ravel()[sl], st[sl])
    return out.reshape(lb.shape)


def build():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with Pool() as pool:
        slabs = []
        for k, s in enumerate(pool.imap(_slab, range(AX_ST['n']))):
            slabs.append(s)
            print(f'\rSt slab {k + 1}/{AX_ST["n"]}  {time.time() - t0:6.1f}s', end='', flush=True)
    print()
    # memory layout: index = (iSt * nEps + iEps) * nB + iB  (b fastest)
    lam3 = np.stack(slabs, axis=0).transpose(0, 2, 1)
    encode(lam3).tofile(os.path.join(OUT, 'lambda3d.bin'))

    lb, le = np.meshgrid(axis(AX_B2), axis(AX_EPS2), indexing='ij')
    lb, le = lb.ravel(), le.ravel()
    lam2 = np.concatenate([lambda_peak(10**lb[s:s + 4000], 10**le[s:s + 4000],
                                       np.full(len(lb[s:s + 4000]), 1e5))
                           for s in range(0, lb.size, 4000)])
    lam2 = lam2.reshape(AX_B2['n'], AX_EPS2['n'])
    encode(lam2.T).tofile(os.path.join(OUT, 'lambda2d.bin'))   # index = iEps * nB2 + iB

    meta = dict(
        encoding=dict(type='uint16le', log10_lo=LOG_LAM_LO, log10_hi=LOG_LAM_HI,
                      nan_code=0, qmin=1, qmax=65535),
        lambda3d=dict(file='lambda3d.bin', order=['St', 'eps', 'b'],
                      b=AX_B, eps=AX_EPS, St=AX_ST),
        lambda2d=dict(file='lambda2d.bin', order=['eps', 'b'], b=AX_B2, eps=AX_EPS2,
                      St=1e5),
    )
    with open(os.path.join(OUT, 'lambda_tables.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'done in {time.time() - t0:.1f}s')


JUMP_DEX = 0.3


def interp_log(table, axes, pts):
    """Multilinear interpolation of log10(Lambda); mirrors the JS code.

    Lambda jumps where the two peaks of the marginal stability curve swap.
    If the cell corners straddle such a jump (spread > JUMP_DEX) the nearest
    corner is used instead of blending the two branches."""
    ll = np.log10(table)
    idx, w = [], []
    for ax, p in zip(axes, pts):
        u = (np.clip(p, ax['lo'], ax['hi']) - ax['lo']) / (ax['hi'] - ax['lo']) * (ax['n'] - 1)
        i = np.clip(np.floor(u).astype(int), 0, ax['n'] - 2)
        idx.append(i); w.append(u - i)
    out = 0
    vals = []
    for corner in np.ndindex(*(2,) * len(axes)):
        wt = 1
        ii = []
        for d, c in enumerate(corner):
            wt = wt * (w[d] if c else 1 - w[d])
            ii.append(idx[d] + c)
        vals.append(ll[tuple(ii)])
        out = out + wt * vals[-1]
    vals = np.array(vals)
    near = ll[tuple(i + (wd > 0.5) for i, wd in zip(idx, w))]
    jump = vals.max(axis=0) - vals.min(axis=0) > JUMP_DEX
    return 10**np.where(jump, near, out)


def check(n=3000, seed=1):
    """Compare interpolated tables with the reference code at random points."""
    rng = np.random.default_rng(seed)
    nb, ne, ns = AX_B['n'], AX_EPS['n'], AX_ST['n']
    lam3 = decode(np.fromfile(os.path.join(OUT, 'lambda3d.bin'), '<u2')).reshape(ns, ne, nb)
    pts = [rng.uniform(AX_ST['lo'], AX_ST['hi'], n), rng.uniform(-2.5, 0.3, n),
           rng.uniform(-3, 2, n)]
    approx = interp_log(lam3, [AX_ST, AX_EPS, AX_B], pts)
    exact = np.array([get_Lambda_vec([10**b], [10**e], [10**s], n_lamb=4000)[0]
                      for s, e, b in zip(*pts)])
    rel = np.abs(approx / exact - 1)
    print('lambda3d vs get_Lambda_vec (n_lamb=4000):')
    for q in (50, 90, 99, 99.9):
        print(f'  {q:5.1f}th percentile rel. error: {np.nanpercentile(rel, q):.2e}')
    print(f'  fraction with error > 5%: {np.mean(rel > 0.05):.3%} (jumps of the bimodal curve)')

    lam2 = decode(np.fromfile(os.path.join(OUT, 'lambda2d.bin'), '<u2')).reshape(AX_EPS2['n'], AX_B2['n'])
    pts2 = [rng.uniform(-2.5, 0.3, n), rng.uniform(-6, 3, n)]
    approx2 = interp_log(lam2, [AX_EPS2, AX_B2], pts2)
    exact2 = np.array([get_Lambda_vec([10**b], [10**e], [1e5], n_lamb=20000)[0]
                       for e, b in zip(*pts2)])
    rel2 = np.abs(approx2 / exact2 - 1)
    print('lambda2d vs marginal stability curve at St=1e5 (n_lamb=20000):')
    for q in (50, 90, 99, 99.9):
        print(f'  {q:5.1f}th percentile rel. error: {np.nanpercentile(rel2, q):.2e}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true', help='only validate existing tables')
    args = p.parse_args()
    if not args.check:
        build()
    check()
