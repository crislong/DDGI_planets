"""
Reference values from the Python model (Lambda at (c_d/c_s)^2, largest root
of eq. C1, 50 bisection iterations) on a coarse (Mdot, r) grid, used by
tests/validate.mjs to check the JavaScript port.

Usage: python make_validation_set.py > ../tests/reference.json
"""
import json
import sys

import numpy as np

from ddgi_reference import dust_quantities_stokes2, disc_model

CASES = [
    dict(vfrag=10, eps_max=1.0, sigma_log=1.0, eps_floor=1e-2, mstar=1.0),
    dict(vfrag=2, eps_max=1.0, sigma_log=1.0, eps_floor=1e-2, mstar=1.0),
    dict(vfrag=5, eps_max=0.5, sigma_log=0.5, eps_floor=3e-3, mstar=0.5),
]

md = np.logspace(-8, -3, 21)
r = np.logspace(0, 3, 21)
out = []
for c in CASES:
    g = disc_model(md, r, mstar=c['mstar'])
    ok = (g['alpha'] > 1e-4) & (g['alpha'] < 0.06)
    sub = {k: v[ok] for k, v in g.items()}
    d = dust_quantities_stokes2(sub, c['vfrag'] * 100, n_iter=50, n_lamb=4000,
                                eps_max=c['eps_max'], sigma_log=c['sigma_log'],
                                eps_floor=c['eps_floor'])
    M, R = np.meshgrid(md, r, indexing='ij')
    pts = []
    for i in range(ok.sum()):
        pts.append(dict(mdot=M[ok][i], r=R[ok][i], alpha=sub['alpha'][i],
                        St_max=d['St_max'][i], eps=d['eps'][i],
                        cdcs=d['cd'][i] / sub['cs'][i], mJ=d['mjeans_d'][i] / 5.972e27))
    out.append(dict(params=c, points=pts))
    print(f"case {c}: {len(pts)} points", file=sys.stderr)

json.dump(out, sys.stdout, default=float)
