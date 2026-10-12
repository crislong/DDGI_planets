"""
Reference implementation of the DDGI model (Longarini & Clarke 2026).

Code provided by the author; used here (i) to generate the precomputed
Lambda tables loaded by the website and (ii) to validate the JavaScript
port. Only constants and imports were added.
"""
import time

import numpy as np

kb = 1.380649e-16   # erg/K
mp = 1.6726e-24     # g
Gcgs = 6.674e-8     # cgs


# ---------------------------------------------------------------------
# Relazione di dispersione (Longarini et al. 2023a) - forma vettoriale
# ---------------------------------------------------------------------

def A(beta):
    return beta / 16

def B_st_BR(lamb, alpha, beta, st):
    return lamb * ((beta + 1 + 1/st**2 + alpha*(1+beta)/st**2
                     + alpha**2 * beta / st**2) * lamb - (alpha + beta)) / 4

def C_st_BR(lamb, alpha, st):
    return (lamb**4 * (1 + st**(-2) + 2*alpha/st**2 + alpha**2/st**2)
            - lamb**3 * (alpha + 1 + (1+alpha)/st**2
                         + 2*alpha*(1+alpha)/st**2 + (1+alpha)*alpha**2/st**2))


def get_Lambda_vec(cdcs, eps, st, n_lamb=1000, lamb_pad=1.0):
    cdcs = np.asarray(cdcs, dtype=float)
    eps  = np.asarray(eps,  dtype=float)
    st   = np.asarray(st,   dtype=float)

    lamb_max = lamb_pad * (1 + np.nanmax(eps))
    lamb = np.linspace(0, lamb_max, n_lamb)

    lamb_b  = lamb[None, :]
    alpha_b = eps[:, None]
    beta_b  = cdcs[:, None]
    st_b    = st[:, None]

    Ab = A(beta_b)
    Bb = B_st_BR(lamb_b, alpha_b, beta_b, st_b)
    Cb = C_st_BR(lamb_b, alpha_b, st_b)

    disc = Bb**2 - 4*Ab*Cb
    with np.errstate(invalid='ignore'):
        disc = np.where(disc < 0, np.nan, disc)
        ms = (-Bb + np.sqrt(disc)) / (2*Ab)

    all_nan   = np.all(np.isnan(ms), axis=1)
    ms_filled = np.where(np.isnan(ms), -np.inf, ms)
    idx       = np.argmax(ms_filled, axis=1)

    lam_val = 2 * lamb[idx]
    return np.where(all_nan, np.nan, lam_val)


def solve_cdcs_vec(eps, st, beta, cdcs_min=1e-8, cdcs_max=1e3,
                    n_iter=50, n_lamb=1000, verbose=False):
    eps, st, beta = np.broadcast_arrays(np.asarray(eps, dtype=float),
                                         np.asarray(st,  dtype=float),
                                         np.asarray(beta, dtype=float))
    shape = eps.shape
    eps_f, st_f, beta_f = eps.ravel(), st.ravel(), beta.ravel()

    def residual_vec(cdcs):
        lam = get_Lambda_vec(cdcs, eps_f, st_f, n_lamb=n_lamb)
        return cdcs - (lam / beta_f)**(1/3)

    lo = np.full_like(eps_f, cdcs_min)
    hi = np.full_like(eps_f, cdcs_max)

    f_lo = residual_vec(lo)
    f_hi = residual_vec(hi)
    valid = (np.sign(f_lo) != np.sign(f_hi)) & ~np.isnan(f_lo) & ~np.isnan(f_hi)

    t_start = time.time()
    for i in range(n_iter):
        mid = np.sqrt(lo * hi)
        f_mid = residual_vec(mid)
        same_sign = (np.sign(f_mid) == np.sign(f_lo))
        lo = np.where(same_sign, mid, lo)
        f_lo = np.where(same_sign, f_mid, f_lo)
        hi = np.where(same_sign, hi, mid)
        if verbose:
            elapsed = time.time() - t_start
            print(f"\riterazione {i+1}/{n_iter} - {elapsed:5.1f}s", end="", flush=True)
    if verbose:
        print()

    cdcs_sol = np.sqrt(lo * hi)
    cdcs_sol = np.where(valid, cdcs_sol, np.nan)
    return cdcs_sol.reshape(shape)


def solve_St_star_vec(eps, betacool, St_min=1e-3, St_max=1e3,
                       n_iter=50, n_lamb=1000):
    eps, betacool = np.broadcast_arrays(np.asarray(eps, dtype=float),
                                         np.asarray(betacool, dtype=float))
    shape = eps.shape
    eps_f, beta_f = eps.ravel(), betacool.ravel()

    def residual2(St_trial):
        cdcs_trial = solve_cdcs_vec(eps_f, St_trial, beta_f, n_iter=n_iter, n_lamb=n_lamb)
        return cdcs_trial - np.sqrt(St_trial / beta_f)

    lo = np.full_like(eps_f, St_min)
    hi = np.full_like(eps_f, St_max)

    f_lo = residual2(lo)
    f_hi = residual2(hi)
    valid = (np.sign(f_lo) != np.sign(f_hi)) & ~np.isnan(f_lo) & ~np.isnan(f_hi)

    for _ in range(n_iter):
        mid = np.sqrt(lo * hi)
        f_mid = residual2(mid)
        same_sign = (np.sign(f_mid) == np.sign(f_lo))
        lo = np.where(same_sign, mid, lo)
        f_lo = np.where(same_sign, f_mid, f_lo)
        hi = np.where(same_sign, hi, mid)

    St_star = np.sqrt(lo * hi)
    St_star = np.where(valid, St_star, np.nan)
    return St_star.reshape(shape)


# ---------------------------------------------------------------------
# Disc model
# ---------------------------------------------------------------------

def eps_of_st(St, eps_max, sigma_log=1.0, eps_floor=1e-2):
    x = np.log10(St)
    return eps_floor + (eps_max - eps_floor) * np.exp(-(x**2) / (2 * sigma_log**2))


def boundary(regime, mdot):
    mdot_6 = mdot / 1e-6
    r = 'nan'
    if regime == 0:
        r = 27 * mdot_6**(2/9)
    if regime == 1:
        r = 17 * mdot_6**(2/9)
    if regime == 2:
        r = 5.5 * mdot_6**(0.24)
    if regime == 3:
        r = 70 * mdot_6**(2/9)
    return r


def disc_model(mdot, radius, mstar=1.0):
    mdot   = np.atleast_1d(np.asarray(mdot,   dtype=float))
    radius = np.atleast_1d(np.asarray(radius, dtype=float))

    mdot_6     = mdot / 1e-6
    radius_eff = radius / (mstar**(1/3))
    radius_100 = radius_eff / 100
    mstar_corr = mstar**(-1/3)

    shape = (len(mdot), len(radius))
    sigma = np.zeros(shape); T = np.zeros(shape); alpha = np.zeros(shape)
    H_R = np.zeros(shape);   t_nu = np.zeros(shape)

    boundaries = np.zeros((4, len(mdot)))
    boundaries[0, :] = boundary(2, mdot) * mstar**(1/3)
    boundaries[1, :] = boundary(1, mdot) * mstar**(1/3)
    boundaries[2, :] = boundary(0, mdot) * mstar**(1/3)
    boundaries[3, :] = boundary(3, mdot) * mstar**(1/3)

    for i in range(len(mdot)):
        m = mdot_6[i]; r = radius_100; b = boundaries[:, i]

        zone0 =  radius <= b[0]
        zone1 = (radius >  b[0]) & (radius <= b[1])
        zone2 = (radius >  b[1]) & (radius <= b[2])
        zone3 = (radius >  b[2]) & (radius <= b[3])
        zone4 =  radius >  b[3]

        sigma[i, zone0] = 170    * m**0.018  * r[zone0]**(-1.64)
        T    [i, zone0] = 500    * m**0.036  * r[zone0]**(-0.27)
        alpha[i, zone0] = 2.1e-4 * m**0.945  * r[zone0]**(0.41)
        H_R  [i, zone0] = 0.68   * m**0.018  * r[zone0]**(0.36)  * mstar_corr
        t_nu [i, zone0] = 2.1e6  * m**(-0.82)* r[zone0]**(0.36)

        sigma[i, zone1] = 29     * m**(1/6)  * r[zone1]**(-9/4)
        T    [i, zone1] = 14     * m**(1/3)  * r[zone1]**(-1.5)
        alpha[i, zone1] = 4.4e-2 * m**(1/2)  * r[zone1]**(9/4)
        H_R  [i, zone1] = 0.11   * m**(1/6)  * r[zone1]**(-1/4)  * mstar_corr
        t_nu [i, zone1] = 3.4e5  * m**(-5/6) * r[zone1]**(-1/4)

        sigma[i, zone2] = 75     * m**(1/21) * r[zone2]**(-12/7)
        T    [i, zone2] = 95     * m**(2/21) * r[zone2]**(-3/7)
        alpha[i, zone2] = 2.5e-3 * m**(6/7)  * r[zone2]**(9/14)
        H_R  [i, zone2] = 0.30   * m**(1/21) * r[zone2]**(2/7)   * mstar_corr
        t_nu [i, zone2] = 9.4e5  * m**(-20/21)*r[zone2]**(2/7)

        sigma[i, zone3] = 14     * m**(1/3)  * r[zone3]**(-3)
        T    [i, zone3] = 3.2    * m**(2/3)  * r[zone3]**(-3)
        alpha[i, zone3] = 0.4    *             r[zone3]**(9/2)
        H_R  [i, zone3] = 0.055  * m**(1/3)  * r[zone3]**(-1)    * mstar_corr
        t_nu [i, zone3] = 1.7e5  * m**(-2/3) * r[zone3]**(-1)

        sigma[i, zone4] = 25     *             r[zone4]**(-1.5)
        T    [i, zone4] = 10
        alpha[i, zone4] = 0.06   * m
        H_R  [i, zone4] = 0.1    *             r[zone4]**(0.5)   * mstar_corr
        t_nu [i, zone4] = 3e5    * m**(-1)   * r[zone4]**(0.5)

    cs     = np.sqrt(kb * T.squeeze() / (2.3 * mp))
    Ljeans = cs**2 / (np.pi * Gcgs * sigma.squeeze())
    mjeans = sigma.squeeze() * Ljeans**2

    return {
        'sigma': sigma.squeeze(), 'T': T.squeeze(), 'alpha': alpha.squeeze(),
        'H_R': H_R.squeeze(), 't_nu': t_nu.squeeze(), 'cs': cs,
        'beta': 4 / (9 * alpha.squeeze()), 'Ljeans': Ljeans, 'mjeans': mjeans
    }


def lambda_fac_vec(eps, xi):
    """Lambda used for the Jeans mass in dust_quantities_stokes2 (Q=1, St=1e5)."""
    eps = np.asarray(eps, float).ravel(); xi = np.asarray(xi, float).ravel()
    l = np.linspace(0, 2, 500)
    lamb_b = l[None, :]; alpha_b = eps[:, None]; beta_b = xi[:, None]
    st_b = np.full_like(beta_b, 100000.0)
    with np.errstate(invalid='ignore'):
        from_marg = (-(lamb_b * ((beta_b + 1 + 1/st_b**2)*lamb_b - (alpha_b+beta_b)) / 4)
                     + np.sqrt((lamb_b * ((beta_b + 1 + 1/st_b**2)*lamb_b - (alpha_b+beta_b)) / 4)**2
                               - 4*A(beta_b)*(lamb_b**4*(1+st_b**(-2))
                                              - lamb_b**3*(1+alpha_b+(1+alpha_b)/st_b**2)))
                     ) / (2*A(beta_b))
    return 2 * l[np.argmax(from_marg, axis=1)]


def dust_quantities_stokes2(gas_disc, vfrag, cdcs_min=1e-8, cdcs_max=1e3,
                            n_iter=5, n_lamb=1000, eps_max=1, sigma_log=0.5,
                            eps_floor=1e-2):
    betacool = gas_disc['beta']
    cs       = gas_disc['cs']

    St_max = betacool * (vfrag / cs)**2
    eps    = eps_of_st(St_max, eps_max, sigma_log=sigma_log, eps_floor=eps_floor)
    vcoll  = (St_max / betacool)**0.5 * cs

    cdcs = solve_cdcs_vec(eps, St_max, betacool, cdcs_min=cdcs_min,
                          cdcs_max=cdcs_max, n_iter=n_iter, n_lamb=n_lamb)
    St_star = solve_St_star_vec(eps, betacool, n_iter=n_iter, n_lamb=n_lamb)

    cd = np.where(St_max < St_star, cdcs * cs, np.sqrt(St_max / betacool) * cs)

    shape_orig = St_max.shape
    lam = lambda_fac_vec(eps, (cd / cs)**2).reshape(shape_orig)
    lambda_fac = lam / (1 + eps)
    mjd = gas_disc['Ljeans']**2 * lambda_fac**2 * gas_disc['sigma']
    Hd = 2 * cd**2 / (Gcgs * gas_disc['sigma'] * eps)

    return {'St_max': St_max, 'vcoll': vcoll, 'cd': cd, 'eps': eps, 'cdcs': cdcs,
            'St_star': St_star, 'lambda_fac': lambda_fac, 'Hd': Hd, 'mjeans_d': mjd}
