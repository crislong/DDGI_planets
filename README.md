# DDGI Explorer

Interactive website for the models of *Planetary core formation through
dust-driven gravitational instability – I. Self-consistent models*
(Longarini & Clarke 2026). It shows the gas and dust discs in the (Ṁ, r)
plane and recomputes everything live as you change the parameters:

- fragmentation velocity v_frag
- ε(St) parameters: ε_bkg, ε_max, σ_ε (eq. 7)
- stellar mass M★ (Clarke 2009 rescaling)
- initial grain size a₀ and material density ρ_s (growth timescale, eq. 14)

Maps: regimes (Fig. 3), any gas quantity (α_GI, Σ_g, T, H/r, β, c_s) or dust
quantity (St_max, ε, c_d/c_s, c_d/(c_s ε), Λ_J, M_J, t_growth, t_DDGI), the
histogram of M_J in the DDGI region (Fig. 4), and a point inspector. "Pin for
comparison" keeps the DDGI boundary and histogram of the current parameters.
The parameters are stored in the URL, so a configuration can be shared as a link.

## How it works

The only expensive ingredient is Λ, from the two-fluid marginal stability
curve. It depends only on dimensionless quantities, so it is precomputed once
and interpolated in the browser:

| table | variables | used for |
|---|---|---|
| `web/data/lambda3d.bin` | Λ(b, ε, St), 321 × 59 × 113 | implicit eq. (C1) for c_d/c_s |
| `web/data/lambda2d.bin` | Λ(b, ε), St → ∞, 961 × 363 | Jeans mass |

Everything else (Table A1, St_max, ε, the bisection for c_d/c_s, M_J,
t_growth) is analytic and is evaluated in JavaScript (`web/js/model.js`):
a 200 × 200 grid takes ~0.1–0.2 s.

## Layout

```
python/ddgi_reference.py      reference Python model (original code)
python/make_lambda_tables.py  builds web/data/*.bin and checks their accuracy
python/make_validation_set.py reference values for tests/validate.mjs
tests/validate.mjs            compares the JS port with the Python code
web/                          static site (index.html, js/, css/, data/)
```

## Running locally

```
python3 -m http.server -d web 8000     # then open http://localhost:8000
node tests/validate.mjs                # JS vs Python check
cd python && python3 make_lambda_tables.py   # rebuild the tables (~2 min, numpy)
```

## Deploying

`.github/workflows/pages.yml` publishes `web/` to GitHub Pages on every push to
`main`. Enable it once in *Settings → Pages → Source: GitHub Actions*.
