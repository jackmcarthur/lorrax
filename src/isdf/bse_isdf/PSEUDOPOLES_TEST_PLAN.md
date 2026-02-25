# Pseudopoles W(0) Reconstruction Test Plan

This document is for an agent who wants to validate that `bse_pseudopoles.py` can build a controlled approximation to

`W_c(omega) = W(omega) - V ≈ sum_p d_p d_p^H / (omega - Omega_p)`

and in particular reconstruct `W_c(omega=0)` accurately, by converging the pseudopoles construction parameters.

The intended “exact” reference for these tests is the **sum-over-states + Dyson** `W_c(0)` (q=0, nohead) already used by
`../tests_isdf/*/compare_w_exact_to_sos.py`.

---

## 0) Ground Truth / What We Compare Against

### Reference definition (RPA, q=0, nohead)

1. Build raw independent-particle susceptibility (transition sum):
   - `chi0_raw[mu,nu] = sum_{k,c,v} M_{cvk}[mu] * (1/ΔE_{cvk}) * conj(M_{cvk}[nu])`
   - `M_{cvk}[mu] = sum_s conj(psi_v[k,v,s,mu]) * psi_c[k,c,s,mu]`
2. Convert to RPA `chi0` with the repo’s normalization:
   - `chi0 = -(2/Nk) * chi0_raw`
3. Dyson solve (do not form inverse explicitly):
   - `W = solve(I - V @ chi0, V)`
   - `Wc_ref = W - V`

This is implemented in `../tests_isdf/cohsex_prod/compare_w_exact_to_sos.py` (and similar scripts).

### Quantity we want from pseudopoles

For `omega=0`:

`Wc_pseudo(0) = sum_p d_p d_p^H / (0 - Omega_p) = - sum_p d_p d_p^H / Omega_p`

Here each residue vector `d_p` is in **r_mu/ISDF space** and is already Coulomb-dressed:

- TDA: `d_p = V * (d X_p)`
- non-TDA: `d_p = V * (d X_p + d* Y_p)`

So `Wc_pseudo(0)` should be directly comparable to the SOS-Dyson `Wc_ref` above.

---

## 1) Required Code Conventions (Do Not Regress These)

### non-TDA Liouvillian convention

All non-TDA code paths assume:

`S = [[A, B], [-B*, -A*]]`

and for density response (and “bright” seeding) the correct driving symmetry is:

`rhs = [f, -fbar]`

with:
- `f    = d^H (V g)` (dagger-coupled vertex)
- `fbar = d^T (V g)` (transpose-coupled vertex)

Do **not** assume `fbar == conj(f)` for complex Bloch spinors.

### Density readout channel

For non-TDA response outputs, use **only**:

`rho = d X + d* Y`

and the code’s “Wc column” readout is:

`w = V rho = V (d X + d* Y)`

This is the only channel we compare against SOS in RPA tests.

---

## 2) Files / Entry Points

- Pole construction: `python -m isdf.bse_isdf.bse_pseudopoles`
  - Writes poles and residue vectors `d_p` to an H5 file, grouped by windows.
- Wc evaluation from poles: `python -m isdf.bse_isdf.pseudopoles_eval`
  - Converts pseudopoles into a `bse_w_exact`-compatible H5 file containing `columns` and `Wc`.
- SOS reference comparison: `../tests_isdf/.../compare_w_exact_to_sos.py`
  - Reads your evaluated `Wc` columns and reports `||diff||/||ref||`.

---

## 3) Minimal Repro Workflow (Recommended Baseline)

From `../tests_isdf/cohsex_prod/`:

1. Generate pseudopoles (RPA, nohead):
```bash
uv run python -m isdf.bse_isdf.bse_pseudopoles \
  -i cohsex_prod.in \
  --n-val 26 --n-cond 44 \
  --rpa --nohead \
  --windows-kpm --windows-kpm-count 6 \
  --m0 12 --p-keep 8 --n-tail 4 \
  --n-quad 16 \
  --gmres-max-iter 200 --gmres-tol 1e-8 \
  --out bse_pseudopoles_rpa.h5
```

2. Evaluate `Wc(0)` columns from pseudopoles:
```bash
uv run python -m isdf.bse_isdf.pseudopoles_eval \
  --poles bse_pseudopoles_rpa.h5 \
  --omega-ev 0.0 --eta-ev 0.0 \
  --cols 0,1,2,3,4,5,6,7,8 \
  --out Wc_from_pseudopoles_cols0_8.h5
```

3. Compare to SOS-Dyson:
```bash
uv run python compare_w_exact_to_sos.py \
  --restart tmp/isdf_tensors_600.h5 \
  --n-val 26 --n-cond 44 \
  --bse Wc_from_pseudopoles_cols0_8.h5
```

Target: `||diff||/||ref||` decreases smoothly as you tighten/raise parameters below.

---

## 4) Convergence Parameters to Sweep (In Priority Order)

The knobs below are coupled; the cleanest strategy is “fix everything else fairly tight” and sweep one knob at a time.

### A) GMRES solve quality (per shifted solve)

Goal: ensure any error is dominated by pseudopole truncation, not inexact shifted solves.

Sweep:
- `--gmres-tol`: `1e-4`, `1e-6`, `1e-8`, (optionally `1e-10`)
- `--gmres-max-iter`: set comfortably above observed needs (e.g. 200)

Check:
- `compare_w_exact_to_sos.py` error should improve with tighter tol (if it doesn’t, you’re truncation-dominated).

### B) Quadrature resolution (filter accuracy)

`--n-quad` controls FEAST quadrature points (per half contour; non-TDA uses full contour internally).

Sweep:
- `8`, `12`, `16`, `24`

Expect:
- Too-small `n-quad` produces window leakage and unstable poles/residues.

### C) Subspace size vs brightness truncation

Controls how much of the window’s filtered subspace you keep and how much is approximated stochastically.

Sweep (keep one variable fixed while sweeping another):
- `--m0`: `6`, `12`, `18`, `24`
- `--p-keep`: `4`, `8`, `12`
- `--n-tail`: `0`, `2`, `4`, `8`
- `--s-cutoff`: `1e-6` to `1e-8` (lower keeps more nearly-dependent vectors; can worsen conditioning)

Heuristic:
- Increase `m0` until the orthonormalized basis size stops collapsing.
- Increase `p-keep` until the error plateaus; then add `n-tail` to pick up discarded brightness.

### D) Windowing strategy

This is typically the hardest to converge if the spectrum is broad and uneven.

Options:
- Default windows (`build_default_windows_eV`) based on estimated spectral max.
- KPM windows (`--windows-kpm`) with `--windows-kpm-count N`.

Sweeps:
- `--windows-kpm-count`: `4`, `6`, `8`, `10`
- KPM resolution: `--kpm-n-moments` (`100`, `200`, `400`) and `--kpm-n-random` (`4`, `8`, `16`)

Checks:
- Ensure the union of windows covers the relevant positive spectrum.
- If error is dominated by missing high-energy contributions, add windows at the top end (or widen the last window).

---

## 5) What To Record for Each Sweep

For each (parameter set):

1. `||diff||/||ref||` vs SOS on a fixed set of columns (e.g. 0..8).
2. The “best scalar fit” alpha from `compare_w_exact_to_sos.py`:
   - If alpha is far from 1 and stable across sweeps, you may have a normalization mismatch.
3. Basic sanity:
   - Check `Wc(0)` is close to Hermitian:
     - `||Wc - Wc^H|| / ||Wc||` on the reconstructed columns.
4. Stability vs a small imaginary part:
   - Repeat evaluation at `eta=1e-3 eV`:
     - poles near zero / numerical issues show up as extreme sensitivity to eta.

---

## 6) Likely Failure Modes (Debug Checklist)

1. **Using the wrong non-TDA density channel**
   - Any code that does `R(X+Y)` instead of `R X + R* Y` will fail for complex Bloch/SOC.
2. **Wrong non-TDA driving symmetry**
   - Seeds or solves using `[f, f]` or `[f, -f]` instead of `[f, -fbar]` can make the static response collapse.
3. **Missing k-coupling in the RPA V term**
   - For q=0 response, `V` must couple all k through total density.
4. **Not enough windows / window leakage**
   - Errors that do not improve with `m0/p_keep/n_tail` often come from window coverage / quadrature.

---

## 7) Cleanup / Structural Suggestions (If You Need to Extend This)

If you need to build a larger “parameter sweep harness”, prefer:

1. Keep pole construction (`bse_pseudopoles.py`) and evaluation (`pseudopoles_eval.py`) separate.
2. Write a thin sweep driver in `../tests_isdf/cohsex_prod/` that:
   - loops over parameter grids,
   - runs pole build,
   - runs eval,
   - runs `compare_w_exact_to_sos.py`,
   - stores a CSV.

This keeps the library code stable while enabling fast iteration.

