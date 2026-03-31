# Sigma/GN-PPM Audit Status (Current)

Date: 2026-03-28  
Primary test directory: `/home/jackm/projects/tests_isdf/qe_co`

This document is the current handoff for the dynamic GN-PPM SOS debugging path.
It is intentionally focused on the active comparisons and code paths in use now.

## 1) Active Code Paths

- Main production pipeline:
  - `src/gw_isdf/gw_jax.py`
  - `src/gw_isdf/ppm_sigma.py`
- SOS dynamic debug generator (ISDF basis from eps0mat + restart):
  - `tests_isdf/qe_co/isdf_w_from_eps0_projection_test.py`
- Current sigma_hp comparison tool (consistent kernel path):
  - `tests_isdf/qe_co/plot_sigmahp_chprime_compare_consistent.py`

## 2) Active Reference Files

BGW references:
- `tests_isdf/qe_co/sigma_hp.log` (now `frequency_dependence 3`)
- `tests_isdf/qe_co/ch_converge.dat` (now dynamic GN)
- `tests_isdf/qe_co/sigma_40b_gn_noepsbroad.out`

ISDF/SOS outputs:
- `tests_isdf/qe_co/isdf_sos_gn_bgw_sx.out`
- `tests_isdf/qe_co/isdf_sos_gn_legacy_sx.out`
- `tests_isdf/qe_co/isdf_sos_gn_bgw_sx_nocut.out`
- `tests_isdf/qe_co/isdf_sos_debug.out`
- `tests_isdf/qe_co/isdf_sos_debug_40b.out`
- `tests_isdf/qe_co/isdf_sos_debug_40b_right.out`

Current comparison artifacts:
- CH-converge comparison:
  - `tests_isdf/qe_co/ch_converge_compare_dynamic_gn_isdf_latest.dat`
  - `tests_isdf/qe_co/ch_converge_compare_dynamic_gn_isdf_latest.png`
- sigma_hp per-band comparisons built from consistent kernels:
  - `tests_isdf/qe_co/sigmahp_chprime_vs_isdf_ch_bgwlike.dat`
  - `tests_isdf/qe_co/sigmahp_chprime_vs_isdf_ch_bgwlike.png`
  - `tests_isdf/qe_co/sigmahp_sxx_vs_isdf_sxx_bgwlike.dat`
  - `tests_isdf/qe_co/sigmahp_sxx_vs_isdf_sxx_bgwlike.png`
  - `tests_isdf/qe_co/sigmahp_cor_vs_isdf_cor_bgwlike.dat`
  - `tests_isdf/qe_co/sigmahp_cor_vs_isdf_cor_bgwlike.png`

## 3) Exact Math/Flags Used In The Current sigma_hp Comparison

Implemented in `plot_sigmahp_chprime_compare_consistent.py`.

### 3.1 W and Wc construction

- Read `eps^{-1}(q=0, ifreq=0/1)` from `eps0mat.h5`.
- Use transposed orientation (`transpose_eps = True`) to match BGW convention used in this debug flow.
- Patch head column:
  - `epsinv[:, G0] = 0`
  - `epsinv[G0, G0] = W_head / V_head`
- Build W with right-side Coulomb application:
  - `W = epsinv @ diag(v)`
- Build correlation part:
  - `Wc = W - v`
- Project to ISDF basis:
  - `W_munu = Z^H W Z`

### 3.2 GN fit in ISDF basis

From `extract_gn_ppm_parameters_from_Wc`:

- Inputs: `Wc(0)`, `Wc(i*omega_p)`
- `omega_p = 2.0 Ry`
- `fallback_omega = 2.0 Ry`
- Per `(mu,nu)`:
  - `denom = Wc0 - Wci`
  - `ratio = Re(Wci / denom)` when `|denom| > 1e-14`, else `0`
  - valid mode if `ratio` finite and `ratio > 0`
  - `Omega = omega_p * sqrt(ratio)` for valid, else fallback `2.0 Ry`
  - `B = -0.5 * Wc0 * Omega`

### 3.3 CH comparison kernel used (BGW-like branch)

Per solved band `n` (1..20), sum over `n1` (1..40):

- `wx = E_n - E_n1` (Ry), clamped to `1e-15` if tiny
- overlap vector:
  - `M_mu = sum_s conj(psi_{n1,s,mu}) * psi_{n,s,mu}`
- `wdiff = wx - Omega_munu`
- `delw = Omega_munu / wdiff` on safe entries
- thresholds:
  - `tol_small = 1e-12`
  - `tol_zero = 1e-15`
  - `gamma = gpp_broadening / RYD2EV`
  - `limittwo = gamma^2`
  - `limitone = 1 / (4*tol_small)`
- active mask:
  - `cond1 = (|wdiff|^2 > limittwo) & (|delw|^2 < limitone) & valid_mode`
- CH kernel:
  - `K = B / (wx - Omega)` on `cond1`, else `0`
- contribution:
  - `CH_n += M^H K M`

Important: this path does **not** use a direct `+i*eta` in CH denominator. It uses BGW-like pole-region branch selection with `gpp_broadening` thresholds.

### 3.4 SX-X comparison kernel used (BGW-like branch)

Computed via `_sigma_sx_gpp_bgw_diag` in `isdf_w_from_eps0_projection_test.py`.

- Uses same `gpp_broadening` threshold logic (`limittwo`, `limitone`)
- Uses BGW-style near-pole branch split
- Applies `gpp_sexcutoff` clipping when `wx < 0`
- Default flags used in current comparison script:
  - `gpp_broadening = 0.5 eV`
  - `gpp_sexcutoff = 4.0`

### 3.5 BGW defaults check (source)

- `gpp_broadening` default `0.5`:
  - `bgw_src/Sigma/inread.f90:154`
- `gpp_sexcutoff` default `4.0`:
  - `bgw_src/Sigma/inread.f90:155`
- Pole-threshold use in SX/CH branch split:
  - `bgw_src/Sigma/mtxel_cor.f90:1701-1702`
  - `bgw_src/Sigma/mtxel_cor.f90:1831-1843`
- `sexcutoff` application:
  - `bgw_src/Sigma/mtxel_cor.f90:1850-1851`

## 4) Current Error Metrics (Updated)

All values below are MAE/bias in eV using latest files from 2026-03-28.

### 4.1 CH-converge (VBM/CBM cumulative vs Nb)

From `ch_converge_compare_dynamic_gn_isdf_latest.dat`:

- VBM `dVBM = ISDF - BGW`
  - all Nb: MAE `1.054e-01`, bias `+1.054e-01`, max `8.012e-01`
  - Nb >= 3: MAE `1.099e-01`, bias `+1.099e-01`, max `8.012e-01`
- CBM `dCBM = ISDF - BGW`
  - all Nb: MAE `1.692e-02`, bias `+1.692e-02`, max `4.595e-02`
  - Nb >= 3: MAE `1.769e-02`, bias `+1.769e-02`, max `4.595e-02`

### 4.2 sigma_hp per-band (n=1..20), consistent kernels

From `sigmahp_*_bgwlike.dat`:

Important BGW convention note:
- In dynamic GPP, BGW has near-pole handling that redistributes occasional
  contribution between SX-X and CH branches (cancellation-safe reassignment).
- Therefore **standalone CH and standalone SX-X MAE are not meaningful
  validation targets** against BGW.
- Only `Cor = (SX-X) + CH` is a physically stable comparison target.

Cor = SX-X + CH:
- all n: MAE `4.847e-02`, bias `+3.648e-02`, max `3.923e-01`
- n >= 3: MAE `1.027e-02`, bias `-3.056e-03`, max `4.567e-02`
- valence (1..10): MAE `9.087e-02`, bias `+6.688e-02`
- valence (3..10): MAE `1.552e-02`, bias `-1.447e-02`
- conduction (11..20): MAE `6.074e-03`, bias `+6.074e-03`

Diagnostics-only observation (not scored):
- CH and SX-X branchwise residuals are large and opposite for some valence
  bands, consistent with branch redistribution behavior.
- The sum (`Cor`) is the correct scored quantity and is tight, especially for
  bands 3..20.

## 5) Current Open Question

Given the above pattern (branchwise mismatch but small `Cor` mismatch), the
highest-priority remaining issue is decomposition-level assignment in dynamic
branch handling/reporting. Total correlation normalization/sign is largely
constrained by the `Cor` agreement.

## 6) Reproduction Commands

From `tests_isdf/qe_co`:

- Build consistent sigma_hp CH/SX-X/Cor comparisons:
  - `uv run -- python plot_sigmahp_chprime_compare_consistent.py`
- Existing SOS table generator:
  - `uv run -- python isdf_w_from_eps0_projection_test.py --help`
