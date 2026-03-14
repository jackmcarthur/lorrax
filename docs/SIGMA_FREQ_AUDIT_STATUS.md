# Sigma(omega) GN-PPM / Minimax Audit Status

Date: 2026-03-14

This note records what has been checked carefully in the full-frequency GN-PPM
`Sigma_c(omega)` path, what has been reproduced numerically, what bugs were
real and fixed, and what remains the live physics problem.

The goal is to keep a single current status file for other reviewers.

---

## Current Bottom Line

The remaining mismatch with BerkeleyGW is **not** presently traced to:

- minimax screening for `W(0)` / `W(i omega_p)`,
- the GN-PPM sign convention for `W^c(0)`,
- Hermiticity of `W`,
- the order of `Re/Im` projection versus band projection,
- the `Sigma(E_DFT)` reference-energy path,
- the `Sigma^+ / Sigma^-` denominator definitions or window routing,
- modest changes in the chosen `omega` range,
- the q->0 head values themselves,
- or the invalid-GN fallback policy by itself.

The live suspect is still **upstream of the denominators**, i.e. in the
effective spectral weights entering `Sigma_c`, most likely the
`W^c -> (B, Omega) -> W^c(t)` amplitude / normalization side rather than the
window algebra.

---

## Confirmed Not A Problem

### 1. Minimax screening for `W(0)` and `W(i omega_p)`

Confirmed:

- `W(0)` and `W(i omega_p)` are built from the canonical minimax screening path.
- GN-PPM is fitted to **`W^c(0)` and `W^c(i omega_p)`**, not directly to `Pi`.
- The static minimax screening path had already been checked earlier against the
  CTSP `W(0)` workflow and gave acceptable agreement.

Relevant code:

- [minimax_screening.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/minimax_screening.py)
- [ppm_sigma.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/ppm_sigma.py)
- [w_isdf.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/w_isdf.py)
- [minimax.py](/home/jackm/projects/isdf_cohsex/docs/minimax.py)

---

### 2. GN-PPM sign convention for `W^c(0)`

This was checked explicitly.

Results:

- `W^c(0) = +2 B / Omega` is wrong by a relative error of about `2.0`.
- `W^c(0) = -2 B / Omega` matches the fitted `W^c(0)` essentially exactly.

Representative log:

- `PPM W^c(0) check (+2B/Ω): rel = 2.000e+00`
- `PPM W^c(0) check (-2B/Ω): rel = 5.582e-17`

So the internal GN-PPM sign convention is settled:

\[
W^c(0) = -\frac{2B}{\Omega}.
\]

This sign check by itself does **not** prove a sigma sign bug; it only fixes
the `W^c` convention being used internally.

Relevant code:

- [ppm_sigma.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/ppm_sigma.py)

---

### 3. Hermiticity of `W(0)` and `W(i omega_p)`

Hermiticity checks were added and run on `cohsex_prod`.

Observed:

- `W(0)` Hermitian residual: `7.637e-08`
- `W(i omega_p=2 Ry)` Hermitian residual: `6.706e-08`
- maximum imaginary diagonal entries are `~1e-8`

Conclusion:

- `W(0)` and `W(i omega_p)` are effectively Hermitian.
- Small non-Hermiticity from numerics is present but tiny and not the source of
  the multi-eV `Sigma_c` mismatch.

Relevant code:

- [ppm_sigma.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/ppm_sigma.py)

---

### 4. Window bounds and energy classification

The code now prints the actual vacuum-referenced energy ranges used for the
`Sigma_c` windows.

Example for `cohsex_prod`, `[-5, 5] eV`:

- `Ev = [-4.856589, -0.024453] Ry`
- `Ec = [0.076820, 1.532999] Ry`
- `Omega = [0.002482, 201.5483] Ry`

Per-window masks also print:

- `A`-range,
- `B`-range,
- mapped `Ev_vac` or `Ec_vac`,
- `T` and `z_edge`.

These printed ranges are consistent with the intended window logic. This rules
out the earlier concern that transitions were silently falling outside the
ranges the minimax builders believed they were covering.

Relevant code:

- [ppm_sigma.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/ppm_sigma.py)

---

### 5. `Re/Im` projection order versus band projection

This was a real issue and was fixed.

Old problem:

- the optimized `kij` path projected complex `Sigma_tau(mu,nu)` to band space
  first,
- then applied window `Re/Im` projection afterward,
- which is not algebraically valid in general.

Reason:

\[
K[\operatorname{Re} X] \neq \operatorname{Re} K[X], \qquad
K[\operatorname{Im} X] \neq \operatorname{Im} K[X]
\]

for the complex-linear band projection map `K`.

Fix:

- the code now contracts two exact band-space channels per `tau` node:
  - `K[Re X_tau]`
  - `K[Im X_tau]`
- and reconstructs each window contribution in band space from those two
  channels.

This was checked numerically in `complex128` to about `5e-15`.

Conclusion:

- the `Re/Im` ordering bug is fixed,
- and is not the current explanation for the BGW mismatch.

Relevant code:

- [gw_jax.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/gw_jax.py)
- [ppm_sigma.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/ppm_sigma.py)
- [GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md](/home/jackm/projects/isdf_cohsex/docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md)

---

### 6. `Sigma(E_DFT)` reference-energy path

This path was traced end to end through the code and rerun.

For `cohsex_prod` with `fermi_reference = midgap`, the active DFT reference is
now explicitly printed and matches the stored values:

- `VBM = -5.178570 eV`
- `CBM = -3.507981 eV`
- `E_F(midgap) = -4.343276 eV`

The code now writes / checks:

- `omega_dft_rel_ev = E_nk^DFT - E_F`
- `sigma_c_at_dft_ev`
- `sigma_xc_at_dft_ev`

and reloads them consistently for text output.

Concrete result:

- the earlier bad `+0.356 eV` effective reference is gone,
- bands 22-25 in `cohsex_prod` now have finite `sigC_EDFT` values in
  `eqp0_noqsym_w.dat`,
- and those values are consistent with the active WFN / band window.

Representative output:

- [eqp0_noqsym_w.dat](/home/jackm/projects/tests_isdf/cohsex_prod/eqp0_noqsym_w.dat)
- [eqp_g0w0.dat](/home/jackm/projects/tests_isdf/cohsex_prod/eqp_g0w0.dat)

Relevant code:

- [gw_jax.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/gw_jax.py)

---

### 7. `omega`-range dependence for the near-gap real part

This was tested directly by comparing saved `sigma_mnk.h5` outputs for:

- `[-5, 5] eV`
- `[-6, 6] eV`

on the common `[-5, 5] eV` grid.

Findings:

- global max difference in `sigma_c_kij_ev`: about `0.286 eV`
- but that worst case is almost entirely an imaginary-part change in a higher band
- for the near-gap `k=0` bands 22-29, the max diagonal difference is only about
  `0.0017` to `0.0048 eV`
- the max **real-part** drift for those bands is only about `3e-05 eV`

Conclusion:

- the near-gap real `Sigma_c(omega)` is effectively invariant under
  `[-5,5] -> [-6,6]`,
- so the large BGW mismatch is not being driven by modest changes in the chosen
  omega range.

Saved reference files:

- [/home/jackm/projects/tests_isdf/cohsex_prod/omega_m5_5_snapshot/sigma_mnk_m5_5.h5](/home/jackm/projects/tests_isdf/cohsex_prod/omega_m5_5_snapshot/sigma_mnk_m5_5.h5)
- [/home/jackm/projects/tests_isdf/cohsex_prod/omega_m6_6_snapshot/sigma_mnk_m6_6.h5](/home/jackm/projects/tests_isdf/cohsex_prod/omega_m6_6_snapshot/sigma_mnk_m6_6.h5)

---

### 8. `Sigma^+ / Sigma^-` denominator definitions and window routing

This was reduced to scalar tests using the **actual** window builders and
prefactors.

The tested target denominators were:

\[
\Sigma^{(-)}: \frac{1}{\omega_{\mathrm{rel}} - (E_c - E_F + \Omega)},
\qquad
\Sigma^{(+)}: \frac{1}{\omega_{\mathrm{rel}} + (E_F - E_v + \Omega)}.
\]

Using the real implementation, the scalar approximations matched the exact
denominators numerically for:

- positive `omega`, `Sigma^-`
- positive `omega`, `Sigma^+`
- negative `omega`, `Sigma^-`
- negative `omega`, `Sigma^+`

with errors at the `1e-8` to `1e-9` level in representative cases.

Conclusion:

- the current branch split is correct,
- the `omega < E_F` routing is correct,
- the minimax window algebra is not the source of the present BGW disagreement.

Relevant code:

- [ppm_sigma.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/ppm_sigma.py)
- [GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md](/home/jackm/projects/isdf_cohsex/docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md)

---

### 9. FFT normalization in the sigma convolution path

The sigma pipeline uses orthonormal FFTs and the explicit convolution
normalization

\[
-\frac{1}{\sqrt{N_k}}
\]

inside [get_sigma_static_mu_nu_jax](/home/jackm/projects/isdf_cohsex/src/gw_isdf/gw_jax.py).

This was traced because it is essential for the `k <-> R` convolution
consistency. There is no evidence that a missing `1/sqrt(N_k)` factor is the
current issue.

---

### 10. q->0 head values are not the main source of the mismatch

Head values were compared against BerkeleyGW mini-BZ output for the same MoS2
setup.

Compared values:

- `V_head`
  - BGW: `1649.143966071123`
  - ours: `1653.797615366955`
  - relative error: `0.2822%`

- `W_head(0)`
  - BGW: `394.450574775649`
  - ours: `362.219409655586`
  - relative error: `8.1712%`

- `W_head(i 2 Ry)`
  - BGW: `1439.483075606150`
  - ours: `1433.952559316620`
  - relative error: `0.3842%`

Then the code was temporarily hard-coded to use the BGW head values for `V`,
`W(0)`, and `W(i 2 Ry)` and rerun on `cohsex_prod`.

Observed effect on near-gap `sigC_EDFT`:

- only about `0.02` to `0.04 eV`

Conclusion:

- the q->0 head mismatch is real,
- but it is far too small to explain the multi-eV `Sigma_c` discrepancy.

---

## BerkeleyGW Source Audit

The local BerkeleyGW tree used for comparison is:

- [/home/jackm/SOURCES/BerkeleyGW](/home/jackm/SOURCES/BerkeleyGW)

### 11. BGW GN epsilon path does not use extra denominator broadening

For `frequency_dependence = 3` in `Epsilon`, BGW uses exactly two frequencies:

- `omega_1 = 0`
- `omega_2 = i * imaginary_frequency`

with defaults:

- `imaginary_frequency = 2.0 * ryd`
- `dBrdning = 0.0`

Relevant source:

- [Epsilon/inread.f90](/home/jackm/SOURCES/BerkeleyGW/Epsilon/inread.f90)
- [Epsilon/chi_summation.f90](/home/jackm/SOURCES/BerkeleyGW/Epsilon/chi_summation.f90)

Conclusion:

- BGW is **not** secretly evaluating `chi0(0)` or `chi0(i omega_p)` with a
  `+ i 0.25 eV` style Lorentzian denominator broadening in the GN epsilon path.

---

### 12. BGW fits `I_eps = 1 - epsinv`, not `W^c`

In `Sigma`, the GN fit is built from

\[
I_\epsilon(\omega) \equiv \delta - \epsilon^{-1}(\omega)
\]

at `omega = 0` and `omega = i omega_p`.

Relevant source:

- [Sigma/mtxel_cor.f90](/home/jackm/SOURCES/BerkeleyGW/Sigma/mtxel_cor.f90)

This is different in representation from our direct fit to `W^c = W - V`, but
the sign difference alone is not enough to diagnose the current mismatch.

What matters more is how invalid modes and dynamic amplitudes are handled after
the fit.

---

### 13. BGW invalid GN mode default is effectively staticization

In the local BGW source:

- `invalid_gpp_mode` initializes to `-1`
- the GN invalid-mode switch has explicit cases `0,1,2,3`
- `case default` falls through to the static-COHSEX-like limit

So for this source tree the **effective default** is:

- invalid GN modes are **staticized**, not forced to `2 Ry`

Relevant source:

- [Sigma/inread.f90](/home/jackm/SOURCES/BerkeleyGW/Sigma/inread.f90)
- [Sigma/mtxel_cor.f90](/home/jackm/SOURCES/BerkeleyGW/Sigma/mtxel_cor.f90)

This was important enough to test directly on our side.

---

### 14. BGW has extra GN sigma safeguards beyond the bare pole fit

The GN path in BGW sigma includes:

- tiny-`I_eps` skipping,
- explicit invalid-mode policy,
- complex-mode option,
- static-limit option,
- near-pole merged SX+CH handling,
- `ssxcutoff` logic for pathological screened-exchange terms.

Conclusion:

- BGW is not simply "fit one pole and use it directly";
- it has production stabilizers in the GN dynamic reconstruction.

This is one reason the remaining mismatch is now being treated as an amplitude /
policy problem rather than a pure windowing problem.

---

## Debugs Tried And What They Showed

### A. Checking `W^c(0)` sign from `B, Omega`

Outcome:

- settled the sign convention to `W^c(0) = -2B/Omega`

Status:

- not a live problem anymore

---

### B. Forcing exact band-space `Re/Im` handling

Outcome:

- fixed a real algebraic bug
- did **not** resolve the BGW `Sigma_c` mismatch

Status:

- resolved and not current

---

### C. Tracing `sigC_EDFT` output path

Outcome:

- fixed the bad DFT reference / stale HDF5 issue
- output now uses the intended `E_nk^DFT - E_F(midgap)`

Status:

- resolved and not current

---

### D. Changing omega range from `[-5,5]` to `[-6,6]`

Outcome:

- near-gap real part barely changes
- major BGW disagreement remains

Status:

- range choice is not the primary problem for the near-gap states

---

### E. Scalar `Sigma^+ / Sigma^-` denominator audit

Outcome:

- branch routing / prefactors reproduce the intended scalar denominators

Status:

- denominator algebra is not the current problem

---

### F. `Omega_{q,mu,nu}` spread and GN fallback classification

This was re-extracted directly from the live `cohsex_prod` restart-backed
minimax path using the same `compute_w0_wiwp_and_ppm_from_minimax` routine that
the production run uses.

For the current setup (`ppm_omega_p = 2.0 Ry`, `ppm_fallback_omega = 2.0 Ry`):

- total entries: `3,240,000`
- positive entries: `3,240,000`
- exact fallback entries: `133,826`
- fallback fraction: `0.04130432` (`4.130432%`)

Percentiles of the assigned `Omega` values:

- `p1  = 0.65368 Ry`
- `p5  = 1.14286 Ry`
- `p50 = 1.99849 Ry`
- `p95 = 2.90576 Ry`
- `p99 = 3.71552 Ry`
- `p99.9 = 6.92974 Ry`
- `max = 201.54834 Ry`

Most importantly, the unfulfilled set was classified directly:

- `BAD_TOTAL = 133,826` (`4.130432%`)
- `BAD_UNSAFE = 0`
- `BAD_NEG_OR_ZERO_RATIO = 133,826`

where

- `BAD_UNSAFE` means `|W^c(0) - W^c(i omega_p)| <= 1e-14`
- `BAD_NEG_OR_ZERO_RATIO` means
  `Re[W^c(i omega_p) / (W^c(0) - W^c(i omega_p))] <= 0`

Conclusion:

- the fallback population is **not** caused by small denominators,
- it is caused entirely by **negative or zero GN ratios**.

The fallback entries are also highly structured:

- diagonal fallback fraction: `0.0`
- off-diagonal fallback fraction: `0.04137328`

So the GN fit succeeds on **all diagonal** `Omega_{q,mu,mu}` entries and fails
only on an off-diagonal subset.

The failures are strongly concentrated at `q=0`:

- `q=0` fallback fraction: `0.13441111`
- nonzero-`q` fallback fractions: about `0.025 .. 0.033`

Head correction increases that concentration substantially. Repeating the
extraction without the head correction gives:

- no-head global fallback fraction: `0.03059444`
- no-head `q=0` fallback fraction: `0.03802222`

whereas with the current head correction:

- headed global fallback fraction: `0.04130432`
- headed `q=0` fallback fraction: `0.13441111`

Status:

- `Omega` assignment is not numerically failing because of near-singular fits,
- but the head-corrected `q=0` off-diagonal sector is still a real outlier.

---

### G. Replacing forced `2 Ry` invalid poles with BGW-like staticization

This was implemented directly.

Current code now carries:

- the GN validity mask,
- the true `W^c(0)` matrix,
- and an `invalid_mode` policy in the sigma driver.

With `invalid_mode = static_limit`:

- invalid entries are removed from the dynamic GN windows,
- then reintroduced as a static correlation correction
  \[
  \Sigma_c^{invalid} \approx \Sigma_x[W^c(0)] - \tfrac12 \Sigma_{RI}[W^c(0)].
  \]

Observed on `cohsex_prod`:

- invalid population: `133826/3240000 (4.13%)`
- static invalid correction size: `max|Sigma| = 1.514467e-02 Ry`

Effect on near-gap `sigC_EDFT`:

- `n=22`: `+0.000033 eV`
- `n=23`: `+0.000035 eV`
- `n=24`: `+0.013205 eV`
- `n=25`: `+0.013194 eV`
- `n=26`: `+0.000536 eV`
- `n=27`: `+0.000541 eV`
- `n=28`: `+0.000740 eV`

Conclusion:

- switching from forced-`2 Ry` invalid poles to BGW-like static invalid
  handling is physically more defensible,
- but it moves the near-gap `Sigma_c` by only `10^-3` to `10^-2 eV`,
- so it is **not** the main source of the multi-eV BGW mismatch.

Relevant code:

- [minimax_screening.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/minimax_screening.py)
- [ppm_sigma.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/ppm_sigma.py)
- [gw_jax.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/gw_jax.py)

---

## What Still Looks Wrong

### 1. The dynamic correlation amplitude is still in the wrong regime

For the current `cohsex_prod` run, representative values are:

From [eqp0_noqsym_w.dat](/home/jackm/projects/tests_isdf/cohsex_prod/eqp0_noqsym_w.dat):

- `n=22`: `sigC_EDFT = 4.563721 + 0.000055i eV`
- `n=24`: `sigC_EDFT = 3.993386 + 0.000009i eV`
- `n=26`: `sigC_EDFT = 3.524744 - 0.001671i eV`
- `n=28`: `sigC_EDFT = 3.481913 + 0.001449i eV`

From [sigma_mnk.h5](/home/jackm/projects/tests_isdf/cohsex_prod/sigma_mnk.h5) at `omega = 0`:

- `n=22`: `4.091401 + 0.004936i eV`
- `n=24`: `3.786359 - 0.000024i eV`
- `n=26`: `3.899663 - 0.002541i eV`
- `n=28`: `3.872633 + 0.002210i eV`

The BerkeleyGW reference still has `Cor` values of order:

- `+1.169 eV`
- `+0.398 eV`
- `-1.468 eV`

So even after the invalid-mode fix, our dynamic correlation remains in a very
different regime.

---

### 2. The strongest remaining suspect is the valid-mode spectral-weight side

Since the denominator routing, window algebra, head values, and invalid-mode
policy have now all been checked, the most plausible remaining issue is one of:

- `W^c(0), W^c(i omega_p) -> (B, Omega)` amplitude normalization,
- how the valid-mode GN fit object for `W^c` maps into the sigma formulas,
- a remaining mismatch between our direct-`W^c` pole fit and the effective
  BGW `I_eps = 1 - epsinv` sigma reconstruction,
- or a missing production-style stabilization in the valid-mode dynamic path.

This is upstream of the minimax window algebra.

---

## Core Files

### Main implementation

- [ppm_sigma.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/ppm_sigma.py)
  - GN-PPM extraction from `W^c(0)` and `W^c(i omega_p)`
  - `Sigma_c` windows
  - `tau` accumulation
  - band-space `Re/Im` two-channel path
  - invalid-mode staticization

- [gw_jax.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/gw_jax.py)
  - driver
  - output path
  - `Sigma(E_DFT)` handling
  - static convolution helper reused by GN-PPM sigma

- [minimax_screening.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/minimax_screening.py)
  - canonical minimax wrappers
  - GN-PPM extraction helpers
  - GN validity mask

- [w_isdf.py](/home/jackm/projects/isdf_cohsex/src/gw_isdf/w_isdf.py)
  - minimax `chi -> W` path
  - head-corrected `W` construction

### Theory documents

- [GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md](/home/jackm/projects/isdf_cohsex/docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md)
- [MINIMAX_CTSP_IMPLEMENTATION_REVISED.md](/home/jackm/projects/isdf_cohsex/docs/MINIMAX_CTSP_IMPLEMENTATION_REVISED.md)
- [PHYSICS_COMPREHENSIVE.md](/home/jackm/projects/isdf_cohsex/docs/PHYSICS_COMPREHENSIVE.md)
- [NEW_WINDOW_MINIMAX_GUIDELINES.md](/home/jackm/projects/isdf_cohsex/docs/NEW_WINDOW_MINIMAX_GUIDELINES.md)

### External comparison source

- [/home/jackm/SOURCES/BerkeleyGW/Epsilon](/home/jackm/SOURCES/BerkeleyGW/Epsilon)
- [/home/jackm/SOURCES/BerkeleyGW/Sigma](/home/jackm/SOURCES/BerkeleyGW/Sigma)

---

## Most Useful Runtime Controls

In `cohsex_prod.in` or related inputs:

- `use_ppm_sigma = true`
- `fermi_reference = vbm | midgap`
- `sigma_debug_split_contrib = true`
- `sigma_debug_quadrature = true`
- `sigma_omega_min_ev`, `sigma_omega_max_ev`, `sigma_omega_step_ev`
- `sigma_regularization_ev`
- `ppm_sigma_scale`
- `ppm_sigma_flip_neg`
- `ppm_invalid_mode = static_limit | fixed_2ry`

---

## Recommended Next Audit

The next audit should target the **valid-mode spectral weight side**, not the
denominator side and not the invalid-mode fallback side.

Best next step:

1. build a scalar / single-pole test for the full
   `W^c(0), W^c(i omega_p) -> B, Omega -> W^c(t) -> Sigma_c(omega)` chain,
2. compare its amplitude against the analytic GN-PPM expression,
3. then compare that with the current `Sigma^+` and `Sigma^-` amplitudes in the
   live code and with the BGW `I_eps`-based dynamic reconstruction.

That is the shortest remaining path to the mismatch.
