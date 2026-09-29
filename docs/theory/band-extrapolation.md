# Σ_c band extrapolation

`use_band_extrapolation` estimates the band-converged correlation self-energy
from one Σ pass that is truncated at `number_bands_sigma`. The estimator is
`spectral_shell`, the pooled denominator shell (owner ruling 2026-09-28). The
code, its constants and its refusals are in `gw/band_extrapolation.py`; the
deck keys are in the [input reference](../input_reference.md).

## Where it runs

Only the plasmon-pole Σ stages (`gn_ppm`, `hl_ppm`) consume it. On every other
`compute_mode`, shared pole included, a defaulted-on key disables itself with a
log note and a named key refuses. On a static Coulomb hole the band limit
anti-converges, so the guard is a correctness rule
([decisions](../architecture/decisions.md)).

## Three sums from one pass

The Green's-function band sum is cut into three disjoint brackets at
N₁ < N₂ < N₃ = `number_bands_sigma`. Each bracket is contracted against the
same W(τ), so the cumulative sums S(N₁), S(N₂), S(N₃) come from one Σ pass.
W, the ISDF basis, the quadrature and the evaluation energy (E_DFT) are shared.
Default cuts: 70 %, 85 % and 100 % of the total Σ band count, moved to the
nearest degeneracy-clean boundary (`band_extrapolation_bracket_scheme`).

## The model

A band A above N₁ adds to Σ_c of state i

$$
c_i(A) = a_i \sum_{\mathbf k} w_{\mathbf k}\,
\big(E_{A\mathbf k} - E_i + \Omega\big)^{-\beta},
$$

with one (β, Ω) for the run and one amplitude a_i per state. A high band is a
plane wave of energy E. Its matrix element and W^c each fall as 1/E and the
denominator of the empty-branch remainder as 1/(E + Ω − E_i), so the leading
exponent is β = 3 with a state-independent amplitude, and the state enters
through E_i in the denominator. The search ranges are physical bounds:
β ∈ [2, 8] (above 3/2 the tail is summable on the E ∝ n^{2/3} ladder; 8 is
twice the first state-dependent exponent) in steps of 0.25, and
Ω ∈ [0, 40] eV (a pole energy; twice a solid's valence plasmon) in steps of
2 eV.

With $G_i(l,h) = \sum_{l<A\le h}\sum_{\mathbf k} w_{\mathbf k}
(E_{A\mathbf k}-E_i+\Omega)^{-\beta}$:

1. For each grid point, $a_i = (S_3 - S_1)/G_i(N_1,N_3)$ and the model predicts
   $\hat S_{2,i} = S_1 + a_i G_i(N_1,N_2)$.
2. The grid point minimising $\sum_i (\hat S_{2,i} - S_{2,i})^2$ over the
   pooled states wins.
3. $\hat S_i = S(N_3) + \big(S(N_3)-S(N_1)\big)\,G_i(N_3,N_T)/G_i(N_1,N_3)$.

N_T = min(n_gk)·n_spinor is the complete plane-wave basis. Bands past the
WFN's own are continued by the Weyl ladder E_n = E₀ + C(n + n₀)^{2/3}, fitted
to the DFT eigenvalues only. The coefficients (−r_i, 0, 1 + r_i) are real and
sum to 1; an off-diagonal element uses (r_i + r_j)/2, so the extrapolated Σ
stays Hermitian and is diagonalized after extrapolation.

The pooled states are the QP window's states below every band above N₁
(E_i < min_k E_{N₁+1,k}). A state with E_i − Ω at or above that band has a
pole of the model inside the sum and keeps S(N₃); the log names it.

## Why pooled

The estimator it replaced solved one β per state from the ratio of the two top
shell increments. On narrow top shells that ratio is band texture, and a
per-state β amplifies it. Pooling fixes the shape from every requested state
and leaves each state one amplitude, which the widest shell determines.

## Cost

Every shell sum is evaluated on a composite Gauss compression of the shell's
spectrum in log(E − E_ref), exact to about 1e-16 relative. The 525-point grid
and the tail cost about 10 ms on Si 4³ (71 pooled states) and 0.08 s for 1200
states with a 152 012-band tail.

## Measured

Si 4³, 25 Ry, scalar, shared-pole W, complete basis (536 bands) as the truth;
71 degeneracy-closed states within ±10 eV of midgap. G truncated at N; W at
536 bands (case a) or at N (case c). Std over the states / max 4v4c difference
error / median 4v4c direct error, meV (sandbox BANDEX study, claims 2898 and
2900; reproduced through this code to 1e-12 meV, sandbox run DEV/602).

| N | cuts | case a | case c |
|---|---|---|---|
| 78 | 50, 64, 78 (default) | 9.2 / 32.9 / 8.7 | 20.5 / 81.0 / 17.9 |
| 78 | 64, 72, 78 | 6.5 / 33.6 / 6.0 | 16.9 / 69.0 / 19.7 |
| 78 | no extrapolation | 29.2 / 90.5 / 35.8 | 34.6 / 121.8 / 38.0 |
| 50 | 20, 34, 50 | 23.2 / 81.9 / 23.0 | 36.8 / 151.7 / 30.7 |
| 34 | 14, 20, 34 | 49.0 / 168.0 / 52.8 | 53.6 / 179.3 / 57.8 |

The per-state form at the old default cuts (64, 72, 78) scored
109 / 278 / 113 meV in case a. The pooled form moves by at most 3 meV std
across the placements measured at 78 bands. The gap error (−19 to +19 meV at
78 bands, case a) is not controlled by the fit. One material: Si. At 34 and 50
bands the default cuts fall inside multiplets on this spectrum and were not
among the stored samples.
