# Σ_c band extrapolation

`use_band_extrapolation` estimates the band-converged correlation self-energy
from one Σ pass that is truncated at `number_bands_sigma`. The estimator is
`spectral_shell`, the pooled denominator shell (owner ruling 2026-09-28). The
code, its constants and its refusals are in `gw/band_extrapolation.py`; the
deck keys are in the [input reference](../input_reference.md).

## Where it runs

The plasmon-pole stages (`gn_ppm`, `hl_ppm`) and the scalar `mpa` stage
(shared pole or MPA fit) consume it: all three run the same pole-sum Σ
executor, which splits the Green band sum into brackets. A static stage and a
bispinor `mpa` stage (a sum of four-current sector bodies, no bracket axis) do
not: a defaulted-on key disables itself with a log note and a named key
refuses. On a static Coulomb hole the band limit anti-converges, so that guard
is a correctness rule ([decisions](../architecture/decisions.md)).

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

The pooled states are those within ±10 eV of E_F, closed over degenerate
multiplets, that lie below every band above N₁ (E_i < min_k E_{N₁+1,k}).
Semicore states in a wide QP window therefore do not set (β, Ω); every
QP-window state in the model's domain still gets its tail. A state with
E_i − Ω at or above that band has a pole of the model inside the sum and
keeps S(N₃); the log names it.

## Why pooled

A β solved per state from the ratio of the two top shell increments reads
band texture on narrow top shells and amplifies it. Pooling fixes the shape from every requested state
and leaves each state one amplitude, which the widest shell determines.

## Cost

Every shell sum is evaluated on a composite Gauss compression of the shell's
spectrum in log(E − E_ref), exact to about 1e-16 relative. The cost is linear
in the pooled states: about 10 ms for the 71 Si 4³ states of the rescore,
0.038 s for 896 pooled states in a production GN-PPM run (compute node), and
0.08 s for 1200 states with a 152 012-band tail (login node, one thread).

## Measured

End to end, Si 4³, 25 Ry, scalar, complete-basis WFN, 78 against 536 bands
with the same ISDF basis and ε; std over the ±10 eV states / max 4v4c /
median 4v4c / gap error, meV. χ₀ is at 78 bands in both arms, so W keeps its
own band truncation.

| route | no extrapolation | pooled |
|---|---|---|
| shared pole, one-shot (claim 2948) | 35.6 / 125.1 / 37.0 / −120.4 | 21.2 / 83.7 / 24.3 / −72.8 (β 4.75, Ω 36 eV) |
| GN-PPM, head off (claim 2942) | 34.4 / 107.1 / 36.8 / −96.4 | 19.9 / 72.0 / 20.5 / −53.3 (β 4.75, Ω 38 eV) |

The shared-pole legs ran with a study Gram-validity dial (the stock gate
refuses 78 bands on 2532 centroids). The extrapolation covers Σ's G sum only:
most of the remainder is W's band truncation, which the study below
separates (case c against case a). The error budget that carries both is
[production QSGW](../how-to/production-qsgw.md#error-budget).

**Study (shared-pole W, stored samples).** Si 4³, 25 Ry, scalar, complete
basis (536 bands) as the truth;
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

The pooled form's std moves by at most
2.7 meV (case a) and 3.6 meV (case c) across the placements measured at 78
bands.

The fitted Ω sits at 36–38 eV, near its 40 eV bound, on 5 of the 6 distinct
placements at 78 bands (the exception is (34, 50, 78), case a, Ω = 12 eV).
The residual is flat along a ridge in (β, Ω). With the bound at 80 eV the
fit moves along it (Ω 44–80 eV, β up to 6.75) and the scores change by
−1.3 to +2.2 meV std at 78 bands, mostly for the worse (default cuts: 9.2 →
9.3 meV in case a, 20.5 → 21.9 meV in case c); at 34 and 50 bands by 0 to
+3.5 meV. The 40 eV bound is kept. The gap error (−19 to +19 meV at
78 bands, case a) is not controlled by the fit. One material: Si. At 34 and 50
bands the default cuts fall inside multiplets on this spectrum and were not
among the stored samples.
