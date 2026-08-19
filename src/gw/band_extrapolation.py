"""Band-convergence extrapolation of the correlation self-energy Σ_c.

TWO ESTIMATORS, ONE SET OF THREE POINTS.  Both read the SAME three cumulative
bracket sums S(N₁), S(N₂), S(N₃) that the τ kernel already produces in one
pass; they differ only in what they do with them.  The deck selects with
``band_extrapolation_estimator``:

  ``spectral_shell``   **the default since 2026-08-17.**  A spectrum-resolved
      SHELL estimator: solve ONE exponent per external state from the ratio of
      the two observed shell increments against spectral moments of the real
      DFT eigenvalues, then integrate the remaining tail out to the finite
      plane-wave basis.  ``SPECTRAL SHELL ESTIMATOR`` below.
  ``band_index_only``  the incumbent two-parameter ``S_∞ + A/N`` least
      squares.  Kept and selectable, unchanged bit-for-bit; renamed because
      "1/N" names the only thing it looks at — the band INDEX — and the
      measurement below is that the band index alone is not enough.

Everything from ``WHAT CONVERGES SLOWLY`` to ``BAND COUNTS, NOT AN ENERGY
CUTOFF`` describes machinery both estimators share (the brackets, what is held
fixed across the three points, the counting law, the mode gate) or the
``band_index_only`` fit and its own history.  Read the two estimator sections
as the fork.

WHAT CONVERGES SLOWLY.  Σ_c's intermediate-state sum runs over every band in
the Green's function,

    Σ_c(ω) = Σ_n  ψ_n ⊗ ψ_n*  ⊗  [W-dependent kernel](ω − E_n),

and its unoccupied tail decays slowly: the states above the QP window
contribute a small, same-signed, slowly-vanishing amount that a brute-force
band count only removes by being enormous.  This module evaluates the SAME
sum at three band counts in ONE pass and extrapolates instead.

⚠ **THE 1/N LAW THIS MODULE FITS IS NOT THE MEASURED DECAY, AND THE ERROR
THAT CAUSES IS ONE-SIGNED.  Measured 2026-08-16 — see
``sandbox:reports/ch_converge_functional_forms_2026-08-16/``.**  On a 508-band
Si arm the local convergence power climbs monotonically 1.28 → 2.29 across
N = 100 → 300 and never approaches 1.  The kinematics are fine (the band
ladder is free-electron to 1 %, and the counting law below reproduces the
low-N behaviour); the drift is entirely in the matrix elements, whose
effective falloff ``|M|² ~ E^-a`` runs a = 1.83 at N = 60 to 3.94 at N = 380.
Because the data decays FASTER than 1/N over the window sampled here, a 1/N
fit projects more remaining tail than exists and lands BELOW the truth on
every state at every band count.  Scored against a MEASURED S(508):
45.8 meV median at N_max = 152, 17.5 at 260, 3.5 at 396, always undershooting.

The form is kept for now because it is what has been validated end to end and
the replacement is not yet implemented; it is documented here so that no
reader takes "decays like 1/N" as established.  The screened-Coulomb cutoff
was tested as the cause and REFUTED (tripling W's G-space moves the exponent
by ≤ 0.001).

THE THREE POINTS COME FROM DISJOINT BRACKETS, NOT THREE RUNS.  Under the
default ``band_extrapolation_bracket_scheme = total_fractions``, the band axis
is cut into three contiguous brackets

    bracket 0   [0, N₁)          everything up to 80 % of the TOTAL band count
    bracket 1   [N₁, N₂)         80 % → 90 %
    bracket 2   [N₂, N₃)         90 % → 100 %

and the τ kernel builds one G(τ) per bracket, contracting each against the
SAME, singly-computed W(τ).  Because the brackets PARTITION the band sum, a
cumulative sum along the bracket axis is the sum at each cut:

    S(N₁) = b₀,   S(N₂) = b₀ + b₁,   S(N₃) = b₀ + b₁ + b₂,

and S(N₃) is — to floating-point associativity — the ordinary full-band Σ_c.
Nothing is computed twice and no band is dropped, which is the property
``tests/test_band_extrapolation.py::test_brackets_partition_the_band_sum``
exists to defend.

WHAT MUST BE HELD FIXED ACROSS THE THREE POINTS, or the fit measures
something other than band convergence:

  * **W is built once.**  χ₀/W and the PPM pole fit happen before any
    bracket exists; only the A-side (Green's function) band range varies.
  * **One ISDF representation.**  Centroids and ζ are fitted once at the
    largest band range; the smaller points are obtained by RESTRICTING the
    band index, never by refitting — a regenerated ISDF basis would mix
    basis error into what is meant to be pure band-sum error.
  * **One quadrature.**  The minimax windows, their τ nodes, E_ref_A/E_ref_B
    and the ω grid are built from the FULL band range and shared verbatim;
    the bracket enters only as a band-index restriction inside the kernel.
  * **One evaluation energy.**  All three points are read off the same ω
    grid at the same E_nk.
  * **Σ_c only.**  Σ_x is a bare-exchange sum over OCCUPIED states; it has no
    slow unoccupied tail and is not extrapolated.

WHERE THE POINTS GO, AND WHY SMOOTH IS NOT ASYMPTOTIC.  This is the one thing
about the feature that is not obvious, and it was measured rather than argued
— see ``BRACKET_FRACTIONS`` for the table.  There are TWO band scales, an
order of magnitude apart, and the fractions have to clear the second:

  * The band sum stops resolving individual states — bumpy becomes smooth —
    at ``n_cond ≈ n_val``.  From there the 1/N model already fits with
    R² ≥ 0.998.
  * The fitted INTERCEPT is not accurate until ``N ≈ 0.8·N_max``.  The
    running exact two-point intercept is still 30–50 meV from its limit at
    71 % of the range and settles inside ±10 meV only above 94 %.

A fit can therefore have R² = 0.9994 and an intercept 47 meV wrong: the
residual is set by how well the model tracks a several-hundred-meV rise,
while the intercept error is set by a small smooth curvature that every point
in a low window shares.  **No residual, R² or scatter statistic can see it**
— which is also why ``Δ_model`` alone is not a sufficient diagnostic and why
:class:`ExtrapolationFit` carries the two SIGNED numbers below beside it.

DEFAULT FRACTIONS ARE OF THE TOTAL BAND COUNT, NOT OF THE CONDUCTION COUNT.
The incumbent model is written in ``N_eff``, and the free-electron counting
law that makes 1/N the right variable is written in the total count measured
from the bottom of the band manifold.  Measured on the Si 4×4×4 SOC deck's
own eigenvalues:

    N_total vs (Ē_N − E_bandbottom)   →  p = 1.481,  R² = 0.9988   (3/2 ✓)
    N_cond  vs (Ē_N − E_CBM)          →  p = 1.212,  R² = 0.9979   (✗)

Only the total count obeys N ∝ (E − E₀)^{3/2}.  The fit's lever arm is
1/N₁ − 1/N₃, which depends on the RATIO N₁/N₃ — again a fraction of the
total.  On a deck with a small occupied manifold the two parametrisations
differ by ~2 % and nothing distinguishes them; on one with a large occupied
manifold ``n_occ + 0.8·n_cond`` sits far below ``0.8·N_max`` and they diverge
badly.

AN EXPLICIT CONDUCTION-COORDINATE GEOMETRY IS ALSO AVAILABLE.  The named
``conduction_energy_midpoint`` scheme places N1 at half of the conduction
bands included in the Sigma sum, snaps it to a multiplet-clean boundary, then
places N2 at the clean rectangular boundary nearest halfway between N1 and N3
in ``mean_k E[k,N-1]``.  It does NOT apply a global E_ck mask: every k keeps
the same static band count.  This changes only cut placement; both estimators
still use absolute band indices and the full DFT energy ladder.

It is deliberately NOT the default.  On the measured Si GN-PPM curve its
indirect-gap spread was 12.29 meV over N3=180..396 and 4.18 meV over
N3=276..440, but 82.96 meV over N3=140..296; the incumbent total-80/90
geometry also remained more accurate per state on that control.  One material
does not establish a universal band threshold.  Existing decks therefore
retain their numerical meaning, while production studies can name the new
scheme and get its resolved cuts in the startup log and HDF5 provenance.

THE FIT HAS EXACTLY TWO FREE PARAMETERS.

    S(N) = S_∞ + A / N_eff,        N_eff = N_occ + N_c = N

fitted by ordinary least squares in 1/N over the three points.  Two
parameters, and the limit point is 1/N → 0.

**THE THREE-PARAMETER FORMS WERE TESTED AGAINST THE DENSE CURVE AND
REJECTED.  Do not re-litigate this without new data.**  Against BerkeleyGW's
``ch_converge.dat`` on the Si 4×4×4 SOC deck at 124 bands, 192 (k, band)
states, MAE / max error in meV against BGW's own band-converged CH:

    3 points, 2 par, (0.80, 0.90)          14.2 /  26.5     noise ×6.6
    3 points, 3 par S_∞+A/N+B/N², wide     18.0 /  69.2     noise ×4.7
    3 points, 3 par, top-weighted          57.5 / 216.9     noise ×112.3
    4 points, 3 par, wide                  15.5 /  86.5     noise ×4.0

**THE SHIFTED LAW ``S_∞ + A/(N + N₀)`` WAS ALSO TESTED, WITH N₀ FITTED ON
THE DENSE CURVE AND THEN HELD FIXED** — the move that makes it two
parameters in production and answers "is there a functional form for the
crossover from non-1/N to 1/N".  **There is not, on this deck.**  N₀ comes
out SMALL and NEGATIVE with a large spread and a strong window dependence
(median −1.5 fitted over [16, 124], drifting to −6.0 over [40, 124], p25/p75
spanning −7.5…+2), and it does **not** correspond to ``n_val = 8``, to the
measured bumpy→smooth transition at ``n_cond ≈ 7–10``, or to anything else
physical.  It is a fitted nuisance, not a crossover scale.

Held fixed at the calibrated value it buys ~15–25 % at LOW fractions and
essentially nothing where the sampling actually sits: at nband 124,
(0.50, 0.75) improves 35.3 → 27.6 meV MAE, but (0.80, 0.90) only 14.2 →
13.7 and its MAX gets WORSE, 26.5 → 31.4.  **And it does not buy the ability
to sample lower, which was the whole reason to want it**: (0.50, 0.75) with
N₀ fitted in-sample on this very deck is 27.6 meV, still 2× worse than
(0.80, 0.90) with N₀ = 0 at 14.2.  Moving the window beats modelling the
crossover.

**THE NEW DATA THE INJUNCTION ABOVE ASKS FOR NOW EXISTS (2026-08-16), AND IT
CHANGES TWO OF THESE CONCLUSIONS.**  A 508-band Si arm gives a MEASURED
S(508), so estimators can be scored held-out instead of against a reference
that is itself a fit or a model.  Under that protocol:

  * The shifted law was re-searched over ``N₀`` AND the exponent JOINTLY,
    rather than with the exponent pinned to 1.  ``S_∞ + A/(N + 50)²`` cuts
    the sliding-window intercept travel 42.7 → 3.9 meV and transfers
    out-of-sample to a different SCF, cutoff and band count (travel 4.8
    against 36.4 for 1/N).  It is still NOT shippable — ``N₀ ≈ +50`` has no
    counterpart in the mean field (the DFT bands give n₀ = 0), and (N₀, q)
    drifts to ~(110, 3.0) over the extended range — so it is a shape
    diagnostic, not a form.  But "there is no crossover form on this deck" is
    too strong as written above; what is true is that no FIXED one describes
    N = 100…500.
  * A FOURTH point does not help this estimator.  Held-out against S(508),
    4 points at (0.70, 0.80, 0.90) are WORSE than 3 at (0.80, 0.90) at every
    band count (47.8 vs 45.8 at N_max = 152, 20.4 vs 17.5 at 260): the extra
    sample extends the lever into the region where the form is more wrong,
    and that costs more than the conditioning gain (noise ×3.5 vs ×5.9) wins.
    The defensible reason to want a fourth point is a held-out prediction of
    the fourth shell as an uncertainty estimate — not accuracy.

The replacement is a spectrum-resolved SHELL estimator: fit the per-state
decay from the ratio of the two observed shell increments against spectral
moments of the real DFT eigenvalues, then integrate the remaining tail to the
finite basis endpoint.  Held-out against S(508) it beats this module's
estimator at every band count, and it needs exactly the three points this
module already computes.  **It is implemented here now (2026-08-17) and is the
DEFAULT**; see ``SPECTRAL SHELL ESTIMATOR`` below.

A CALIBRATED BIAS CORRECTION was tested too, and the result is the argument
for leaving the centre alone.  Scaling the applied correction by an optimal
``g`` gives g = 0.788 at (nband 76, 0.50/0.75) rising to **g = 0.9885 at the
shipped (nband 124, 0.80/0.90)**, where the in-sample gain is 14.2 → 13.7
meV — nothing.  At the recommended fractions the estimator is already
unbiased; there is no centre left to tune, only an honest bar to report.

The three-parameter form absorbs the curvature and does help at low
``nband``, but (a) its MAX error is no better, and the worst states are what
a QP window is judged on; (b) it is catastrophically ill-conditioned at
exactly the top-weighted spacings the two-parameter form wants — 112× noise
amplification, against 6.6× — so the two changes fight each other; (c)
through three points it is an INTERPOLANT: zero residual by construction, no
third-point check, ``Δ_model`` undefined.  The one respectable variant is a
FOURTH bracket with a 3-parameter least squares, and it is beaten by three
points at (0.80, 0.90) while costing another G(τ) build.  ("noise ×" is the
2-norm of the row of the design-matrix pseudo-inverse that produces S_∞: the
factor by which an INDEPENDENT per-point error reaches the intercept.  A
common-mode error passes through 1:1 for every estimator.)

THE DIAGNOSTICS ARE THE POINT — BUT THEY SEE DIFFERENT THINGS.  A
two-parameter fit through three points always returns a number; whether that
number means anything is decided by whether the three points are already in
the asymptotic 1/N regime.  The exact two-point intercepts

    S_∞^(ij) = (N_j·S_j − N_i·S_i) / (N_j − N_i)

are free (they are the closed-form solution of the same model on a pair) and
if the model held, all three would coincide.  Three numbers are reported and
they are NOT interchangeable:

``Δ_model`` — the pairwise intercepts' spread about the fit.  A measure of
    SCATTER.  It sees preasymptotic curvature that makes the three points
    disagree; **it cannot see a bias the three points share.**  Measured on
    a clean BerkeleyGW curve at counts (52, 76, 100): ``Δ_model`` median
    18.8 meV, ratio to ``Δ_tail`` 0.057, verdict ``consistent`` on 100 % of
    768 state-fits with ZERO sign reversals — while the true intercept error
    was 55 meV MAE and 167 meV max.  So a passing ratio is necessary and not
    sufficient, and a ``NOT TRUSTWORTHY`` verdict on a real deck is evidence
    about the Σ_c being fitted, not about the sampling.

``pair_split`` = S_∞^(1,2) − S_∞^(2,3) — the same information UNSIGNED in
    ``Δ_model``, kept SIGNED.  Its magnitude bounds the residual curvature
    and its sign says which way the bias runs, which the absolute value
    throws away.  8.2–52.1 meV at the old fractions, 2–20 meV at these.

``a_over_n_last`` = A / N₃ — how much correction the model is still applying
    AT the largest computed point.  A one-sided number: it is large exactly
    when the sum has not finished, and unlike ``Δ_model`` it does not go
    quiet when all three points share the same error.

``Δ_tail`` — how far the extrapolation moved the largest computed point.
    The size of the correction, not its error.

SPECTRAL SHELL ESTIMATOR — THE DEFAULT SINCE 2026-08-17.  The band index is
the wrong variable and the section above says so with a measurement: the local
convergence power climbs 1.28 → 2.29 over N = 100 → 300 because the MATRIX
ELEMENTS fall off, and their falloff is a property of the ENERGY the
intermediate state sits at, not of its position in a list.  So use the
energies.  Nothing else about the run changes — the same three brackets, the
same one pass, the same three numbers.

Form the two SHELL increments actually observed,

    D₂ = S(N₂) − S(N₁),        D₃ = S(N₃) − S(N₂),

and assume the per-intermediate-state contribution is locally
``c(E) ~ A·(E − E₀)^(−β)`` with an amplitude A and an exponent β that are
constant across the top of the computed range and out into the tail.  Build
spectral moments of the DFT eigenvalues over each shell and over the tail,

    I_j(β)    = Σ_{i ∈ (N_{j-1}, N_j]}  w_i · ((ε_i − E₀)/E*)^(−β)
    I_tail(β) = Σ_{i = N₃+1}^{N_T}      w_i · ((ε_i − E₀)/E*)^(−β)

with ``w_i`` the k-point weights and ``ε_i`` the DFT eigenvalues.  Then

    D₃/D₂ = I₃(β)/I₂(β)

is ONE scalar equation in ONE unknown per external state, solved by bracketed
root find on ``log I₃(β) − log I₂(β) − log|D₃/D₂|``, and

    Ŝ = S(N₃) + D₃ · I_tail(β)/I₃(β).

**A AND THE INTERCEPT ARE ELIMINATED ANALYTICALLY.**  A cancels in the ratio
that determines β and again in the ratio that applies it; the intercept never
appears because the estimator adds up the REMAINING tail rather than fitting a
limit.  There is no nonlinear fit anywhere in this estimator and none is to be
added — the whole reason it works at three points is that it has one unknown.
``E*`` is an arbitrary conditioning scale: it enters every moment as the same
multiplicative ``E*^β`` and cancels from both ratios exactly, so it can be any
positive energy and is chosen only to keep the powers off the exponent rails.

**β IS PER-STATE AND IS NEVER POOLED.  Owner ruling, not negotiable.**  The
point of the estimator is that valence and conduction states genuinely
converge differently; a pooled β averages that away and returns the estimator
to a one-size-fits-all correction, which is the defect being fixed.  There is
no pooling option, and pooling is not to be benchmarked back in.

**E₀ AND THE BAND LADDER COME FROM THE DFT EIGENVALUES ONLY, NEVER FROM THE
THREE Σ VALUES.**  E₀ is the bottom of the band manifold, obtained by fitting
the free-electron (Weyl) form ``E_n = E₀ + C·(n + n₀)^(2/3)`` to the k-mean of
the DFT ladder.  On the Si 50 Ry deck that fits with n₀ = 0 and R² = 0.99975.
Letting any part of the ladder be informed by S(N₁..₃) would make the
estimator a fit of the data to itself.

**N_T, THE TARGET ENDPOINT, IS THE FINITE BASIS — NOT INFINITY.**  ``S(∞)``
corresponds to no physical quantity: the band sum is EXACTLY complete at the
plane-wave basis's own dimension, ``N_PW = ngk · nspinor`` read from the WFN,
and there is nothing beyond it to converge to.  ``ngk`` varies by k-point
(1604…1639 on the Si deck), and the default uses the MINIMUM — the band count
at which no k-point is still short.  Where eigenvalues are needed past the
WFN's own band count, and they always are (512 bands against N_PW = 3208 on
that deck), the ladder is continued with the SAME Weyl form.  That
continuation is used ONLY to extend the eigenvalue SEQUENCE; no self-energy,
no matrix element and no exponent is ever taken from it.

**FAILURE IS A NAMED REFUSAL, NEVER A CLIP.**  Three signatures, all per
state, all refusals: (a) ``D₂`` and ``D₃`` of opposite sign or either exactly
zero — the shells do not agree that there is a tail, so there is no ratio to
invert; (b) no bracketed root of the β equation in
:data:`SHELL_EXPONENT_BRACKET`; (c) a root pressed against a bracket edge,
which is the clip a naive implementation would return as a value.  A failed
state gets ``NaN`` and a named reason, never a substituted number, and the
driver turns any failure inside the Σ band sum into a
:class:`SpectralShellExtrapolationFailed` naming every failed state.  The
alternative — falling back to ``S(N₃)`` or to the 1/N fit on the failed states
— was rejected: it would ship a Σ built from two different estimators with
nothing in the artifact saying which states came from which.

MEASURED, HELD OUT, AGAINST A NUMBER BERKELEYGW COMPUTED.  The 508-band Si arm
(50 Ry, ``sandbox:reports/band_tail_exponent_50ry_2026-08-16/``) gives a
MEASURED ``S(508)``, so both estimators can be scored on data neither fitted.
Median |error| over the 28 Fermi-window states, in meV:

    N_max      band_index_only      spectral_shell
    -------------------------------------------------
    152             45.8                  4.7
    204             29.7                 14.7
    260             17.5                 12.5
    296             12.6                  0.7
    396              3.5                  0.0

β came out 3.4–5.3 across those rungs, independently consistent with ``a + 1``
from the dense increment fit (a = 1.83 → 3.94), which is the check that it is
measuring the matrix-element falloff and not absorbing an arbitrary fit
parameter.  **THE ERRORS ARE NON-MONOTONE IN N_max, AND THAT IS THE METHOD'S
OWN BEHAVIOUR** — 4.7 at 152 against 14.7 at 204 is not a porting bug and is
not to be tuned away.  The estimator is a two-shell local fit; which pair of
shells it gets, and how well the local power law describes them, is not a
monotone function of where the top of the range sits.

WHAT THE ESTIMATOR IS APPLIED TO, AND WHY THAT NEEDED A RULING.  The fit is
per external state, so its three combination coefficients ``[0, −r, 1 + r]``
carry a ``(nk, nb)`` shape rather than being three scalars.  The Σ object that
drives the iteration is the full ``(nω, nk, nb, nb)`` cube, whose element
``(i, j)`` has TWO external states.  The coefficient applied there is the mean
of the two states' own, ``r_ij = ½(r_i + r_j)``.  That choice is forced, not
free: it is the unique symmetric rule that is exact on the diagonal (where the
estimator is defined and measured) and that keeps ``Σ_b c_b = 1``, and
symmetry is what makes the extrapolated Σ exactly Hermitian — the same
argument :func:`extrapolation_weights` makes for the scalar case, which is
required for "extrapolate Σ, then diagonalize" to yield a legitimate static
self-energy.  A per-row rule would produce a non-Hermitian Σ and a spectrum
belonging to no Hamiltonian.

GN/HL-PPM ONLY IS A CORRECTNESS GUARD, NOT A SCOPE LIMITATION.  The refusal
in ``gw.sigma_dispatch`` on a non-PPM ``compute_mode`` reads like "we only
wired it up for PPM".  It is stronger than that: **the 1/N → 0 limit point
is itself mode-dependent, and it is wrong for a static Coulomb hole.**
Measured against BerkeleyGW's EXACT static CH (the closure sum — no band
sum, no extrapolation in it), same deck, same estimator, MAE in meV:

    nband                         60      76     100     124
    static COHSEX, 1/N → 0      94.9    96.6   202.8   288.2    ← WORSE
    GN-PPM,        1/N → 0     171.3    97.4    55.1    32.8    ← better

The COHSEX arm ANTI-CONVERGES: more bands determine the line better and
drive it more confidently past the right answer, overshooting the exact
value by ~340 meV.  The GPP self-energy's high-energy intermediate states
are suppressed by the pole denominator, so its band sum genuinely exhausts
itself inside the 1/N regime; the static Coulomb hole has no such
suppression and its tail keeps contributing past where the 1/N law was
calibrated.  Routing this feature at a static mode would produce a number
that looks converged, carries a ``consistent`` verdict, and gets worse the
more you spend on it.

BAND COUNTS, NOT AN ENERGY CUTOFF.  The cut is a number of bands because
JAX's shapes are static: a band count is a compile-time slice, an energy
cutoff is not.  ``mean_energy_ev`` is reported per cut for readers who want
a cutoff-like number, and is a REPORTED quantity only — nothing keys off it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.band_degeneracy import (
    DEGENERACY_TOL_RY,
    BandWindowDegeneracyError,
    boundary_min_gaps,
    snap_cut_to_clean_boundary,
)
from common.units import RYD_TO_EV


#: The two interior sampling fractions of the TOTAL band count ``N_max``.
#: The third point is always the full range, so it needs no fraction.
#:
#: MEASURED, 2026-08-15, against BerkeleyGW's dense ``ch_converge.dat`` on the
#: Si 4×4×4 SYM/SOC deck (25 Ry, 8 irreducible k, n_occ = 8) at the largest
#: degeneracy-legal band count it supports, 124.  The score is against BGW's
#: OWN band-converged GN-PPM Coulomb hole — ``sigma_hp.log`` col 6 from a
#: ``frequency_dependence 3`` + ``exact_static_ch 1`` arm, i.e. partial sum
#: plus static remainder, an independent quantity with no extrapolation in it
#: — over all 192 (k, band) states.  MAE / max, meV:
#:
#:     fractions        nband 76      nband 100     nband 124
#:     ------------------------------------------------------------
#:     raw, no extrap  470.2 / 813   350.8 / 616   281.4 / 500
#:     (0.50, 0.75)     97.4 / 353    55.1 / 167    32.8 /  89.7   ← was
#:     (0.60, 0.80)     73.4 / 221    44.8 / 122    21.5 /  63.2
#:     (0.70, 0.85)     73.3 / 234    39.3 / 115    17.3 /  41.0
#:     (0.80, 0.90)     73.4 / 179    31.8 /  82.4  14.2 /  26.5   ← is
#:     dense LSQ over [0.8·N_max, N_max], every band:  14.1 /  26.8
#:
#: Two things that decided it.  The gain is 1.7× in MAE and 3.4× in max error
#: at nband 124 for ZERO extra compute — same three brackets, same one pass,
#: same fit.  And at (0.80, 0.90) the three-point estimate is already AT the
#: floor: it matches what a least squares over every band in [100, 124]
#: achieves, so nothing is left on the table from better sampling and the
#: residual ~14 meV is the 1/N model's own error against BerkeleyGW.
#:
#: Why not higher: the gain saturates ((0.85, 0.925) measured 12.2 meV MAE,
#: no better than 0.80/0.90 within the reference's own spread) while the
#: lever arm 1/N₁ − 1/N₃ shrinks and the noise amplification rises 6.6 → 10.2.
#:
#: Why not fractions of ``n_cond``: see the module docstring — the counting
#: law that makes 1/N right is written in the TOTAL count, p = 1.481 measured,
#: against 1.212 for the conduction-only alternative.
#:
#: ⚠ THE REFERENCE THESE FRACTIONS WERE CHOSEN AGAINST IS A MODEL, NOT A
#: MEASUREMENT.  "partial sum plus static remainder ... no extrapolation in
#: it" is true about extrapolation and misleading about bias: BerkeleyGW's
#: static remainder assumes the neglected GPP tail is 1/2 the static one, and
#: the measured ratio on this deck runs 2.26 at N = 60 to 3.34 at N = 124,
#: rising.  Scored against a dense fit instead, that oracle is +14.0 meV mean
#: / +15.3 median off over 192 states, ONE-SIGNED — 3-5x the sampling error
#: these fractions were selected to minimise.  So the RANKING above is
#: probably sound (the bias is common-mode across rows) but the absolute
#: numbers are not, and (0.80, 0.90) has never been re-derived against a
#: reference-free target.  A 508-band arm now provides one — a MEASURED
#: S(508) — and re-deriving the fractions under it is open work.
#: See ``sandbox:reports/ch_converge_functional_forms_2026-08-16/``.
#:
#: Report: ``sandbox:reports/ch_converge_band_extrapolation_2026-08-15/``,
#: which carries the three ``ch_converge.dat`` arms and the analysis scripts.
BRACKET_FRACTIONS: tuple[float, float] = (0.80, 0.90)

#: Named geometries for the three cumulative band sums.  The incumbent stays
#: the default: changing the meaning of existing decks would invalidate their
#: convergence history, and the conduction-coordinate arm has only been
#: measured on one Si curve.  ``conduction_energy_midpoint`` is therefore an
#: explicit production option, not an inferred replacement.
BRACKET_SCHEMES: tuple[str, str] = (
    "total_fractions",
    "conduction_energy_midpoint",
)
BRACKET_SCHEME_DEFAULT: str = "total_fractions"

#: First point of the conduction-coordinate geometry.  The middle point is
#: not another fraction: it is chosen halfway in k-mean DFT energy between
#: this snapped boundary and N3.
CONDUCTION_HALF_FRACTION: float = 0.50


#: Extrapolation uncertainty, as a fraction of the correction actually applied
#: (``Delta_tail``).  ``(p90, p99)``.
#:
#: A PER-STATE ERROR BAR IS NOT AVAILABLE AND THIS IS NOT ONE.  Regressing the
#: true error on every per-state number the fit carries — ``A/N3``,
#: ``Delta_model``, ``pair_split``, ``Delta_tail`` — gives R² <= 0 at the
#: shipped fractions (measured: A/N3 R² = -0.213, Delta_model -0.178,
#: pair_split -0.237 at nband 124 / (0.80, 0.90)), and a bar calibrated from
#: A/N3 covers 13.5 % of states at 1σ and 41.7 % at 3σ.  The reason is that at
#: these fractions the residual error is no longer the smooth 1/N² bias those
#: numbers track; it is per-state band-structure texture, which is invisible
#: to all four.
#:
#: What IS stable is the error as a FRACTION OF THE CORRECTION.  Over 16
#: (nband x fractions) configurations on the Si 4x4x4 SOC deck, scored against
#: BerkeleyGW's band-converged GN-PPM CH, |error| / Delta_tail:
#:
#:     configuration                    median     p90     max
#:     nband 124, (0.80, 0.90)   ←ship    4.31%  14.44%  22.46%
#:     nband 100, (0.80, 0.90)            9.08%  17.55%  23.14%
#:     nband 124, (0.50, 0.75)           11.11%  23.17%  30.23%
#:     worst of all 16 configurations    18.04%  35.47%  45.03%
#:
#: The p90 ratio moves by only 2.5x across every configuration measured and
#: shrinks monotonically as the sampling moves up and nband rises, so 0.15 is
#: a p90 bar at the shipped configuration and a conservative one below it.
#:
#: THIS IS THE EXTRAPOLATION UNCERTAINTY ONLY.  It does not include, and must
#: not be added to, the difference from BerkeleyGW: the reference it was
#: calibrated against is BGW's static remainder, which for a GPP self-energy
#: is itself a construction (the factor-of-1/2 form, exact only for the static
#: Coulomb hole).  Nor does it cover the ISDF basis, the W-side band count
#: (~100 meV at the band edges on this deck, and NOT extrapolated by this
#: feature), or anything else in the run.  One deck, one system: treat the
#: coefficients as calibrated rather than universal.
TAIL_UNCERTAINTY_FRACTION: tuple[float, float] = (0.15, 0.25)


class BandExtrapolationRefused(ValueError):
    """``sigma_band_extrapolation`` was requested but cannot be honored.

    Raised by name rather than silently disabling the feature: a run that
    quietly did not extrapolate would report a converged-looking Σ with no
    indication that the thing being tested never happened.
    """


@dataclass(frozen=True)
class BandBracketPlan:
    """Which band ranges the τ kernel sums, and what each cumulative cut means.

    Attributes
    ----------
    bounds : tuple[tuple[int, int], ...]
        Half-open ``(lo, hi)`` band-index brackets, contiguous and covering
        ``[0, nb_padded)`` exactly.  Length 1 in the ordinary case.
    counts : tuple[int, ...]
        ``N_i`` — the LOGICAL band count reached after cumulating brackets
        ``0..i``.  Excludes the mesh pad bands, whose ψ is exactly zero and
        which therefore add nothing to any sum (``common/meta.py``).
    requested : tuple[int, ...]
        The unsnapped targets, for the log line.  Same length as ``counts``.
    n_occ : int
        Occupied bands in the Σ band sum (the ``N_occ`` of ``N_eff``).
    n_cond : int
        Unoccupied bands in the Σ band sum.
    mean_energy_ev : tuple[float, ...]
        Mean band energy over ``[0, N_i)``, in eV relative to nothing in
        particular — a cutoff-flavoured REPORTING number, per the module
        docstring.  Never consumed.
    bracket_scheme : str
        Named rule that chose the interior cuts.  Stored in the output
        artifact and printed before compilation; it is not reconstructed
        from the counts because degeneracy snapping makes that ambiguous.
    boundary_mean_energy_ev : tuple[float, ...]
        ``mean_k E[k, N_i-1]`` in eV for each cut.  This is the coordinate
        consumed by ``conduction_energy_midpoint`` and provenance for every
        scheme.  Unlike ``mean_energy_ev`` it is a band-edge coordinate, not
        the mean over all included states.
    enabled : bool
        False for the trivial single-bracket plan.
    notes : tuple[str, ...]
        Anything the planner had to decide that the caller would otherwise
        not know — currently the degeneracy-snap fallbacks.  Carried as data
        rather than printed here so it is testable without capturing stdout;
        the driver prints it beside the plan line, and an empty tuple is the
        ordinary case.
    """

    bounds: tuple[tuple[int, int], ...]
    counts: tuple[int, ...]
    requested: tuple[int, ...]
    n_occ: int
    n_cond: int
    mean_energy_ev: tuple[float, ...]
    enabled: bool
    bracket_scheme: str = BRACKET_SCHEME_DEFAULT
    boundary_mean_energy_ev: tuple[float, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def n_brackets(self) -> int:
        return len(self.bounds)


def trivial_plan(nb_padded: int, n_occ: int, nb_logical: int) -> BandBracketPlan:
    """The ordinary (non-extrapolating) plan: ONE bracket over every band.

    This is what the default path runs, and it is a plan rather than a
    ``None`` so that the kernel, the accumulator and the Σ cube carry a
    length-1 leading bracket axis unconditionally — one code path, no
    ``if extrapolating`` fork anywhere below this module.
    """
    nb_padded = int(nb_padded)
    return BandBracketPlan(
        bounds=((0, nb_padded),),
        counts=(int(nb_logical),),
        requested=(int(nb_logical),),
        n_occ=int(n_occ),
        n_cond=int(nb_logical) - int(n_occ),
        mean_energy_ev=(float("nan"),),
        enabled=False,
        bracket_scheme="disabled",
        boundary_mean_energy_ev=(float("nan"),),
    )


def plan_band_brackets(
    *,
    enabled: bool,
    enk_ry: np.ndarray,
    n_occ: int,
    nb_logical: int,
    nb_padded: int,
    tol_ry: float = DEGENERACY_TOL_RY,
    fractions: tuple[float, ...] = BRACKET_FRACTIONS,
    bracket_scheme: str = BRACKET_SCHEME_DEFAULT,
) -> BandBracketPlan:
    """Build the bracket plan for one Σ stage.

    Parameters
    ----------
    enabled : bool
        The deck's ``sigma_band_extrapolation``.  False returns
        :func:`trivial_plan` — the length-1 axis, bit-identical default.
    enk_ry : (nk, nb) float array
        Band energies in Ry over the Σ band sum's band range, ascending in
        the band axis.  Only ``[:, :nb_logical]`` is read: the mesh pad
        bands carry a sentinel energy and zero ψ.
    n_occ : int
        Occupied bands in the Σ band sum (``b2 - b0``).
    nb_logical, nb_padded : int
        Real and mesh-padded band counts **of the Σ band sum** —
        ``b4_sigma - b0`` in both cases, with ``b4_sigma`` carrying the mesh
        pad only when Σ is the LARGER of the two counts (``common.meta``).
        **Never the χ count and never the loaded extent.**  Since 2026-08-16
        those are three different numbers on a split deck: the ψ is loaded
        over ``max(chi, sigma)`` and Σ sums a window inside it, so reading
        ``b_id_4_user`` / ``s.nb_full`` here would bracket a curve this run
        never evaluates.  The last bracket runs to ``nb_padded`` so the plan
        still covers every band the un-bracketed path summed; the pad bands
        contribute exactly zero, so ``counts`` stays logical.
    bracket_scheme : {"total_fractions", "conduction_energy_midpoint"}
        ``total_fractions`` preserves the incumbent 80/90/100 total-band
        geometry.  ``conduction_energy_midpoint`` places N1 at half the
        included conduction manifold (then snaps it), and N2 at the clean
        rectangular boundary nearest the midpoint in ``mean_k E[k,N-1]``.
        The latter is an explicit compatibility spelling, never inferred.

    Raises
    ------
    BandExtrapolationRefused
        When extrapolation is requested and ``n_cond < n_occ`` (the tail
        being extrapolated is then shorter than the occupied block and the
        1/N model has no room to be tested), or when degeneracy snapping
        collapses two of the three cuts onto each other.

        THE THRESHOLD IS ``n_cond >= n_occ``, i.e.
        ``number_bands_sigma >= 2*n_occ``.  Owner ruling 2026-08-16, whose
        words were "kill the calculation if the number of bands requested is
        not >= 2*N_electrons".  It is written in ``n_occ`` rather than in
        ``N_electrons`` because that is the SPIN-CONVENTION-INDEPENDENT form
        of the same condition: under SOC/noncolin ``n_occ = N_electrons``,
        without SOC ``n_occ = N_electrons/2``, and in BOTH cases the
        condition says "at least as many conduction bands as valence".
        Writing it in the electron count would silently mean two different
        things on the two deck families.

        **WHICH COUNT THE GATE IS ABOUT — merge ruling, 2026-08-16.**  The
        owner's rule was written before ``number_bands`` split into
        ``number_bands_chi`` / ``number_bands_sigma``, so "nband" in it is
        now ambiguous.  IT IS THE Σ COUNT, and only the Σ count.  The
        argument is not a preference:

          * The gate is a statement about the object being fitted.  The
            feature fits ``S(N) = S_inf + A/N`` to the **Σ** band sum's
            partial sums; ``n_cond`` and ``n_occ`` are the unoccupied and
            occupied halves of THAT sum.  ``nb_logical`` is
            ``b4_sigma - b0`` (see above), so the arithmetic in this
            function is already Σ-only and a χ-flavoured reading of the
            threshold would not correspond to anything computed here.
          * A χ-side floor would refuse runs that are fine.  The physically
            motivated configuration the split exists for is "χ at full
            bands, Σ short and extrapolated"; there χ >= Σ, so a gate on
            max(chi, sigma) is merely a weaker gate that lets an
            under-banded Σ through — the exact failure the gate exists to
            stop.  In the opposite split (Σ > χ) a χ-side floor would kill a
            run whose Σ sum is perfectly well conditioned for the fit, over
            a band count no part of this feature reads.
          * χ0 has no 1/N fit and no occupied/unoccupied ratio this gate
            could be a statement about.  A χ-side ``2*n_occ`` rule would be
            a new, unmeasured physics claim smuggled in as a merge detail.

        The refusal text therefore names ``number_bands_sigma`` explicitly
        and says outright that raising ``number_bands_chi`` will not help.
        Pinned by ``tests/test_band_extrapolation_sigma_count.py`` and
        ``tests/test_band_extrapolation_split_sc.py``.

        This RELAXES the threshold that shipped before 2026-08-16, which
        refused on ``n_cond <= n_occ`` (strictly greater).  The equality
        case ``n_cond == n_occ`` now runs.  There is ONE gate at ONE
        threshold: the "counts collapsed" refusal below reports the same
        ``2*n_occ`` floor rather than a second, tighter one.
    """
    n_occ = int(n_occ)
    nb_logical = int(nb_logical)
    nb_padded = int(nb_padded)
    n_cond = nb_logical - n_occ

    bracket_scheme = str(bracket_scheme).strip().lower()
    if bracket_scheme not in BRACKET_SCHEMES:
        raise ValueError(
            f"band_extrapolation_bracket_scheme = {bracket_scheme!r} is "
            f"not known; choose one of {BRACKET_SCHEMES}.")
    if (bracket_scheme == "conduction_energy_midpoint"
            and tuple(fractions) != BRACKET_FRACTIONS):
        raise ValueError(
            "plan_band_brackets: fractions and "
            "bracket_scheme='conduction_energy_midpoint' cannot be combined; "
            "that named scheme fixes N1 at half the conduction manifold and "
            "derives N2 from energy, so fractions would be ignored.")

    if not enabled:
        return trivial_plan(nb_padded, n_occ, nb_logical)

    # ── THE ACTIVATION GATE ─────────────────────────────────────────────
    # Requested-but-impossible REFUSES.  Silently running the ordinary path
    # would produce a log with no extrapolation block and a Σ that looks
    # converged, and the operator would have no way to tell the feature was
    # off (measurement-discipline rule 1: an ignored deck key is how a green
    # A/B comes to measure nothing).
    if n_cond < n_occ:
        raise BandExtrapolationRefused(
            f"use_band_extrapolation is ON, but the Σ_c band sum has "
            f"n_cond = {n_cond} unoccupied bands against n_occ = {n_occ} "
            f"occupied ones (number_bands_sigma = {nb_logical}).  The feature "
            f"extrapolates the UNOCCUPIED tail, and it is only meaningful "
            f"when that tail is at least as large as the occupied block: it "
            f"requires n_cond >= n_occ, i.e. number_bands_sigma >= 2*n_occ = "
            f"{2 * n_occ}.\n"
            f"  The owner's form of this rule is "
            f"'nband >= 2*N_electrons'.  It is enforced here in n_occ "
            f"because that is the SPIN-CONVENTION-INDEPENDENT statement of "
            f"the same thing: under SOC/noncolin n_occ = N_electrons, "
            f"without SOC n_occ = N_electrons/2, and in both cases the "
            f"condition means 'at least as many conduction bands as "
            f"valence'.\n"
            f"  THE COUNT THIS GATE IS ABOUT IS THE Σ COUNT (merge ruling, "
            f"2026-08-16).  The feature extrapolates the Σ band sum, so "
            f"n_cond and n_occ here are both edges of THAT sum; raising "
            f"`number_bands_chi` will NOT help, because this planner never "
            f"reads the χ count.  There is deliberately NO 2*n_occ floor on "
            f"the χ side: χ0 is a full band sum with no 1/N fit and no "
            f"occupied/unoccupied ratio for the gate to be a statement "
            f"about.\n"
            f"  Raise the deck's `number_bands_sigma` (or the umbrella "
            f"`number_bands`, which sets both counts) to at least "
            f"{2 * n_occ} (n_occ is set by the electron count, not by a deck "
            f"key), or set use_band_extrapolation = false.")

    e = np.asarray(enk_ry, dtype=np.float64)[:, :nb_logical]
    if e.ndim != 2 or e.shape[1] != nb_logical:
        raise ValueError(
            f"plan_band_brackets: expected (nk, >={nb_logical}) energies, "
            f"got shape {np.shape(enk_ry)}")

    # ``total_fractions`` is the incumbent geometry.  Keep this expression
    # exactly as it was: it is the compatibility arm and its counts must not
    # move.  The alternate geometry is resolved below because its N2 depends
    # on the *snapped* N1, not on the raw conduction-half request.
    requested = tuple(int(round(f * nb_logical)) for f in fractions)
    if bracket_scheme == "conduction_energy_midpoint":
        requested = (
            n_occ + int(round(CONDUCTION_HALF_FRACTION * n_cond)),
        )

    # ── SNAPPING THE INTERIOR CUTS IS A PREFERENCE, NOT A CONSTRAINT ────
    # The interior cuts are SAMPLING POINTS ON A PARTIAL-SUM CURVE, not
    # truncations of a delivered Σ.  Nothing downstream consumes a bracket
    # boundary as a window: S(N₁) and S(N₂) are read, fitted and discarded,
    # and only N₃ = nb_logical is the band sum the run actually reports.  So
    # a cut that lands inside a multiplet costs accuracy, not correctness —
    # MEASURED at ≤ 6.4 meV on the Si 4×4×4 SOC deck, and in half the cases
    # the UNSNAPPED points were slightly better, because snapping moves the
    # sample away from the requested fraction and the fraction is the thing
    # that matters (report §8).
    #
    # It is kept as a preference because a clean cut is free when one is
    # available.  It is NOT kept as a refusal: under SOC every Kramers pair
    # is exactly degenerate, so clean boundaries are sparse by construction,
    # and at (0.80, 0.90) the old behaviour REFUSED outright on decks where
    # the unsnapped points work fine (measured: nband = 60 on this very
    # deck).  Trading ≤ 6.4 meV for a run that stops is the wrong trade.
    #
    # ⚠ THE ≤ 6.4 meV IS ABOUT THE FIT RESIDUAL AND IS SILENT ABOUT STAR
    # COVARIANCE.  It was measured against BerkeleyGW's k-RESOLVED
    # ch_converge curve, where star covariance is not visible at all.  The
    # part that measurement could not see (established 2026-08-15 on the
    # orbit-closed set): S(N) is a deterministic POINTWISE function of three
    # partial sums, each exactly the Σ_c of a run at that band cut; a clean
    # cut is exactly star-covariant (0.0000 meV, floor 0.0010) and a SLICED
    # cut is not (1.957 meV of sigCOH spread at nband = 60).  A pointwise
    # function of star-covariant inputs is star-covariant, so:
    #
    #     S_∞ is exactly star-covariant IFF every interior cut is clean.
    #
    # So the fallback is a real trade, not a free one, and it must be
    # RECORDED every time it fires — which is why every branch below appends
    # to ``notes`` and none of them is allowed to be silent.
    # ``tests/test_band_extrapolation_star_covariance.py`` accepts a refusal
    # or a recorded fallback and fails only on silence.
    #
    # N₃ is not snapped here at all — it is the caller's `nb_logical`, the
    # band sum the deck asked for, and it is BerkeleyGW's `number_bands`
    # degeneracy check (not this one) that governs it.
    notes: list[str] = []
    snapped: list[int] = []
    lo_bound = n_occ + 1
    n_interior = 2 if bracket_scheme == "conduction_energy_midpoint" \
        else len(requested)
    for i_cut, req in enumerate(requested):
        req = int(req)
        # RESERVE ROOM FOR THE CUTS THAT COME AFTER THIS ONE.  Without this,
        # snapping the first cut UPWARD can eat the last clean boundary and
        # leave the second with nowhere legal to go — which turned into a
        # refusal on a spectrum that has three perfectly good sampling points
        # in it (seen at nband = 17 on the Si deck: cut 1 snapped 14 -> 16,
        # and 16 is nb_logical - 1).  One band per remaining interior cut is
        # the minimum that keeps the counts strictly ascending.
        hi_bound = nb_logical - 1 - (n_interior - 1 - i_cut)
        if lo_bound > hi_bound:
            # No room left below N₃ for another cut.  Record the raw request
            # so the distinctness check below refuses with the actionable
            # message, rather than letting snap_cut_to_clean_boundary raise
            # its own ValueError about inverted bounds.
            #
            # NOTE EVEN HERE.  This branch normally ends in the refusal
            # below, but it must not be the one path that can emit an
            # unsnapped cut SILENTLY: an unsnapped interior cut costs the
            # star covariance of S_∞ (see below), and
            # ``tests/test_band_extrapolation_star_covariance.py`` fails on
            # silence, not on the fallback.
            notes.append(
                f"interior cut {req} kept UNSNAPPED: no room left between "
                f"{lo_bound} and {hi_bound} for another cut below N3 = "
                f"{nb_logical}.")
            snapped.append(req)
            lo_bound = req + 1
            continue
        req = int(min(max(req, lo_bound), hi_bound))
        try:
            cut = int(snap_cut_to_clean_boundary(
                e, req, tol_ry=tol_ry, lo=lo_bound, hi=hi_bound))
        except BandWindowDegeneracyError:
            cut = req
            notes.append(
                f"interior cut {req} kept UNSNAPPED: no multiplet-clean "
                f"boundary exists in [{lo_bound}, {hi_bound}] at tol "
                f"{tol_ry * RYD_TO_EV * 1e3:.3f} meV.  The cut is a sampling "
                f"point on a partial-sum curve, not a Σ window, so this "
                f"costs accuracy (<= ~6 meV measured) and not correctness.")
        snapped.append(cut)
        lo_bound = cut + 1

    if bracket_scheme == "conduction_energy_midpoint" and snapped:
        # One rectangular band count at every k is non-negotiable: a literal
        # global E_ck threshold gives k-dependent shapes and is a different
        # kernel.  Instead use the k-mean DFT ladder as a scalar coordinate.
        # N1 has already been snapped, so the target describes the geometry
        # that will actually be fitted rather than the unsnapped request.
        n1 = int(snapped[0])
        boundary_energy = np.mean(e, axis=0)
        target_ry = 0.5 * (
            float(boundary_energy[n1 - 1])
            + float(boundary_energy[nb_logical - 1]))
        lo2, hi2 = n1 + 1, nb_logical - 1
        if lo2 <= hi2:
            raw_n2 = min(
                range(lo2, hi2 + 1),
                key=lambda n: (abs(float(boundary_energy[n - 1]) - target_ry),
                               n),
            )
            gaps = boundary_min_gaps(e, is_full_spectrum=False)
            clean = [n for n in range(lo2, hi2 + 1)
                     if float(gaps[n]) > float(tol_ry)]
            if clean:
                n2 = min(
                    clean,
                    key=lambda n: (
                        abs(float(boundary_energy[n - 1]) - target_ry), n),
                )
            else:
                n2 = int(raw_n2)
                notes.append(
                    f"interior energy-midpoint cut {raw_n2} kept "
                    f"UNSNAPPED: no multiplet-clean boundary exists in "
                    f"[{lo2}, {hi2}] at tol "
                    f"{tol_ry * RYD_TO_EV * 1e3:.3f} meV.  The cut is a "
                    f"sampling point on a partial-sum curve, not a Σ "
                    f"window, so this costs accuracy and not correctness.")
            requested = (int(requested[0]), int(raw_n2))
            snapped.append(int(n2))
    counts = tuple(snapped) + (nb_logical,)

    # What survives as a refusal: three counts that are not DISTINCT and
    # ascending.  That means the selected geometry has no room for three
    # points below this N3 — a real "raise nband", not a spectrum accident.
    if len(set(counts)) != len(counts) or list(counts) != sorted(counts):
        # The smallest fraction gap has to be worth at least one band, plus
        # the floor at n_occ+1 that the first cut is clamped to.
        fraction_gaps = np.diff(np.sort(
            np.asarray(tuple(fractions) + (1.0,), float)))
        # ONE GATE, ONE THRESHOLD.  The activation gate above is
        # ``n_cond >= n_occ`` (nband >= 2*n_occ); this floor must be the SAME
        # number, or a deck that clears one refusal lands on a second with a
        # different answer to "how many bands do I need".
        if bracket_scheme == "total_fractions":
            need = max(
                int(np.ceil(1.0 / max(float(fraction_gaps.min()), 1e-12))),
                2 * n_occ)
            why = "the sampling fractions are less than one band apart"
        else:
            # Four conduction bands are the smallest manifold that can put
            # one point at half and still leave a distinct middle and N3.
            need = max(n_occ + 4, 2 * n_occ)
            why = "the conduction manifold has no room for two interior cuts"
        geometry = (
            f"fractions {tuple(fractions)} of number_bands_sigma"
            if bracket_scheme == "total_fractions" else
            "conduction-half / k-mean-energy-midpoint geometry")
        raise BandExtrapolationRefused(
            f"use_band_extrapolation: the three band counts collapsed onto "
            f"{counts} (requested {requested + (nb_logical,)} from "
            f"{geometry}, number_bands_sigma = {nb_logical}).  "
            f"Three DISTINCT, ascending counts are required — two coincident "
            f"points cannot determine a two-parameter fit.  At "
            f"number_bands_sigma = {nb_logical} {why}.  Raise the deck's "
            f"`number_bands_sigma` (or the umbrella `number_bands`, which "
            f"sets both counts) to at least {need}, or set "
            f"use_band_extrapolation = false.  `number_bands_chi` is not "
            f"read by this planner and will not move these cuts.")

    bounds = tuple(
        (int(a), int(b)) for a, b in
        zip((0,) + counts[:-1], counts[:-1] + (nb_padded,)))
    mean_e = tuple(
        float(np.mean(e[:, :c])) * RYD_TO_EV for c in counts)
    boundary_mean_e = tuple(
        float(np.mean(e[:, c - 1])) * RYD_TO_EV for c in counts)
    return BandBracketPlan(
        bounds=bounds,
        counts=counts,
        requested=requested + (nb_logical,),
        n_occ=n_occ,
        n_cond=n_cond,
        mean_energy_ev=mean_e,
        enabled=True,
        bracket_scheme=bracket_scheme,
        boundary_mean_energy_ev=boundary_mean_e,
        notes=tuple(notes),
    )


class BandBracketCountMismatch(BandExtrapolationRefused):
    """The bracket partition and the OLS abscissae describe different sums.

    A separate name from :class:`BandExtrapolationRefused` because it is a
    different KIND of fault: the other refusals are about a deck that asked
    for something the physics cannot deliver, and the operator fixes them by
    changing a key.  This one is about the CODE having wired the planner to a
    band count the Σ sum does not walk, and the operator cannot fix it at all.
    Subclassed rather than freestanding so an ``except
    BandExtrapolationRefused`` that means "the extrapolation cannot run here"
    still catches it.
    """


def assert_brackets_match_ols_abscissae(plan: BandBracketPlan, slices, *,
                                        meta=None, where: str = "") -> None:
    """Refuse unless the partition and the abscissae are the SAME band sum.

    WHY THIS EXISTS, AND WHY NOTHING DOWNSTREAM CAN REPLACE IT.  Bracketing
    the wrong band count is invisible in every weight-level diagnostic there
    is.  :func:`extrapolation_weights` solves OLS in ``x = 1/N``, and its
    coefficients depend only on the RATIOS of the abscissae (the ``x - xbar``
    terms are homogeneous of degree 1 in ``1/N``, and ``xbar*(x-xbar)/Sxx``
    is therefore scale-free).  The sampling fractions are the same
    0.80/0.90/1.00 of whichever count is used, so the ratios are the same
    too, and the weights barely move:

        counts (80, 90, 100)     -> c = [-4.295082, +0.663934, +4.631148]
        counts (198, 223, 248)   -> c = [-4.254729, +0.663885, +4.590844]

    — 0.94 % apart at the largest coefficient, and ``sum(c) == 1`` in both.
    So a run that brackets the WRONG count applies a nearly-correct operator
    to the wrong three partial sums.  ``c`` is still real, so Σ_∞ is still
    exactly Hermitian; the SC loop still converges; ``trust_verdict`` still
    reads ordinary, because it inspects the fit's own residual structure and
    that structure is self-consistent on the wrong curve.  There is no
    downstream check that can see it.  Only the site where the plan meets the
    band axis it will slice can, which is why this is called there.

    WHAT IS COMPARED — the actual objects, never a second derivation of the
    same number.  Re-deriving "the Σ count" here would be worthless: a rewire
    that fed the planner the χ count would rewire the expectation with it and
    the check would pass vacuously.  So:

      * the abscissae are checked against THE PARTITION ITSELF — every
        interior ``counts[i]`` must BE the cumulative top of bracket ``i``,
        because those are the band sums the kernel will actually deliver;
      * the partition is checked against ``slices``, the SAME
        :class:`~gw.wavefunction_bundle.BandSlices` object from which
        ``ppm_sigma._run_sigma_branch`` builds the ψ/E/mask operands it then
        slices.  ``slices.nb_sigma_sum`` is the Σ band sum's extent; a plan
        built from ``nb_full`` / ``b_id_4_user`` (the LOADED extent,
        ``max(chi, sigma)``) or from the χ count disagrees with it the moment
        the deck is split.

    Parameters
    ----------
    plan
        The plan whose ``bounds`` the τ kernel will slice by and whose
        ``counts`` the OLS will use as abscissae.
    slices
        The band slices the Σ operands are built from.
    meta
        Optional; when given, ``meta.b_id_4_sigma_user`` supplies the LOGICAL
        Σ top so the unpadded abscissa is checked too.
    where
        Call-site tag for the message.

    Raises
    ------
    BandBracketCountMismatch
    """
    tag = f" ({where})" if where else ""
    bounds = tuple(plan.bounds)
    counts = tuple(plan.counts)

    def refuse(what: str, detail: str) -> None:
        raise BandBracketCountMismatch(
            f"band extrapolation{tag}: {what}.\n"
            f"  {detail}\n"
            f"  THE BRACKET PARTITION AND THE OLS ABSCISSAE MUST BE THE SAME "
            f"BAND SUM.  The partition is what the τ kernel slices "
            f"(`ppm_tau_kernel._bracketed`, operands `[..., lo:hi]`); the "
            f"abscissae are the N_i that `extrapolation_weights` inverts.  "
            f"They are checked here and nowhere else because a mismatch is "
            f"INVISIBLE downstream: OLS in 1/N depends only on the RATIOS of "
            f"the abscissae, and the fractions are the same 0.80/0.90/1.00 "
            f"of whichever count, so the weights move under 1 % — "
            f"(80, 90, 100) gives [-4.295, +0.664, +4.631] and "
            f"(198, 223, 248) gives [-4.255, +0.664, +4.591].  A wrong-count "
            f"run therefore produces a Hermitian Σ, converges, and prints "
            f"entirely ordinary numbers.\n"
            f"  THE COUNT THIS FEATURE BRACKETS IS `number_bands_sigma`, via "
            f"`BandSlices.sigma_sum` / `nb_sigma_sum` and "
            f"`meta.b_id_4_sigma_user`.  It is NEVER `number_bands_chi`, and "
            f"NEVER the LOADED extent max(chi, sigma) — which is what "
            f"`BandSlices.full` / `nb_full` and `meta.b_id_4_user` carry, "
            f"both of which read like 'all the bands' and are not.  Raising "
            f"`number_bands_chi` will not clear this; it is not a deck fault "
            f"at all.  Fix the planner call site "
            f"(`gw.ppm_pipeline`), not the deck.")

    # (1) THE ABSCISSAE MUST BE THE PARTITION.  Read off the bounds, so this
    # cannot be satisfied by a plan whose counts were computed from anything
    # other than the cuts the kernel will actually make.
    if len(counts) != len(bounds):
        refuse(f"the plan has {len(bounds)} bracket(s) but {len(counts)} "
               f"OLS abscissa(e)",
               f"bounds={bounds}, counts={counts}")
    if bounds[0][0] != 0:
        refuse("the bracket partition does not start at band 0",
               f"first bracket is {bounds[0]}; the cumulative sums the "
               f"abscissae name are all measured from band 0")
    for i in range(len(bounds) - 1):
        if bounds[i][1] != bounds[i + 1][0]:
            refuse(f"the bracket partition has a gap or overlap at cut {i}",
                   f"bracket {i} ends at {bounds[i][1]} and bracket {i + 1} "
                   f"starts at {bounds[i + 1][0]}; a non-partition makes the "
                   f"cumulative sums something other than S(N_i)")
        if bounds[i][1] != counts[i]:
            refuse(f"OLS abscissa {i} is not the band count bracket {i} "
                   f"delivers",
                   f"counts[{i}]={counts[i]} but the cumulative top of "
                   f"bracket {i} is {bounds[i][1]}")

    # (2) THE PARTITION MUST BE THE Σ BAND SUM.  ``slices`` is the object
    # ``_run_sigma_branch`` builds the operands from, so this ties the plan to
    # the band axis it will actually slice rather than to a repeat of the
    # arithmetic that built it.
    nb_sigma_padded = int(slices.nb_sigma_sum)
    if int(bounds[-1][1]) != nb_sigma_padded:
        extra = ""
        nb_full = int(getattr(slices, "nb_full", -1))
        nb_chi = int(getattr(slices, "nb_chi_sum", -1))
        if int(bounds[-1][1]) == nb_full:
            extra = (f"  That value IS the LOADED extent nb_full={nb_full} "
                     f"= max(chi, sigma) — the classic wrong reach.")
        elif int(bounds[-1][1]) == nb_chi:
            extra = (f"  That value IS the χ count nb_chi_sum={nb_chi}.")
        refuse(
            f"the bracket partition spans {bounds[-1][1]} bands but the Σ "
            f"band sum is {nb_sigma_padded} bands",
            f"last bracket is {bounds[-1]}, so the τ kernel would slice to "
            f"band {bounds[-1][1]}, while the Σ operands are built over "
            f"slices.nb_sigma_sum = {nb_sigma_padded}.{extra}")

    # (3) THE LARGEST ABSCISSA MUST BE THE LOGICAL Σ TOP.  N₃ is the band sum
    # the run reports; the padded top may exceed it by the mesh pad, whose ψ
    # is exactly zero, but nothing else may.
    if int(counts[-1]) > nb_sigma_padded:
        refuse(f"the largest OLS abscissa {counts[-1]} exceeds the Σ "
               f"partition's {nb_sigma_padded} bands",
               f"counts={counts}, last bracket={bounds[-1]}")
    if meta is not None:
        b0 = int(getattr(slices, "b0", 0))
        logical_top = int(getattr(meta, "b_id_4_sigma_user", 0) or 0)
        if logical_top:
            nb_sigma_logical = logical_top - b0
            if int(counts[-1]) != nb_sigma_logical:
                refuse(
                    f"the largest OLS abscissa is {counts[-1]} but the Σ "
                    f"band sum is {nb_sigma_logical} logical bands",
                    f"meta.b_id_4_sigma_user={logical_top}, slices.b0={b0}; "
                    f"N₃ must be the band count the run reports, since the "
                    f"fit's intercept is quoted as the N→∞ limit of THAT "
                    f"series")


# ---------------------------------------------------------------------------
#  The fit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtrapolationFit:
    """Result of the two-parameter 1/N fit, plus its quality diagnostics.

    Every array field carries the CALLER's trailing shape (e.g. ``(nk, nb)``
    for a band-diagonal Σ_c); only the leading three-point axis is consumed.
    """

    counts: np.ndarray            # (3,) int   — N₁, N₂, N₃
    s_at_counts: np.ndarray       # (3, ...)   — S(N₁), S(N₂), S(N₃)
    s_inf: np.ndarray             # (...)      — least-squares intercept
    amplitude: np.ndarray         # (...)      — A, the 1/N coefficient
    pair_s_inf: dict              # {(i, j): (...)} exact two-point intercepts
    delta_tail: np.ndarray        # (...)      |S_∞^fit − S₃|
    delta_model: np.ndarray       # (...)      max_ij |S_∞^(ij) − S_∞^fit|
    residual: np.ndarray          # (...)      max_i |S_i − model(N_i)|

    # ── THE TWO SIGNED, ONE-SIDED DIAGNOSTICS ───────────────────────────
    # Derived rather than stored: both are exact functions of fields the fit
    # already carries, so ``at()`` restricts them for free and there is no
    # second place that can go stale.  See the module docstring for what
    # each one can and cannot see; the short version is that ``delta_model``
    # is a SCATTER metric and is blind to an error the three points share,
    # which is the error that actually dominates.

    @property
    def pair_split(self) -> np.ndarray:
        """S_∞^(1,2) − S_∞^(2,3), SIGNED.

        The same information ``delta_model`` reduces to an absolute value.
        Its magnitude bounds the residual curvature; its sign says which way
        the preasymptotic bias runs, and the sign is the half that says
        whether the extrapolation is over- or under-correcting.
        """
        return self.pair_s_inf[(0, 1)] - self.pair_s_inf[(1, 2)]

    def uncertainty(self, quantile: str = "p90") -> np.ndarray:
        """Extrapolation uncertainty as a fraction of the applied correction.

        ``|S_inf - S_true| <~ f * Delta_tail`` with ``f`` from
        :data:`TAIL_UNCERTAINTY_FRACTION` — an ENVELOPE calibrated against
        BerkeleyGW on one deck, not a per-state error bar (there is no
        working per-state predictor; see the constant's comment for the
        R² <= 0 measurement that rules the obvious candidates out).

        Extrapolation only.  Not the difference from BerkeleyGW, not the
        ISDF basis, not the W-side band count.
        """
        p90, p99 = TAIL_UNCERTAINTY_FRACTION
        f = {"p90": p90, "p99": p99}[quantile]
        return f * np.abs(self.delta_tail)

    @property
    def a_over_n_last(self) -> np.ndarray:
        """A / N₃ — the correction the model is still applying AT the last point.

        One-sided: large exactly when the band sum has not finished.  Unlike
        ``delta_model`` it does not go quiet when all three points carry the
        same error, which is why it is reported beside it rather than
        instead of it.
        """
        return self.amplitude / float(self.counts[-1])

    def at(self, index) -> "ExtrapolationFit":
        """Restrict every field to a subset of the trailing state axes.

        The fit is elementwise in (k, band) — every state has its own two
        parameters — so a subset of it IS the fit of that subset, and this is
        an indexing operation rather than a refit.  It exists because an
        aggregate over ALL states is the wrong summary: Σ_c at the top of the
        QP window is both large and the worst-converged thing in the run, so
        a max over (k, band) reports that state's tail as if it were the
        calculation's, and its verdict as if it were the calculation's.  The
        states a GW run is FOR are the band edges.
        """
        # ``index`` addresses the TRAILING state axes; ``s_at_counts`` has the
        # three-point axis in front of them, so the tuple has to be SPLICED
        # after a full slice, not passed as one more index.  Wrapping it
        # instead — ``[(slice(None), index)]`` — makes numpy read a (k, n)
        # pair as fancy indexing along axis 1 and returns a (3, 2) array where
        # a scalar was meant; that reached a driver run.
        idx = index if isinstance(index, tuple) else (index,)
        return ExtrapolationFit(
            counts=self.counts,
            s_at_counts=self.s_at_counts[(slice(None),) + idx],
            s_inf=self.s_inf[index],
            amplitude=self.amplitude[index],
            pair_s_inf={k: v[index] for k, v in self.pair_s_inf.items()},
            delta_tail=self.delta_tail[index],
            delta_model=self.delta_model[index],
            residual=self.residual[index],
        )


def fit_band_extrapolation(
    counts, s_at_counts: np.ndarray) -> ExtrapolationFit:
    """Ordinary least squares of ``S(N) = S_∞ + A/N`` over three points.

    Parameters
    ----------
    counts : sequence of int, length 3
        ``N_eff`` at each point.  ``N_eff = N_occ + N_c`` is just the total
        band count of the sum, which is what the brackets cumulate to.
    s_at_counts : (3, ...) complex array
        The CUMULATIVE bracket sums — S(N₁), S(N₂), S(N₃) — with any
        trailing shape.  Complex is carried through: the model is linear, so
        fitting the complex value is exactly fitting Re and Im separately.

    Returns
    -------
    ExtrapolationFit
    """
    N = np.asarray(counts, dtype=np.float64)
    S = np.asarray(s_at_counts)
    # EXACTLY three, not ">= 2".  The fit itself is well posed on two points,
    # but every consumer indexes the (1, 2) pair by name — ``pair_split``,
    # ``trust_verdict``, the log block, the h5 payload — so a two-point fit
    # CONSTRUCTS and then dies with ``KeyError((1, 2))`` at whichever consumer
    # runs first.  Verified 2026-08-16.  Refusing here turns a confusing
    # downstream crash into a statement about the input.
    if N.ndim != 1 or N.size != 3:
        raise ValueError(
            f"fit_band_extrapolation: need exactly 3 counts, got {N}.  "
            f"The two-parameter fit needs three points to have a residual at "
            f"all, and the diagnostics are built on the three pairwise "
            f"intercepts.")
    if S.shape[0] != N.size:
        raise ValueError(
            f"fit_band_extrapolation: S leading axis {S.shape[0]} != "
            f"{N.size} counts")

    x = 1.0 / N                                   # the regressor
    n = float(N.size)
    xbar = float(np.mean(x))
    # Ordinary least squares, written out rather than via lstsq so the
    # broadcast over the trailing (k, band) axes is explicit and allocation
    # free.  Sxx is a scalar; the covariance carries the trailing shape.
    Sxx = float(np.sum((x - xbar) ** 2))
    if Sxx <= 0.0:
        raise ValueError(
            "fit_band_extrapolation: the three band counts are degenerate "
            f"in 1/N ({N}) — no slope is determined.")
    xb = x.reshape((-1,) + (1,) * (S.ndim - 1))
    Sbar = np.mean(S, axis=0)
    Sxy = np.sum((xb - xbar) * (S - Sbar), axis=0)
    A = Sxy / Sxx
    s_inf = Sbar - A * xbar

    pair = {}
    for i in range(N.size):
        for j in range(i + 1, N.size):
            # Exact two-point solution of the same model:
            #   N_j·S_j − N_i·S_i = (N_j − N_i)·S_∞
            pair[(i, j)] = (N[j] * S[j] - N[i] * S[i]) / (N[j] - N[i])

    model = s_inf[None, ...] + A[None, ...] * xb
    residual = np.max(np.abs(S - model), axis=0)
    delta_tail = np.abs(s_inf - S[-1])
    delta_model = np.max(
        np.abs(np.stack([p - s_inf for p in pair.values()], axis=0)), axis=0)

    return ExtrapolationFit(
        counts=np.asarray(N, dtype=np.int64),
        s_at_counts=S,
        s_inf=s_inf,
        amplitude=A,
        pair_s_inf=pair,
        delta_tail=delta_tail,
        delta_model=delta_model,
        residual=residual,
    )


def extrapolation_weights(counts) -> np.ndarray:
    """The REAL coefficients ``c`` with ``S_inf = sum_i c_i * S(N_i)``.

    WHY THIS EXISTS SEPARATELY FROM :func:`fit_band_extrapolation`.  That
    function fits the band DIAGONAL and returns diagnostics; this one returns
    the three numbers that let a caller apply the same fit to an object it
    would be absurd to run a regression on — the full ``(nomega, nk, nb, nb)``
    Σ_c cube that becomes the QP Hamiltonian.  Both are the SAME estimator,
    and ``tests/test_band_extrapolation.py::
    test_weights_reproduce_the_fit_intercept`` pins them together so the
    number the log reports and the number that drives the iteration cannot
    drift apart.

    THIS IS WHY "EXTRAPOLATE Σ, THEN DIAGONALIZE" IS THE ONLY DEFENSIBLE
    ORDER, and the reason is visible in the return type.  Ordinary least
    squares of ``S_inf + A/N`` is LINEAR in the observations, so its
    intercept is a fixed affine combination of them whose coefficients depend
    only on the band COUNTS -- not on Σ, not on k, not on the band index.
    Two consequences, both load-bearing:

      * ``c`` is REAL.  A real linear combination of Hermitian matrices is
        Hermitian, ELEMENTWISE and to machine precision: ``S_b[j, i]`` is
        exactly ``conj(S_b[i, j])`` out of the kernel, a real scalar multiply
        commutes with conjugation exactly in IEEE arithmetic, and the sum is
        formed in the same order for both elements.  So the extrapolated Σ is
        a legitimate static self-energy and the next iteration's eigenvectors
        stay consistent with its own eigenvalues.
      * The alternative -- diagonalizing each bracket and extrapolating the
        EIGENVALUES -- is not the same operation and does not correspond to
        any Hamiltonian.  Eigenvalues are not linear in the matrix, so the
        extrapolated set is not the spectrum of anything; feeding it back
        would pair iteration ``i+1``'s energies with eigenvectors belonging
        to a different operator.  The band-sum tail is a property of Σ, not
        of the spectrum, so Σ is where the fit belongs.

    ``sum(c) == 1`` identically (the ``x_i - xbar`` terms cancel), which is
    the statement that the estimator is an affine combination -- it preserves
    a Σ that does not depend on the band count, so a converged band sum comes
    through unchanged rather than being scaled.

    Parameters
    ----------
    counts : sequence of int, length 3
        ``N_eff`` at each point, as in :func:`fit_band_extrapolation`.

    Returns
    -------
    (3,) float64 ndarray
    """
    N = np.asarray(counts, dtype=np.float64)
    if N.ndim != 1 or N.size != 3:
        raise ValueError(
            f"extrapolation_weights: need exactly 3 counts, got {N}.  Same "
            f"requirement as fit_band_extrapolation -- the two must describe "
            f"the same estimator.")
    x = 1.0 / N
    xbar = float(np.mean(x))
    Sxx = float(np.sum((x - xbar) ** 2))
    if Sxx <= 0.0:
        raise ValueError(
            "extrapolation_weights: the three band counts are degenerate in "
            f"1/N ({N}) -- no slope is determined.")
    # s_inf = mean(S) - A*xbar with A = sum_i (x_i - xbar) S_i / Sxx, so
    # c_i = 1/n - xbar*(x_i - xbar)/Sxx.  Written out rather than via lstsq
    # so the realness is manifest at the call site.
    return (1.0 / float(N.size)) - xbar * (x - xbar) / Sxx


# ---------------------------------------------------------------------------
#  The spectrum-resolved shell estimator
# ---------------------------------------------------------------------------

#: The band-convergence estimators a deck may select, and the default.
#:
#: ``spectral_shell`` is the default since 2026-08-17 (owner ruling): held out
#: against a MEASURED S(508) it beats ``band_index_only`` at every band count
#: on the Si 50 Ry arm — see the module docstring for the table.
#: ``band_index_only`` is the incumbent ``S_∞ + A/N`` least squares under its
#: honest name; it is kept, selectable, and bit-for-bit what it always was.
BAND_EXTRAPOLATION_ESTIMATORS: tuple[str, ...] = (
    "spectral_shell", "band_index_only")
BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT: str = "spectral_shell"

#: Bracket for the per-state exponent β, and the bisection depth.
#:
#: NOT A MEASURED THRESHOLD, and it must not become one by being tuned.  β is
#: only meaningful where the local power law is: the measured values run
#: 3.4–5.3 on the Si arm, an order of magnitude inside both edges.  The
#: interval is wide on purpose so that a state whose root is NOT in the
#: physical range hits :data:`SHELL_EXPONENT_EDGE_TOL` and REFUSES rather than
#: being quietly clipped to a plausible-looking number.  Narrowing it would
#: convert refusals into clips, which is exactly the failure the owner's
#: ruling forbids.
SHELL_EXPONENT_BRACKET: tuple[float, float] = (0.05, 40.0)
SHELL_EXPONENT_BISECTIONS: int = 80

#: A root this close (in β) to either bracket edge is a REFUSAL, not a value.
#: 80 halvings of the interval leave the root located to ~3e-23, so a β that
#: is still within 4e-5 of an edge means the bisection walked to the wall:
#: ``f`` did not change sign inside, and what would be returned is the edge
#: itself dressed up as a solution.
SHELL_EXPONENT_EDGE_TOL: float = 1.0e-6 * (
    SHELL_EXPONENT_BRACKET[1] - SHELL_EXPONENT_BRACKET[0])

#: Offsets scanned when fitting ``E_n = E₀ + C·(n + n₀)^(2/3)``.  n₀ is a
#: single integer shift of the band ladder, so an integer scan is the whole
#: parameter space; on the Si deck the minimum lands at n₀ = 0.
WEYL_N0_SCAN: tuple[float, float, float] = (0.0, 200.0, 1.0)

#: Per-state failure codes.  Index into :data:`SHELL_FAILURE_REASONS`.
SHELL_OK = 0
SHELL_FAIL_SIGN = 1
SHELL_FAIL_ZERO = 2
SHELL_FAIL_NO_ROOT = 3
SHELL_FAIL_EDGE = 4

SHELL_FAILURE_REASONS = {
    SHELL_OK: "ok",
    SHELL_FAIL_SIGN: (
        "D2 and D3 have OPPOSITE SIGN: the two observed shells disagree "
        "about which way the band sum is still moving, so there is no tail "
        "to integrate and |D3/D2| is not a decay ratio"),
    SHELL_FAIL_ZERO: (
        "D2 or D3 is exactly zero: the shell increment carries no "
        "information about the falloff and the ratio is undefined"),
    SHELL_FAIL_NO_ROOT: (
        "no bracketed root: log I3(b) - log I2(b) - log|D3/D2| does not "
        "change sign across the exponent bracket, so the observed shell "
        "ratio is not reachable by ANY power law on this spectrum"),
    SHELL_FAIL_EDGE: (
        "the root is pressed against a bracket edge: the bisection walked "
        "to the wall rather than converging inside, and returning the edge "
        "would be a CLIP reported as a fit"),
}


class SpectralShellExtrapolationFailed(BandExtrapolationRefused):
    """The spectral-shell estimator refused on at least one external state.

    A separate name from :class:`BandExtrapolationRefused` because the fix is
    different in kind: the other refusals are about a deck asking for
    something the band counts cannot support, and the operator clears them by
    raising ``number_bands_sigma``.  This one is about the DATA — two shells
    that disagree about the sign of the remaining tail, or a shell ratio no
    power law on this spectrum can produce.  Subclassed so an ``except
    BandExtrapolationRefused`` that means "the extrapolation cannot run here"
    still catches it.

    WHY THIS IS A RUN-LEVEL REFUSAL AND NOT A PER-STATE FALLBACK.  The owner's
    ruling is that a failed state says so and no value is substituted for it.
    A ``NaN`` left in Σ would propagate into the QP Hamiltonian and take the
    whole spectrum with it silently; falling back to ``S(N₃)`` or to the 1/N
    fit on the failed states would ship a Σ assembled from two estimators with
    nothing in the artifact recording which states came from which.  So the
    run stops, names every failed state and its reason, and offers the two
    honest ways forward: more bands, or ``band_extrapolation_estimator =
    band_index_only``.
    """


@dataclass(frozen=True)
class BandLadder:
    """The DFT-only spectral ladder the shell moments are built on.

    Nothing in here is derived from Σ.  It is the mean-field eigenvalue
    sequence, its k-point weights, the band-manifold bottom ``E₀`` from the
    Weyl fit, the conditioning scale ``E*``, and the finite-basis endpoint
    ``N_T`` — the four things the estimator needs and the only four.

    Band indices are 1-BASED and ABSOLUTE (band 1 is the lowest band in the
    WFN), because the Weyl counting law ``E_n = E₀ + C(n + n₀)^(2/3)`` counts
    from the bottom of the band manifold.  Callers whose Σ band sum starts
    above band 1 pass their cuts through :meth:`absolute`.

    Attributes
    ----------
    e_dft_ev : (n_dft, nk) float
        DFT eigenvalues in eV, band-major.  Exactly what the WFN carries.
    e_weyl_ev : (n_target - n_dft,) float
        The ladder continued past the WFN's band count by the Weyl form.
        k-INDEPENDENT by construction: the fit is to the k-mean, and at these
        energies the k-dispersion of a band is a vanishing fraction of its
        distance from E₀.  Used ONLY to extend the eigenvalue sequence.
    w_k : (nk,) float
        k-point weights, normalised to sum 1.  Ratios are weight-scale
        invariant; the normalisation is for conditioning only.
    e0_ev, n0, c_ev, r2 : float
        The Weyl fit: ``E_n = e0_ev + c_ev·(n + n0)^(2/3)``, and its R².
    estar_ev : float
        The conditioning scale.  Cancels from every ratio (see the module
        docstring); it exists so the powers stay off the exponent rails.
    n_dft, n_target : int
        The WFN's band count and the endpoint the tail is integrated to.
    fit_window : (int, int)
        The 1-based inclusive band range the Weyl fit used.
    b0 : int
        The absolute index of the band the caller's cuts are measured from
        (0 when the Σ band sum starts at the first band, which is the
        ordinary case).
    """

    e_dft_ev: np.ndarray
    e_weyl_ev: np.ndarray
    w_k: np.ndarray
    e0_ev: float
    n0: float
    c_ev: float
    r2: float
    estar_ev: float
    n_dft: int
    n_target: int
    fit_window: tuple[int, int]
    b0: int = 0

    def absolute(self, count: int) -> int:
        """A caller's Σ-relative band count as an absolute band index."""
        return int(count) + int(self.b0)

    # ── the moments ─────────────────────────────────────────────────────
    def log_moment(self, lo: int, hi: int, beta) -> np.ndarray:
        """``log I(β)`` over ABSOLUTE bands ``(lo, hi]``, vectorised over β.

        Computed as a log-sum-exp so that the ``x^(-β)`` powers cannot
        overflow or flush to zero at the large β the bracket admits — which
        is the only reason ``E*`` is not load-bearing here.

        ``β`` may be a scalar or any array; the return has ``β``'s shape.
        Bands whose ``ε − E₀`` is non-positive (there are none above the
        occupied manifold on a converged mean field, but the ladder is not
        assumed) contribute nothing and are dropped, not clamped.
        """
        b = np.asarray(beta, dtype=np.float64)
        flat = b.ravel()
        terms = []                      # each (n_terms, n_beta)
        lo = int(lo)
        hi = int(hi)

        lo_d, hi_d = min(lo, self.n_dft), min(hi, self.n_dft)
        if hi_d > lo_d:
            x = (self.e_dft_ev[lo_d:hi_d] - self.e0_ev) / self.estar_ev
            ok = x > 0.0
            lg = np.where(ok, -np.log(np.where(ok, x, 1.0)), np.nan)
            lg = lg.reshape(-1, 1) * flat[None, :]
            lgw = np.log(self.w_k)
            lg = lg + np.tile(lgw, hi_d - lo_d)[:, None]
            terms.append(np.where(np.isnan(lg), -np.inf, lg))

        lo_w, hi_w = max(lo, self.n_dft), max(hi, self.n_dft)
        if hi_w > lo_w:
            x = ((self.e_weyl_ev[lo_w - self.n_dft:hi_w - self.n_dft]
                  - self.e0_ev) / self.estar_ev)
            ok = x > 0.0
            lg = np.where(ok, -np.log(np.where(ok, x, 1.0)), np.nan)
            # The extended ladder is k-independent, so each extended band
            # carries the FULL k weight (which sums to 1) exactly once.
            lg = lg.reshape(-1, 1) * flat[None, :]
            terms.append(np.where(np.isnan(lg), -np.inf, lg))

        if not terms:
            return np.full(b.shape, -np.inf)
        allt = np.concatenate(terms, axis=0)
        m = np.max(allt, axis=0)
        out = np.where(
            np.isfinite(m),
            m + np.log(np.sum(np.exp(allt - np.where(np.isfinite(m), m, 0.0)),
                              axis=0)),
            -np.inf)
        return out.reshape(b.shape)

    def moment(self, lo: int, hi: int, beta) -> np.ndarray:
        """``I(β)`` over ABSOLUTE bands ``(lo, hi]``."""
        return np.exp(self.log_moment(lo, hi, beta))

    def describe(self) -> str:
        return (
            f"DFT ladder: E0 = {self.e0_ev:.4f} eV, n0 = {self.n0:g}, "
            f"C = {self.c_ev:.6f} eV, R^2 = {self.r2:.5f} over bands "
            f"[{self.fit_window[0]}, {self.fit_window[1]}]; E* = "
            f"{self.estar_ev:.3f} eV; N_dft = {self.n_dft}, "
            f"N_T = {self.n_target}")


def weyl_ladder_fit(e_mean_ev: np.ndarray, lo: int, hi: int) -> tuple:
    """Fit ``E_n = E₀ + C·(n + n₀)^(2/3)`` to the k-mean DFT ladder.

    THE FREE-ELECTRON COUNTING LAW, USED AS A LADDER AND NOTHING ELSE.  In a
    finite volume the number of plane-wave states below E grows as
    ``(E − E₀)^{3/2}``, so the n-th one sits at ``E₀ + C·n^{2/3}``.  That is
    a statement about counting, not about the material, which is why it
    describes a real band ladder well enough to extend it (R² = 0.99975 on
    the Si 50 Ry deck) while saying nothing about any matrix element.

    ``n₀`` is scanned over :data:`WEYL_N0_SCAN` rather than fitted, because
    for fixed ``n₀`` the model is LINEAR in ``(1, (n + n₀)^{2/3})`` and the
    scan is therefore a sequence of exact least squares rather than a
    nonlinear optimisation.  The module has a standing rule against
    introducing a nonlinear fit into this estimator.

    Parameters
    ----------
    e_mean_ev : (n_band,) float
        Band energies in eV, averaged over k, indexed from band 1.
    lo, hi : int
        1-based inclusive band range to fit over.

    Returns
    -------
    (e0_ev, n0, c_ev, r2)
    """
    e_mean_ev = np.asarray(e_mean_ev, dtype=np.float64)
    lo, hi = int(lo), int(hi)
    if not (1 <= lo < hi <= e_mean_ev.size):
        raise ValueError(
            f"weyl_ladder_fit: band window [{lo}, {hi}] is not inside "
            f"[1, {e_mean_ev.size}]")
    ns = np.arange(lo, hi + 1, dtype=np.float64)
    e = e_mean_ev[ns.astype(np.int64) - 1]
    ss_tot = float(np.sum((e - e.mean()) ** 2))
    best = None
    for n0 in np.arange(*WEYL_N0_SCAN):
        x = (ns + n0) ** (2.0 / 3.0)
        design = np.vstack([np.ones_like(x), x]).T
        coef = np.linalg.lstsq(design, e, rcond=None)[0]
        r = e - design @ coef
        ss = float(r @ r)
        if best is None or ss < best[0]:
            best = (ss, float(coef[0]), float(n0), float(coef[1]))
    ss, e0, n0, c = best
    r2 = 1.0 - ss / ss_tot if ss_tot > 0 else float("nan")
    return e0, n0, c, r2


def plane_wave_band_count(ngk, nspinor: int) -> int:
    """``N_PW`` — the band count at which the sum is EXACTLY complete.

    The one-particle Hilbert space at k has dimension ``ngk(k)·nspinor``, so
    a band sum reaching that count has summed every state there is and there
    is nothing beyond it.  ``ngk`` varies by k (1604…1639 on the Si deck);
    the MINIMUM is the count at which no k-point is still short, and it is
    the conservative choice — a larger endpoint would ask the ladder to
    continue past where some k-point has run out of basis.
    """
    ngk = np.asarray(ngk).ravel()
    if ngk.size == 0:
        raise ValueError("plane_wave_band_count: empty ngk")
    return int(np.min(ngk)) * int(max(int(nspinor), 1))


def build_band_ladder(
    *,
    enk_ry: np.ndarray,
    kweights=None,
    n_target: int,
    b0: int = 0,
    fit_window: "tuple[int, int] | None" = None,
    estar_window: "tuple[int, int] | None" = None,
) -> BandLadder:
    """Assemble the :class:`BandLadder` the shell moments read.

    Parameters
    ----------
    enk_ry : (nk, n_dft) float
        DFT eigenvalues in Ry over the WFN's own k-set and band set —
        ABSOLUTE band indexing, band 1 first.  This is ``wfn.energies[0]``.
    kweights : (nk,) float or None
        k-point weights.  ``None`` means uniform, which is the correct
        reading of a full-BZ eigenvalue set.  Normalised here.
    n_target : int
        ``N_T``.  Ordinarily :func:`plane_wave_band_count`.
    b0 : int
        Absolute index of the band the CALLER's cuts are measured from.
    fit_window : (int, int) or None
        1-based inclusive band range for the Weyl fit.  ``None`` uses the
        top 90 % of the DFT ladder, floored at ``b0 + 1``: the Weyl form is
        asymptotic, so the fit belongs where the ladder already is.
    estar_window : (int, int) or None
        1-based inclusive band range whose median ``ε − E₀`` sets ``E*``.
        ``None`` uses the upper half of the DFT ladder.  ``E*`` cancels from
        every ratio, so this choice cannot move a result; it only keeps the
        powers conditioned.
    """
    e = np.asarray(enk_ry, dtype=np.float64) * RYD_TO_EV
    if e.ndim != 2:
        raise ValueError(
            f"build_band_ladder: expected (nk, n_band) energies, got "
            f"shape {np.shape(enk_ry)}")
    nk, n_dft = e.shape
    n_target = int(n_target)
    b0 = int(b0)
    if n_target <= n_dft and n_target <= b0:
        raise ValueError(
            f"build_band_ladder: n_target = {n_target} does not reach past "
            f"the band offset b0 = {b0}")

    w = (np.full(nk, 1.0 / nk) if kweights is None
         else np.asarray(kweights, dtype=np.float64).ravel()[:nk])
    if w.size != nk or not np.all(w > 0):
        raise ValueError(
            f"build_band_ladder: need {nk} positive k weights, got {w}")
    w = w / w.sum()

    e_mean = e.mean(axis=0)
    if fit_window is None:
        lo = max(b0 + 1, int(round(0.10 * n_dft)), 1)
        fit_window = (min(lo, n_dft - 1), n_dft)
    e0, n0, c, r2 = weyl_ladder_fit(e_mean, *fit_window)

    if estar_window is None:
        estar_window = (max(1, n_dft // 2), n_dft)
    lo_s, hi_s = int(estar_window[0]), int(estar_window[1])
    estar = float(np.median(e[:, lo_s - 1:hi_s] - e0))
    if not (estar > 0.0):
        raise ValueError(
            f"build_band_ladder: E* came out {estar}; the conditioning "
            f"scale must be positive (bands {estar_window} lie at or below "
            f"the fitted band bottom E0 = {e0} eV)")

    n_ext = max(0, n_target - n_dft)
    ns_ext = np.arange(n_dft + 1, n_dft + 1 + n_ext, dtype=np.float64)
    e_weyl = e0 + c * (ns_ext + n0) ** (2.0 / 3.0)

    return BandLadder(
        e_dft_ev=np.ascontiguousarray(e.T),          # (n_dft, nk)
        e_weyl_ev=e_weyl,
        w_k=w,
        e0_ev=float(e0),
        n0=float(n0),
        c_ev=float(c),
        r2=float(r2),
        estar_ev=estar,
        n_dft=int(n_dft),
        n_target=n_target,
        fit_window=(int(fit_window[0]), int(fit_window[1])),
        b0=b0,
    )


def solve_shell_exponents(ladder: BandLadder, shell2, shell3, ratio):
    """Bracketed root find for ``β``, one per external state.

    Solves ``log I₃(β) − log I₂(β) − log(ratio) = 0`` on
    :data:`SHELL_EXPONENT_BRACKET` by bisection, vectorised over the state
    axis.  ``g(β) = log I₃ − log I₂`` is strictly DECREASING — its derivative
    is ``⟨log x⟩₂ − ⟨log x⟩₃`` and shell 3 sits above shell 2 — so the root
    is unique where it exists, which is what makes a bracketed find the right
    tool and a clip the wrong answer.

    Parameters
    ----------
    ladder : BandLadder
    shell2, shell3 : (int, int)
        ABSOLUTE band ranges ``(lo, hi]`` of the two observed shells.
    ratio : array
        ``|D₃/D₂|`` per state, strictly positive and finite.

    Returns
    -------
    (beta, code)
        ``beta`` is NaN wherever ``code`` is not :data:`SHELL_OK`.
    """
    r = np.asarray(ratio, dtype=np.float64)
    shape = r.shape
    flat = r.ravel()
    beta = np.full(flat.shape, np.nan)
    code = np.full(flat.shape, SHELL_OK, dtype=np.int64)

    bad = ~np.isfinite(flat) | (flat <= 0.0)
    code[bad] = SHELL_FAIL_ZERO
    live = ~bad
    if not live.any():
        return beta.reshape(shape), code.reshape(shape)

    logr = np.log(flat[live])
    lo_b, hi_b = SHELL_EXPONENT_BRACKET

    def f(b):
        return (ladder.log_moment(*shell3, b)
                - ladder.log_moment(*shell2, b) - logr)

    lo = np.full(logr.shape, float(lo_b))
    hi = np.full(logr.shape, float(hi_b))
    flo, fhi = f(lo), f(hi)
    ok = np.isfinite(flo) & np.isfinite(fhi) & (flo * fhi <= 0.0)

    for _ in range(SHELL_EXPONENT_BISECTIONS):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        ok &= np.isfinite(fm)
        take_hi = (flo * fm) <= 0.0
        hi = np.where(take_hi, mid, hi)
        fhi = np.where(take_hi, fm, fhi)
        lo = np.where(take_hi, lo, mid)
        flo = np.where(take_hi, flo, fm)
    b = 0.5 * (lo + hi)

    # A ROOT ON THE WALL IS A CLIP, NOT A FIT.  After 80 halvings a genuine
    # interior root is located to ~1e-22; still sitting on an edge means the
    # bisection never found a sign change inside.
    on_edge = ((b - lo_b) <= SHELL_EXPONENT_EDGE_TOL) | (
        (hi_b - b) <= SHELL_EXPONENT_EDGE_TOL)

    sub_code = np.where(~ok, SHELL_FAIL_NO_ROOT,
                        np.where(on_edge, SHELL_FAIL_EDGE, SHELL_OK))
    sub_beta = np.where(sub_code == SHELL_OK, b, np.nan)
    beta[live] = sub_beta
    code[live] = sub_code
    return beta.reshape(shape), code.reshape(shape)


@dataclass(frozen=True)
class SpectralShellFit:
    """Result of the spectrum-resolved shell estimator.

    Elementwise in the trailing (k, band) state axes, exactly as
    :class:`ExtrapolationFit` is — every external state carries its own β and
    its own tail ratio, which is the whole point (see the module docstring's
    ruling on pooling).
    """

    counts: np.ndarray            # (3,) int — N₁, N₂, N₃, Σ-RELATIVE
    s_at_counts: np.ndarray       # (3, ...) — S(N₁), S(N₂), S(N₃)
    s_inf: np.ndarray             # (...)    — Ŝ; NaN where the state failed
    beta: np.ndarray              # (...)    — the per-state exponent
    tail_ratio: np.ndarray        # (...)    — I_tail(β)/I₃(β)
    delta_tail: np.ndarray        # (...)    — |Ŝ − S(N₃)|
    d2: np.ndarray                # (...)    — S(N₂) − S(N₁)
    d3: np.ndarray                # (...)    — S(N₃) − S(N₂)
    failure: np.ndarray           # (...) int — SHELL_* code
    ladder: BandLadder
    shells: tuple                 # ((lo,hi), (lo,hi), (lo,hi)) ABSOLUTE

    @property
    def n_failed(self) -> int:
        return int(np.count_nonzero(np.asarray(self.failure) != SHELL_OK))

    @property
    def n_states(self) -> int:
        return int(np.size(self.failure))

    def weights(self) -> np.ndarray:
        """``c`` with ``Ŝ = Σ_b c_b · S(N_b)``, shape ``(3,) + state shape``.

        ``Ŝ = S(N₃) + D₃·r = 0·S(N₁) − r·S(N₂) + (1 + r)·S(N₃)``, so the
        coefficients are REAL and sum to 1 identically — the same affine
        property :func:`extrapolation_weights` documents, and for the same
        reason: a Σ that does not depend on the band count must come through
        unchanged rather than scaled.

        Unlike the scalar case these carry the state shape, because β does.
        Refuses on a failed state rather than emitting a coefficient for it.
        """
        if self.n_failed:
            raise SpectralShellExtrapolationFailed(self.failure_report())
        r = np.asarray(np.real(self.tail_ratio), dtype=np.float64)
        return np.stack([np.zeros_like(r), -r, 1.0 + r], axis=0)

    def uncertainty(self, quantile: str = "p90") -> np.ndarray:
        """Extrapolation uncertainty as a fraction of the applied correction.

        THE FRACTION IS THE ONE CALIBRATED FOR ``band_index_only`` and it has
        NOT been re-calibrated here.  It is carried so the h5 payload and the
        SC-tolerance block keep the same shape under both estimators, and it
        is an ENVELOPE, not a per-state bar — see
        :data:`TAIL_UNCERTAINTY_FRACTION`.  Because ``spectral_shell`` applies
        a SMALLER correction where it is more accurate, this bar is
        conservative on the states the held-out test scored; nothing has
        measured it on the states it did not.
        """
        p90, p99 = TAIL_UNCERTAINTY_FRACTION
        f = {"p90": p90, "p99": p99}[quantile]
        return f * np.abs(self.delta_tail)

    def at(self, index) -> "SpectralShellFit":
        """Restrict every field to a subset of the trailing state axes.

        An indexing operation, not a refit: the estimator is elementwise in
        (k, band).  Same splice discipline as :meth:`ExtrapolationFit.at` —
        ``s_at_counts`` carries the three-point axis in front.
        """
        idx = index if isinstance(index, tuple) else (index,)
        return SpectralShellFit(
            counts=self.counts,
            s_at_counts=self.s_at_counts[(slice(None),) + idx],
            s_inf=self.s_inf[index],
            beta=self.beta[index],
            tail_ratio=self.tail_ratio[index],
            delta_tail=self.delta_tail[index],
            d2=self.d2[index],
            d3=self.d3[index],
            failure=np.asarray(self.failure)[index],
            ladder=self.ladder,
            shells=self.shells,
        )

    def failure_report(self, *, limit: int = 12) -> str:
        """Every failed state, named, with its reason.  '' when none failed."""
        fail = np.asarray(self.failure)
        idx = np.argwhere(fail != SHELL_OK)
        if idx.size == 0:
            return ""
        d2 = np.real(np.asarray(self.d2))
        d3 = np.real(np.asarray(self.d3))
        lines = [
            f"spectral_shell band extrapolation FAILED on {len(idx)} of "
            f"{self.n_states} external states.  The estimator does not "
            f"substitute a value for a failed state and does not clip an "
            f"exponent to its bracket, so the run stops here.",
            f"  band counts {tuple(int(c) for c in self.counts)}; shells "
            f"(absolute band index) 2 = {self.shells[0]}, 3 = "
            f"{self.shells[1]}, tail = {self.shells[2]}; "
            f"{self.ladder.describe()}",
        ]
        for row in idx[:limit]:
            key = tuple(int(v) for v in row)
            reason = SHELL_FAILURE_REASONS[int(fail[key])]
            lines.append(
                f"    state {key}: D2 = {d2[key]:+.6e}, D3 = {d3[key]:+.6e} "
                f"-- {reason}")
        if len(idx) > limit:
            lines.append(f"    ... and {len(idx) - limit} more.")
        lines += [
            "  TWO HONEST WAYS FORWARD.  Raise `number_bands_sigma` so the "
            "two shells sit further into the tail where the local power law "
            "holds; or set `band_extrapolation_estimator = band_index_only` "
            "to run the incumbent S_inf + A/N fit, which always returns a "
            "number and whose error against a measured S(508) is 3-10x "
            "larger (see gw.band_extrapolation's module docstring).",
        ]
        return "\n".join(lines)


def fit_band_extrapolation_spectral(
    counts, s_at_counts: np.ndarray, ladder: BandLadder,
) -> SpectralShellFit:
    """The spectrum-resolved shell estimator over three points.

    Parameters
    ----------
    counts : sequence of int, length 3
        ``N₁, N₂, N₃`` — the band counts the three cumulative bracket sums
        reach, in the CALLER's indexing (``ladder.b0`` converts them to
        absolute band indices).
    s_at_counts : (3, ...) complex array
        The cumulative bracket sums, any trailing shape.
    ladder : BandLadder
        Built from the DFT eigenvalues alone.

    Returns
    -------
    SpectralShellFit

    Notes
    -----
    The shell increments are formed on the REAL part for the purpose of the
    sign test and the ratio: β is a decay exponent of a real, positive
    spectral falloff, and the imaginary part of Σ_c at a QP energy is a
    lifetime, not a piece of the same tail.  The correction ``D₃·r`` is then
    applied to the COMPLEX increment, so Im Σ_c is extrapolated with the same
    per-state ratio rather than being dropped or fitted separately.
    """
    N = np.asarray(counts, dtype=np.int64)
    S = np.asarray(s_at_counts)
    if N.ndim != 1 or N.size != 3:
        raise ValueError(
            f"fit_band_extrapolation_spectral: need exactly 3 counts, got "
            f"{N}.  The estimator reads two shell increments, which is three "
            f"cumulative points.")
    if S.shape[0] != N.size:
        raise ValueError(
            f"fit_band_extrapolation_spectral: S leading axis {S.shape[0]} "
            f"!= {N.size} counts")
    if not (N[0] < N[1] < N[2]):
        raise ValueError(
            f"fit_band_extrapolation_spectral: band counts {tuple(N)} are "
            f"not strictly ascending; the shells would be empty or inverted.")

    a1, a2, a3 = (ladder.absolute(int(c)) for c in N)
    if a3 >= ladder.n_target:
        raise BandExtrapolationRefused(
            f"spectral_shell band extrapolation: the largest band count "
            f"N3 = {int(N[2])} (absolute band {a3}) is already at or past "
            f"the finite-basis endpoint N_T = {ladder.n_target} "
            f"(= min(ngk)*nspinor).  There is no tail left to integrate: "
            f"the band sum is complete, and the honest report is S(N3) "
            f"itself.  Set `use_band_extrapolation = false`.")
    shells = ((a1, a2), (a2, a3), (a3, ladder.n_target))

    D2 = S[1] - S[0]
    D3 = S[2] - S[1]
    r2, r3 = np.real(D2), np.real(D3)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(r3) / np.abs(r2)
    beta, code = solve_shell_exponents(ladder, shells[0], shells[1], ratio)

    # The two data-side refusals, applied AFTER the solve so that a state
    # carrying both gets the more specific reason.
    zero = (r2 == 0.0) | (r3 == 0.0)
    sign = ~zero & (np.sign(r2) != np.sign(r3))
    code = np.where(zero, SHELL_FAIL_ZERO, np.where(sign, SHELL_FAIL_SIGN,
                                                    code))
    beta = np.where(code == SHELL_OK, beta, np.nan)

    good = code == SHELL_OK
    safe_beta = np.where(good, beta, 1.0)
    log_i3 = ladder.log_moment(*shells[1], safe_beta)
    log_it = ladder.log_moment(*shells[2], safe_beta)
    tail_ratio = np.where(good, np.exp(log_it - log_i3), np.nan)

    s_inf = S[2] + D3 * tail_ratio
    delta_tail = np.abs(s_inf - S[2])

    return SpectralShellFit(
        counts=N,
        s_at_counts=S,
        s_inf=s_inf,
        beta=beta,
        tail_ratio=tail_ratio,
        delta_tail=delta_tail,
        d2=D2,
        d3=D3,
        failure=code,
        ladder=ladder,
        shells=shells,
    )


def spectral_trust_verdict(fit: SpectralShellFit) -> str:
    """One line saying whether the shell ratios support a local power law.

    Unlike :func:`trust_verdict` there is no residual to inspect: the
    estimator has one unknown and two shells, so it is an interpolant of the
    ratio by construction and a "fit quality" number would be identically
    zero.  What CAN be reported is (a) whether every state solved at all, and
    (b) the spread of the per-state β, which is the estimator's own statement
    about how differently the states are converging.  A tight spread is not
    evidence the extrapolation is right; it is evidence the states agree.
    """
    if fit.n_failed:
        return (f"NOT TRUSTWORTHY - {fit.n_failed} of {fit.n_states} states "
                f"FAILED to produce an exponent (see the failure report); "
                f"no value was substituted for them.")
    b = np.asarray(np.real(fit.beta), dtype=np.float64).ravel()
    b = b[np.isfinite(b)]
    if b.size == 0:
        return "NOT TRUSTWORTHY - no finite exponent on any state."
    lo, med, hi = (float(np.percentile(b, 10)), float(np.median(b)),
                   float(np.percentile(b, 90)))
    note = ""
    if lo < 1.0:
        note = ("  b < 1 on the lowest decile: a tail that shallow is not "
                "summable in the way the estimator assumes -- read the "
                "correction as a lower bound.")
    return (f"solved on all {fit.n_states} states - beta median {med:.2f}, "
            f"p10/p90 {lo:.2f}/{hi:.2f}.  The per-state spread IS the "
            f"resolution this estimator exists to provide; it is not a "
            f"quality metric.{note}")


def trust_verdict(fit: ExtrapolationFit, *, ratio_warn: float = 0.35) -> str:
    """One line saying whether the three points support the 1/N model.

    Two failure signatures, both read off the REAL part (the part that moves
    a QP energy):

    * **sign reversal** — the (1,2) and (2,3) intervals imply corrections of
      opposite sign, i.e. the sum is not monotonically approaching anything
      on this range;
    * **Δ_model comparable to Δ_tail** — the pairwise intercepts disagree by
      an appreciable fraction of the correction being applied, so the fitted
      correction is smaller than its own model error.

    Reported as prose, not as a refusal: the numbers are the product, and a
    run that stops on a soft quality metric would be worse than one that
    prints the metric.

    ``ratio_warn = 0.35`` IS NOT A MEASURED THRESHOLD.  Every other constant
    in this module carries a provenance paragraph; this one does not, and it
    should be read as a round number chosen to sit well above the 0.057 ratio
    measured on a clean curve (module docstring) and well below 1.  Nothing
    has calibrated where between those it belongs, and the docstring's own
    warning applies with full force: a passing ratio was measured to coexist
    with a 55 meV MAE / 167 meV max intercept error, so tightening this number
    would not have caught that and loosening it would not have missed it.
    Treat a verdict as a statement about SCATTER only.
    """
    d_tail = float(np.max(np.abs(np.real(fit.delta_tail))))
    d_model = float(np.max(np.abs(np.real(fit.delta_model))))
    a12 = np.real(fit.pair_s_inf[(0, 1)] - fit.s_at_counts[1])
    a23 = np.real(fit.pair_s_inf[(1, 2)] - fit.s_at_counts[2])
    reversed_sign = bool(np.any(np.sign(a12) * np.sign(a23) < 0))
    ratio = d_model / d_tail if d_tail > 0.0 else float("inf")

    if reversed_sign:
        return (f"NOT TRUSTWORTHY — the (1,2) and (2,3) intervals imply "
                f"corrections of OPPOSITE SIGN on at least one state: the "
                f"band sum is not monotone in 1/N over these counts, so the "
                f"extrapolation is not measuring a tail.")
    if ratio > ratio_warn:
        return (f"NOT TRUSTWORTHY — Δ_model/Δ_tail = {ratio:.2f} > "
                f"{ratio_warn:.2f}: the pairwise intercepts disagree by an "
                f"appreciable fraction of the correction, so these three "
                f"counts do not resolve the 1/N tail.  Use more bands.")
    return (f"consistent — Δ_model/Δ_tail = {ratio:.2f}; the three pairwise "
            f"intercepts agree to well within the applied correction.")


def tolerance_bar_ev(fit, quantile: str = "p90") -> tuple:
    """``(median, max)`` of the extrapolation uncertainty over all states, eV.

    Takes EITHER fit type.  Both :class:`ExtrapolationFit` and
    :class:`SpectralShellFit` expose ``uncertainty(quantile)`` as a fraction
    of their own ``Delta_tail``, and that is the only thing this reads — so
    the SC-tolerance block below is one code path under both estimators
    rather than two that could drift.

    Two numbers because they answer different questions and the module has a
    standing rule against reporting the max alone: a max over (k, band) is set
    by the top of the QP window, whose Σ_c is the largest and least converged
    quantity in the run, so it describes that state rather than the
    calculation (see :meth:`ExtrapolationFit.at`).  The MEDIAN is the bar on a
    typical state and is what the ruling below triggers on; the MAX is quoted
    beside it as the envelope.
    """
    u = np.abs(np.real(np.asarray(fit.uncertainty(quantile))))
    return float(np.median(u)), float(np.max(u))


def sc_tolerance_ruling(fit, tol_ev: float,
                        *, quantile: str = "p90") -> tuple:
    """Is the SC convergence tolerance inside the extrapolation's own bar?

    Takes either fit type; see :func:`tolerance_bar_ev`.

    Returns ``(inside: bool, text: str)``.  ``text`` is always a block worth
    printing; ``inside`` says whether it is a warning or a statement.

    THE RULING, 2026-08-16: this WARNS, unmissably and every iteration.  It
    does NOT refuse.  Both halves were argued, and the two reasons for
    warning are independent -- either alone would settle it.

    (1) A REFUSAL WOULD FIRE ON THE SHIPPED DEFAULT.  ``sc_tol_ev`` defaults
        to 1.0e-4 eV = 0.1 meV (``gw_config._DEFAULTS``) and
        ``use_band_extrapolation`` now defaults to TRUE, while the bar on
        this deck family runs to tens of meV.  The default configuration is
        therefore ALWAYS "inside the bar" by two to three orders of
        magnitude.  A gate that refuses the combination the code ships with
        is not a safety property; it is a build that cannot run, and it
        would be routed around within a day by the first operator who needs
        a number.

    (2) THE TWO NUMBERS DO NOT MEASURE THE SAME THING, so "inside" is not an
        inconsistency to refuse -- it is a misreading to prevent.
        ``sc_tol_ev`` bounds the ITERATION-TO-ITERATION displacement of E_nk:
        it asks "has the fixed point been reached".  The extrapolation bar is
        a SYSTEMATIC uncertainty on the absolute Σ_c: it asks "where is the
        fixed point".  A systematic, iteration-independent bias in Σ_c does
        not stop the loop reaching its fixed point to 0.1 meV; it MOVES the
        fixed point.  So a run that reports "converged, RMS dE = 8e-5 eV" is
        making a TRUE statement about the iteration and would be making a
        FALSE one about the accuracy of E_nk -- and nothing in the loop
        distinguishes those two readings on the operator's behalf.  That is
        what this block exists to do, and refusing would be answering a
        question about accuracy by breaking a mechanism about convergence.

    WHAT WOULD ACTUALLY JUSTIFY A REFUSAL, and is a different measurement:
    the extrapolated correction WOBBLING between iterations by more than
    ``tol_ev``.  That is not a systematic bias, it is iteration noise
    injected into the fixed-point map, and it can genuinely prevent
    convergence rather than relocate it.  The driver reports the
    per-iteration change in ``Delta_tail`` for exactly this reason
    (``gw.sc_iteration``); this function cannot see it, because it is handed
    one iteration's fit.
    """
    med, mx = tolerance_bar_ev(fit, quantile)
    tol = float(tol_ev)
    inside = bool(tol < med)
    head = ("*** SC TOLERANCE IS INSIDE THE EXTRAPOLATION BAR ***"
            if inside else
            "SC tolerance vs extrapolation bar")
    lines = [
        f"     {head}",
        f"       sc_tol_ev              = {tol * 1e3:11.4f} meV  "
        f"(per-band RMS dE between SC iterations)",
        f"       extrapolation {quantile:<4s}     = {med * 1e3:11.4f} meV "
        f"median over states, {mx * 1e3:.4f} meV max "
        f"({100 * TAIL_UNCERTAINTY_FRACTION[0]:.0f} % of Delta_tail)",
    ]
    if inside:
        lines += [
            f"       The loop is being asked to converge "
            f"{med / tol if tol > 0 else float('inf'):,.0f}x TIGHTER than "
            f"the uncertainty on the quantity it is converging.",
            f"       THESE ARE NOT THE SAME NUMBER AND THIS IS NOT A "
            f"CONTRADICTION.  sc_tol_ev bounds the ITERATION-TO-ITERATION "
            f"displacement -- 'has the fixed point been reached'.  The bar "
            f"is a SYSTEMATIC uncertainty on absolute Sigma_c -- 'where IS "
            f"the fixed point'.  A converged run here is genuinely converged "
            f"and its E_nk is genuinely uncertain by the bar.",
            f"       SO: do NOT quote the SC residual as the accuracy of "
            f"E_nk.  Quote {med * 1e3:.3f} meV (median) / {mx * 1e3:.3f} meV "
            f"(max envelope), and note it covers the EXTRAPOLATION ONLY -- "
            f"not the difference from BerkeleyGW, the ISDF basis, or the "
            f"W-side band count.",
        ]
    else:
        lines.append(
            f"       Tolerance is outside the median bar; the SC residual is "
            f"a meaningful statement at this scale.  The bar still covers "
            f"the EXTRAPOLATION only.")
    return inside, "\n".join(lines)


#: How large the static-limit term's own band tail may be, as a fraction of
#: the extrapolation bar the run reports, before
#: :func:`static_limit_tail_ruling` escalates from a statement to a warning.
#:
#: NOT A MEASURED THRESHOLD, and said so plainly rather than dressed up: it is
#: 1.0, i.e. "escalate exactly when the unreported error term reaches the size
#: of the reported one".  That is a definition, not a calibration, and it is
#: the only bar that does not require a constant nobody has measured.  The
#: numbers it is meant to catch are in
#: ``sandbox:reports/ppm_static_limit_extrapolation_2026-08-16/``.
STATIC_LIMIT_TAIL_WARN_FRACTION: float = 1.0

#: Absolute floor, eV, below which the omitted static tail is not escalated no
#: matter how it compares to the bar.
#:
#: THIS EXISTS BECAUSE A RATIO OF TWO NEAR-ZERO NUMBERS IS NOT A SIGNAL, and
#: that was MEASURED rather than anticipated: on a Si arm whose three bracket
#: points came out bit-identical (the brackets above the deck's real band
#: window contributed exactly nothing, so ``Delta_tail`` was 0), the ruling
#: divided a ~1e-9 meV tail by a ~1e-14 meV bar and reported
#: ``ratio = 122880``, escalating on a run where the true omission was zero to
#: every digit printed.  ``1e-4 eV`` = 0.1 meV is the default ``sc_tol_ev``,
#: i.e. the tightest tolerance anything in this code asks for; below it there
#: is no quantity this term could change.
STATIC_LIMIT_TAIL_FLOOR_EV: float = 1.0e-4


def static_limit_tail_ruling(
    fit: ExtrapolationFit,
    static_coh_at_counts,
    *,
    quantile: str = "p90",
    warn_fraction: float = STATIC_LIMIT_TAIL_WARN_FRACTION,
    floor_ev: float = STATIC_LIMIT_TAIL_FLOOR_EV,
) -> tuple:
    """How much of ``S_inf`` is a static Coulomb hole that was NOT extrapolated.

    Returns ``(exceeds: bool, text: str, stats: dict)``.

    THE HOLE THIS CLOSES.  ``sigma_dispatch``'s PPM-only guard keeps this
    estimator away from a static Coulomb hole, because the ``1/N → 0`` limit
    ANTI-converges for one — 94.9 → 288.2 meV MAE as nband goes 60 → 124,
    ~340 meV past BerkeleyGW's exact closure (module docstring).  That guard
    reads ``compute_mode``.  ``ppm_invalid_mode = "static_limit"`` — the
    SHIPPING DEFAULT — adds an analytic static-COHSEX term for every pole whose
    fitted ``Ω² < 0``, and it does so *underneath* the mode: the
    ``compute_mode`` genuinely IS ``gn_ppm``, so a per-``compute_mode`` check
    cannot observe a per-MODE contaminant and this survived registration
    unmeasured.  **This function is that check moved to the level where the
    contamination happens** — it triggers on the term itself, at whatever
    ``compute_mode``, and it reports a number rather than an opinion.

    WHAT IS ALREADY RIGHT, AND MUST NOT BE "FIXED".  ``ppm_sigma`` folds the
    static term into bracket 0 ONLY, so it is a CONSTANT on the band-sum
    series.  This estimator is affine with ``sum(c) == 1``, so a constant
    passes into ``S_inf`` exactly 1:1 and contributes NOTHING to ``A``,
    ``Δ_tail``, ``Δ_model``, the residual or :func:`trust_verdict`.  That is
    equivalent to extrapolating the dynamical part alone and adding the static
    part back afterwards, which is the correct treatment and is what the
    anti-convergence measurement demands.  Band-resolving the term — making
    ``S(N_i)`` carry ``Σ_static(N_i)`` — would feed a static Coulomb hole to
    the 1/N law and is the one change this ruling exists to argue against.

    WHAT IS LEFT, AND WHY IT IS INVISIBLE WITHOUT THIS.  A constant is not
    band-count independent just because it is treated as one.  The term's
    Coulomb-hole half runs over ``s.full`` with no occupation projector
    (``cohsex_sigma.sigma_coh``), so it carries the same slowly convergent
    unoccupied tail everything else here is about — and folding it in as a
    constant pins it at ``N₃`` and never extrapolates it.  ``S_inf`` therefore
    contains a band-truncated static Coulomb hole whose remaining tail the
    reported bar does NOT cover, and *because* it is a constant, every
    diagnostic above is blind to it by construction.  Nothing in the fit can
    see this.  Only a separate measurement can, which is what
    ``ppm_sigma._invalid_static_coh_by_bracket`` supplies.

    THE NUMBER.  ``delta_static = Σ_i c_i·C(N_i) − C(N₃)`` — the correction
    this estimator WOULD apply to the static term if it were allowed to.  It
    is not applied; it is the SCALE of what is missing.  Its true magnitude is
    smaller (the 1/N law overshoots a static CH), so ``delta_static`` is an
    upper bound on the omitted tail rather than an estimate of it, and it is
    reported that way.

    ``span`` = ``C(N₃) − C(N₁)`` is quoted beside it as the direct refutation
    of "this term is band-count independent": if that claim were true the span
    would be zero.

    Parameters
    ----------
    fit
        The band-diagonal fit whose ``S_inf`` this qualifies.
    static_coh_at_counts : ``(3, ...)`` array, eV
        CUMULATIVE static-limit Coulomb-hole term at each of ``fit.counts``,
        on the SAME trailing state axes as ``fit.s_inf``.
    """
    C = np.asarray(static_coh_at_counts)
    if C.shape[0] != fit.counts.size:
        raise ValueError(
            f"static_limit_tail_ruling: static term has {C.shape[0]} band "
            f"points but the fit has {fit.counts.size}.  They must be the "
            f"SAME cumulative band counts — a mismatch means the two were "
            f"built from different bracket plans, and comparing them would "
            f"be meaningless rather than merely wrong.")
    if C.shape[1:] != np.shape(fit.s_inf):
        raise ValueError(
            f"static_limit_tail_ruling: static term state axes {C.shape[1:]} "
            f"!= fit state axes {np.shape(fit.s_inf)}.")

    w = extrapolation_weights(fit.counts)
    wb = w.reshape((-1,) + (1,) * (C.ndim - 1))
    delta_static = np.sum(wb * C, axis=0) - C[-1]
    span = C[-1] - C[0]

    d = np.abs(np.real(delta_static))
    d_med, d_max = float(np.median(d)), float(np.max(d))
    s = np.abs(np.real(span))
    bar_med, bar_max = tolerance_bar_ev(fit, quantile)
    ratio = d_med / bar_med if bar_med > 0.0 else float("inf")
    # BOTH conditions, and the floor is the one that stops a 0/0.  A tail
    # below STATIC_LIMIT_TAIL_FLOOR_EV cannot move any quantity this code
    # reports, so however it compares to a bar that is itself ~0, there is
    # nothing to escalate.  See that constant for the run that proved it.
    exceeds = bool(ratio > warn_fraction and d_med > floor_ev)

    stats = {
        "delta_static_median_ev": d_med,
        "delta_static_max_ev": d_max,
        "span_median_ev": float(np.median(s)),
        "span_max_ev": float(np.max(s)),
        "bar_median_ev": bar_med,
        "bar_max_ev": bar_max,
        "ratio_median": ratio,
        "floor_ev": float(floor_ev),
    }

    head = ("*** STATIC-LIMIT TERM'S BAND TAIL EXCEEDS THE REPORTED "
            "EXTRAPOLATION BAR ***" if exceeds else
            "static-limit term inside the extrapolation bar")
    lines = [
        f"     -- ppm_invalid_mode = static_limit, INSIDE a band-extrapolated "
        f"Sigma_c --",
        f"     {head}",
        f"       band-count SPAN  C(N3)-C(N1) = {stats['span_median_ev'] * 1e3:11.4f} meV "
        f"median over states, {stats['span_max_ev'] * 1e3:.4f} meV max",
        f"       omitted tail (upper bound)   = {d_med * 1e3:11.4f} meV "
        f"median over states, {d_max * 1e3:.4f} meV max",
        f"       reported extrapolation {quantile:<4s}  = {bar_med * 1e3:11.4f} meV "
        f"median over states, {bar_max * 1e3:.4f} meV max",
        f"       ratio (median/median)        = {ratio:11.4f}  "
        f"(warn above {warn_fraction:.2f} AND tail above "
        f"{floor_ev * 1e3:.3f} meV)",
        f"       WHAT THIS IS.  The static-COHSEX term this run adds for its "
        f"invalid PPM poles is folded into bracket 0 as a CONSTANT, which is "
        f"CORRECT: a static Coulomb hole must not be run through the 1/N law "
        f"(it anti-converges, ~340 meV past the exact answer, and gets WORSE "
        f"with more bands).  The constant reaches S_inf 1:1 and moves no "
        f"diagnostic above.",
        f"       WHAT IT COSTS.  A SPAN of {stats['span_median_ev'] * 1e3:.3f} meV is a "
        f"measurement that the term is NOT band-count independent, so pinning "
        f"it at N3={int(fit.counts[-1])} leaves a static tail in S_inf that "
        f"the bar above does NOT cover.  The omitted-tail number is what this "
        f"estimator WOULD have applied and is an UPPER BOUND on the true "
        f"omission, not an estimate of it.",
    ]
    if exceeds:
        lines += [
            f"       SO: the quoted extrapolation bar UNDERSTATES the band-"
            f"convergence error of S_inf on a typical state, because the "
            f"largest single band-truncation term left in it is not in the "
            f"fit at all.  Either raise nband until the span collapses, or "
            f"set `ppm_invalid_mode = zero` (BGW mode 0), which DROPS the "
            f"invalid poles instead of making them static and leaves nothing "
            f"here to omit.  Both are deck changes; neither is a code fix.",
        ]
    elif d_med <= floor_ev:
        lines.append(
            f"       The omitted static tail is below {floor_ev * 1e3:.3f} meV "
            f"in absolute terms, so it cannot move any quantity this run "
            f"reports and the ratio beside it is a ratio of two numbers that "
            f"are both ~0.  Nothing to act on.")
    else:
        lines.append(
            f"       The omitted static tail is smaller than the bar already "
            f"reported, so quoting the bar is not misleading at this band "
            f"count.  It is still a SEPARATE error term and does not belong "
            f"inside it.")
    return exceeds, "\n".join(lines), stats


#: Dataset names the fit contributes to ``sigma_mnk.h5``, all ``(nk, nb)``
#: band-diagonal and all in eV — the same unit and the same band diagonal as
#: ``sigma_c_kij_ev``'s, evaluated at the same E_nk on the same ω grid.
EXTRAP_DATASETS = (
    "sigma_c_extrap_inf_kn_ev",     # S_∞, the extrapolated Σ_c
    "sigma_c_extrap_last_kn_ev",    # S(N₃), the ordinary full-band Σ_c
    "sigma_c_extrap_ampl_kn_ev",    # A, the 1/N coefficient
    "sigma_c_extrap_sigma_kn_ev",   # the p90 uncertainty envelope
)

#: The same, for ``spectral_shell``.  THREE OF THE FOUR NAMES ARE SHARED, and
#: the fourth is DIFFERENT ON PURPOSE.  ``inf``, ``last`` and ``sigma`` mean
#: the same thing under both estimators, so a consumer that reads the
#: extrapolated Σ_c, the un-extrapolated one and the bar needs no fork.  But
#: ``ampl`` is ``A``, the coefficient of ``1/N`` — a quantity the spectral
#: estimator does not have and must not appear to have.  Writing β into a
#: dataset named ``ampl`` would silently redefine a shipped array; the
#: estimator-specific fourth name is how a reader can tell which estimator
#: produced the file even without reading the ``estimator`` attribute.
SPECTRAL_EXTRAP_DATASETS = (
    "sigma_c_extrap_inf_kn_ev",     # Ŝ, the extrapolated Σ_c
    "sigma_c_extrap_last_kn_ev",    # S(N₃), the ordinary full-band Σ_c
    "sigma_c_extrap_beta_kn",       # β, the per-state exponent (dimensionless)
    "sigma_c_extrap_sigma_kn_ev",   # the p90 uncertainty envelope
)


def _bracket_h5_attrs(plan: BandBracketPlan) -> dict:
    """Artifact provenance shared by both estimators.

    ``bracket_fractions`` stays for compatibility, but is empty when fractions
    did not select the cuts.  Writing 0.80/0.90 for the conduction-coordinate
    scheme would be worse than omitting provenance: it would be a false fact.
    """
    if plan.bracket_scheme == "total_fractions":
        # Preserve the incumbent artifact exactly: absence of the new scheme
        # attribute means the historical/default total-fractions geometry.
        return {
            "bracket_fractions": np.asarray(
                BRACKET_FRACTIONS, dtype=np.float64),
        }
    return {
        "band_extrapolation_bracket_scheme": str(plan.bracket_scheme),
        "bracket_fractions": np.asarray((), dtype=np.float64),
        "bracket_boundary_mean_energy_ev": np.asarray(
            plan.boundary_mean_energy_ev, dtype=np.float64),
    }


def _bracket_geometry_text(plan: BandBracketPlan) -> str:
    """One log spelling of the planner semantics for both estimators."""
    if plan.bracket_scheme == "total_fractions":
        return (f"fractions = {BRACKET_FRACTIONS} of the TOTAL band count "
                f"{plan.counts[-1]}")
    if plan.bracket_scheme == "conduction_energy_midpoint":
        return (
            "bracket scheme = conduction_energy_midpoint "
            "(N1 at 50% of included conduction bands; N2 halfway in "
            "k-mean DFT boundary energy)")
    return f"bracket scheme = {plan.bracket_scheme}"


def _bracket_report_lines(plan: BandBracketPlan, unit: str) -> list[str]:
    """The resolved cuts, written identically by both estimator reports."""
    lines: list[str] = []
    for i, (req, got, (lo, hi), me, edge) in enumerate(zip(
            plan.requested, plan.counts, plan.bounds, plan.mean_energy_ev,
            plan.boundary_mean_energy_ev)):
        snap = "" if req == got else f"  (requested {req}, snapped/derived)"
        lines.append(
            f"     N{i + 1} = {got:5d}   bracket [{lo:5d}, {hi:5d})   "
            f"<E> = {me:9.3f} {unit}   "
            f"mean_k E[N-1] = {edge:9.3f} {unit}{snap}")
    lines.extend(f"     NOTE: {note}" for note in plan.notes)
    return lines


def spectral_h5_payload(plan: BandBracketPlan, fit: SpectralShellFit,
                        *, scale: float = 1.0) -> dict:
    """``sigma_mnk.h5``'s payload for a ``spectral_shell`` run.

    The same four-array contract :func:`extrapolation_h5_payload` documents,
    with β in place of the 1/N amplitude — see
    :data:`SPECTRAL_EXTRAP_DATASETS` for why the name changes rather than the
    meaning of an existing one.  ``β`` is dimensionless, so ``scale`` (a unit
    conversion) is deliberately NOT applied to it.

    The attributes carry the two facts a reader needs to reproduce the number
    and cannot get from the arrays: which estimator ran, and the DFT-only
    ladder it ran on (``E₀``, ``n₀``, ``E*``, ``N_T``, the shells).
    """
    def _arr(a):
        return np.asarray(a, dtype=np.complex128) * scale

    lad = fit.ladder
    return {
        "arrays": {
            "sigma_c_extrap_inf_kn_ev": _arr(fit.s_inf),
            "sigma_c_extrap_last_kn_ev": _arr(fit.s_at_counts[-1]),
            "sigma_c_extrap_beta_kn": np.asarray(fit.beta,
                                                 dtype=np.complex128),
            "sigma_c_extrap_sigma_kn_ev": _arr(fit.uncertainty("p90")),
        },
        "attrs": {
            "band_extrapolation_estimator": "spectral_shell",
            "band_counts": np.asarray(plan.counts, dtype=np.int64),
            "band_counts_requested": np.asarray(plan.requested,
                                                dtype=np.int64),
            **_bracket_h5_attrs(plan),
            "n_occ": int(plan.n_occ),
            "n_cond": int(plan.n_cond),
            "shell_bands_absolute": np.asarray(fit.shells, dtype=np.int64),
            "ladder_e0_ev": float(lad.e0_ev),
            "ladder_n0": float(lad.n0),
            "ladder_c_ev": float(lad.c_ev),
            "ladder_r2": float(lad.r2),
            "ladder_estar_ev": float(lad.estar_ev),
            "ladder_n_dft": int(lad.n_dft),
            "ladder_n_target": int(lad.n_target),
            "ladder_fit_window": np.asarray(lad.fit_window, dtype=np.int64),
            "uncertainty_fraction_p90_p99": np.asarray(
                TAIL_UNCERTAINTY_FRACTION, dtype=np.float64),
            "verdict": str(spectral_trust_verdict(fit)),
            "planner_notes": " | ".join(plan.notes) if plan.notes else "",
        },
    }


def format_spectral_report(
    plan: BandBracketPlan,
    fit: SpectralShellFit,
    *,
    states: "list[tuple[str, object]] | None" = None,
    label: str = "Sigma_c",
    unit: str = "eV",
    scale: float = 1.0,
) -> str:
    """The log block for ``spectral_shell``.  Same requirement as the 1/N one.

    ONE log must carry the full-band value and the extrapolated value side by
    side with everything that produced it — here the three band counts, the
    three shells in ABSOLUTE band index, the DFT-only ladder, and the
    per-state β and tail ratio.  A reader must be able to recompute Ŝ from
    this block without opening the h5.
    """
    def _sg(a):
        return float(np.real(np.asarray(a))) * scale

    def _mx(a):
        return float(np.max(np.abs(np.real(np.asarray(a))))) * scale

    nan = float("nan")
    lad = fit.ladder
    geometry = _bracket_geometry_text(plan)
    lines = [
        f"  -- {label} band-convergence extrapolation "
        f"(estimator = spectral_shell: one exponent PER STATE from the two "
        f"shell increments, tail integrated to the finite basis) --",
        f"     N_occ = {plan.n_occ}   N_cond = {plan.n_cond}   {geometry}",
    ]
    lines.extend(_bracket_report_lines(plan, unit))
    lines += [
        f"     {lad.describe()}",
        f"       shells (ABSOLUTE band index, half-open at the low end): "
        f"I2 over {fit.shells[0]}, I3 over {fit.shells[1]}, "
        f"I_tail over {fit.shells[2]}",
        f"       N_T = {lad.n_target} is the FINITE PLANE-WAVE BASIS "
        f"(min(ngk)*nspinor), not infinity: the band sum is exactly complete "
        f"there and S(inf) names no physical quantity.  Bands "
        f"{lad.n_dft + 1}..{lad.n_target} come from the Weyl ladder, used to "
        f"extend the EIGENVALUE SEQUENCE only.",
    ]

    for slabel, index in (states or []):
        f1 = fit.at(index)
        code = int(np.asarray(f1.failure))
        if code != SHELL_OK:
            lines += [
                f"     [{slabel}]",
                f"       *** FAILED: {SHELL_FAILURE_REASONS[code]}",
                f"       D2 = {_sg(f1.d2):+12.6f}   D3 = {_sg(f1.d3):+12.6f} "
                f"{unit}   -- no value is substituted for this state.",
            ]
            continue
        lines += [
            f"     [{slabel}]",
            f"       S(N1={plan.counts[0]}) = {_sg(f1.s_at_counts[0]):+12.6f}   "
            f"S(N2={plan.counts[1]}) = {_sg(f1.s_at_counts[1]):+12.6f}   "
            f"S(N3={plan.counts[2]}) = {_sg(f1.s_at_counts[2]):+12.6f} {unit}",
            f"       D2 = {_sg(f1.d2):+12.6f}   D3 = {_sg(f1.d3):+12.6f} "
            f"{unit}   |D3/D2| = "
            f"{(abs(_sg(f1.d3) / _sg(f1.d2)) if _sg(f1.d2) else nan):.6f}",
            f"       beta = {float(np.real(f1.beta)):8.4f}   "
            f"I_tail/I3 = {float(np.real(f1.tail_ratio)):10.6f}",
            f"       S(N3) [full {plan.counts[-1]}-band Sigma_c] = "
            f"{_sg(f1.s_at_counts[-1]):+12.6f} {unit}   ->   "
            f"S_hat = {_sg(f1.s_inf):+12.6f} {unit}",
            f"       Delta_tail = {_mx(f1.delta_tail):.6f} {unit}   "
            f"= D3 * I_tail/I3, the whole correction",
            f"       S_hat = {_sg(f1.s_inf):+.6f} +/- "
            f"{_mx(f1.uncertainty('p90')):.6f} (p90) / "
            f"{_mx(f1.uncertainty('p99')):.6f} (p99) {unit}"
            f"   <- see below: this bar is INHERITED, not re-calibrated",
        ]

    b = np.asarray(np.real(fit.beta), dtype=np.float64).ravel()
    b = b[np.isfinite(b)]
    lines += [
        f"     [envelope over ALL (k, band) of the QP window -- an upper "
        f"bound, NOT the result: it is set by the top of the window, whose "
        f"Sigma_c is the least converged quantity in the run]",
        f"       max|S(N3)| = {_mx(fit.s_at_counts[-1]):.6f}   "
        f"max|S_hat| = {_mx(fit.s_inf):.6f} {unit}   "
        f"max Delta_tail = {_mx(fit.delta_tail):.6f} {unit}",
        f"       beta over {fit.n_states} states: median "
        f"{(float(np.median(b)) if b.size else float('nan')):.3f}, "
        f"p10/p90 "
        f"{(float(np.percentile(b, 10)) if b.size else float('nan')):.3f}/"
        f"{(float(np.percentile(b, 90)) if b.size else float('nan')):.3f}, "
        f"min/max "
        f"{(float(b.min()) if b.size else float('nan')):.3f}/"
        f"{(float(b.max()) if b.size else float('nan')):.3f}   "
        f"({fit.n_failed} FAILED)",
        f"       verdict: {spectral_trust_verdict(fit)}",
        # WHAT THIS ESTIMATOR DOES AND DOES NOT REPORT.  The 1/N block's four
        # diagnostics do not exist here and their absence must not read as an
        # omission: they are statements about a two-parameter fit's residual
        # structure, and this estimator has one unknown and no residual.
        f"     [reading these] There is NO Delta_model, pair_split, residual "
        f"or A/N3 here and their absence is not an omission: those are "
        f"properties of a two-parameter LEAST SQUARES through three points.  "
        f"This estimator solves ONE unknown from ONE ratio, so it reproduces "
        f"|D3/D2| exactly by construction and a residual would be "
        f"identically zero.",
        f"       The per-state beta spread IS the diagnostic.  It is the "
        f"resolution the estimator exists to provide -- valence and "
        f"conduction states converge differently -- and it is NOT a quality "
        f"metric: states agreeing on beta is not evidence that beta is "
        f"right.",
        f"       The +/- is {100*TAIL_UNCERTAINTY_FRACTION[0]:.0f} % / "
        f"{100*TAIL_UNCERTAINTY_FRACTION[1]:.0f} % of Delta_tail -- the "
        f"envelope CALIBRATED FOR band_index_only AND NOT RE-DERIVED HERE.  "
        f"Held out against a measured S(508) this estimator's median error "
        f"is 3-10x smaller than that one's, so the bar is conservative on "
        f"the states that test covered and unmeasured on the rest.  It "
        f"covers the EXTRAPOLATION only -- not the difference from "
        f"BerkeleyGW, the ISDF basis, or the W-side band count.",
    ]
    if fit.n_failed:
        lines.append("     " + fit.failure_report().replace("\n", "\n     "))
    return "\n".join(lines)


def extrapolation_h5_payload(plan: BandBracketPlan, fit: ExtrapolationFit,
                             *, scale: float = 1.0) -> dict:
    """The arrays and attributes ``sigma_mnk.h5`` should carry for this fit.

    WHY THIS EXISTS.  Until 2026-08-15 the fitted ``S_∞`` was PRINTED and
    written nowhere, so a run with the feature ON and one with it OFF
    produced byte-identical artifacts — ``sigma_diag.dat`` identical, every
    dataset of ``sigma_mnk.h5`` identical to 8e-15 — while the log reported
    an 848 meV correction at the VBM.  The feature's entire output lived in
    a log line: it could not be gated, diffed, regression-tested, or
    consumed downstream, and a star-spread test on the written Σ passed
    VACUOUSLY because it was measuring the un-extrapolated cube.

    Four ``(nk, nb)`` arrays go in, not one.  ``S_∞`` alone tells a reader
    the answer but not how far it moved or whether to believe it, and the
    other three are already computed: ``S(N₃)`` is the un-extrapolated value
    at the same states (so the correction is a subtraction, not a
    reconstruction), ``A`` is how much correction is still being applied at
    the largest count, and the uncertainty is the envelope from
    :data:`TAIL_UNCERTAINTY_FRACTION`.

    The verdict and the band counts ride as ATTRIBUTES rather than arrays:
    they are per-run, not per-state.
    """
    def _arr(a):
        return np.asarray(a, dtype=np.complex128) * scale

    return {
        "arrays": {
            "sigma_c_extrap_inf_kn_ev": _arr(fit.s_inf),
            "sigma_c_extrap_last_kn_ev": _arr(fit.s_at_counts[-1]),
            "sigma_c_extrap_ampl_kn_ev": _arr(fit.amplitude),
            "sigma_c_extrap_sigma_kn_ev": _arr(fit.uncertainty("p90")),
        },
        "attrs": {
            "band_counts": np.asarray(plan.counts, dtype=np.int64),
            "band_counts_requested": np.asarray(plan.requested,
                                                dtype=np.int64),
            **_bracket_h5_attrs(plan),
            "n_occ": int(plan.n_occ),
            "n_cond": int(plan.n_cond),
            "uncertainty_fraction_p90_p99": np.asarray(
                TAIL_UNCERTAINTY_FRACTION, dtype=np.float64),
            "verdict": str(trust_verdict(fit)),
            "planner_notes": " | ".join(plan.notes) if plan.notes else "",
        },
    }


def format_extrapolation_report(
    plan: BandBracketPlan,
    fit: ExtrapolationFit,
    *,
    states: "list[tuple[str, object]] | None" = None,
    label: str = "Sigma_c",
    unit: str = "eV",
    scale: float = 1.0,
) -> str:
    """The log block.  Numbers first; the layout is deliberately not final.

    The requirement this satisfies is that ONE log carries the full-band
    value and the extrapolated value side by side, with the three band
    counts, the fit parameters and every diagnostic — unambiguously, and
    without the reader having to reconstruct anything.

    ``states`` is ``[(label, index), ...]``: the individual (k, band) states
    to report in full, each on its own row with SIGNED values, plus its own
    verdict.  The aggregate row that follows is a max over every state and is
    deliberately labelled as an envelope, not as a result — on a real deck it
    is dominated by the top of the QP window, whose Σ_c is both the largest
    and the least converged quantity in the run.
    """
    def _sg(a):
        return float(np.real(np.asarray(a))) * scale

    def _mx(a):
        return float(np.max(np.abs(np.real(np.asarray(a))))) * scale

    geometry = _bracket_geometry_text(plan)
    lines = [
        f"  -- {label} band-convergence extrapolation "
        f"(S(N) = S_inf + A/N, 2 parameters, 3 points) --",
        f"     N_occ = {plan.n_occ}   N_cond = {plan.n_cond}   {geometry}",
    ]
    lines.extend(_bracket_report_lines(plan, unit))

    for slabel, index in (states or []):
        f1 = fit.at(index)
        lines += [
            f"     [{slabel}]",
            f"       S(N1={plan.counts[0]}) = {_sg(f1.s_at_counts[0]):+12.6f}   "
            f"S(N2={plan.counts[1]}) = {_sg(f1.s_at_counts[1]):+12.6f}   "
            f"S(N3={plan.counts[2]}) = {_sg(f1.s_at_counts[2]):+12.6f} {unit}",
            f"       S(N3) [full {plan.counts[-1]}-band Sigma_c] = "
            f"{_sg(f1.s_at_counts[-1]):+12.6f} {unit}   ->   "
            f"S_inf = {_sg(f1.s_inf):+12.6f} {unit}   "
            f"(A = {_sg(f1.amplitude):+.4f} {unit}*band)",
            f"       S_inf^(12) = {_sg(f1.pair_s_inf[(0, 1)]):+12.6f}   "
            f"S_inf^(23) = {_sg(f1.pair_s_inf[(1, 2)]):+12.6f}   "
            f"S_inf^(13) = {_sg(f1.pair_s_inf[(0, 2)]):+12.6f} {unit}",
            f"       Delta_tail = {_mx(f1.delta_tail):.6f} {unit}   "
            f"Delta_model = {_mx(f1.delta_model):.6f} {unit}   "
            f"residual = {_mx(f1.residual):.3e} {unit}",
            f"       pair_split = {_sg(f1.pair_split):+12.6f} {unit}   "
            f"A/N3 = {_sg(f1.a_over_n_last):+12.6f} {unit}   "
            f"(both SIGNED -- see below)",
            f"       S_inf = {_sg(f1.s_inf):+.6f} +/- "
            f"{_mx(f1.uncertainty('p90')):.6f} (p90) / "
            f"{_mx(f1.uncertainty('p99')):.6f} (p99) {unit}"
            f"   <- EXTRAPOLATION uncertainty only",
            f"       verdict: {trust_verdict(f1)}",
        ]

    lines += [
        f"     [envelope over ALL (k, band) of the QP window -- an upper "
        f"bound, NOT the result: it is set by the top of the window, whose "
        f"Sigma_c is the least converged quantity in the run]",
        f"       max|S(N3)| = {_mx(fit.s_at_counts[-1]):.6f}   "
        f"max|S_inf| = {_mx(fit.s_inf):.6f} {unit}",
        f"       max Delta_tail = {_mx(fit.delta_tail):.6f} {unit}   "
        f"max Delta_model = {_mx(fit.delta_model):.6f} {unit}   "
        f"max residual = {_mx(fit.residual):.3e} {unit}",
        f"       max |pair_split| = {_mx(fit.pair_split):.6f} {unit}   "
        f"max |A/N3| = {_mx(fit.a_over_n_last):.6f} {unit}",
        f"       verdict: {trust_verdict(fit)}",
        # WHAT EACH DIAGNOSTIC CAN AND CANNOT SEE.  Printed rather than left
        # to the docstring because the verdict line above is what an operator
        # reads, and a "consistent" that is necessary-but-not-sufficient has
        # to say so where it is read.  Measured on a clean BerkeleyGW curve:
        # verdict "consistent" on 100 % of 768 state-fits, zero sign
        # reversals, while the true intercept error was 55 meV MAE / 167 max.
        f"     [reading these] Delta_model is a SCATTER metric: it sees the "
        f"three points DISAGREEING, and is blind to an error they SHARE -- "
        f"which is the error that dominates.  A 'consistent' verdict is "
        f"necessary, not sufficient.",
        f"       pair_split is Delta_model kept SIGNED: its sign says which "
        f"way the preasymptotic bias runs (negative => the fit is "
        f"over-correcting).  A/N3 is one-sided: it is the correction still "
        f"being applied at the largest band count, and stays loud when "
        f"Delta_model goes quiet.",
        f"       Both shrink when the sampling moves up the band range; "
        f"neither is a substitute for raising nband when A/N3 is comparable "
        f"to the accuracy you need.",
        f"       The +/- is {100*TAIL_UNCERTAINTY_FRACTION[0]:.0f} % / "
        f"{100*TAIL_UNCERTAINTY_FRACTION[1]:.0f} % of Delta_tail -- an "
        f"ENVELOPE calibrated against BerkeleyGW on one deck, NOT a per-state "
        f"bar (no per-state predictor works: R^2 <= 0 for all four numbers "
        f"above).  It covers the EXTRAPOLATION only -- not the difference "
        f"from BerkeleyGW, the ISDF basis, or the W-side band count.",
    ]
    return "\n".join(lines)


__all__ = [
    "BRACKET_FRACTIONS",
    "BRACKET_SCHEMES",
    "BRACKET_SCHEME_DEFAULT",
    "CONDUCTION_HALF_FRACTION",
    "EXTRAP_DATASETS",
    "SPECTRAL_EXTRAP_DATASETS",
    "TAIL_UNCERTAINTY_FRACTION",
    "BAND_EXTRAPOLATION_ESTIMATORS",
    "BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT",
    "SHELL_EXPONENT_BRACKET",
    "SHELL_EXPONENT_BISECTIONS",
    "SHELL_EXPONENT_EDGE_TOL",
    "SHELL_FAILURE_REASONS",
    "SHELL_OK",
    "SHELL_FAIL_SIGN",
    "SHELL_FAIL_ZERO",
    "SHELL_FAIL_NO_ROOT",
    "SHELL_FAIL_EDGE",
    "extrapolation_h5_payload",
    "extrapolation_weights",
    "spectral_h5_payload",
    "BandBracketCountMismatch",
    "BandBracketPlan",
    "BandExtrapolationRefused",
    "BandLadder",
    "SpectralShellExtrapolationFailed",
    "SpectralShellFit",
    "assert_brackets_match_ols_abscissae",
    "build_band_ladder",
    "ExtrapolationFit",
    "fit_band_extrapolation",
    "fit_band_extrapolation_spectral",
    "format_extrapolation_report",
    "format_spectral_report",
    "plan_band_brackets",
    "plane_wave_band_count",
    "sc_tolerance_ruling",
    "solve_shell_exponents",
    "spectral_trust_verdict",
    "tolerance_bar_ev",
    "trivial_plan",
    "trust_verdict",
    "weyl_ladder_fit",
]
