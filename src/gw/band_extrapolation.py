"""Band-convergence extrapolation of the correlation self-energy Σ_c.

ONE ESTIMATOR, THREE POINTS.  It reads the three cumulative bracket sums
S(N₁), S(N₂), S(N₃) that the τ kernel already produces in one pass.  The deck
key ``band_extrapolation_estimator`` has one value, ``spectral_shell``.
Since 2026-09-28 (owner ruling) it is the POOLED DENOMINATOR SHELL: band A
adds ``a_i · Σ_k w_k (E_Ak − E_i + Ω)^(−β)`` to state i, with ONE (β, Ω)
pooled over the requested states and a per-state amplitude; the tail is
integrated out to the finite plane-wave basis.  ``POOLED DENOMINATOR SHELL``
below.

The two-parameter ``S_∞ + A/N`` least squares (deck value
``band_index_only``) was deleted on 2026-09-27 (owner ruling).  Against the
accurate BANDTRUTH truth it was the worst estimator, 43 / 79 meV median / max
at N₃ = 152 (sandbox claims 2860, 2866).  A deck that names it refuses by name
(``gw.gw_config``).  :func:`extrapolation_weights` keeps its three OLS
coefficients because :func:`static_limit_tail_ruling` still reads them.

WHAT CONVERGES SLOWLY.  Σ_c's intermediate-state sum runs over every band in
the Green's function,

    Σ_c(ω) = Σ_n  ψ_n ⊗ ψ_n*  ⊗  [W-dependent kernel](ω − E_n),

and its unoccupied tail decays slowly: the states above the QP window
contribute a small, same-signed, slowly-vanishing amount that a brute-force
band count only removes by being enormous.  This module evaluates the SAME
sum at three band counts in ONE pass and extrapolates instead.

THE THREE POINTS COME FROM DISJOINT BRACKETS, NOT THREE RUNS.  Under the
default ``band_extrapolation_bracket_scheme = total_fractions``, the band axis
is cut into three contiguous brackets

    bracket 0   [0, N₁)          everything up to 70 % of the TOTAL band count
    bracket 1   [N₁, N₂)         70 % → 85 %
    bracket 2   [N₂, N₃)         85 % → 100 %

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

WHERE THE POINTS GO.  Default 70 / 85 / 100 % of the total Σ band count
(owner ruling 2026-09-28; ``BRACKET_FRACTIONS``).  The pooled form reads
the widest shell for each state's amplitude and one interior point for the
shape, so it wants the shells wide; its std moves by ≤ 2.7 meV (case a)
and ≤ 3.6 meV (case c) across placements at 78 bands on Si, where the
per-state form moved by up to 100 meV.

THE DEFAULT FRACTIONS ARE OF THE TOTAL BAND COUNT, NOT OF THE CONDUCTION
COUNT — and since 2026-08-22 that is a NAMED default
(``bracket_scheme = total_fractions``) rather than the only reading.
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

THE COORDINATE IS A NAMED SPELLING, NOT AN INFERENCE.  Two conduction-
coordinate geometries are available by name, and neither is reachable by
accident:

``conduction_fractions`` reads the SAME fractions in the coordinate the owner
named on 2026-08-18 — ``N_i = n_occ + round(f_i * (N_max - n_occ))``.  It
exists because the two readings are indistinguishable on a small occupied
manifold and far apart on a large one: on Si (``n_occ = 8``) 0.80 of 396 is
317 against 318, while on a CrI3-scale occupied manifold every shell boundary
moves by tens of bands.  Both readings live in exactly one function,
:func:`bracket_counts_from_fractions`, whose ``coordinate`` argument is
keyword-only with NO default — at a call site the difference is invisible, so
it has to be said out loud.

``conduction_energy_midpoint`` places N1 at half of the conduction bands
included in the Sigma sum, snaps it to a multiplet-clean boundary, then places
N2 at the clean rectangular boundary nearest halfway between N1 and N3 in
``mean_k E[k,N-1]``.  It does NOT apply a global E_ck mask: every k keeps the
same static band count.  This changes only cut placement; the estimator
still uses absolute band indices and the full DFT energy ladder.

Neither is the default, and that is measured rather than cautious.  On the
dense Si GN-PPM control, merely reinterpreting 80/90 as conduction fractions
WORSENED the ``N=180..396`` indirect-gap spread/max from 29.79/17.29 meV to
38.12/25.63 meV (Perlmutter JID 57267197,
``runs/Si/69_conduction_fraction_geometry_20260818/``).  The owner's stated
intent fixes what the coordinate MEANS; it does not by itself rule which one
converges better, and one material does not establish a universal
cut-placement law.  So existing decks keep their numerical meaning and a
production study names the coordinate it wants.

It is deliberately NOT the default.  On the measured Si GN-PPM curve its
indirect-gap spread was 12.29 meV over N3=180..396 and 4.18 meV over
N3=276..440, but 82.96 meV over N3=140..296; the incumbent total-80/90
geometry also remained more accurate per state on that control.  One material
does not establish a universal band threshold.  Existing decks therefore
retain their numerical meaning, while production studies can name the new
scheme and get its resolved cuts in the startup log and HDF5 provenance.

POOLED DENOMINATOR SHELL — THE ESTIMATOR SINCE 2026-09-28 (owner ruling).
The model: a band A above the sampled range adds to Σ_c of state i

    c_i(A) = a_i · Σ_k w_k (E_Ak − E_i + Ω)^(−β),

with ONE (β, Ω) pooled over the requested states and one amplitude a_i per
state.  Physics: the empty-branch remainder is
``−Σ_{A,p} M_iAp / (E_A + Ω_p − E_i)``.  A high band is a plane wave of energy
E; its matrix element and W^c each fall as 1/E and the denominator as
1/(E + Ω − E_i), so the leading exponent is β = 3 with a state-independent
amplitude, and the state enters through ``E_i`` in the denominator.  Keeping
``E_i`` there lets one (β, Ω) describe every state; ``a_i`` carries the
matrix-element size.

WHY POOLED.  The form it replaced (2026-08-17 to 2026-09-28) solved one β
per state from the ratio of the two top shell increments.  On narrow top
shells that ratio carries band texture, and a per-state β amplifies it.  On
Si 4³ with the complete basis (536 bands) as truth, 78 bands and the old
default cuts (64, 72, 78), the per-state β gave 109 meV std over the ±10 eV
states and 278 meV max 4v4c error, worse than no extrapolation (29 / 90);
the pooled form gives 6.5 / 34 meV (sandbox claims 2898, 2900).  Pooling
fixes the SHAPE from every requested state and leaves each state one
amplitude, which the widest shell determines.

THE FIT.  With ``G_i(lo, hi) = Σ_{lo<A≤hi} Σ_k w_k ((E_Ak − E_i + Ω)/E*)^(−β)``
(E* cancels from every ratio), for each (β, Ω) on
:data:`SHELL_BETA_GRID` × :data:`SHELL_OMEGA_GRID_EV`:

    a_i = (S₃ − S₁) / G_i(N₁, N₃),     Ŝ₂,i = S₁ + a_i · G_i(N₁, N₂),

and the grid point minimising ``Σ_i (Ŝ₂,i − S₂,i)²`` over the pooled states
wins.  Then

    Ŝ_i = S(N₃) + (S(N₃) − S(N₁)) · G_i(N₃, N_T) / G_i(N₁, N₃).

The pooled set is the states within ±10 eV of E_F (closed over multiplets,
:func:`pooled_state_mask`) that lie below every band above N₁,
``E_i < min_k E[N₁+1, k]``, so every one of them is in the model's domain at
every grid point.  Every state of the QP window in the domain gets its tail.  The grid is searched, not optimised: a
closed-form evaluation per point, no nonlinear solver.  The shell sums are
evaluated on a composite Gauss compression of each shell's spectrum in
``log(E − E_ref)`` (:func:`_log_energy_rule`), exact to ~1e-16 relative, so
the whole grid costs ~10 ms on Si 4³ (71 pooled states) and the N_T tail
costs ~100 terms per state however many Weyl bands it spans.

NO DOMAIN, NO TAIL.  A state with ``E_i − Ω ≥ min_k E[N₁+1, k]`` puts a pole
of the model inside the band sum.  It keeps its computed sum, ``Ŝ = S(N₃)``
(:data:`SHELL_FAIL_POLE`).  If no requested state can be pooled, every state
keeps ``S(N₃)`` (:data:`SHELL_FAIL_NO_FIT`).  No other estimator is
substituted, and the log names every such state.

MEASURED.  BANDEX (``sandbox:runs/Si_scalar/42_bandex_20260927``), Si 4³
25 Ry scalar, shared-pole W, truth S(536) with the complete basis, 71
degeneracy-closed states within ±10 eV of midgap, G truncated at N and W
built at 536 (case a) or at N (case c); std over the states / max 4v4c /
median 4v4c direct, meV.  This function reproduces the study's scores to
1e-12 meV (sandbox run DEV/602):

    N    cuts          case a               case c
    78   50, 64, 78    9.2 / 32.9 / 8.7    20.5 / 81.0 / 17.9   <- default
    78   64, 72, 78    6.5 / 33.6 / 6.0    16.9 / 69.0 / 19.7
    50   20, 34, 50   23.2 / 81.9 / 23.0   36.8 / 151.7 / 30.7
    34   14, 20, 34   49.0 / 168.0 / 52.8  53.6 / 179.3 / 57.8

No extrapolation at 78 bands: 29.2 / 90.5 / 35.8 (a), 34.6 / 121.8 / 38.0 (c).
The pooled form's std moves by ≤ 2.7 meV (case a) and ≤ 3.6 meV (case c)
across placements at 78 bands; the per-state form moved by up to 100 meV.  One material; the gap error (−19 to
+19 meV at 78 bands, case a) is not controlled by the fit.

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

WHAT THE ESTIMATOR IS APPLIED TO, AND WHY THAT NEEDED A RULING.  The tail
ratio r is per external state, so the three combination coefficients ``[−r, 0, 1 + r]``
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

import dataclasses
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
#: 0.70 / 0.85, owner ruling 2026-09-28, with the pooled denominator shell.
#: The estimator takes each state's amplitude from the widest shell
#: (N₁, N₃] and the shape from the interior point, so wider shells carry more
#: of the spectrum into the fit.  On Si 4³ at 78 bands (complete-basis truth,
#: sandbox run DEV/602) these fractions snap to (50, 64, 78): 9.2 meV std over
#: the ±10 eV states and 32.9 meV max 4v4c error with W at 536 bands; the old
#: (0.80, 0.90) snap to (64, 72, 78): 6.5 / 33.6 meV.  The pooled form's std
#: moves by ≤ 2.7 meV (case a) and ≤ 3.6 meV (case c) over the placements
#: measured at 78 bands, so the choice is not tuned on that difference.  The cuts are then degeneracy-snapped by
#: :func:`plan_band_brackets`.
BRACKET_FRACTIONS: tuple[float, float] = (0.70, 0.85)

#: Named geometries for the three cumulative band sums.  The incumbent stays
#: the default: changing the meaning of existing decks would invalidate their
#: convergence history, and the conduction-coordinate arms have only been
#: measured on one Si curve.  They are therefore explicit production options,
#: never inferred replacements.
#:
#: ``conduction_fractions`` is the SAME fractions in the conduction
#: coordinate the owner named on 2026-08-18 —
#: ``N_i = n_occ + round(f_i * (N_max - n_occ))`` — and it exists because the
#: two coordinates are indistinguishable on a small occupied manifold and far
#: apart on a large one.  On Si (``n_occ = 8``) 0.80 of 396 is 317 and the
#: conduction reading is 8 + 0.80*388 = 318; on a CrI3-scale occupied
#: manifold they diverge by tens of bands and every shell boundary moves.
#:
#: IT IS NOT THE DEFAULT, and the reason is measured rather than
#: conservative: on the dense Si GN-PPM control, merely reinterpreting 80/90
#: as conduction fractions WORSENED the N=180..396 indirect-gap
#: spread/max from 29.79/17.29 meV to 38.12/25.63 meV (Perlmutter JID
#: 57267197, ``runs/Si/69_conduction_fraction_geometry_20260818/``).  Intent
#: alone is not a numerical-default ruling — one material does not establish
#: a universal cut-placement law — so the coordinate is a spelling a deck
#: chooses, and existing decks keep their numerical meaning.
BRACKET_SCHEMES: tuple[str, str, str] = (
    "total_fractions",
    "conduction_fractions",
    "conduction_energy_midpoint",
)
BRACKET_SCHEME_DEFAULT: str = "total_fractions"

#: The two schemes that CONSUME ``band_extrapolation_fractions``.  The third
#: derives its cuts from energy and refuses the key.
BRACKET_FRACTION_SCHEMES: tuple[str, str] = (
    "total_fractions",
    "conduction_fractions",
)


def bracket_counts_from_fractions(fractions, n_occ: int, nb_logical: int,
                                  *, coordinate: str) -> tuple[int, ...]:
    """Turn sampling fractions into requested band counts — ONE conversion.

    **THE COORDINATE IS NAMED, NEVER INFERRED.**  ``coordinate`` is
    keyword-only and has no default: the two readings of "0.80" differ by
    ``n_occ`` bands and nothing at a call site makes the choice visible.
    This function is the only place either arithmetic lives, so a third
    consumer cannot grow a fourth reading.

    * ``total_fractions``      -> ``round(f * N_max)``
    * ``conduction_fractions`` -> ``n_occ + round(f * (N_max - n_occ))``

    Both return counts measured from band 0 — absolute band indices, which is
    what every downstream consumer (the snapper, the partial sums, the fit)
    reads.  The coordinate changes WHERE the cuts are, not what a cut means.
    """
    coordinate = str(coordinate).strip().lower()
    n_occ = int(n_occ)
    nb_logical = int(nb_logical)
    if coordinate == "total_fractions":
        return tuple(int(round(float(f) * nb_logical)) for f in fractions)
    if coordinate == "conduction_fractions":
        n_cond = nb_logical - n_occ
        return tuple(n_occ + int(round(float(f) * n_cond)) for f in fractions)
    raise ValueError(
        f"bracket_counts_from_fractions: coordinate {coordinate!r} consumes "
        f"no fractions; the fraction-consuming schemes are "
        f"{BRACKET_FRACTION_SCHEMES}.")

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


def require_extrapolation_band_floor(n_occ: int, nb_logical: int) -> None:
    """Refuse ``use_band_extrapolation`` when the Σ sum holds fewer unoccupied than occupied bands.

    One owner for the rule: ``gw.gw_jax`` calls it at startup, once the band
    slices exist and before the ζ fit and the W build, and
    :func:`plan_band_brackets` calls it again at the Σ stage.
    """
    n_occ = int(n_occ)
    n_cond = int(nb_logical) - n_occ
    nb_logical = int(nb_logical)
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
        ``total_fractions`` preserves the incumbent 70/85/100 total-band
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
    if (bracket_scheme not in BRACKET_FRACTION_SCHEMES
            and tuple(fractions) != BRACKET_FRACTIONS):
        raise ValueError(
            f"plan_band_brackets: fractions and "
            f"bracket_scheme={bracket_scheme!r} cannot be combined; that "
            f"named scheme fixes N1 at half the conduction manifold and "
            f"derives N2 from energy, so fractions would be ignored.  The "
            f"fraction-consuming schemes are {BRACKET_FRACTION_SCHEMES} — "
            f"'conduction_fractions' is the one that reads these same "
            f"fractions in the conduction coordinate.")

    if not enabled:
        return trivial_plan(nb_padded, n_occ, nb_logical)

    # ── THE ACTIVATION GATE ─────────────────────────────────────────────
    # Requested-but-impossible REFUSES.  Silently running the ordinary path
    # would produce a log with no extrapolation block and a Σ that looks
    # converged, and the operator would have no way to tell the feature was
    # off (measurement-discipline rule 1: an ignored deck key is how a green
    # A/B comes to measure nothing).
    require_extrapolation_band_floor(n_occ, nb_logical)

    e = np.asarray(enk_ry, dtype=np.float64)[:, :nb_logical]
    if e.ndim != 2 or e.shape[1] != nb_logical:
        raise ValueError(
            f"plan_band_brackets: expected (nk, >={nb_logical}) energies, "
            f"got shape {np.shape(enk_ry)}")

    # ONE conversion, and the coordinate is named at the call.  The two
    # readings of "0.80" differ by n_occ bands; on Si (n_occ = 8) that is one
    # band and on a CrI3-scale occupied manifold it moves every shell
    # boundary.  ``total_fractions`` is the incumbent arm and its counts do
    # not move.  ``conduction_energy_midpoint`` is resolved separately because
    # its N2 depends on the *snapped* N1, not on the raw conduction-half
    # request.
    if bracket_scheme == "conduction_energy_midpoint":
        requested = bracket_counts_from_fractions(
            (CONDUCTION_HALF_FRACTION,), n_occ, nb_logical,
            coordinate="conduction_fractions")
    else:
        requested = bracket_counts_from_fractions(
            fractions, n_occ, nb_logical, coordinate=bracket_scheme)

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
    gaps = boundary_min_gaps(e, is_full_spectrum=False)
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
        # AND NEVER PAST THE NEXT CUT'S REQUEST.  On Si 6x6x6 SOC at 64 bands
        # the only 1-meV-clean boundaries in 39..63 over 216 k were 40 and 60,
        # so cut 1 snapped 51 -> 60, past cut 2's request of 58; cut 2 was
        # left at 61 and both shells sat in the top four bands (1 + 3), where
        # the spectral estimator's exponent is shell texture (lane BX2).
        if i_cut + 1 < len(requested):
            hi_bound = min(hi_bound, int(requested[i_cut + 1]) - 1)
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
            # The LEAST-degenerate boundary in range (largest min gap, then
            # nearest the request, then downward), never blindly the request:
            # under SOC with inversion every odd count is a Kramers split at
            # EVERY k, which makes S(N) gauge-dependent (Γ quartet partners
            # differed by 1e-4 eV in S(61) and by 3.6 eV after extrapolation).
            cand = range(lo_bound, hi_bound + 1)
            cut = max(cand, key=lambda n: (float(gaps[n]), -abs(n - req), -n))
            notes.append(
                f"interior cut {req} -> {cut} kept UNSNAPPED: no "
                f"multiplet-clean boundary exists in [{lo_bound}, {hi_bound}] "
                f"at tol {tol_ry * RYD_TO_EV * 1e3:.3f} meV; {cut} is the "
                f"least-degenerate boundary there (min gap "
                f"{float(gaps[cut]) * RYD_TO_EV * 1e3:.3f} meV).  The cut is a "
                f"sampling point on a partial-sum curve, not a Σ window, so "
                f"this costs accuracy and not correctness.")
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
    exactly Hermitian; the SC loop still converges; a residual-based verdict
    still reads ordinary, because the fit's own residual structure is
    self-consistent on the wrong curve.  There is no
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
        the shared MPA Sigma executor builds the psi/E/mask operands it then
        slices.  ``slices.nb_sigma_sum`` is the Sigma band sum's extent; a plan
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
            f"the abscissae, and the fractions are the same "
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
    # The shared MPA executor builds the operands from this bundle, so this
    # ties the plan to the band axis it will actually slice rather than to a repeat of the
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

def extrapolation_weights(counts) -> np.ndarray:
    """The REAL coefficients ``c`` with ``S_inf = sum_i c_i * S(N_i)``.

    The OLS coefficients of ``S_∞ + A/N`` in ``x = 1/N``.  The estimator
    that applied them to Σ (``band_index_only``) was deleted on 2026-09-27;
    :func:`static_limit_tail_ruling` still reads them to size the static tail
    a 1/N limit would omit, and ``tests/test_band_extrapolation_split_sc.py``
    uses them to show a wrong bracket count is invisible in the weights.
    Historical note on the original pairing, kept for the argument below: the
    weights and the diagonal fit were pinned together so the
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
        ``N_eff`` at each point.

    Returns
    -------
    (3,) float64 ndarray
    """
    N = np.asarray(counts, dtype=np.float64)
    if N.ndim != 1 or N.size != 3:
        raise ValueError(
            f"extrapolation_weights: need exactly 3 counts, got {N}.")
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
#: ``spectral_shell`` is the only estimator.  Since 2026-09-28 (owner ruling)
#: it is the POOLED DENOMINATOR SHELL (BANDEX candidate B9); the per-state
#: exponent it replaced has no name left to select.  ``band_index_only`` (the
#: 1/N least squares) was deleted 2026-09-27 and refuses by name in
#: ``gw.gw_config``.
BAND_EXTRAPOLATION_ESTIMATORS: tuple[str, ...] = ("spectral_shell",)
BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT: str = "spectral_shell"

#: The pooled exponent β: ``(first, last, step)``, both ends included.
#:
#: PHYSICS, NOT TUNING.  A band far above the requested states is a plane
#: wave of energy E; the matrix element, the correlation part of W and the
#: energy denominator each fall as 1/E, so its contribution falls as E^-3
#: (β = 3, the 1/N_PW law), and the first state-dependent correction as E^-4.
#: The lower end, 2, keeps the tail summable on the Weyl ladder
#: (``ε_n ∝ n^{2/3}`` sums ``n^{-2β/3}`` only for β > 3/2).  The upper end,
#: 8, is twice the first correction's exponent: a shell that decays faster
#: carries no information about the tail.  The step is the resolution of the
#: grid search (BANDEX study grid; the residual is smooth on this scale).
SHELL_BETA_GRID: tuple[float, float, float] = (2.0, 8.0, 0.25)

#: The pooled denominator offset Ω, eV: ``(first, last, step)``.
#:
#: Ω is the pole energy of W^c that a high band sees in ``E_A + Ω − E_i``.
#: A pole energy is non-negative, and 40 eV is twice the valence plasmon of
#: any solid the code targets (Si 16.6 eV).  The step is the grid resolution
#: of the BANDEX study.
SHELL_OMEGA_GRID_EV: tuple[float, float, float] = (0.0, 40.0, 2.0)

#: The pooled set: states within this distance of E_F, eV, closed over
#: degenerate multiplets at each k (:func:`pooled_state_mask`).  It is the
#: owner's ±10 eV requested budget and the set the BANDEX study validated
#: the form on; semicore states in a wide QP window must not set (β, Ω).
SHELL_POOL_WINDOW_EV: float = 10.0

#: Two states at one k closer than this, eV, are one multiplet for the pool
#: (the 0.1 meV threshold of the BANDEX scorer).
SHELL_POOL_DEGENERACY_EV: float = 1.0e-4


def pooled_state_mask(e_rel_ev) -> np.ndarray:
    """States that set (β, Ω): ``|E − E_F| ≤`` :data:`SHELL_POOL_WINDOW_EV`, closed over multiplets.

    ``e_rel_ev`` is ``(nk, nb)``, E − E_F in eV.  A state enters if any state
    of its own k within :data:`SHELL_POOL_DEGENERACY_EV` is inside the window.
    """
    e = np.asarray(e_rel_ev, dtype=np.float64)
    inside = np.abs(e) <= SHELL_POOL_WINDOW_EV
    same = np.abs(e[..., :, None] - e[..., None, :]) <= SHELL_POOL_DEGENERACY_EV
    return np.any(same & inside[..., None, :], axis=-1)


#: Offsets scanned when fitting ``E_n = E₀ + C·(n + n₀)^(2/3)``.  n₀ is a
#: single integer shift of the band ladder, so an integer scan is the whole
#: parameter space; on the Si deck the minimum lands at n₀ = 0.
WEYL_N0_SCAN: tuple[float, float, float] = (0.0, 200.0, 1.0)

#: Per-state codes.  Index into :data:`SHELL_FAILURE_REASONS`.
SHELL_OK = 0
SHELL_FAIL_POLE = 1
SHELL_FAIL_NO_FIT = 2

SHELL_FAILURE_REASONS = {
    SHELL_OK: "ok",
    SHELL_FAIL_POLE: (
        "the state sits at or above the lowest band of the extrapolated "
        "range (E_i - Omega >= min_k E[N1+1, k]), so the denominator "
        "(E_A - E_i + Omega) of the model changes sign inside the band sum "
        "and the model has no tail to give"),
    SHELL_FAIL_NO_FIT: (
        "no state lies below the lowest band of the extrapolated range, so "
        "no pooled (beta, Omega) could be fitted"),
}

#: Composite Gauss compression of a shell's spectral measure.  Every shell
#: sum is ``Σ_m w_m (ε_m − E_ref + c)^(−β)`` with ``c ≥ 0``, a function of
#: ``y = log(ε − E_ref)`` analytic in the strip ``|Im y| < π`` for every
#: state and every (β, Ω) on the grid.  The y range is cut into panels of
#: width :data:`_SHELL_PANEL_WIDTH` and each panel's measure is replaced by its
#: :data:`_SHELL_PANEL_NODES`-point Gauss rule.  On a panel of half-width 0.25
#: inside a strip of half-width 0.8π the Bernstein ellipse has ρ ≈ 20, so an
#: 8-point rule is exact to ρ^(-16) ≈ 1e-21 before the (2/0.6)^β growth of
#: the integrand on the ellipse: ≤ 1e-16 relative at β = 8.  A panel with at
#: most 8 points keeps its points.  This is what makes the (β, Ω) grid cost
#: ~100 terms per state instead of (N3 − N1)·nk, and the N_T tail ~100
#: instead of up to 1.5e5.
_SHELL_PANEL_WIDTH = 0.5
_SHELL_PANEL_NODES = 8


def _gauss_rule(y: np.ndarray, w: np.ndarray, q: int):
    """The ``q``-point Gauss rule of the discrete measure ``Σ w δ(y − y_m)``.

    Lanczos on ``diag(y)`` from ``sqrt(w)`` with full reorthogonalisation
    (Golub–Welsch).  Stops early when the measure has fewer than ``q`` distinct
    points, where the shorter rule is already exact.
    """
    c = float(y.mean())
    s = max(float(y.max() - y.min()), 1e-300)
    z = (y - c) / s
    wt = float(w.sum())
    Q = np.zeros((q, z.size))
    Q[0] = np.sqrt(w / wt)
    a = np.zeros(q)
    b = np.zeros(q)
    k = q
    # einsum, not BLAS: a threaded BLAS on a shared host core turned these
    # small products into the whole cost of the fit (2 s against 0.07 s).
    for j in range(q):
        u = z * Q[j]
        a[j] = np.einsum("m,m->", Q[j], u)
        for _ in range(2):
            u = u - np.einsum("jm,j->m", Q[:j + 1],
                              np.einsum("jm,m->j", Q[:j + 1], u))
        if j + 1 == q:
            break
        b[j] = float(np.sqrt(np.einsum("m,m->", u, u)))
        if b[j] <= 1e-12:
            k = j + 1
            break
        Q[j + 1] = u / b[j]
    T = np.diag(a[:k]) + np.diag(b[:k - 1], 1) + np.diag(b[:k - 1], -1)
    th, V = np.linalg.eigh(T)
    return c + s * th, wt * V[0] ** 2


def _log_energy_rule(de: np.ndarray, w: np.ndarray):
    """Composite Gauss rule in ``y = log(de)`` for the measure ``Σ w δ(ε − ε_m)``.

    ``de = ε − E_ref > 0``.  Returns ``(de_nodes, weights)``; sums against it
    equal the raw sums to ~1e-16 relative for every ``(de + c)^(−β)`` with
    ``c ≥ 0`` and β on the grid (see :data:`_SHELL_PANEL_WIDTH`).
    """
    y = np.log(de)
    order = np.argsort(y, kind="stable")
    y, w = y[order], w[order]
    panel = np.floor((y - y[0]) / _SHELL_PANEL_WIDTH).astype(np.int64)
    cuts = np.flatnonzero(np.diff(panel)) + 1
    ys, ws = [], []
    for yp, wp in zip(np.split(y, cuts), np.split(w, cuts)):
        if yp.size <= _SHELL_PANEL_NODES:
            ys.append(yp)
            ws.append(wp)
        else:
            yn, wn = _gauss_rule(yp, wp, _SHELL_PANEL_NODES)
            ys.append(yn)
            ws.append(wn)
    return np.exp(np.concatenate(ys)), np.concatenate(ws)


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

    # ── the shell measures ──────────────────────────────────────────────
    def _terms(self, lo: int, hi: int):
        """Energies (eV) and weights of every (band, k) term in ABSOLUTE bands ``(lo, hi]``.

        DFT bands carry their k weights; each Weyl band is k-independent and
        carries the full weight 1 once.  Clipped to the ladder's extent.
        """
        lo, hi = int(lo), int(hi)
        e, w = [], []
        lo_d, hi_d = min(lo, self.n_dft), min(hi, self.n_dft)
        if hi_d > lo_d:
            e.append(self.e_dft_ev[lo_d:hi_d].reshape(-1))
            w.append(np.tile(self.w_k, hi_d - lo_d))
        n_end = self.n_dft + int(self.e_weyl_ev.size)
        lo_w, hi_w = (min(max(v, self.n_dft), n_end) for v in (lo, hi))
        if hi_w > lo_w:
            e.append(self.e_weyl_ev[lo_w - self.n_dft:hi_w - self.n_dft])
            w.append(np.ones(hi_w - lo_w))
        if not e:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(e), np.concatenate(w)

    def floor_ev(self, lo: int) -> float:
        """Lowest energy (eV) of any term in ABSOLUTE bands above ``lo``.

        A state with ``E_i − Ω`` at or above it puts a pole of the
        denominator model inside the band sum (:data:`SHELL_FAIL_POLE`).
        """
        e, _ = self._terms(lo, self.n_target)
        return float(e.min()) if e.size else float("inf")

    def shell_rule(self, lo: int, hi: int, e_ref_ev: float):
        """The compressed measure of ABSOLUTE bands ``(lo, hi]``: ``(ε_node − E_ref, w_node)``.

        ``E_ref`` must lie below every term; every state the rule is applied
        to must have ``E_ref − E_i + Ω ≥ 0`` (see :func:`_log_energy_rule`).
        """
        e, w = self._terms(lo, hi)
        if e.size == 0:
            return np.zeros(0), np.zeros(0)
        de = e - float(e_ref_ev)
        if not np.all(de > 0.0):
            raise ValueError(
                f"BandLadder.shell_rule: E_ref = {e_ref_ev} eV is not below "
                f"every term of bands ({lo}, {hi}] (min {e.min()} eV)")
        return _log_energy_rule(de, w)

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


@dataclass(frozen=True)
class SpectralShellFit:
    """Result of the pooled denominator-shell estimator.

    ONE ``(β, Ω)`` for the run, fitted on the pooled states; every per-state
    field is elementwise in the trailing (k, band) state axes.
    """

    counts: np.ndarray            # (3,) int — N₁, N₂, N₃, Σ-RELATIVE
    s_at_counts: np.ndarray       # (3, ...) — S(N₁), S(N₂), S(N₃)
    s_inf: np.ndarray             # (...)    — Ŝ
    beta: float                   # pooled exponent; NaN when no fit
    omega_ev: float               # pooled denominator offset, eV; NaN when no fit
    tail_ratio: np.ndarray        # (...)    — r = G(N₃, N_T)/G(N₁, N₃)
    delta_tail: np.ndarray        # (...)    — |Ŝ − S(N₃)|
    d2: np.ndarray                # (...)    — S(N₂) − S(N₁)
    d3: np.ndarray                # (...)    — S(N₃) − S(N₂)
    failure: np.ndarray           # (...) int — SHELL_* code
    fit_mask: np.ndarray          # (...) bool — states the (β, Ω) was pooled over
    residual_ev: float            # rms middle-point residual of the pooled fit, eV
    fit_seconds: float            # wall of fit + apply
    ladder: BandLadder
    shells: tuple                 # ((lo,hi), (lo,hi), (lo,hi)) ABSOLUTE
    held: bool = False            # (β, Ω) held from an earlier SC map, not refitted

    @property
    def n_failed(self) -> int:
        return int(np.count_nonzero(np.asarray(self.failure) != SHELL_OK))

    @property
    def n_states(self) -> int:
        return int(np.size(self.failure))

    def weights(self) -> np.ndarray:
        """``c`` with ``Ŝ = Σ_b c_b · S(N_b)``, shape ``(3,) + state shape``.

        ``Ŝ = S(N₃) + (S(N₃) − S(N₁))·r = −r·S(N₁) + 0·S(N₂) + (1 + r)·S(N₃)``:
        REAL coefficients that sum to 1, so a Σ that does not depend on the
        band count comes through unchanged and the symmetrised Σ stays
        Hermitian.  ``N₂`` fixes (β, Ω) and carries no weight.  A state
        without a tail has ``r = 0``: ``[0, 0, 1]``.
        """
        r = np.asarray(np.real(self.tail_ratio), dtype=np.float64)
        return np.stack([-r, np.zeros_like(r), 1.0 + r], axis=0)

    def beta_per_state(self) -> np.ndarray:
        """The pooled β on every state that got a tail, NaN elsewhere."""
        ok = np.asarray(self.failure) == SHELL_OK
        return np.where(ok, float(self.beta), np.nan)

    def uncertainty(self, quantile: str = "p90") -> np.ndarray:
        """Extrapolation uncertainty as a fraction of the applied correction.

        An ENVELOPE inherited from the deleted 1/N fit's calibration, not a
        per-state bar and not re-derived for this estimator; see
        :data:`TAIL_UNCERTAINTY_FRACTION`.
        """
        p90, p99 = TAIL_UNCERTAINTY_FRACTION
        f = {"p90": p90, "p99": p99}[quantile]
        return f * np.abs(self.delta_tail)

    def at(self, index) -> "SpectralShellFit":
        """Restrict every per-state field to a subset of the trailing state axes."""
        idx = index if isinstance(index, tuple) else (index,)
        return dataclasses.replace(
            self,
            s_at_counts=self.s_at_counts[(slice(None),) + idx],
            s_inf=self.s_inf[index],
            tail_ratio=self.tail_ratio[index],
            delta_tail=self.delta_tail[index],
            d2=self.d2[index],
            d3=self.d3[index],
            failure=np.asarray(self.failure)[index],
            fit_mask=np.asarray(self.fit_mask)[index],
        )

    def failure_report(self, *, limit: int = 12) -> str:
        """Every state without a tail, named, with its reason.  '' when none."""
        fail = np.asarray(self.failure)
        idx = np.argwhere(fail != SHELL_OK)
        if idx.size == 0:
            return ""
        (a1, _), _, (a3, n_t) = self.shells
        lines = [
            f"spectral_shell: {len(idx)} of {self.n_states} external states "
            f"have NO TAIL and keep their computed sum, S_hat = S(N3) "
            f"(r = 0).  The denominator model needs every band above "
            f"N1 = {a1} to lie above E_i - Omega "
            f"(lowest such band {self.ladder.floor_ev(a1):.4f} eV).  "
            f"Remedy: more bands (`number_bands_sigma`)."]
        for row in idx[:limit]:
            key = tuple(int(v) for v in row)
            lines.append(f"    state {key}: "
                         f"{SHELL_FAILURE_REASONS[int(fail[key])]}")
        if len(idx) > limit:
            lines.append(f"    ... and {len(idx) - limit} more.")
        return "\n".join(lines)


def _pooled_shell_grid(ladder: BandLadder, a1: int, a2: int, a3: int,
                       e_fit: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                       y3: np.ndarray):
    """``(β, Ω, rms residual)`` minimising the middle-point residual over the grid.

    For every grid point the amplitude of each pooled state comes from the
    widest shell, ``a_i = (S₃ − S₁)/G_i(N₁, N₃)``, and the model predicts
    ``S₂ = S₁ + a_i·G_i(N₁, N₂)``; the residual is summed over the pooled
    states.  The first minimum in β-major order wins (the study's order).
    Every pooled state satisfies ``E_i < floor``, so ``c = E_ref − E_i + Ω``
    is ≥ 0 at every Ω ≥ 0 with ``E_ref = max E_i``.
    """
    betas = np.arange(SHELL_BETA_GRID[0],
                      SHELL_BETA_GRID[1] + 0.5 * SHELL_BETA_GRID[2],
                      SHELL_BETA_GRID[2])
    omegas = np.arange(SHELL_OMEGA_GRID_EV[0],
                       SHELL_OMEGA_GRID_EV[1] + 0.5 * SHELL_OMEGA_GRID_EV[2],
                       SHELL_OMEGA_GRID_EV[2])
    e_ref = float(e_fit.max())
    de2, w2 = ladder.shell_rule(a1, a2, e_ref)
    de3, w3 = ladder.shell_rule(a2, a3, e_ref)
    de = np.concatenate([de2, de3])
    logw = np.log(np.concatenate([w2, w3]))
    n2 = de2.size
    d13 = y3 - y1
    target = y2 - y1
    step = float(betas[1] - betas[0]) if betas.size > 1 else 0.0
    res = np.empty((betas.size, omegas.size))
    for jo, om in enumerate(omegas):
        # (state, node) log of the scaled denominator; E* cancels in every
        # ratio and only keeps the powers conditioned.
        lx = np.log((de[None, :] + (e_ref - e_fit + om)[:, None])
                    / ladder.estar_ev)
        v = np.exp(logw[None, :] - betas[0] * lx)
        dv = np.exp(-step * lx)
        for jb in range(betas.size):
            if jb:
                v *= dv
            g12 = v[:, :n2].sum(axis=1)
            g13 = g12 + v[:, n2:].sum(axis=1)
            res[jb, jo] = float(np.sum((d13 * (g12 / g13) - target) ** 2))
    jb, jo = np.unravel_index(int(np.argmin(res)), res.shape)
    return (float(betas[jb]), float(omegas[jo]),
            float(np.sqrt(res[jb, jo] / max(e_fit.size, 1))))


def fit_band_extrapolation_spectral(
    counts, s_at_counts: np.ndarray, ladder: BandLadder, *,
    e_state_ev, fit_mask=None, held=None,
) -> SpectralShellFit:
    """The pooled denominator-shell estimator over three points.

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
    e_state_ev : array broadcastable to the trailing shape
        DFT energy of each external state, eV, on the ladder's reference.
    fit_mask : bool array of the trailing shape, optional
        The states (β, Ω) is pooled over.  Default: every state.  Production
        passes :func:`pooled_state_mask` (±10 eV of E_F).
    held : (β, Ω) or None
        A pooled pair from an earlier map of the same SC run.  Given, the grid
        is not searched: r_i then depends only on the DFT ladder and E_i, so
        the extrapolation is the same linear map on Σ at every SC map.  A grid
        argmin re-selected per map would jump between grid points and inject
        a discontinuity into the fixed-point map.

    The model (module docstring, POOLED DENOMINATOR SHELL): band A adds
    ``a_i · Σ_k w_k (E_Ak − E_i + Ω)^(−β)`` to state i, with ONE (β, Ω) pooled
    over the requested states and a per-state amplitude ``a_i`` fixed by the
    widest shell.  Then ``Ŝ_i = S₃ + (S₃ − S₁)·G_i(N₃, N_T)/G_i(N₁, N₃)``.

    The fit is on the REAL part (β, Ω describe a real spectral falloff); the
    ratio ``r_i`` is applied to the COMPLEX increment, so Im Σ_c is carried
    with the same per-state ratio.
    """
    import time
    t0 = time.perf_counter()
    N = np.asarray(counts, dtype=np.int64)
    S = np.asarray(s_at_counts)
    if N.ndim != 1 or N.size != 3:
        raise ValueError(
            f"fit_band_extrapolation_spectral: need exactly 3 counts, got "
            f"{N}.  The estimator reads the widest shell and one interior "
            f"point, which is three cumulative points.")
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

    shape = S.shape[1:]
    e = np.broadcast_to(np.asarray(e_state_ev, dtype=np.float64),
                        shape).reshape(-1)
    y1, y2, y3 = (np.real(S[j]).reshape(-1).astype(np.float64)
                  for j in range(3))
    pool = (np.ones(e.size, dtype=bool) if fit_mask is None else
            np.broadcast_to(np.asarray(fit_mask, dtype=bool),
                            shape).reshape(-1))
    floor = ladder.floor_ev(a1)
    # The pooled set: requested states below every band of the shells at
    # Ω = 0, so each one is in the model's domain at every grid point.
    fit = pool & (e < floor)
    if held is not None:
        beta, omega = (float(v) for v in held)
        rms = float("nan")
    elif fit.any():
        beta, omega, rms = _pooled_shell_grid(
            ladder, a1, a2, a3, e[fit], y1[fit], y2[fit], y3[fit])
    else:
        beta = omega = rms = float("nan")

    r = np.zeros(e.size)
    code = np.full(e.size, SHELL_FAIL_NO_FIT, dtype=np.int64)
    if np.isfinite(beta):
        ok = (e - omega) < floor
        code = np.where(ok, SHELL_OK, SHELL_FAIL_POLE)
        if ok.any():
            e_ok = e[ok]
            e_ref = float(np.max(e_ok - omega))
            c = e_ref - e_ok + omega
            sums = []
            for lo, hi in ((a1, a3), (a3, ladder.n_target)):
                de, w = ladder.shell_rule(lo, hi, e_ref)
                x = (de[None, :] + c[:, None]) / ladder.estar_ev
                sums.append(np.exp(np.log(w)[None, :]
                                   - beta * np.log(x)).sum(axis=1))
            r[ok] = sums[1] / sums[0]
    r = r.reshape(shape)
    code = code.reshape(shape)

    s_inf = S[2] + (S[2] - S[0]) * r
    return SpectralShellFit(
        counts=N,
        s_at_counts=S,
        s_inf=s_inf,
        beta=float(beta),
        omega_ev=float(omega),
        tail_ratio=r,
        delta_tail=np.abs(s_inf - S[2]),
        d2=S[1] - S[0],
        d3=S[2] - S[1],
        failure=code,
        fit_mask=fit.reshape(shape),
        residual_ev=float(rms),
        fit_seconds=float(time.perf_counter() - t0),
        ladder=ladder,
        shells=shells,
        held=held is not None,
    )


def spectral_trust_verdict(fit: SpectralShellFit) -> str:
    """One line: the pooled (β, Ω), its fit set, residual and the no-tail count.

    A statement of what was fitted, not a quality metric.  ``β`` or ``Ω`` on a
    grid edge is named because the model then wanted a value the physical
    bounds exclude.
    """
    if not np.isfinite(fit.beta):
        return ("NOT TRUSTWORTHY - no requested state lies below the "
                "extrapolated bands; S_hat = S(N3) on every state.")
    edge = []
    if fit.beta in SHELL_BETA_GRID[:2]:
        edge.append(f"beta at its bound {fit.beta:g}")
    if fit.omega_ev in SHELL_OMEGA_GRID_EV[:2]:
        edge.append(f"Omega at its bound {fit.omega_ev:g} eV")
    no_tail = (f"  {fit.n_failed} of {fit.n_states} states have no tail and "
               f"keep S(N3)." if fit.n_failed else "")
    if fit.held:
        return (f"pooled beta = {fit.beta:.2f}, Omega = {fit.omega_ev:.1f} eV "
                f"held from SC map 0.{no_tail}")
    return (f"pooled beta = {fit.beta:.2f}, Omega = {fit.omega_ev:.1f} eV over "
            f"{int(np.count_nonzero(fit.fit_mask))} states; rms middle-point "
            f"residual {fit.residual_ev * 1e3:.3f} meV"
            f"{' (' + '; '.join(edge) + ')' if edge else ''}.{no_tail}")


def tolerance_bar_ev(fit, quantile: str = "p90") -> tuple:
    """``(median, max)`` of the extrapolation uncertainty over all states, eV.

    :class:`SpectralShellFit` exposes ``uncertainty(quantile)`` as a fraction
    of its own ``Delta_tail``, and that is the only thing this reads.

    Two numbers because they answer different questions and the module has a
    standing rule against reporting the max alone: a max over (k, band) is set
    by the top of the QP window, whose Σ_c is the largest and least converged
    quantity in the run, so it describes that state rather than the
    calculation.  The MEDIAN is the bar on a
    typical state and is what the ruling below triggers on; the MAX is quoted
    beside it as the envelope.
    """
    u = np.abs(np.real(np.asarray(fit.uncertainty(quantile))))
    return float(np.median(u)), float(np.max(u))


def sc_tolerance_ruling(fit, tol_ev: float,
                        *, quantile: str = "p90") -> tuple:
    """Is the SC convergence tolerance inside the extrapolation's own bar?

    See :func:`tolerance_bar_ev`.

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
    fit: SpectralShellFit,
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
    ``Δ_tail`` or β.  That is
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
#: band-diagonal and in eV except β.  The deleted 1/N fit wrote
#: ``sigma_c_extrap_ampl_kn_ev`` (its ``A``) in β's place; β has its own name
#: so an old file cannot be read as a new one.
SPECTRAL_EXTRAP_DATASETS = (
    "sigma_c_extrap_inf_kn_ev",     # Ŝ, the extrapolated Σ_c
    "sigma_c_extrap_last_kn_ev",    # S(N₃), the ordinary full-band Σ_c
    "sigma_c_extrap_beta_kn",       # the pooled β on states with a tail, NaN elsewhere
    "sigma_c_extrap_sigma_kn_ev",   # the p90 uncertainty envelope
)


def _bracket_h5_attrs(plan: BandBracketPlan) -> dict:
    """Artifact provenance for the estimator's h5 payload.

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
    """One log spelling of the planner semantics."""
    if plan.bracket_scheme == "total_fractions":
        return (f"fractions = {BRACKET_FRACTIONS} of the TOTAL band count "
                f"{plan.counts[-1]}")
    if plan.bracket_scheme == "conduction_fractions":
        return (f"fractions = {BRACKET_FRACTIONS} of the CONDUCTION manifold "
                f"(N_i = n_occ + f*(N_max - n_occ)); n_occ = {plan.n_occ}, "
                f"n_cond = {plan.n_cond}, N_max = {plan.counts[-1]}")
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

    Four arrays.  ``sigma_c_extrap_beta_kn`` carries the POOLED β on every
    state that got a tail and NaN on the rest; the pooled β and Ω are also
    attributes.  ``β`` is dimensionless, so ``scale`` (a unit conversion) is
    deliberately NOT applied to it.

    The attributes carry what a reader needs to reproduce the number and
    cannot get from the arrays: the estimator, the pooled (β, Ω), and the
    DFT-only ladder it ran on (``E₀``, ``n₀``, ``E*``, ``N_T``, the shells).
    """
    def _arr(a):
        return np.asarray(a, dtype=np.complex128) * scale

    lad = fit.ladder
    return {
        "arrays": {
            "sigma_c_extrap_inf_kn_ev": _arr(fit.s_inf),
            "sigma_c_extrap_last_kn_ev": _arr(fit.s_at_counts[-1]),
            "sigma_c_extrap_beta_kn": np.asarray(fit.beta_per_state(),
                                                 dtype=np.complex128),
            "sigma_c_extrap_sigma_kn_ev": _arr(fit.uncertainty("p90")),
        },
        "attrs": {
            "band_extrapolation_estimator": "spectral_shell",
            "spectral_shell_form": "pooled_denominator_shell",
            "pooled_beta": float(fit.beta),
            "pooled_omega_ev": float(fit.omega_ev),
            "pooled_residual_rms_ev": float(fit.residual_ev),
            "pooled_state_count": int(np.count_nonzero(fit.fit_mask)),
            # A wall time, so it differs run to run; the verdict above does not.
            "pooled_fit_seconds": float(fit.fit_seconds),
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
    """The log block for ``spectral_shell``.

    ONE log carries the full-band value and the extrapolated value side by
    side with everything that produced it: the three band counts, the shells
    in ABSOLUTE band index, the DFT-only ladder, the pooled (β, Ω) and each
    named state's tail ratio.  A reader can recompute Ŝ from this block.
    """
    def _sg(a):
        return float(np.real(np.asarray(a))) * scale

    def _mx(a):
        return float(np.max(np.abs(np.real(np.asarray(a))))) * scale

    lad = fit.ladder
    geometry = _bracket_geometry_text(plan)
    lines = [
        f"  -- {label} band-convergence extrapolation "
        f"(estimator = spectral_shell: pooled denominator shell, band A adds "
        f"a_i * sum_k w_k (E_Ak - E_i + Omega)^-beta, tail integrated to the "
        f"finite basis) --",
        f"     N_occ = {plan.n_occ}   N_cond = {plan.n_cond}   {geometry}",
    ]
    lines.extend(_bracket_report_lines(plan, unit))
    lines += [
        f"     {lad.describe()}",
        f"       shells (ABSOLUTE band index, half-open at the low end): "
        f"amplitude over {(fit.shells[0][0], fit.shells[1][1])}, interior "
        f"point at {fit.shells[0][1]}, tail over {fit.shells[2]}",
        f"       N_T = {lad.n_target} is the FINITE PLANE-WAVE BASIS "
        f"(min(ngk)*nspinor), not infinity.  Bands "
        f"{lad.n_dft + 1}..{lad.n_target} come from the Weyl ladder, used to "
        f"extend the EIGENVALUE SEQUENCE only.",
        f"       pooled fit: {spectral_trust_verdict(fit)}  "
        f"(fit + apply {fit.fit_seconds:.3f} s)",
    ]
    for slabel, index in (states or []):
        f1 = fit.at(index)
        code = int(np.asarray(f1.failure))
        head = [
            f"     [{slabel}]",
            f"       S(N1={plan.counts[0]}) = {_sg(f1.s_at_counts[0]):+12.6f}   "
            f"S(N2={plan.counts[1]}) = {_sg(f1.s_at_counts[1]):+12.6f}   "
            f"S(N3={plan.counts[2]}) = {_sg(f1.s_at_counts[2]):+12.6f} {unit}"]
        if code != SHELL_OK:
            lines += head + [
                f"       *** NO TAIL: {SHELL_FAILURE_REASONS[code]}  "
                f"-> S_hat = S(N3) = {_sg(f1.s_inf):+12.6f} {unit}"]
            continue
        lines += head + [
            f"       r = G(N3,N_T)/G(N1,N3) = "
            f"{float(np.real(f1.tail_ratio)):10.6f}   S_hat = S(N3) + "
            f"(S(N3) - S(N1)) r = {_sg(f1.s_inf):+12.6f} {unit}   "
            f"Delta_tail = {_mx(f1.delta_tail):.6f} {unit}",
        ]
    lines += [
        f"     [envelope over ALL (k, band) of the QP window -- set by the "
        f"top of the window, not the result]",
        f"       max|S(N3)| = {_mx(fit.s_at_counts[-1]):.6f}   "
        f"max|S_hat| = {_mx(fit.s_inf):.6f} {unit}   "
        f"max Delta_tail = {_mx(fit.delta_tail):.6f} {unit}   "
        f"({fit.n_failed} of {fit.n_states} without a tail)",
        f"       The +/- envelope ({100*TAIL_UNCERTAINTY_FRACTION[0]:.0f} % / "
        f"{100*TAIL_UNCERTAINTY_FRACTION[1]:.0f} % of Delta_tail) is inherited "
        f"from the deleted 1/N fit and not re-derived.  It covers the "
        f"extrapolation only -- not the difference from BerkeleyGW, the ISDF "
        f"basis, or the W-side band count.",
    ]
    if fit.n_failed:
        lines.append("     " + fit.failure_report().replace("\n", "\n     "))
    return "\n".join(lines)


__all__ = [
    "BRACKET_FRACTIONS",
    "BRACKET_SCHEMES",
    "BRACKET_SCHEME_DEFAULT",
    "CONDUCTION_HALF_FRACTION",
    "SPECTRAL_EXTRAP_DATASETS",
    "TAIL_UNCERTAINTY_FRACTION",
    "BAND_EXTRAPOLATION_ESTIMATORS",
    "BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT",
    "SHELL_FAILURE_REASONS",
    "SHELL_OK",
    "SHELL_FAIL_POLE",
    "SHELL_FAIL_NO_FIT",
    "SHELL_BETA_GRID",
    "SHELL_OMEGA_GRID_EV",
    "SHELL_POOL_WINDOW_EV",
    "SHELL_POOL_DEGENERACY_EV",
    "pooled_state_mask",
    "extrapolation_weights",
    "spectral_h5_payload",
    "BandBracketCountMismatch",
    "BandBracketPlan",
    "BandExtrapolationRefused",
    "BandLadder",
    "SpectralShellFit",
    "assert_brackets_match_ols_abscissae",
    "build_band_ladder",
    "fit_band_extrapolation_spectral",
    "format_spectral_report",
    "plan_band_brackets",
    "plane_wave_band_count",
    "sc_tolerance_ruling",
    "spectral_trust_verdict",
    "tolerance_bar_ev",
    "trivial_plan",
    "weyl_ladder_fit",
]
