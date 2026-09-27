# Sigma windows: one plan and two band classes

Decision, 2026-09-27, WINDEC. This page owns window policy.
The rule construction remains in [minimax quadrature](minimax-quadrature.md).
The SC equations and artifact contract remain in [self consistency](../self_consistency.md).

## User specification

1. `nval` and `ncond` request the occupied and empty QP bands around `nelec`.
2. `number_bands` sets the complete rotating and sum-band carrier.
3. `sigma_omega_min_ev` only enlarges the negative sampled extent, in eV relative to the Sigma reference.
4. `sigma_omega_max_ev` only enlarges the positive extent; either endpoint may be omitted.
5. `sigma_omega_step_ev` sets sampling; `sigma_window_ev`, `sigma_out_of_grid`, `sigma_omega_patches_ev`, `sigma_window_edge_factor`, and `sc_frozen_core_bands` refuse by name.

## Decisions and cost

| Decision | Reason | Tau-pair cost |
|---|---|---|
| Close requested bands outward across adjacent DFT gaps ≤ eta, independently at each k; freeze those identities. | A requested edge must not bisect an unresolved manifold. | Only the resulting support extrema change the boxes; measure the total below. |
| Bound automatic class-edge promotion to 2 eV per side; refuse a more distant gap. | A dense ladder cannot silently turn a small request into all states. | Zero pairs for accepted decks; no enlarged fallback. |
| Keep the 2 eV outer pad on one-shot and SC map 0. | Both start from the same support policy. | Prior WINDESIGN Si estimate +50 pairs; remeasure at epsilon 1e-4. |
| Include the ±0.5 eV Z stencil before adding the outer pad. | These are actual Sigma reads, not another motion allowance. | Included in each measured total. |
| Plan once; never grow the grid or refit inside the SC loop. | Anderson must see one map with fixed interpolation and quadrature. | Zero in-loop rule-build cost; dormant boxes consume pairs only when populated. |
| Clamp protected reads to the nearest sampled endpoint. | This is continuous and bounded when a state leaves support. | Zero extra pairs during iteration. |
| Check at convergence and rebuild once if needed; restart the history then. | An early excursion does not justify permanent support growth. | At most one additional complete plan and SC solve. |
| Fix the selector margin at eta. | It bounds the non-crossing denominator away from zero without a tunable geometry key. | Saves the former extra 0.5 eta of crossing width. |
| Use one crossing rectangle per causal branch, with separate non-crossing tails. | The state and pole selectors give a direct denominator bound. | No crossing subwindows or per-state rule families. |
| Use the provable insulator short side, including poles down to zero. | A trend in the gap is not a certificate. | Prior MoS2 estimate +61 pairs; remeasure at epsilon 1e-4. |
| Use four times the initial far state, pole and damping extents. | Far-side growth costs logarithmically and cannot enlarge the crossing edge. | Prior estimate +8–9 pairs; remeasure with derived rules. |
| Default epsilon = 1e-4 on every W tier; use QUADWIRE's derived rules. | One explicit accuracy target and one rule owner. | Measured per-deck totals decide the 500/1000 limits. |
| Route rotating Fermi-crossing bands through the existing rigid Fermi shift. | They need a rotating diagonal law; this keeps their bandwidth without a new fit parameter. | Zero pairs; map 0 is unchanged. |
| Retain the existing scissor fit behind its existing API. | SPCOST B has map-0 comparisons but no converged fit verdict; window geometry does not choose the fit form. | No quadrature pairs. |

## Hamiltonian

Protected bands P receive full QSGW Sigma and full mixing. Every other
loaded band belongs to R. The R diagonal is the existing conduction scissor,
the current Fermi displacement for ordinary valence and Fermi-crossing bands,
or DFT for deep bands.
There are no R–R off-diagonal entries. A P–R entry uses the Hermitian part
of Sigma_ij(E_i), i in P. P–P uses the usual Hermitian endpoint half-sum.
The fixed DFT partition is transported through the current eigenvectors;
energy sorting does not redefine a protected identity.

The edge closure is a chosen approximation, not a bound on remote coupling.
Its numerical acceptance requires the wide-reference test below. It may
promote a connected manifold within 2 eV of each requested edge; a more
distant closure refuses and requires an explicit larger band request.
A wide requested band range, including deep states, is intentionally costly;
no hidden active-band cutoff overrides the user's requested states.

## Geometry

Let X = kBT log(1/1e-5) for FD occupations, and X = 0 for an insulator.
The retained occupation support gives E >= -X on either causal branch.
For a crossing half with omega <= W, selectors use E, Omega <= W + eta + X.
The complement is a state tail and a pole tail, both non-crossing.
The opposite-frequency branch needs only its small E/Omega/omega corner;
its remaining bulk, pole tail and frequency tail are non-crossing.
These selectors, the initial omega grid and the far ceilings define each
immutable denominator box. Outward cache snapping supplies numerical
closure. There is no extra percentage pad or empirical zero-side cap.
Dormant product boxes are certified in the initial plan, so a later
population change does not require a new rule or a changed topology.

A protected state leaving the omega support is clamped and counted per map.
At a converged candidate, any remaining clipped stencil or violated far box
requests the sole rebuild. Failure after that rebuild refuses by name.
The support is never enlarged merely to cover rotating states.

## Acceptance and expected changes

Band classes, endpoint clamping, deep requested states, fixed boxes, and
epsilon can all move energies relative to R55. Mean shifts are reported,
not used as an accuracy verdict. Compare common physical bands with one
wide reference per deck using claim 2866: mean, standard deviation, MAD,
maximum deviation about the mean, and 2/4-band edge differences.
The standard deviation within ±5 eV of DFT E_F must be ≤1 meV.
Every SC map reports its support and active tau pairs; limits are 500 for
insulators and 1000 for metals. HDF5 comparisons authenticate shapes and
values independently. A short SC budget is a trajectory check, not proof
of convergence or of the one-rebuild path.

## Implementation status

Candidate implementation is on `feat/sigma-window-decision-2026-09-27`.
Numerical acceptance is pending. The existing full-band SC matrix and
identity-assignment seams still replicate matrices / use host axis loops;
these are recorded prerequisites to certifying the complete rotating carrier
at scale. Small-deck evidence does not resolve that architectural defect.
