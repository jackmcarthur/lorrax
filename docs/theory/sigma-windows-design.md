# Sigma windows: one plan and two band classes

Decision, 2026-09-27, WINDEC. This page owns window policy.
The implementation is a rejected landing candidate: its short-map accuracy and cost gates fail.
The rule construction remains in [minimax quadrature](minimax-quadrature.md).
The SC equations remain in [self consistency](../self_consistency.md).

## User specification

1. `nval` and `ncond` count the protected QP states below and above E_F at each k (at `nelec` on an insulator).
2. `number_bands` sets the complete rotating and sum-band carrier.
3. `sigma_omega_min_ev` protects every state at or above it and lowers the sampled lower endpoint, in eV relative to the Sigma reference.
4. `sigma_omega_max_ev` protects every state at or below it and raises the sampled upper endpoint; either endpoint may be omitted.
5. `sigma_omega_step_ev` sets sampling; `sigma_window_ev`, `sigma_out_of_grid`, `sigma_omega_patches_ev`, `sigma_window_edge_factor`, and `sc_frozen_core_bands` refuse by name.

## Decisions and cost

T below is the measured joint cost of the chosen support and box geometry.
A marginal pair saving is not inferred from a run in which several choices changed.
Classification, interpolation and Hamiltonian masks add no quadrature pairs themselves.

| Decision | Reason | Tau-pair cost |
|---|---|---|
| Close requested bands outward across adjacent DFT gaps ≤ eta at each k; freeze identities. | This avoids cutting unresolved multiplets, but is not a remote-coupling certificate. | Fe T=750; its class-accuracy gate fails. |
| Limit automatic class promotion to 2 eV per side; refuse a more distant gap. | A dense ladder cannot silently promote an entire remote spectrum. | Zero classification pairs; accepted extrema enter T. |
| Use the 2 eV outer pad on one-shot and SC map 0. | Both start from the same support policy. | Included in Si T=384 and the SC totals below; no isolated pad-cost measurement. |
| Include the ±0.5 eV Z stencil before the outer pad. | Those are actual Sigma reads. | Included in T; no second motion pad. |
| Plan once, including dormant product boxes; freeze selectors, grid and rules. | Anderson must see a fixed interpolation and quadrature map. | No in-loop rule build; dormant activation adds 11 pairs on MoS2 and up to 18 on Na. |
| Clamp protected reads to the nearest endpoint during iteration. | This is continuous and bounded after an excursion. | Zero added pairs; Fe bispinor still fails the short-map accuracy gate. |
| Check at convergence and rebuild at most once, retaining the old grid and restarting history. | An early excursion alone must not enlarge permanent support. | At most one additional complete plan and solve; no second outer pad. |
| Use selector margin eta and one crossing rectangle per causal branch, plus non-crossing tails. | These selectors give a denominator bound without an empirical geometry key. | All crossing boxes, including opposite-frequency corners: Fe 606, MoS2 859, Si 323, bispinor 552 per sector, Na 1776. |
| Use the provable insulator short side, including poles down to zero. | A gap trend is not a certificate. | MoS2 T=913/924; crossing cost alone exceeds 500. |
| Use four times the initial far state, pole and damping extents. | Far-side reserve cannot enlarge the crossing edge. | Non-crossing costs: Fe 144, MoS2 54/65, Si 61, bispinor 139 per sector, Na 114/123/132. |
| Use epsilon=1e-4 and QUADWIRE's derived rules on every W tier. | Keep one accuracy target and one rule owner. | T is measured at 1e-4; no tolerance relaxation is selected. |
| Give every rotating band its DFT energy plus one scissor per side of mu, the mean H_ii - E_i of the protected occupied (empty) states; deep bands take the occupied-side scissor. | A diagonal error reaches a protected state at second order, |V|^2 dbeta/Delta^2; with exact couplings this law leaves 0.9-1.3 meV at the Si fixed point against 2.4-3.0 for a static-QSGW-plus-correlation diagonal (CLASSMIX). | Zero pairs. |
| Read every rotating endpoint beyond the near support from far patches planned once at map 0 (rotating DFT energies +/- 2 eV, merged across holes <= 4 eV), one plan per eta: 1 eV above E_F, 2 eV below; the rotating diagonal is its own-energy patch read. With patches the side scissor is the fallback only. | The P-R coupling at the protected energy alone costs 33 meV (Fe) and 2-4 meV (Si) at the fixed point; far reads recover 67-85% of it (CLASSMIX round 2). Semicore endpoints are eta-insensitive to 2 eV. | Fe 748 -> ~1110, MoS2 338 -> 462 pairs/map; each far plan pays a 117-192 pair box floor. |
| Reject the tested global-envelope automatic boundary replacement. | Its Fe spread is 1988.003 meV, against 1 meV. | 1613 pairs on its common broad diagnostic grid; no production saving established. |

| Gate deck | Maps | T per map | Fixed sampled support, eV |
|---|---|---|---|
| Fe 4³ charge | 0–3 | 750/750/750/750 | [-12,26.5] |
| MoS2 3×3 | 0–3 | 913/924/924/924 | [-64,13.5] |
| Si 4³ SOC | one-shot | 384 | [-15,12.25] |
| Fe 4³ bispinor | 0–2 | 691 per sector; 2764 total each map | [-12,21.5] |
| Na 8³ | 0–2 | 1890/1899/1908 | [-56.5,98.5] |

These are executed pairs; reserves that remain empty do not execute.
A wide band request can include deep states: the MoS2 deck requests all 44 bands.
The implementation honors that request rather than silently applying an energy cutoff.

## Hamiltonian and geometry

Protected bands P receive full QSGW Sigma and full mixing. Every other loaded
band is R. A rotating endpoint beyond the near support reads the held far
patch (P-R: the usual half-sum; diagonal: own energy). Without patches (sector
routes) its diagonal is its DFT energy plus the QP-correction scissor of its
side of mu.
R–R off-diagonal entries vanish. A P–R entry uses the half-sum with the rotating endpoint read from its far patch (herm Sigma_ij(E_i), i in P, on sector routes);
P–P uses the usual Hermitian endpoint half-sum. The fixed DFT partition follows
state identities through the eigenvectors; energy sorting never reclassifies P.

Let X=kBT log(1/1e-5) for retained metallic occupations and X=0 for an insulator.
On each causal branch E≥−X. For the crossing half with omega≤W,
selectors use E, Omega≤W+eta+X. The complement is a state tail and a pole tail.
The opposite-frequency branch has its small crossing corner and non-crossing
bulk, pole and frequency tails. The selectors, grid and far ceilings define
immutable denominator boxes; outward cache snapping supplies numerical enclosure.
There is no empirical zero-side cap, percentage pad or escape refit.

Each map records clipped protected stencils and escaped boxes. A converged
candidate with either condition requests the sole rebuild. A second failed
check refuses with `GATE sigma_plan_fixed_point`. Intermediate escapes are not
certified evaluations merely because the run continues. Rotating states never
force sampled-support growth. Omitted endpoints add no implicit zero anchor.

## Numerical verdict and limits

Claim 2895 records the same-node R57 comparisons, metrics, HDF5 checks and costs.
Band classes, clamping, deep requested states, fixed boxes and epsilon can move
energies. Compare common physical rows using claim 2866: mean, centered standard
deviation, MAD, maximum centered deviation, and band-edge differences.
The mean is not the acceptance statistic. Required spread within ±5 eV of DFT
E_F is ≤1 meV; pair limits are 500 for insulators and 1000 for metals.

Fe's covered fixed-fit wide-class reference gives spreads
2064.752/628.587/1074.265/1012.491 meV on maps 0–3. All alternatives fail:
occupied-only widening 75.093 meV; empty-only 2070.196; global envelope 1988.003.
Their common broad-grid cost is 1613 pairs; the identity control is 0.000225 meV.
Thus eta-gap closure is degeneracy hygiene, not a sufficient class-accuracy remedy.
No tested automatic boundary is certified for the supplied small Fe request.
Fe bispinor frequency-only spread reaches 6.270 meV; MoS2 and Na fail pair budgets.
SPCOST B (claim 2893) supplies no fixed-point scissor ranking; QCOST (claim 2891)
supplies no tested node-count reduction that resolves these broader requests.

Source 8f155c121 is on `feat/sigma-window-decision-2026-09-27-r2`, not main.
Short SC trajectories do not prove convergence or W/ISDF accuracy. Existing
replicated full-band SC matrices and host identity-assignment loops remain
scale defects. Small-deck gates cannot certify the complete carrier at scale.
