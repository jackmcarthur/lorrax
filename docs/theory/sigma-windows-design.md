# Sigma windows: one plan and two band classes

This page owns the Σ window policy of self-consistent QSGW: which states are
protected, which rotate, where Σ is sampled, and what happens when a state
leaves the plan. The scheme is the owner's of 2026-09-28 (rulings Q2–Q5,
TRACKER). The rule construction is in [minimax quadrature](minimax-quadrature.md)
and the SC equations are in [self consistency](../self_consistency.md).

## Input

1. `nval` and `ncond` request states below and above E_F at each k (from
   `nelec` on an insulator). `number_bands` sets the whole rotating and
   sum-band carrier.
2. `sigma_omega_min_ev` and `sigma_omega_max_ev` (eV from the Σ Fermi
   reference, either may be omitted) only enlarge: every state between them
   is protected, closed over its η-resolved manifold, and the sampled support
   reaches them.
3. `sigma_omega_step_ev` sets the near sampling step. `sigma_window_ev`,
   `sigma_out_of_grid`, `sigma_omega_patches_ev`, `sigma_window_edge_factor`
   and `sc_frozen_core_bands` refuse by name.

## Classes

- **Protected P.** Every requested state whose DFT energy lies within
  `WINDOW_CLIP_EV` = 10 eV of μ (μ at each map's entry on a metal, the DFT
  midgap on an insulator), closed over exact degeneracies, plus every state
  inside the ω endpoints. Requested edges close outward to the next gap
  larger than η; an automatic promotion beyond 2 eV refuses
  (`GATE sigma_band_edge_gap`) for an edge inside the clip.
- **Requested states outside the window** (semicore, far conduction) rotate.
  They are still reported; their energies come from the far patches below
  (sub-eV class).
- **Rotating R.** Every other loaded band. Classes are fixed at map 0 by DFT
  identity and follow the eigenvectors; energy sorting never reclassifies.

## One plan at map 0, held

- **Near support.** The protected DFT energies, the ±0.5 eV Z stencil and one
  2 eV outer pad, enlarged to the ω endpoints; sampled at the deck η and
  ε = 1e-4 (ruling Q2). Crossing and non-crossing product windows are planned
  once with their reserves (`sigma_box_plan`).
- **Map-0 probe.** Map 0 first evaluates Σ on that plan. If a protected
  state's map-0 QP estimate lies outside the support, or a rotating one
  outside the near support and every far patch, the plan is made once from
  the DFT energies and those estimates, with the same pad, and map 0 is
  re-evaluated (`sc_iteration.gw_iteration_map`). The estimates are the
  diagonal of the map-0 QSGW Hamiltonian and the output eigenvalue each DFT
  identity carries into map 1 (`sc_state_identity.assign_qp_identity`);
  strong P–R mixing separates the two by several eV (Fe 4³ bispinor). A
  scalar route re-reads the same file-resident W; a sector route rebuilds W,
  because Σ releases its device-resident models. An insulator's Σ frame is the DFT
  midgap, so the map-0 gap opening is not absorbed by the frame: MoS2 3×3
  conduction states move +2.4 to +3.1 eV and the Si 4³ conduction top
  +2.3 eV at map 0, beyond a 2 eV pad about the DFT energies.
- **Far patches.** Every rotating DFT energy (and map-0 estimate) outside the
  near support, padded by 6 eV and merged across holes ≤ 12 eV
  (`qp_support.far_patches_ev`); a patch that reaches the near support starts
  1e-3 eV past its edge, so no energy falls between the two. The 6 eV pad
  keeps each rotating own-energy fixed point inside its patch in the frame
  Σ is read in (Fe 4³ 3s: 3.3 eV below its DFT energy, plus a +1.3 eV move
  of the metal μ); a fixed point outside flips between
  the own read and the side scissor from map to map, and the SC map has no
  fixed point. A
  patch above E_F is broadened to η_far = 1 eV, one below to 2 eV (ruling
  Q4; INVARIANTS 12 exception for rotating-endpoint reads only), sampled at
  η_far/2. Far patches join the near plan: each crossing window splits by the
  η of the frequencies it serves and certifies at ε_far = 1e-2
  (`FAR_PATCH_EPS`); sign-definite windows serve far frequencies at the near η.
- **Held.** No map repads, refits or rebuilds a window. The plan is one
  object from one call, `qp_support.plan_sigma_windows` → `SigmaPlan`: the
  protected support `protected_support_ev` (deck η, 2 eV pad and Z stencil
  included) and the far patches `far_patches_ev` with their `far_eta_ev`. The
  SC map carries it as `SCSupport.plan` and the session's `"sigma_plan"`;
  the W sampling ladder reads it there. At map 0 the W of the probe pass is
  built before the probe's plan exists; maps ≥ 1 read the held plan.
- **Escape refuses** (ruling Q5). A protected state whose Σ read (its
  current energy) leaves the held support refuses with
  `GATE sigma_plan_escape`, naming the state. There is no clamp and no
  rebuild. The ±0.5 eV Z stencil is one-sided at an edge by design
  (`eqp_bgw.compute_z_factor_from_omega_grid`) and is counted, not refused.
  A product window whose live box leaves its held box (W poles move between
  maps) keeps its rule and is counted in the receipt (`escaped`).

## Hamiltonian

In the DFT basis, with endpoints read at the current QP energies:

| block | rule |
|---|---|
| P–P | the half-sum ½[Σ_ij(E_i) + Σ_ij(E_j)], Hermitian part |
| P–R | the half-sum, the rotating endpoint read from the far patch that covers E_o (or from the near cube inside the near support) |
| R diagonal | Σ_oo(E_o) where a patch or the near support covers E_o (ruling Q3); elsewhere E_DFT,o + β_side (scissor law A: β is the k-star-weighted mean H_ii − E_i of the protected occupied or empty states) |
| R–R off-diagonal | zero (CLASSMIX: keeping the static block moves protected states ≤ 0.3 meV) |

A rotating state that no patch covers contributes no endpoint of its own: its
P–R entries are read at the protected energy (`qsgw_utils._qsgw_far_kernel`).
Sector (bispinor) routes have no far patches; there every rotating diagonal
is the side scissor and P–R is read at the protected energy.

## Why

- A rotating-diagonal error δβ reaches a protected state at second order,
  |V|²δβ/Δ²; a coupling error δV at first order, 2|V|δV/Δ. Reading P–R
  couplings only at the protected energy makes δV = ½[Σ_io(E_o) − Σ_io(E_i)],
  of order V; this set Fe at 33 meV and Si at 2–4 meV at the fixed point
  (CLASSMIX, claim 2896). Far reads at η_far 1–2 eV recover it (claims 2902,
  2905, 2908).
- A far crossing window's node count follows its short side over η,
  N ≈ 2.7 s/η + 20, so pole or state splits add windows; ε_far and η_far are
  the levers (claim 2908).
- Protecting every requested band exposes deep states that cost crossing
  pairs beyond the budgets (MoS2 859, Na 1776; WINDEC, claim 2895). The
  ±10 eV clip keeps a band-count request within ≤ 500 τ pairs per map on
  insulators and ≤ 1000 on metals.

## Measured (PARTITION, 2026-09-28; not main)

Same node, P4; reference = the same tree with every band protected at the deck
η and ε 1e-4. Protected states within ±10 eV of E_F; std and maxdev in meV.

| deck | τ pairs/map (main) | map 0 std / maxdev | fixed point std / maxdev | maps |
|---|---|---|---|---|
| Si 4³, 8v/8c, ω [−6, 6] | 416 (414) | 5.8 / 44 | 3.4 / 8.9, mean +11 | 7 |
| Si 4³, 8v/8c, ω [−13, 10] | 446 | 1.1 / 7.7 | 1.9 / 5.9, mean +1.0 | 9 |
| MoS2 3×3, 10v/12c, ω [−10, 10] | 429 (392) | 1.3 / 5.7 | 3.3 / 13 | 8 |
| Fe 4³ charge, 6v/8c, ω [−12, 8] | 765 (829) | 28 / 160 | reference stalls | 14 |
| Na 8³, all 86 bands requested | 654 (1621) | 1.7 / 7.0 | SC-2 only | – |

- **Occupied rotating states set the fixed-point error.** From map 1 on,
  Σ_x and W are built from every occupied orbital. A rotating occupied
  state shifts every protected state: the Si Γ valence bottom (−12.3 eV, cut
  by the ±10 eV clip) moves the Si fixed point +11 meV; protecting it
  (ω_min −13 eV) leaves +1.0 meV. MoS2's rotating semicore (read at
  η_far 2 eV, 0.1–0.4 eV off) opens its gap 6 meV. A frozen-W replay cannot
  see this.
- **An interleaved P/R edge sets the map-0 error.** Requested bands that
  end inside a dense spectrum leave rotating states at the same energies as
  protected ones (Si bands 15–16 at +9.7 eV: 45 meV at map 0; with every
  state below +10 eV protected: 7.7 meV).
- **Fe is not attributed.** Protecting every requested state (ω_max +24 eV,
  924 pairs) moves Fe's map-0 std from 28 to 21 meV and its ±1 eV shell from
  30 to 4.3 meV; single rows near E_F differ by up to 156 meV in both, and
  the all-protected Fe reference stalls at 73 meV after 30 maps.
- Requested states outside the window read far patches: Si Γ valence bottom
  16 meV at map 0; Fe +10.4…+23.8 eV median 163 meV; Na (2357 (k,state))
  median 0.22 eV, max 1.2 eV.
