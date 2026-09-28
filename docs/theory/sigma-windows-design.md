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
- **Far patches.** Every rotating DFT energy outside the near support, padded
  by 2 eV and merged across holes ≤ 4 eV (`qp_support.far_patches_ev`). A
  patch above E_F is broadened to η_far = 1 eV, one below to 2 eV (ruling
  Q4; INVARIANTS 12 exception for rotating-endpoint reads only), sampled at
  η_far/2. Far patches join the near plan: each crossing window splits by the
  η of the frequencies it serves and certifies at ε_far = 1e-2
  (`FAR_PATCH_EPS`); sign-definite windows serve far frequencies at the near η.
- **Held.** No map repads, refits or rebuilds a window. RECIPE3's W line
  ladder reads this one map-0 support.
- **Escape refuses** (ruling Q5). A protected read stencil outside the held
  support, or a product window whose current states or poles leave its held
  box, refuses with `GATE sigma_plan_escape`, naming the state or window.
  There is no clamp and no rebuild. The remedy is a deck change: protect
  fewer edge states or move the ω endpoints.

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

## Measured

Claim PARTITION (this lane) holds the gate numbers: τ pairs per map, the
±10 eV std and maxdev against a same-tree all-protected reference at map 0
and at the fixed point, and the cost against main, on Si 4³, MoS2 3×3,
Fe 4³ charge, Na 8³ and Fe 4³ bispinor.
