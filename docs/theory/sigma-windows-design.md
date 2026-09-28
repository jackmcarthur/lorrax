# Sigma windows: one plan and three band classes

This page owns the Σ window policy of self-consistent QSGW: which states are
active, which rotate, where Σ is sampled, and what happens when a state
leaves the plan. The scheme is the owner's of 2026-09-28 (rulings Q2–Q5,
TRACKER; round 2 of lane PARTITION). The rule construction is in
[minimax quadrature](minimax-quadrature.md) and the SC equations are in
[self consistency](../self_consistency.md).

## Input

1. `nval` and `ncond` request states below and above E_F at each k (from
   `nelec` on an insulator); a request larger than a k holds takes all.
   `number_bands` sets the whole rotating and sum-band carrier.
2. `sigma_omega_min_ev` and `sigma_omega_max_ev` (eV from the Σ Fermi
   reference, either may be omitted) only enlarge: every state between them
   is protected at the deck η.
3. `sigma_omega_step_ev` sets the near sampling step. `sigma_window_ev`,
   `sigma_out_of_grid`, `sigma_omega_patches_ev`, `sigma_window_edge_factor`
   and `sc_frozen_core_bands` refuse by name.

## Classes (`band_partition.sc_band_classes`, from the DFT ladder at map 0)

- **Every occupied state is active.** None rotates or is scissored: from
  map 1 on Σ_x and W are built from every occupied orbital, so a rotating
  occupied state shifts every protected state (round 1: the Si Γ valence
  bottom +11 meV, MoS2 semicore +6 meV on the gap).
- **Protected P, read on the near grid at the deck η.** The valence
  manifold, from E_F down through every all-k gap narrower than
  `SEMICORE_GAP_EV` = 5 eV (twice the padded near halfwidth, so a narrower
  gap is sampled anyway), and every state up to the cut.
- **The cut is a spectral gap.** `top` is the highest requested energy,
  clipped to μ + 10 eV (`WINDOW_CLIP_EV`) and raised to ω_max; the cut is
  the midpoint of the widest gap of the all-k spectrum that opens in
  [top, top + 2 eV). Every state below it is protected at every k, so no
  rotating state sits among protected ones.
- **Semicore S.** Occupied states below a gap wider than 5 eV: active (full
  mixing, their rows kept), read at their own energy on held patches at
  `SEMICORE_ETA_EV` = 1 eV. States inside [ω_min, ω_max] stay protected. On
  a sector (bispinor) route there are no patches and S is protected at η.
- **Rotating R.** The empty states above the cut. Classes are fixed by DFT
  identity at map 0 and follow the eigenvectors.

## One plan at map 0, held

- **Near support.** The protected DFT energies, the ±0.5 eV Z stencil and one
  2 eV outer pad, enlarged to the ω endpoints; sampled at the deck η and
  ε = 1e-4 (ruling Q2). Crossing and non-crossing product windows are planned
  once with their reserves (`sigma_box_plan`).
- **Map-0 probe ("plan at the first iteration").** Map 0 first evaluates Σ
  on a provisional plan. The plan is then made once from the DFT energies
  and the map-0 estimates (the diagonal of the map-0 QSGW Hamiltonian and
  the output eigenvalue each DFT identity carries into map 1,
  `sc_state_identity.assign_qp_identity`), and map 0 is re-evaluated. This
  happens whenever a protected estimate leaves the provisional support or
  the deck has semicore or rotating states (their pads derive from the
  probe). A scalar route re-reads the same file-resident W; a sector route
  rebuilds W. The insulator Σ frame is the DFT midgap, so the map-0 gap
  opening is not absorbed by the frame (MoS2 3×3 conduction +2.4 to
  +3.1 eV, Si 4³ conduction top +2.3 eV at map 0).
- **Patches.** Rotating (above E_F, η_far = 1 eV) and semicore (below,
  η_semi = 1 eV) energies outside the near support, each padded by a
  derived pad, max over the class of |E_map0 − E_DFT| + 2 eV
  (`qp_support.derived_pad_ev`), merged across holes shorter than twice the
  pad (`far_patches_ev`); a patch that reaches the near support starts
  1e-3 eV past its edge. Sampled at η/2. Patches join the near plan: each
  crossing window splits by the η of its frequencies and certifies at
  ε_far = 1e-2 (`FAR_PATCH_EPS`); sign-definite windows serve patch
  frequencies at the near η. Ruling Q4 (INVARIANTS 12) admits η 1–2 eV here.
- **Read mode fixed at map 0 (Q3 continuity).** A rotating state reads its
  own energy if the near grid or a rotating patch covers its DFT energy and
  every map-0 estimate, and takes the side scissor otherwise, for good.
- **Held, one object.** `qp_support.plan_sigma_windows` → `SigmaPlan`:
  `protected_support_ev` (deck η, pad and Z stencil included),
  `far_patches_ev`/`far_eta_ev` (rotating), `semicore_ev`/`semicore_eta_ev`,
  and the two pads. The SC map carries it as `SCSupport.plan` and the
  session's `"sigma_plan"`; the W sampling ladder reads it there. At map 0
  the probe pass's W precedes the final plan; maps ≥ 1 read the held plan.
- **Escape refuses** (ruling Q5). A protected read outside the near support,
  a semicore read outside its patches, or an own-energy rotating read
  outside the near support and its patches refuses with
  `GATE sigma_plan_escape`, naming the state. No clamp, no rebuild. The
  ±0.5 eV Z stencil is one-sided at an edge by design and is counted. A
  product window whose live box leaves its held box (W poles move) keeps its
  rule and is counted in the receipt (`escaped`).

## Hamiltonian

In the DFT basis, with endpoints read at the current QP energies:

| block | rule |
|---|---|
| P–P, P–S, S–S | the half-sum ½[Σ_ij(E_i) + Σ_ij(E_j)], Hermitian part; a P endpoint reads the near grid, an S endpoint its semicore patch |
| P–R, S–R | the half-sum, the rotating endpoint read where its fixed mode says (near grid or rotating patch); a scissored R contributes no endpoint (read at the active energy) |
| R diagonal | own-energy read, or E_DFT,o + β_above (scissor law A) for a scissored state |
| R–R off-diagonal | zero (CLASSMIX: keeping the static block moves protected states ≤ 0.3 meV) |

The QSGW kernel takes the two read masks from the SC map
(`sc_sigma_protected_kn`, `sc_sigma_far_kn`; `qsgw_utils._qsgw_far_kernel`).

## Why

- A rotating-diagonal error δβ reaches a protected state at second order,
  |V|²δβ/Δ²; a coupling error δV at first order, 2|V|δV/Δ. Reading P–R
  couplings only at the protected energy makes δV = ½[Σ_io(E_o) − Σ_io(E_i)],
  of order V (CLASSMIX, claim 2896); far reads at η_far 1–2 eV recover most
  of it (claims 2902, 2905, 2908).
- From map 1 on Σ_x and W are built from every occupied orbital, so an
  occupied state that rotates shifts every protected state (claim 2928).
- A crossing window's node count follows its short side over η,
  N ≈ 2.7 s/η + 20 (claim 2908): semicore reads at the deck η cost MoS2
  1580 pairs per map and Fe 1685; at η_semi 1 eV, 490 and 909.

## Measured (PARTITION round 2, 2026-09-28; not main)

Same node, P4; reference = the same tree with every band protected at the
deck η and ε 1e-4. Protected states within ±10 eV of E_F; std / maxdev in meV.

| deck | τ pairs/map (main) | map 0 | fixed point | maps |
|---|---|---|---|---|
| Si 4³ 8v/8c (ω [−6,6] or [−13,10]) | 476 (414) | 1.10 / 7.7 | 1.14 / 2.6 | 9 |
| MoS2 3×3 10v/12c ω [−10,10] | 490 (392) | 0.84 / 3.9 | 2.32 / 8.5; ±1 eV shell 2.0 | 10 |
| Fe 4³ charge 6v/8c ω [−12,8] | 909 (829) | 26.7 / 158 | stalls (3s, 0.33 eV) | – |
| Na 8³ 86 requested | 755 (1621) | 1.79 / 7.2 | SC-2 only | – |

Attribution, one arm per class:

- **Semicore** (read at η_semi on patches): MoS2 with the semicore protected
  at η, 0.04 / 0.14 meV at the fixed point but 1580 pairs; η_semi 0.5 eV,
  1.46 / 4.8 meV at 592 pairs. The error scales with η_semi through the P–S
  couplings; ε_semi 1e-4 changes nothing (< 0.2 meV) and costs 80 pairs.
  On Fe the 3s read at η_semi 1 eV does not converge (0.32 eV between
  maps); protected at η it converges in 22 maps (1685 pairs).
- **Rotating and edge**: Si with no rotating state (ω_max +20) matches the
  reference to 0.03 / 0.10 meV. The Si cut sits in a 0.41 eV gap
  ([10.24, 10.65] eV); the protected states just below it carry the error
  (bands 17–20, 7.9 meV at map 0). With the cut in the 1.8 eV gap at
  +14.3…+16.1 eV (ω_max +13, 8 rotating (k,state)): 0.23 / 0.88 meV at
  520 pairs. Fe with no rotating state: map 0 1.3 / 8.3 against 26.7 / 158.
- **Valence**: protected in every arm; no separate error.
