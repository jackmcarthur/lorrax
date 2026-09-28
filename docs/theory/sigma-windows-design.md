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
  the midpoint of the first all-k gap at least 4η wide (`CUT_GAP_ETAS`)
  that opens in [top, top + 5 eV) (`CUT_SEARCH_EV`), else of the widest gap
  there. Every state below it is protected at every k, so no rotating state
  sits among protected ones, and none within ~2η of the cut.
- **Semicore S.** Occupied states below a gap wider than 5 eV: active (full
  mixing, their rows kept, never scissored). `SEMICORE_READ = "sigma0"`
  (default, main's rule for states below E_F − 15 eV) reads every semicore
  endpoint at Σ(ω = 0) on the near grid: no semicore window. `"patch"` reads
  them at their own energy on held patches at `SEMICORE_ETA_EV` (None = the
  deck η, step and tolerance). States inside [ω_min, ω_max] stay protected.
  On a sector (bispinor) route there are no patches and S is protected at η.
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
- **Patches.** Rotating (above E_F, η_far = 1 eV, ε 1e-2) and semicore
  (below, deck η, ε 1e-4) energies outside the near support, each padded by a
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

## Measured (PARTITION rounds 2–4, 2026-09-28; not main)

Same node, P4. Reference: the same tree with every band protected at the deck
η and ε 1e-4 (semicore at its own energy); for Σ(0) (D) also D's own reference
(every non-semicore band protected, semicore at Σ(0)). Protected states within
±10 eV of E_F; fixed-point std / maxdev in meV; τ pairs per map.

| deck | semicore at η, patches (A) | η_semi 1 eV (B) | η_semi 0.5 eV (C) | Σ(0) (D), own ref / own-energy ref |
|---|---|---|---|---|
| MoS2 3×3 | 982; 0.05 / 0.14 | 490; 2.3 / 8.5 | 592; 1.5 / 4.8 | 391; 0.03 / 0.08 / 23 / 46 |
| Fe 4³ charge | 1659; converges | 909; 3s stalls | 1064; converges | 753; converges; D ref stalls |
| Na 8³ (map 0) | 1284; 1.4 / 6.8 | 755; 1.8 / 7.2 | – | 654; – / 3.4 / 15 |

- **Si 4³** has no semicore: 520 pairs, 0.23 / 0.88 meV with the cut rule;
  482 pairs, 0.03 / 0.10 with every band protected (ω_max +20, a deck choice).
- **Semicore at η costs the budget.** A crossing window's node count is
  N ≈ 2.7 s/η + 20, s its short side, here the depth of the deepest semicore
  frequency: MoS2 s = 66.5 eV → 738 predicted, 725 measured; Fe s = 100.6 eV
  → 1106, 1090. Narrow patches do not shorten s; the derived pad merges them.
- **Σ(0) is cheap but far from the own-energy semicore.** Semicore QP
  energies sit 4–9 eV (MoS2) and 12–18 eV (Fe) below their own-energy values,
  and protected states move 23 ± 23 meV (MoS2) and 36 ± 32 meV (Fe) against
  the own-energy reference.
- **Cached rules.** A crossing request is served only by a cached rule
  inside its twice-widened build box (`_rule_cache_lookup`); before, the
  map-0 probe pass's semicore rule served MoS2's ω ≥ E_F conduction window
  (1472 instead of 982 pairs).
- **The cut gap.** Si's first gap after the request is 0.41 eV wide at
  +10.4 eV (7.9 meV at map 0); the ≥ 4η rule takes the 1.8 eV gap at
  +15.2 eV.
