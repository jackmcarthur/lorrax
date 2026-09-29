# Sigma windows: the band cut, one plan, one semicore patch

This page owns the Σ window policy of self-consistent QSGW: which states are
in the QP matrix, how each is read, where Σ is sampled, and what happens
when a state leaves the plan. The scheme is the owner's of 2026-09-28
(rulings Q2–Q5, "only fitted ISDF pairs", the absolute band cut and the
21:40 semicore rule). The rule construction is in
[minimax quadrature](minimax-quadrature.md) and the SC equations are in
[self consistency](../self_consistency.md).

## Input

1. `nval` and `ncond` request states below and above E_F at each k (from
   `nelec` on an insulator); a request larger than a k holds takes all.
   `number_bands` sets the loaded bands: the QP matrix plus the scissored tail.
2. `sigma_omega_min_ev` and `sigma_omega_max_ev` (eV from the Σ Fermi
   reference, either may be omitted) only enlarge the near window; ω_max
   also raises the band cut.
3. `sigma_omega_step_ev` sets the near sampling step. `sigma_window_ev`,
   `sigma_out_of_grid`, `sigma_omega_patches_ev`, `sigma_window_edge_factor`
   and `sc_frozen_core_bands` refuse by name.

## The band cut (`band_partition.qp_band_cut`, before the ζ fit)

- **The QP matrix is the ζ fit's left range**, [0, b3) at every k. The ζ fit
  is least squares on pairs ψ_i*ψ_j, i in the left range; a QP state above
  it would carry Σ on unfitted pairs. A run whose QP matrix reaches past the
  left range (for example `zeta_nband` < b3) refuses:
  `GATE qp_matrix_zeta_left` (`gw_init.assert_qp_matrix_fitted`).
- **b3 is one band index**, decided from the DFT ladder in
  `gw_init.qp_band_cut_for_deck`. The need is every occupied state, every
  requested state within μ ± 10 eV (`WINDOW_CLIP_EV`) and every state below
  μ + ω_max; b3 is the smallest boundary holding it, moved up to the first
  band gap ≥ 4η (`CUT_GAP_ETAS`) opening within 20η (`CUT_SEARCH_ETAS`),
  else to the least-overlapping boundary there. A boundary that cuts a
  degenerate multiplet at any k is skipped.
- **A band gap is not a level gap.** A band gap at boundary n is
  min_k E[n] − max_k E[n−1] > 0; a gap in the union of all k levels does not
  fix the band count below it, which varies with k. Above E_F a dispersive
  ladder has no band gap on Si 4³, Fe 4³ or Na 8³, so their cut overlaps:
  the matrix's top band reaches above the tail's lowest band at another k.
- **The tail [b3, nband)** is the SC sum-band tail: DFT ψ, energies shifted
  by main's conduction law (`sc_iteration._fit_sum_band_tail`: one rigid
  shift, the Z-weighted mean QP correction of the conduction states that
  read Σ; α = 1), in G and χ only. No Σ row, no mixing.
- **Requested states above the cut** (beyond the 10 eV clip) are tail states
  and the log counts them.

## Classes inside the QP matrix (`sc_iteration._sc_band_classes`)

- **Semicore:** occupied bands below a band gap of at least 4 eV
  (`SEMICORE_GAP_EV`) under the valence manifold, and below every requested
  band within the clip. They keep full mixing and are read at their own
  energy on one held patch at η_semi = 1 eV (`SEMICORE_ETA_EV`, fixed by the
  owner; its systematic error is reported apart from the 1 meV budget). A
  sector route (bispinor) has no patches: its semicore reads the near grid
  at the deck η.
- **Protected:** every other QP-matrix state, read on the near grid at the
  deck η.
- The SC log lists every semicore band with its DFT range at startup, and the
  semicore Z of the patch stencil (min / median / max, states outside
  (0, 1]) every map.

## One plan at map 0, held

- **Near support.** The protected DFT energies, the ±0.5 eV Z stencil and one
  2 eV outer pad, enlarged to the ω endpoints; sampled at the deck η and
  ε = 1e-4 (ruling Q2). Crossing and non-crossing product windows are planned
  once with their reserves (`sigma_box_plan`).
- **Map-0 probe.** Map 0 first evaluates Σ on a provisional plan. The plan is
  then made once from the DFT energies and the map-0 estimates, and map 0 is
  re-evaluated. This happens whenever a protected estimate leaves the
  provisional support or the deck has semicore states (their pad derives
  from the probe). The insulator Σ frame is the DFT midgap, so the map-0 gap
  opening is not absorbed by the frame.
- **The semicore patch.** One window from the deepest semicore energy minus
  its pad to just above the highest semicore band, pad = max over the class
  of |E_map0 − E_DFT| + 2 eV (`qp_support.derived_pad_ev`), sampled at
  η_semi/2; it joins the near plan, and its crossing windows certify at
  ε = 1e-2 (`FAR_PATCH_EPS`) unless η_semi equals the deck η. A semicore
  state inside the near window reads the near grid.
- **Held, one object.** `qp_support.plan_sigma_windows` → `SigmaPlan`
  (`protected_support_ev`, `semicore_ev`/`semicore_eta_ev`, the pad), carried
  as `SCSupport.plan` and the session's `"sigma_plan"`.
- **Escape refuses** (ruling Q5). A protected read outside the near support
  or a semicore read outside its patch refuses with
  `GATE sigma_plan_escape`, naming the state. No clamp, no rebuild.

## Hamiltonian

Every QP-matrix block is the half-sum ½[Σ_ij(E_i) + Σ_ij(E_j)], Hermitian
part; a protected endpoint reads the near grid, a semicore endpoint below it
the semicore patch (`sc_sigma_protected_kn`, `sc_sigma_far_kn`;
`qsgw_utils._qsgw_far_kernel`). There is no rotating class and no
replaced diagonal: the cut, not a class, ends the matrix.

## Why

- **Only fitted pairs** (owner): in the PARTITION rounds the QP matrix
  exceeded the ζ left range (Fe 1–35 vs 1–26, Si 1–28 vs 1–16, MoS2 1–44 vs
  1–38), so those accuracy numbers used unfitted pairs.
- From map 1 on Σ_x and W are built from every occupied orbital, so an
  occupied state is never in the tail (claim 2928).
- A crossing window's node count follows its short side over η,
  N ≈ 2.7 s/η + 20 (claim 2908): semicore reads at the deck η cost
  1000–1700 pairs per map; η_semi 1 eV brings them near the budget.

## Measured (PARTITION round 5, 2026-09-28; not main)

Same node, P4, SC to 1e-4 eV (Anderson, depth 20, at most 30 maps). The
reference is the same tree with every band protected at the deck η and
ε 1e-4. Protected states within ±10 eV of E_F; std / maxdev in meV at map 0
and at the last map; τ pairs per held map; steady-map wall in s (main
c5257230b, Σ(0) semicore, SC-3, in brackets; Na map 2). Run dir
`runs/DEV/593_partition_r5_20260928`.

| deck | (a) η_semi 1 eV, scissor | (b) η_semi 0.5 eV, scissor | (c) = (b) + own-energy reads above the cut |
|---|---|---|---|
| Si 4³ (no coarse; a = b) | 461; 4.17 / 26.7, 6.22 / 22.0; stalls (15 maps); 2.9 s [414; 2.8 s] | same as (a) | 520; 0.06 / 0.25, 0.23 / 0.88; 9 maps; 3.1 s |
| MoS2 3×3 (nothing above the cut; b = c) | 490; 0.84 / 3.94, 2.32 / 8.53; 10 maps; 4.9 s [392; 4.9 s] | 592; 0.48 / 1.68, 1.46 / 4.77; 11 maps; 5.1 s | same as (b) |
| Fe 4³ charge | 804; 77.7 / 320, 170 / 421; 20 maps, 3s converges; 9.1 s [829; 9.0 s] | 959; 77.6 / 320, 144 / 362; 3s stalls (15 maps); 9.7 s | 1062; 26.7 / 158, 20.6 / 104; 21 maps, 3s converges; 10.0 s |
| Na 8³ (map 0 + SC 2) | 580; 9.47 / 44.4, 44.3 / 140; 40 s [1621; 79 s] | 682; 9.27 / 44.0, 44.2 / 140; 44 s | 854; 1.55 / 6.91, 4.26 / 22.5; 52 s |

- **The plain scissor is the error.** Every arm whose states above the cut
  take E_DFT + β fails ≤ 1 meV where anything lies above the cut: Si
  (26.7 meV at map 0, SC stalls at 0.275 eV), Fe, Na (requested states above
  the cut about 3.3–3.5 eV off, median). Own-energy reads there (c) recover
  Si to 0.25 meV at map 0 for 59 more pairs.
- **η_semi sets MoS2.** Nothing sits above MoS2's cut, so its error is the
  coarse read: 3.94 meV at 1 eV, 1.68 meV at 0.5 eV (map 0). Semicore at the
  deck η (rounds 2–4) gave 0.14 meV at 982 pairs.
- **Semicore QP energies** (bands below the fine window, map 0 / last map,
  mean and max |Δ| in meV against the reference): MoS2 (a) −184/333,
  −72/213; (b) −71/156, −22/121. Fe (a) −563/1713, −1448/6206; (b) −185/653,
  −599/1605; (c) −160/645, −135/489. Na (a) −231/440, +23/161; (b) −67/159,
  +154/290; (c) −56/127, +5.5/53.
- **Under-request** (arm b settings, `nval` 4/4/2 instead of 10/8/6, no ω
  endpoints). The bands between the old and new floor turn coarse. Requested
  states are the top `nval` valence and `ncond` conduction at each k, map 0 /
  last map std/maxdev in meV; coarse states in ±10 eV mean/max|Δ|:

  | deck | pairs | requested states | coarse in ±10 eV | maps |
  |---|---|---|---|---|
  | MoS2 3×3 nval 4 | 548 (592) | 0.34/1.29, 0.41/1.00 (b: 0.35/0.86, 1.10/2.38) | −11.5/74, −7.8/20 | 11, converges |
  | Si 4³ nval 4 | 436 (461) | 2.88/10.2, 4.83/7.8 (b: same) | −15.0/52, −25.4/42 | 15, stalls (scissor) |
  | Fe 4³ nval 2 | 882 (959) | 68.3/288, 179/354 (b: same) | −73.7/327, −232/389 | 15, 3s stalls |

  Deep semicore at the last map, mean/max|Δ| in meV: MoS2 bands 1–12
  −25/125 (b: −22/121), Fe bands 1–8 −631/2051 (b: −599/1605). A deck
  that changes `nval` needs its dipole regenerated
  (`GATE dft_head_dipole_provenance`, full head).
- **Node law.** A crossing window's node count is N ≈ 2.7 s/η + 20, s its
  short side, here the depth of the deepest semicore frequency (claim 2908):
  semicore at the deck η costs MoS2 1580 and Fe 1685 pairs per map.
- **The cut gap.** Si's first gap after the request is 0.41 eV wide at
  +10.4 eV; the ≥ 4η rule takes the 1.8 eV gap at +15.2 eV.
- **Σ(0) semicore (main's rule, D)** is cheap (MoS2 391, Fe 753 pairs) but
  its semicore QP energies sit 4–9 eV (MoS2) and 12–18 eV (Fe) from the
  own-energy values (claim 2933).
