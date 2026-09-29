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

## Measured (TWOCLASS, 2026-09-28; branch, not main)

One A100 node, P4, SC to 1e-4 eV (Anderson, depth 20). η_semi = 1 eV against
the reference (the same tree, the semicore patch at the deck η 0.25 eV and
ε 1e-4). Protected states within ±10 eV of μ; std / maxdev in meV. Every leg's
QP matrix equals its ζ left range. Run dir `runs/DEV/598_twoclass_20260928`.

| deck | QP matrix (main) | semicore | map 0 | fixed point | ±1 eV shell (fixed point) | semicore QP, mean / max abs(Δ), map 0 → fixed point |
|---|---|---|---|---|---|---|
| Si 4³ | 1–24 (1–16) | none | arms identical | arms identical | – | – |
| MoS2 3×3 | 1–44 (1–38) | 1–12 | 0.97 / 4.40 | 1.95 / 7.57 | 1.79 / 3.79 | −171 / 323 → −39 / 200 |
| Fe 4³ charge | 1–26 (1–26) | 1–8 | 1.26 / 8.10 | the reference stalls; SC map 3: 15.7 / 62.0 | map 0: 0.25 / 0.73 | −539 / 1691 at map 0 |
| Na 8³ | 1–10 (1–86) | 1–4 | 1.40 / 7.62 | 0.09 / 0.52 (16 vs 20 maps) | 0.06 / 0.21 | −379 / 1032 → −184 / 211 |

- **Semicore Z** (the SC stencil, ±0.5 eV). At η_semi = 1 eV every semicore
  state has Z in (0, 1] at every map: MoS2 0.53–0.72, Fe 0.35–0.70, Na
  0.60–0.91. At the deck η the Fe 3s (band 1) leaves (0, 1] on 59 of 64 k
  from map 1 (Z −4.0 to +7.6 at map 3, −14.5 to +8.2 at map 17) and the
  reference stalls. The Na 2s (band 1) leaves (0, 1] on every k at map 0
  (Z up to +3.0) and returns from map 3; MoS2 stays in (0, 1].
- **Cost against main** (τ pairs per held map; steady-map wall): Si 463 vs
  409, 2.9 vs 2.6 s; MoS2 490 vs 374 (semicore patch 109), 4.9 vs 4.6 s;
  Fe 919 vs 790 (patch 159), 9.2 vs 7.2 s; Na 635 vs 1469, 36.6 vs 73.3 s.
  The references cost MoS2 982, Fe 1669 and Na 1168 pairs.
- **The cut overlaps on Si.** With 968 centroids (0.65 of the pair-set rank
  1480, selected on [0, 28) × [0, 28)), b3 = 24 against b3 = 28 moves the
  ±1 eV shell by 9.7 meV std (20.8 max) at map 0 and the gap by 20 meV.
  At map 0 the tail carries no shift yet, so that move is the dropped Σ
  coupling to the overlapping bands plus the ζ-fit change (about 1 meV RMS
  at 0.65 of the rank, claim 2860). A cut needs a real band gap to be cheap;
  Si 4³, Fe 4³ and Na 8³ have none above E_F in their loaded bands.
