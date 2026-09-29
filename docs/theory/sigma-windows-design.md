# Sigma windows: one plan, fine and coarse windows

This page owns the Σ window policy of self-consistent QSGW: which states are
read finely, coarsely or by scissor, where Σ is sampled, and what happens
when a state leaves the plan. The scheme is the owner's of 2026-09-28
(rulings Q2–Q5 and the round-5 rule, TRACKER). The rule construction is in
[minimax quadrature](minimax-quadrature.md) and the SC equations are in
[self consistency](../self_consistency.md).

## Input

1. `nval` and `ncond` request states below and above E_F at each k (from
   `nelec` on an insulator); a request larger than a k holds takes all.
   `number_bands` sets the whole rotating and sum-band carrier.
2. `sigma_omega_min_ev` and `sigma_omega_max_ev` (eV from the Σ Fermi
   reference, either may be omitted) only enlarge the fine window.
3. `sigma_omega_step_ev` sets the near sampling step. `sigma_window_ev`,
   `sigma_out_of_grid`, `sigma_omega_patches_ev`, `sigma_window_edge_factor`
   and `sc_frozen_core_bands` refuse by name.

## Classes (`band_partition.sc_band_classes`, from the DFT ladder at map 0)

The owner's rule, literally: "evaluate Σ(E_nk) the valence way with small
broadening down to an energy just below min of the user requested lowest
protected Σ band, and start large broadening right above max of the band
below that one". No gap threshold decides a class.

- **Fine (protected) window, deck η, ε 1e-4.** From the minimum energy of the
  lowest requested band (a band counts if it reaches μ ± 10 eV,
  `WINDOW_CLIP_EV`; ω_min only lowers the floor) up to the cut. Every state
  in it is protected at every k.
- **The cut is a spectral gap.** `top` is the highest requested energy,
  clipped to μ + 10 eV and raised to ω_max; the cut is the midpoint of the
  first all-k gap at least 4η wide (`CUT_GAP_ETAS`) that opens in
  [top, top + 20η) (`CUT_SEARCH_ETAS`), else of the widest gap there.
- **Coarse.** Every occupied state below the fine window: active (full
  mixing, rows kept, never scissored), read at its own energy on one coarse
  window at `SEMICORE_ETA_EV` (0.5 eV) and ε 1e-2 (`SEMICORE_READ =
  "patch"`; `"sigma0"` reads Σ(0) instead). The SC log lists every coarse band
  with its DFT range at startup. A user who requests too few valence bands
  pays for it here: those bands are read coarsely. A coarse state inside the
  fine window (its lower pad, the owner's "overlap") reads the fine grid;
  below it, the coarse window; no coarse patch enters the fine grid.
- **Above the cut.** `ROTATING_READ = "scissor"` (default): every state takes
  E_DFT + β_above (scissor law A), with no far patch and no own-energy read;
  Q3's own-energy read is not used in this arm. `"own"` keeps the far
  patches and own-energy reads.
- Classes are fixed by DFT identity at map 0 and follow the eigenvectors.
- **Remaining eV constants:** the 10 eV clip, the 2 eV outer pad
  (`SUPPORT_PAD_EV`, derived pads = max |E_map0 − E_DFT| + 2 eV), the
  ±0.5 eV Z stencil, η_semi 0.5 eV and η_far 1 eV (ruling Q4: 1–2 eV).

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
- **Patches.** Rotating (above E_F, η_far = 1 eV, ε 1e-2, only with
  `ROTATING_READ = "own"`) and coarse (below, one window from the deepest
  coarse energy minus its pad to just above the highest coarse band,
  η_semi, ε 1e-2) energies outside the near support, each padded by a
  derived pad, max over the class of |E_map0 − E_DFT| + 2 eV
  (`qp_support.derived_pad_ev`), merged across holes shorter than twice the
  pad (`far_patches_ev`); a patch that reaches the near support starts
  1e-3 eV past its edge. Sampled at η/2. Patches join the near plan: each
  crossing window splits by the η of its frequencies and certifies at
  ε_far = 1e-2 (`FAR_PATCH_EPS`); sign-definite windows serve patch
  frequencies at the near η. The split is decided at map 0 and held, so
  moving poles never add a window. Ruling Q4 (INVARIANTS 12) admits η 1–2 eV
  here.
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
| P–P, P–S, S–S | the half-sum ½[Σ_ij(E_i) + Σ_ij(E_j)], Hermitian part; a P endpoint reads the fine window, an S (coarse) endpoint the coarse window |
| P–R, S–R | scissor arm: read at the active energy only (R contributes no endpoint); own arm: the half-sum with the rotating endpoint on its near grid or far patch |
| R diagonal | scissor arm: E_DFT,o + β_above (law A); own arm: own-energy read |
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
