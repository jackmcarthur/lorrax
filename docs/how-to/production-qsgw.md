# Production QSGW: the recipe and its error budget

**The production GW calculation is full-frequency QSGW with the shared-pole
W, built from residues.** GN-PPM is not the production route
([decisions, 2026-09-29](../architecture/decisions.md#production-gw-route)).

This page owns the production recipe: the route, the options it sets, the
owner's requirements, the error budget and which parts are on main. Key
meanings are in the [input reference](../input_reference.md); the QSGW map and
its stop rules in [self-consistency](../self_consistency.md); the W model in
[shared-pole W](../theory/shared-pole-w-model.md); metal-only rules in
[metals](metals.md). Numbers cite sandbox claims.

## The route

- **W.** `compute_mode = mpa` with `sigma_w_model = shared_pole`. Each map
  rebuilds W_c as one set of real poles per q, shared by every matrix element,
  with positive residues, from samples of W_c and ∂W_c and the exact moments
  ([theory](../theory/shared-pole-w-model.md)). `compute_mode = auto` never
  selects `mpa`, so the deck names it. The tier `sigma_w_accuracy =
  production` is the default.
- **Σ.** Σ_c(ω) on the real axis from the pole sum, by the certified
  denominator-box quadrature ([Σ quadrature](../theory/sigma-quadrature-problem.md)).
  `sigma_quadrature_eps = 1e-4` and `sigma_regularization_ev = 0.25` are the
  defaults. Rules are built in each run and held across SC maps; none is stored.
- **QSGW.** `qp_solver = self_consistent`. Stop at 1 meV
  (`sc_tol_ev = 1e-3`), the reproducibility the default ε is chosen for.
- **One band count.** `number_bands` = N serves both the χ₀/W sum and the Σ sum.
  Do not set `number_bands_chi` or `number_bands_sigma` apart from it. Choose
  N for χ₀, which is not extrapolated.
- **Band extrapolation.** On by default (`use_band_extrapolation`). Σ's G sum
  is extrapolated past N by the pooled `spectral_shell` fit, on the shared-pole
  Σ as on GN/HL-PPM ([band extrapolation](../theory/band-extrapolation.md)).
  With extrapolation on, N ≥ 2·n_occ, or the run refuses at startup.
- **Heads.** The default `head_correction = full`; the metal head is in
  [metals](metals.md).

The keys that differ from the defaults:

```ini
[cohsex]
sys_dim = 3             ; 2 for a slab
compute_mode = mpa
sigma_w_model = shared_pole
qp_solver = self_consistent
sc_tol_ev = 1e-3
number_bands = N        ; >= 2 n_occ, the one count for chi0 and Sigma
```

## Owner requirements (2026-09-03)

- At least **20 conduction bands** in the Σ window (`ncond >= 20`).
- **Centroids ≥ 10 × `number_bands`**, taking the nearest orbit-closed count.
  About 6 per band is diagnostic only. Select them on the Σ pair set; the
  exchange error follows N_μ/r of that pair set's rank r
  ([ISDF exchange accuracy](../theory/isdf-exchange-accuracy.md)).
- The ζ fit is built on the Gram of **all bands that enter Σ**
  (`zeta_nband = number_bands`). If the strict rank ceiling refuses, the owner
  decides `zeta_rcond`; the run does not drop to a smaller basis.
- Band extrapolation stays on. An explicit `use_band_extrapolation` on a
  stage that does not consume it refuses.
- **Band structures** come from htransform fitted on the whole WFN band set,
  returning at least 16 corrected conduction bands with at least 8 guard
  bands. The htransform coarse k-grid is its own convergence parameter,
  independent of the GW screening grid. Take it from a dedicated uniform
  NSCF WFN, and use a separate QE `calculation='bands'` run along the same
  path as the reference. Densify it until the energy-ordered,
  per-path-VBM-aligned QE certificate is at most **20 meV** for every plotted
  cell whose QE energy lies in [−8, +8] eV. Report the all-state maximum too;
  cells outside the window do not gate. Start Si-class cells at 8×8×8.

## Error budget

Two classes, reported apart. The controllable sources each have a knob that
reaches about 1 meV. The systematic sources are reported separately with their
measured size; the owner's estimate for semicore plus band extrapolation is
about 10 meV together.

| source | class | measured (scope) | knob |
|---|---|---|---|
| Σ quadrature ε | controllable | eqp0 within 0.89 meV of ε 1e-6 at the default 1e-4 (Fe 4³ charge SC, E_F ± 15 eV; claims 2881, 2887) | `sigma_quadrature_eps` |
| ISDF basis | controllable | Σ_x RMS ≤ 1 meV near N_μ = 0.5 r, exact to 0.1 meV near 1.3 r, r the rank of the Σ pair set (Si 4³ SOC; claim 2860) | centroid count and pair set |
| W model | controllable | QP RMS 0.80 meV against contour deformation at the pole cap (Si, near-gap window); 8.2 meV non-rigid RMS in a wider window at the same supports (claim 2431) | poles and supports ([pole count](../theory/shared-pole-w-model.md#shared-pole-pole-count)) |
| semicore read at η_semi = 5 eV | systematic | converged E_F ± 1 eV 3.6/15.9 meV std/max (Fe 4³, against η_semi 1 eV), 2.9/20.0 (MoS2 3×3, against the deck η) (claim 2960) | pending, below |
| Σ band extrapolation (G tail) | systematic | 9.2 meV std, 32.9 meV max 4v4c at 78 against 536 bands with χ complete (Si 4³, shared-pole samples; claims 2898, 2900, reproduced in 2942) | N |
| W's band truncation (χ at N) | systematic | with χ also at 78 the extrapolated score is 20.5 meV std (same study); end to end on the shared-pole route 21.2 meV std, gap −72.8 meV, against 35.6 meV unextrapolated (Si 4³ one-shot; claim 2948). Before extrapolation the 78-band error splits into χ tail 8.0, G tail 25.8 and their cross term 15.1 meV std (CHIEXT report, 2026-09-29, Si only) | N |
| scissored tail | systematic, hard to control, possibly large | moving the QP-matrix top b3 from 24 to 28 moves the ± 1 eV shell 9.7 meV std at map 0 (Si 4³, 968 centroids; claim 2945); from 16 to 24, the ± 10 eV states +48 meV mean (Si 4³, 192 centroids, ISDF-limited; claim 2952) | the QP window top |

The scissored tail is the sum-band range above the QP window, `[b3, N)`,
which moves by one rigid Z-weighted shift per map
([self-consistency §2](../self_consistency.md#2-band-treatment)). χ₀
extrapolation is not implemented: on the stored Si samples a state-independent
χ shape cuts the χ-truncation error about 4× at 78 bands, for an estimated
+20–40 % per map (CHIEXT report).

## What is on main

- The shared-pole W rebuilt every map, the full-frequency Σ and the QSGW map
  above.
- ε 1e-4 on every Σ route, with rules built cold in each run
  (claim 2941) and planned once at map 0, then held
  ([self-consistency §4](../self_consistency.md#sigma-grid-and-quadrature)).
- The pooled `spectral_shell` extrapolation on the shared-pole Σ (claim 2948)
  and on GN/HL-PPM (claim 2942).
- The QP window `[0, nelec + ncond)` with the rigid tail above it.
- Semicore under the default `sigma_out_of_grid = cover`: read at the deck η
  on the grown grid, or at Σ(ω = 0) below the active depth
  ([self-consistency §4](../self_consistency.md#sigma-grid-and-quadrature)).
  Do not freeze it with `sc_frozen_core_bands`: the frozen law fails on Fe
  3s/3p and CrI3 I 5s (claim 2859).

## Pending the owner

Branch `feat/qsgw-production-partition-2026-09-29-r3` (lane TWOPORT,
claim 2952) holds the rest of the recipe. It is not on main.

- **b3 counts bands, as on main** (owner 2026-09-29): the QP matrix
  [b0, nelec + `ncond`) rotates among itself and must lie inside the ζ left
  range (`GATE qp_matrix_zeta_left`); the tail above is scissored.
- **The request.** `number_bands_protected = N` (the documented form): every
  occupied band plus conduction bands up to N; its semicore (coarse) class is
  every occupied band below a ≥ 4 eV band gap. `nval` / `ncond` remain; there
  the coarse class is every occupied state below the lowest requested valence
  band. Both forms refuse by name. On MoS2 3×3 and Fe 4³ charge the two forms
  give identical results (claim 2952).
- **Coarse windows.** Coarse states are read on held windows at η_semi = 5 eV,
  one per coarse manifold, certified at max(`sigma_quadrature_eps`, 3e-3); the
  Σ plan groups them to the least closed-form node count of the boxes it
  builds, which is one window on both decks (MoS2 125 against 248 nodes
  split, Fe 188 against 308; the law equals the certified count). A split
  at a gap pays only when the deeper window's η can rise faster than its
  depth: at equal feedback on E_F it needs q < (1 − r)²/(4r), r the depth
  ratio shallow/deep and q the feedback slope ratio deep/shallow; measured
  q is 0.69 (MoS2, needs < 0.58) and 0.54 (Fe, needs < 0.05) (claim 2960).
  `sigma_omega_patches_ev` takes `lo:hi:eta` user windows. Against the
  semicore-at-deck-η read at the converged fixed point: E_F ± 1 eV 2.9 / 20.0
  meV std / max (MoS2). From map 1, protected conduction states broader
  than the deck η at map 0 read on far windows at η_n = max(η, |Im Σ_nn(E_n)|);
  the protected end of their off-diagonals stays at the deck η (mixing kept).
  Semicore sits at its DFT block by default (`sc_semicore = dft`). Slab decks
  (`sys_dim = 2`) should keep `number_bands_protected` below the vacuum states:
  no vacuum-level check exists yet, so the far rule would read them.
- **Owner calls.** (a) Cost of the one ε: at 1e-4 the coarse windows take
  MoS2 530 against 448 τ pairs per map and Fe 1087 against 959 (over the
  1000-pair metal budget); splitting does not win it back. (b) Fe 4³ map 2 misses 2 meV: at matched
  ε 1e-4 it is 7.57 meV, with 3 of 172 states over 2 meV and the cause not
  isolated; the fixed points agree within 0.78 meV (claim 2952).
