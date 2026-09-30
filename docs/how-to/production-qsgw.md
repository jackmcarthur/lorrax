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
- **Band classes.** `number_bands_protected = M` requests the QP matrix
  [b0, b3): every occupied band plus conduction bands up to M in total. Only
  the QP matrix rotates. The bands [b3, N) are a scissored tail: DFT orbitals
  with one rigid Z-weighted shift per map. Occupied bands below a band gap of
  at least 4 eV are the semicore class: they stay in the matrix, their Σ is
  read on held windows at η_semi = 5 eV, and they are pinned at their DFT
  block by default (`sc_semicore = dft`). b3 is a convergence parameter
  ([error budget](#error-budget)). The rules are owned by
  [self-consistency §2](../self_consistency.md#2-band-treatment).
- **One band count.** `number_bands` = N serves both the χ₀/W sum and the Σ sum.
  Do not set `number_bands_chi` or `number_bands_sigma` apart from it. Choose
  N for χ₀, which is not extrapolated.
- **Band extrapolation.** On by default (`use_band_extrapolation`). Σ's G sum
  is extrapolated past N by the pooled `spectral_shell` fit, on the shared-pole
  Σ as on GN/HL-PPM. The exponent is fixed at β = 3 and only the shell offset
  Ω is fitted. Ω is bounded below on every map so that every requested state
  keeps a tail ([band extrapolation](../theory/band-extrapolation.md#the-model)).
  With extrapolation on, N ≥ 2·n_occ, or the run refuses at startup.
  One-shot runs also write the raw Σ (no tail) as the `sigC_raw`, `eqp0_raw`,
  `eqp1_raw` columns of `sigma_diag.dat`; compare those to BerkeleyGW at the
  same N. SC runs write no raw columns.
- **Heads.** The default `head_correction = full`; the metal head is in
  [metals](metals.md). An SC deck that does not name `sc_head_update` takes
  `parallel_transport` when the link artifact exists (the dipole step writes
  it by default), else `dft_velocity` from `dipole.h5`, else `off`. This holds
  on scalar decks and on `bispinor_gw = full_shared_pole`, whose direct Γ head
  reads the same velocity. On a metal, `bare_transverse` refuses
  `parallel_transport`. The run keeps that head on every map, and
  complete links always serve the Σ term D_kΔH. The link error is a
  k-convergence measure: each map logs it and its bound on the Σ term, and
  it gates nothing (a large value means the k grid is underconverged). Only
  links that are not usable (incomplete, or a stencil or window-hybridization
  gate fails) set D_kΔH to zero, on every map. There is no switch to another
  head and no refusal ([self-consistency §7](../self_consistency.md#metals-direct-drude-head)).
  The one-shot head and every SC velocity head take their velocity from one
  owner, `qsgw_head.qp_velocity`. The links sit on a point-group-closed
  shell ([input reference](../input_reference.md), `parallel_transport_file`).
  A link artifact written before that shell (schema 3) refuses under
  `parallel_transport`: rerun the dipole step.
- **Outputs.** An SC run's `eqp0.dat` and `eqp1.dat` hold the SC eigenvalues
  of the accepted map, tail scissor and semicore pin included; both equal that
  map's `eqp0_iterNNNN.dat`. The fixed-DFT-state diagonal of the final H is
  only in `sigma_diag.dat`, and `python -m gw.eqp_bgw` refuses an SC
  `sigma_mnk.h5` ([self-consistency §1](../self_consistency.md#1-the-map)).
  Every velocity head writes `dipole_qsgw.h5`. With `WFN_qp.h5` it is bound
  to that WFN, so a GW run on `WFN_qp.h5` can use it as its `dipole.h5`
  ([QSGW dipoles](../self_consistency.md#interband-commutator-head)).
  `WFN_qp.h5` files written before 2026-09-30 from a WFN that stores both k
  and −k have broken rows; regenerate them
  ([self-consistency §8](../self_consistency.md#8-seeding-restart-and-outputs)).

The keys that differ from the defaults:

```ini
[cohsex]
sys_dim = 3             ; 2 for a slab
compute_mode = mpa
sigma_w_model = shared_pole
qp_solver = self_consistent
sc_tol_ev = 1e-3
number_bands_protected = M  ; b3, the QP-matrix top; a convergence parameter
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
| semicore read at η_semi = 5 eV | systematic | converged E_F ± 1 eV 3.6/15.9 meV std/max (Fe 4³, against η_semi 1 eV), 2.9/20.0 (MoS2 3×3, against the deck η) (claim 2960) | η_semi; `sigma_omega_patches_ev` sets user windows at their own η |
| Σ band extrapolation (G tail) | systematic | 9.2 meV std, 32.9 meV max 4v4c at 78 against 536 bands with χ complete (Si 4³, shared-pole samples, free exponent; claims 2898, 2900, reproduced in 2942) | N |
| W's band truncation (χ at N) | systematic | with χ also at 78 the extrapolated score is 16.7 meV std, gap −53 meV at β = 3 (same samples; claim 2984). Unextrapolated, with G and χ both at 78, end to end on the shared-pole route: 35.6 meV std, gap −120.4 meV (Si 4³ one-shot; claim 2948). Before extrapolation the 78-band error splits into χ tail 8.0, G tail 25.8 and their cross term 15.1 meV std (CHIEXT report, 2026-09-29, Si only) | N |
| scissored tail, b3 | systematic, converges slowly in b3 | direct-gap error against b3 = N: −107, −89, −59, −39 meV at b3 = 16, 24, 40, 64; std/max over the ± 10 eV states 46/103, 44/88, 36/64, 19/37 meV (scope below) | b3 (`number_bands_protected`) |

The scissored tail is the sum-band range above the QP matrix, `[b3, N)`,
which moves by one rigid Z-weighted shift per map
([self-consistency §2](../self_consistency.md#2-band-treatment)).

**b3 is a convergence parameter.** Scope of the b3 row: Si 4³ SOC,
shared-pole QSGW run to its fixed point at P4, N = 100 bands, 1100
centroids, ε 1e-4, band extrapolation on with the free exponent (code
a17143cf9, before β was fixed at 3); the reference is b3 = N = 100
(indirect/direct gap 1.4749/3.4925 eV). No b3 ≤ 64 came within 5 meV of the
reference. The reference's conduction corrections grow with energy above
about 25 eV, by +0.13 to +0.17 eV per eV, so a rigid tail leaves the high
bands too low. Empty bands placed too low over-screen and close the gap,
which is the sign measured. An energy-linear tail shift anchored at the
cutoff moved the gaps by at most 0.5 meV, and a free linear fit helped only
at b3 = 16. Cost per map: 32.5 s and 914–989 τ
pairs at b3 = 100, 12.4 s and 390 at b3 = 16. Legs at b3 ≥ 32 stalled at
5 meV to 0.12 eV on bands just below b3, while the ± 10 eV states were
stable. b3 = 48 refused: it cuts a Γ multiplet. Source: sandbox TAILLIN
report, 2026-09-30 (runs/Si/120_taillin_20260930).

χ₀ extrapolation is not implemented: on the stored Si samples a
state-independent χ shape cuts the χ-truncation error about 4× at 78 bands,
for an estimated +20–40 % per map (CHIEXT report).

## On main

- The shared-pole W rebuilt every map, the full-frequency Σ and the QSGW map
  above.
- ε 1e-4 on every Σ route, with rules built cold in each run
  (claim 2941) and planned once at map 0, then held
  ([self-consistency §4](../self_consistency.md#sigma-grid-and-quadrature)).
- The pooled `spectral_shell` extrapolation at β = 3 on the shared-pole Σ and
  on GN/HL-PPM.
- The band classes above: `number_bands_protected` (or `nval`/`ncond`; giving
  both refuses), the QP matrix inside the ζ left range
  (`GATE qp_matrix_zeta_left`), the semicore windows at η_semi = 5 eV
  certified at max(`sigma_quadrature_eps`, 3e-3), `sc_semicore = dft`, and the
  rigid tail with min(Z, 1/Z) weights. Do not freeze semicore with
  `sc_frozen_core_bands`: the frozen law fails on Fe 3s/3p and CrI3 I 5s
  (claim 2859).
- The velocity heads and their per-map head block, as above.
- W's line sites held across maps while they drift by at most
  max(3 meV, 0.1 × the previous map's max|dE|)
  ([self-consistency §6](../self_consistency.md#shared-pole-w-with-retained-quadrature)).

## Open

- b3 convergence (b3 row above): no tail law tried converges faster than
  the rigid one.
- Slab decks (`sys_dim = 2`) should keep `number_bands_protected` below the
  vacuum states: no vacuum-level check exists.
