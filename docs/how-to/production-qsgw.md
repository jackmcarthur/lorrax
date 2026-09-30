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

## What to converge, and what each costs {#error-budget}

ε, the centroid count and the W model each reach about 1 meV. The band
counts (N, b3), the k grid and the semicore read are systematic and are
reported apart. "std / max / gap" is the standard deviation and the centred
maximum of the eqp0 error over the multiplet-closed states within ±10 eV of
midgap, and the error of the smallest gap on the k grid.

**1. Bands for W (`number_bands` = N).** χ₀ is summed to N and is not
extrapolated, by design: the ISDF basis is fitted on pairs of bands inside
the band set ([scope](../theory/band-extrapolation.md#scope)). W's band
convergence is the largest remaining dependence on N. Si 4³ scalar, 25 Ry,
one-shot, headless, shared-pole Σ_c read at E_DFT, 71 states, against the
complete basis of 536 bands (claim 3027):

| N | W at N, G complete | W at N, G at N and extrapolated (production) |
|---|---|---|
| 64 | 28.0 / 82 / −89 meV | 19.7 / 58 / −67 meV |
| 72 | 24.8 / 73 / −79 meV | 18.5 / 54 / −58 meV |
| 78 | 22.8 / 67 / −73 meV | 16.7 / 49 / −53 meV |
| 142 | 10.8 / 31 / −34 meV | 7.0 / 20 / −21 meV |

The W error falls roughly as N^−1.2 over these four counts and does not
reach 5 meV in the gap at any measured N. Most of it is a rigid shift (+196
meV at 64, +70 meV at 142), which cancels in a band structure. Si 4³ SOC, one-shot, 1100
centroids: from N = 100 to 116 the eqp0 gap rises 25 meV with extrapolation
and 21 meV without (claim 2984). Cost: the G and χ₀ builds are linear in N,
and the centroid rule (≥ 10 N) grows W and the k-convolution as N_μ²
([Σ quadrature §3](../theory/sigma-quadrature-problem.md#3-cost)).

**2. Σ's G tail (`use_band_extrapolation`).** Σ's band sum is extrapolated
past N by the pooled `spectral_shell` fit: β = 3, one fitted Ω, samples at
0.70 N, 0.85 N and N
([band extrapolation](../theory/band-extrapolation.md#the-model)). With W
at 536 bands the fit leaves |gap error| ≤ 10.7 meV and std ≤ 12.6 meV for
N from 64 to 142; at N = 78, 9.0 / 27 / −2.6 meV against 29.2 / 68 / −90 meV
without extrapolation (Si 4³ scalar, as in 1). Five samples in place of
three give the same numbers at N = 78. Under W at N the same fit opens the
gap by 13 to 23 meV, so the production error is smaller than W's alone.
Cost: Σ τ time +5 to +11 % per SC map (Na 8³).

**3. Protected bands (`number_bands_protected` = b3).** Only the QP matrix
[b0, b3) rotates; the bands [b3, N) keep their DFT orbitals and take one
rigid shift per map
([self-consistency §2](../self_consistency.md#2-band-treatment)). Si 4³ SOC
QSGW at its fixed point, P4, N = 100, 1100 centroids, ε 1e-4, extrapolation
with a free exponent (code a17143cf9), against b3 = N (indirect / direct
gap 1.4749 / 3.4925 eV):

| b3 | 16 | 24 | 40 | 64 |
|---|---|---|---|---|
| direct-gap error (meV) | −107 | −89 | −59 | −39 |
| ±10 eV std / max (meV) | 46 / 103 | 44 / 88 | 36 / 64 | 19 / 37 |

Cost per map: 12.4 s and 390 τ pairs at b3 = 16, 32.5 s and 914–989 at
b3 = N. The error is the restricted rotation: the occupied orbitals miss
their admixture of tail bands. No tail-energy law removes it (linear in
energy, exchange-anchored, or the reference's own tail energies), and a
second-order fold of the matrix–tail coupling makes it worse (−118 meV at
b3 = 16). Raise b3 toward N. Legs at b3 ≥ 32 stall at 5 meV to 0.12 eV on
bands just below b3 while the ±10 eV states are stable; b3 = 48 refuses
because it cuts a Γ multiplet. Source: sandbox TAILLIN report
(runs/Si/120_taillin_20260930); TAILX and LOWDIN, same date.

**4. k grid and the parallel-transport head.** The default SC head,
`parallel_transport`, takes the Σ part of the velocity, D_kΔH, from finite
differences between neighbouring k points (links). The link error is the
relative error of v_DFT rebuilt through the same links, on the elements the
head reads. It is k-convergence error: every map prints it in its head
block, and it gates nothing ([the route](#the-route), Heads). On the
point-group-closed link shell it is 9 % on MoS2 3×3, 4.6 % on Fe 4³ and
3.4 % on Si 4³. Refine the k grid to reduce it.

**5. Centroids.** The Σ_x error follows N_μ / r, r the rank of the Σ pair
set: RMS ≤ 1 meV near 0.5 r, 0.1 meV near 1.3 r (Si 4³ SOC; claim 2860;
[ISDF exchange accuracy](../theory/isdf-exchange-accuracy.md#practical-rule)).

**6. Σ quadrature ε and η.** At the default `sigma_quadrature_eps` = 1e-4,
eqp0 is within 0.89 meV of ε = 1e-6 (Fe 4³ charge SC, E_F ± 15 eV; claims
2881, 2887; [Σ quadrature §6](../theory/sigma-quadrature-problem.md#6-error-currencies-and-the-delivered-bound)).
η (`sigma_regularization_ev`, 0.25 eV) broadens Σ(ω); the node count of a
crossing window grows with its bandwidth over η
([§7](../theory/sigma-quadrature-problem.md#7-the-rule-and-its-node-laws)).

The recipe fixes two more sources:

- **W model.** QP RMS 0.80 meV against contour deformation at the pole cap
  (Si, near-gap window); 8.2 meV non-rigid RMS in a wider window at the same
  supports (claim 2431;
  [pole count](../theory/shared-pole-w-model.md#shared-pole-pole-count)).
- **Semicore read at η_semi = 5 eV.** E_F ± 1 eV std / max 3.6 / 15.9 meV
  (Fe 4³, against η_semi = 1 eV) and 2.9 / 20.0 meV (MoS2 3×3, against the
  deck η), both converged SC (claim 2960). `sigma_omega_patches_ev` sets
  user windows at their own η.

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

- b3 convergence ([3 above](#error-budget)): only raising b3 reduces the
  error.
- Slab decks (`sys_dim = 2`) should keep `number_bands_protected` below the
  vacuum states: no vacuum-level check exists.
