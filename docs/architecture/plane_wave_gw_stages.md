# Real-space (ISDF-free) GW

Space-time GW with no ISDF: ψ_nk(G) stays on its plane-wave sphere, and χ₀ and Σ are
one pair convolution through real space. This page owns what exists on main and on
branches, how to run it, what it has been checked against, and which production owner
each stage reuses. The path is opt-in; **`gw.gw_jax` does not import any of it.**

## Status

**On main.** Every piece below is unwired: no driver imports it.

| piece | module | release, claim |
|---|---|---|
| pair convolution, χ₀ product `'trace'` (K1) | `gw/mixed_basis_pair_convolution.py` | R30, claim 2787 |
| r′-column space-group wedge, `ColumnWedge` (`wedge=`) (K1b) | same | R34, claim 2800 |
| Σ product `'scalar'`, Σ = −G ⊙ W (K2) | same | R37b, claim 2809 |
| p′→r′ expand at the k-parents, per-stage chunk law (K1d); final stage with no copy of T (K1e) | same | R41b, claims 2837, 2838 |
| W_q(G, G′) on the q wedge: v_q(G), Γ cell, Dyson, W^c, MPA fit (K3) | `gw/plane_wave_screening.py` | R37b, claim 2811 |
| one-shot stages ψ(G) → χ₀ → W → Σ_x (K4) | `gw/plane_wave_pipeline.py` | R49, 6506c97b0 + 35d115027, claim 2857 |
| Σ_c(ω) through the Σ owner; χ at the MPA samples through the MPA owner (K4b) | same, plus the owner hooks in [one frequency integration](#one-frequency-integration) | R51, 3dff467d1, claim 2864 |

**Wiring.** The only entry is `python -m gw.plane_wave_pipeline`. The deck key
`screened_coulomb_cutoff` ([input reference](../input_reference.md)) is parsed, but no
driver reads it; the pipeline takes the same value as `--screened-coulomb-cutoff`. No
test drives the pipeline. The pair convolution and the screening have CPU suites
(`tests/test_mixed_basis_pair_convolution.py`, `tests/test_plane_wave_screening.py`)
and P4 gates (`tests/multi_device/mixed_basis_pair_conv_p4.py`,
`tests/multi_device/plane_wave_screening_p4.py`).

**On branches, not main.**

- K4c, shared-pole W on the plane-wave sphere with Σ_c through the shared-pole route:
  `feat/rsgw-shared-pole-pw-2026-09-26-r2` @9ec68fb0f, on 33f7f35f7, claim 2868. It
  does not land until `construct_shared_poles` takes a bank reader and a basis axis;
  the branch repeats the constructor's round loop.
- K5 (the bispinor pathway) and K6 (convergence against ISDF gwjax and BerkeleyGW)
  have no code.

## How to run

One process per GPU; the module brings the runtime up on GPU only:

    python -m gw.plane_wave_pipeline --wfn WFN.h5 --nval NV --nb NB --out pw.json \
        [--wedge] [--screened-coulomb-cutoff RY] [--omega-ry 0.0,0.5 | --deck gw.in]

| flag | meaning |
|---|---|
| `--wfn` | the WFN; ψ(G), E and the symmetry tables are read once |
| `--nval` | occupied bands; the lowest `nval` bands are occupied at every k |
| `--nb` | bands in G and in the χ₀ and Σ sums; read to a multiple of the rank count, which must not exceed the WFN's bands |
| `--wedge` | the r′-column wedge; off is the dense-column schedule |
| `--screened-coulomb-cutoff` | the χ/W sphere cutoff (Ry); unset is the ψ sphere's cutoff |
| `--omega-ry` | without `--deck`: the imaginary frequencies of χ and W (Ry) |
| `--deck` | a gw_jax deck with `compute_mode = mpa`: Σ_x and the MPA Σ_c(ω) on the deck's Σ grid, η, ε and MPA plan |
| `--out` | JSON written by rank 0 |

Without `--deck` the JSON holds Σ_x at the k-parents, ε⁻¹₀₀(q; iω) at q ≠ 0 and the
per-stage walls and peaks. With `--deck` it holds Σ_x, the Σ_c(ω) diagonal and Σ_c
interpolated at E_DFT, in eV, at the k-parents. No eqp file is written; stage 9 below is
not wired.

**Limits of the first cut.**

- Refuses antiunitary k rows: `GATE pw-pipeline-antiunitary`. The WFN's IBZ must fold
  with unitary operations (inversion present). A TR-broken deck needs the W partner
  rule of [symmetry and time](#symmetry-and-time).
- Insulators (the MPA plan and the Σ executor run with `material_class = insulator`),
  bulk (`SphereScreening` is built with `sys_dim = 3`), and the MPA W only: the deck's
  `sigma_w_model` is not read, and `compute_sigma_c_mpa_omega_grid` refuses a
  `tau_kernel_factory` on the shared-pole route.
- No Γ head: v(q + G = 0) = 0 and the Γ W is the head-removed body. A deck compared
  against it needs `head_correction = off` and `mc_average_vcoul_body = false`; the
  comparison decks also set `bare_coulomb_cutoff = 25` (claims 2864, 2868).
- No band-tail extrapolation.
- Other refusals come from the owners: `GATE screened-coulomb-cutoff` (the cutoff at or
  above the box's alias cap) and `GATE pairconv-capacity`. The W stage has no budget
  check: `GATE pw-screening-budget` is raised only by `SphereScreening.plan_q_chunks`,
  which the pipeline does not call.

## Validation

Si_scalar deck: 4³, 25 Ry, 24³ box, 34 bands, 8 IBZ k, P4 A100-40GB.

| check | against | result | claim |
|---|---|---|---|
| pair convolution: `'trace'` at n_s 1/2/4, `'scalar'` at n_s 1/2; typed parents incl. Fe 4³ TR-broken SymMaps, wedge, expand at the parents | dense supercell references, CPU P1/P4 and GPU P4 | ≤ 1.2e-15 of max\|X\| | 2787, 2809, 2800, 2837, 2838 |
| χ normalization | a BerkeleyGW band sum | χ_q = −s·X/(Ω·N_r²), s = 2/(n_spin·n_spinor) | 2811 |
| W_q, Γ cell, antiunitary conj rule | dense references, CPU/GPU, P1/P4, n_s 1/2 | round-off | 2811 |
| pipeline, wedge vs dense (Si) | the dense pipeline run | Σ_x 3.6e-11 eV, ε⁻¹ 4e-12 | 2857 |
| Σ_x (Si), head off, plain v | ISDF Σ_x, centroids and ζ fit on the Σ_x pair set (rank 1450) | max 13.7 / 1.36 / 0.54 meV at 480 / 920 / 1440 centroids | 2864 |
| eqp0 (Si), MPA 8 poles, η 0.25 eV, ε 3e-5, head off, no tail | ISDF gwjax, 1956 centroids | gap 0.9 meV; max 19.5 meV (bands 1–4), 6.2 meV (5–8), 85 meV (9–34) | 2864 |

The bands 9–34 residual barely moves between 1292 and 1956 centroids (max 82 → 85 meV),
so it is not ISDF basis error. Its cause is not proven: the plane-wave Σ_c itself moves by up to
53 meV between 8 and 12 MPA poles (claim 2864). Not checked: any comparison with a
BerkeleyGW Σ or eqp, n_s = 2 through the pipeline, slabs, metals.

**Cost** (P4 A100-40GB). One τ node at Fe 8³ n_s = 2, warm: χ₀ 31.1 s dense (claim
2787), 3.07 s with the wedge (claim 2800), 2.51 s with the expand at the parents
(claim 2837); Σ 2.95 s, peak 24.68 GB per rank (claim 2838). The Si one-shot with the
wedge, one run including compiles: Σ_x 1.4 s, χ at 16 MPA samples 35 s, Dyson and MPA
fit 9.6 s, Σ_c τ sweep (1048 τ nodes) 70 s, peak 6.58 GB per rank (claim 2864). No
N_k·N_r² object is formed. The per-stage memory laws are
`MixedBasisPairConvolution.describe()`, which the pipeline prints, and
`SphereScreening.describe()`, which it writes to the JSON as `screen_law`. The kernels (mode 6 and
`LocalFourierPlan`) are in [the FFI layer](ffi_layout.md#k-convolution-router-and-the-mathdx-family).

## The one idea

ψ_nk(G) sits in the arrays the ISDF path uses for ψ_nk(r_μ). The μ axis is read as the
sphere slot of k, zero-padded to one carrier `padded_axis(ngkmax, P)`:

| array (ISDF name) | shape | spec | holds, plane-wave path |
|---|---|---|---|
| `ParentGreenCarrier.psi_mun` (`wavefunction_bundle.py:229`) | (n_par, s, μ, n) | P(None,None,'x','y') | ψ_nk(G_slot), direct operand of the G builder |
| `ParentGreenCarrier.psi_nmu` | (n_par, n, s, μ) | P(None,'x',None,'y') | the same values; conjugated operand and projection face |

A pad slot holds a zero coefficient, so it is inert in every contraction; the loader's
pad G-vector is the FFT-box sentinel (`WfnLoader.load`, `services/wfn_loader/src/wfn_loader/loader.py:1711`).
Every k shares one carrier, so one static shape serves every program.

With ψ(G) in those arrays, `build_G_parents` (`greens_function_kernel.py:147`) returns
G_k(G, G'; τ) at the k-parents exactly as it returns G_k(μ, ν; τ). Symmetry acts on
the compact G, G' tiles through `SphereTransport.typed`
(`mixed_basis_pair_convolution.py:277`), never on ψ.

## Stages and their owners

| # | stage | output | owner (file:line) |
|---|---|---|---|
| 0 | inputs | WFN tables, SymMaps, v geometry | `WfnLoader.symmetry` (`services/wfn_loader/src/wfn_loader/loader.py:720`), `vcoul.CoulombGeometry.from_wfn` |
| 1 | ψ store | ψ(G) in the r_μ faces, E_nk | `WfnLoader.load(k='ibz')`; `ParentGreenCarrier` |
| 2 | v_q(G) | v on the χ sphere at the q-IBZ; exact zero at q + G = 0 | `plane_wave_screening.sphere_coulomb` (:155) → `compute_vcoul.compute_v_q_per_G` (:55) |
| 3 | G(τ) | G_k(p, p'; τ) at the parents | `build_G_parents` (:147) |
| 4 | χ₀(τ) | raw pair sum X_q(G, G') | `MixedBasisPairConvolution(product='trace')` (:765) |
| 5 | χ(z) | χ at each frequency sample | `plane_wave_screening.accumulate_chi` (:144) with the rules of `minimax_screening.solve_laplace_minimax_interval` (:1065) and `solve_laplace_minimax_imag_interval` (:1121) |
| 6 | W(z) | Dyson, W^c, poles; the Γ head and wings not wired (`wcoul0 = vc0 = 0`) | `plane_wave_screening.SphereScreening` (:215) → `w_isdf.solve_w`, `mpa/pade_fit.fit_mpa_poles_batched` (:1103); the unwired head fold would be `head_correction.fold_small_head_wings_sharded` |
| 7 | Σ_x | Σ_x,k(p, p') | `MixedBasisPairConvolution(product='scalar')` with B = v |
| 8 | Σ_c(τ) → Σ(ω) | Σ_nm(k, ω) | the Σ owner's τ executor, `mpa/sigma.py:1088` `_integrate_sigma_batches`, with a plane-wave `sigma_kij` (below) |
| 9 | QP | eqp0, eqp1 | not wired; the owners would be `qsgw_utils.solve_qp` (:877), `sigma_dispatch.sigma_result_on_kset` (:326), `eqp_bgw` |

Front stages (kmeans, dipole, kin_ion) are not needed by the first cut. A Γ head would
read the dipoles (stage 6), and a QP stage kin_ion.

## One frequency integration

The Σ(ω) quadrature is reused whole ([Σ quadrature](../theory/sigma-quadrature-problem.md)).
`compute_sigma_c_mpa_omega_grid` passes a `tau_kernel_factory` through
`integrate_sigma_store` to `_integrate_sigma_batches`, and
`ppm_tau_kernel.get_shared_sigma_tau_kernel` takes a caller's `_sigma_kij`. The windows,
the box rules (`sigma_box_plan.plan_sigma_windows`, :1539), `ppm_tau_kernel.build_shared_w_tau` (:333)
and `DeviceOmegaAccumulator` (`ppm_accumulators.py:76`) are unchanged. Only the τ body's
middle differs from the ISDF one (`get_sigma_spatial_kernel`):

    sigma_kij(...):
        G   = build_G_tau(ψ_mun, ψ_nmu, E_A, i·t, mask=mask_A, e_ref=E_ref_A)   # the G builder
        Σ_k = −X/(Ω·N_r²),  X = pair_conv['scalar'](G, W_t)
        return project(ψ_nmu, ψ_mun, Σ_k)     # contract_bands_block_reshard, Σ_k spin-major

χ at the MPA samples goes through `mpa.model._evaluate_samples` with this basis as its
χ producer (`chi=`), on `make_mpa_plan` over `build_static_quadrature`: the z grid and
every rule are the ISDF run's. Each τ node is one `'trace'` pair convolution per
particle–hole orientation it takes; the conduction Green carries e^{-(E−μ)τ} and the valence Green e^{+(E−μ)τ̄}, so each pair
carries e^{-Δτ}. Static and imaginary points take the minimax rule on both
particle–hole orientations; line points the damped-line rule on one orientation
(`w_isdf._chi0_contour_alpha_rows` about E_gap = 0). W then goes
`SphereScreening.solve_samples` → `correlation` → `fit_poles`. The poles enter
`MemoryPoleSource(axis=SphereScreening.axis)` on the q-IBZ rows, with the pair
convolution's tables in the residue carrier's `load` slot (`SphereResidues`). The k
tables are a `SphereKPlan`, whose `parent_rows` is `centroid_k_unfold.parent_rows`.

## Symmetry and time

G is built at the k-parents only and W at the q-IBZ, which equals the k-parents on a
Γ-centred grid; both unfold on load. At real τ an antiunitary W row reads conj W. The Σ
executor's nodes and MPA poles are complex, and there the rule is the ISDF path's pair
transpose (the partner tile from conj residues at the same time factor). The `'scalar'`
product refuses a W partner (`GATE pairconv-scalar-partner`), so a TR-broken deck needs
that partner path first.

## Not built

- The Γ head and wings: the wing producer and the head owner's S(z)
  (`SphereScreening.gamma_head` exists; the pipeline does not call it).
- Stage 9 as an owner call, eqp files, a deck key and `gw_jax` wiring.
- Resume positions at committed SlabIO writes (ψ store, v_q, χ(z), poles, Σ(ω)).
- The SC loop over stages 3–9 with a rotated ψ store (`rotate_wavefunctions`).
- The wedge needs operands covariant under the group (complete multiplets at the band
  cut). It equals dense for both products at real τ (claim 2857); the Σ_c arm at
  complex τ was not re-checked (claim 2864).
