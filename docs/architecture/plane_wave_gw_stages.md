# Plane-wave (ISDF-free) GW: the stage graph

This page is the design of the non-ISDF, space-time GW path: which stage owns what,
where each owner already lives, and what the first cut leaves out. The path is opt-in
and unwired from `gw.gw_jax`. Its first cut is `gw/plane_wave_pipeline.py`.

## The one idea

ψ_nk(G) sits in the arrays the ISDF path uses for ψ_nk(r_μ). The μ axis is read as the
sphere slot of k, zero-padded to one carrier `padded_axis(ngkmax, P)`:

| array (ISDF name) | shape | spec | holds, plane-wave path |
|---|---|---|---|
| `ParentGreenCarrier.psi_mun` (`wavefunction_bundle.py:229`) | (n_par, s, μ, n) | P(None,None,'x','y') | ψ_nk(G_slot), direct operand of the G builder |
| `ParentGreenCarrier.psi_nmu` | (n_par, n, s, μ) | P(None,'x',None,'y') | the same values; conjugated operand and projection face |

A pad slot holds a zero coefficient, so it is inert in every contraction; the loader's
pad G-vector is the FFT-box sentinel (`WfnLoader.load`, `loader.py:1711`). Every k
shares one carrier, so one static shape serves every program (no size ladders).

With ψ(G) in those arrays, `build_G_parents` (`greens_function_kernel.py:134`) returns
G_k(G, G'; τ) at the k-parents exactly as it returns G_k(μ, ν; τ) today. Symmetry
acts on the compact G, G' tiles through `SphereTransport.typed`
(`mixed_basis_pair_convolution.py:277`), never on ψ.

## Stages and their owners

| # | stage | output | owner (file:line) |
|---|---|---|---|
| 0 | inputs | WFN tables, SymMaps, v geometry | `WfnLoader.symmetry` (`loader.py:720`), `vcoul.CoulombGeometry.from_wfn` |
| 1 | ψ store | ψ(G) in the r_μ faces, E_nk | `WfnLoader.load(k='ibz')`; `ParentGreenCarrier` |
| 2 | v_q(G) | v on the χ sphere at the q-IBZ, Γ head from `q0_average` | `plane_wave_screening.sphere_coulomb` (:175) → `compute_vcoul.compute_v_q_per_G` (:55) |
| 3 | G(τ) | G_k(p, p'; τ) at the parents | `build_G_parents` (:134) |
| 4 | χ₀(τ) | raw pair sum X_q(G, G') | `MixedBasisPairConvolution(product='trace')` (:767), K1 |
| 5 | χ(z) | χ at each frequency sample | `plane_wave_screening.accumulate_chi` (:144) with the rule of `minimax_screening` (:1061, :1117) |
| 6 | W(z) | Dyson, Γ head, W^c, poles | `plane_wave_screening.SphereScreening` (:215) → `w_isdf.solve_w`, `head_correction.fold_small_head_wings_sharded`, `pade_fit.fit_mpa_poles_batched` (:1103) |
| 7 | Σ_x | Σ_x,k(p, p') | `MixedBasisPairConvolution(product='scalar')`, K2, with B = v |
| 8 | Σ_c(τ) → Σ(ω) | Σ_nm(k, ω) | the Σ owner's τ executor, `mpa/sigma.py:988` `_integrate_sigma_batches`, with a plane-wave `sigma_kij` (below) |
| 9 | QP | eqp0, eqp1 | `qsgw_utils.solve_qp` (:877), `sigma_dispatch.sigma_result_on_kset` (:326), `eqp_bgw` |

Front stages (kmeans, dipole, kin_ion) stay as they are. The plane-wave path needs no
kmeans. It reads kin_ion for the QP Hamiltonian and dipoles for the Γ head (stage 6).

## One frequency integration

The Σ(ω) quadrature is reused whole, not copied. `compute_sigma_c_mpa_omega_grid`
passes a `tau_kernel_factory` through `integrate_sigma_store` to
`_integrate_sigma_batches`, and `ppm_tau_kernel.get_shared_sigma_tau_kernel` takes a
caller's `_sigma_kij`. The planned windows,
the box rules (`sigma_box_plan.plan_sigma_windows`, :1588), the pole windows,
`build_shared_w_tau` (:329) and `DeviceOmegaAccumulator` (`ppm_accumulators.py:76`)
are unchanged. The τ body enters the Σ owner, then leaves for the plane-wave product:

    sigma_kij(ψ_mun, ψ_nmu, ψ_nmu, ψ_mun, E_A, mask_A, E_ref_A, t, W_t, W_pt, load):
        G   = build_G_tau(ψ_mun, ψ_nmu, E_A, i·t, mask=mask_A, e_ref=E_ref_A)   # the G builder
        Σ_k = −X/(Ω·N_r²),  X = pair_conv['scalar'](G, W_t)                     # K2
        return project(ψ_nmu, ψ_mun, Σ_k)                                        # the face projector

The ISDF `sigma_kij` differs only in its middle line (`get_sigma_spatial_kernel`:
the ISDF k-convolution). The projector is the one face projector,
`common.contract_bands.contract_bands_block_reshard`, on Σ_k reordered spin-major.

χ at the MPA samples goes through the MPA owner's route dispatch,
`mpa.model._evaluate_samples`, with this basis as its χ producer (`chi=`): the plan is
`make_mpa_plan` on `build_static_quadrature`, so the z grid and every rule are the ISDF
run's. Each τ node is one `'trace'` pair convolution, summed into every sample row by
`accumulate_chi`. The conduction Green carries e^{-(E−μ)τ}, the valence Green
e^{+(E−μ)τ̄} (the right operand is read conjugated), so each pair carries e^{-Δτ}:
- static and imaginary points: the minimax rule on both particle–hole orientations;
- line points: the damped-line rule, nodes ±i·t on one orientation, weights
  `w_isdf._chi0_contour_alpha_rows` about E_gap = 0.

W then goes `SphereScreening.solve_samples` → `correlation` → `fit_poles`. The poles
enter `MemoryPoleSource(axis=SphereScreening.axis)` on the q-IBZ rows; the plane-wave
τ body reads them through its own sphere transport. The k tables are a `SphereKPlan`;
its `parent_rows` is `centroid_k_unfold.parent_rows`, the centroid plan's own.

## Symmetry and time

- G is built at the k-parents only; W at the q-IBZ, which equals the k-parents on a
  Γ-centred grid. Both unfold on load (K1, K2).
- At a real imaginary time, an antiunitary W row reads conj W (K3). The Σ executor's
  nodes and MPA poles are complex. There the rule is the ISDF path's pair transpose:
  the partner tile is built from conj residues at the same time factor. K2's
  `'scalar'` product refuses a W partner today. So a TR-broken deck needs that partner
  path first, and the first cut refuses antiunitary k rows (`GATE pw-pipeline-antiunitary`).

## Resume positions (later)

Every stage above ends in an object a slab_io writer can commit: the ψ store (the
WFN), v_q, χ(z), the poles, and Σ(ω). Resume positions are those committed writes,
keyed by the stage's inputs. SC maps loop over stages 3–9 with a rotated ψ store,
which the ISDF path already does through `rotate_wavefunctions`. None of this is built
in the first cut.

## What the first cut leaves out, on purpose

- The Γ head and wings. v(q+G=0) = 0, and the Γ W is the head-removed body. The
  wing producer (G4) and the head owner's S(z) are not wired.
- Stage 9 as an owner call. `--deck` computes Σ_x and Σ_c(ω) and interpolates
  Σ_c at E_DFT; eqp0 is assembled by the caller from the ISDF run's kin_ion and V_H.
- Band-tail extrapolation. The plane-wave Σ_c sums the deck's bands only.
- n_s = 2 and bispinors. The pair convolution supports them; the cut refuses
  antiunitary k rows.
- The r'-column wedge is an option (`--wedge`). It needs operands covariant under the
  group, which means complete multiplets at the band cut.
- Resume positions and deck keys. The entry is `python -m gw.plane_wave_pipeline`.
