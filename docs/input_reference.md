# Input reference

The GW deck is one `[cohsex]` INI section, read by
`gw.gw_config.read_lorrax_input`; the BSE, htransform and exciton-band drivers
read the same deck. `gw_config._DEFAULTS` is the accepted key set. Keys are
case-insensitive. An unknown key refuses, in one error naming every such key and
its line; a retired key refuses by name and states its replacement. Energies are
in Ry unless the key name ends in `_ev`. Environment overrides are listed in
[env_vars.md](dev/env_vars.md); longer discussions of the load-bearing keys are
in [drivers.md](drivers.md).

An optional `K_POINTS {crystal_b}` block after the section gives the band path
for htransform and exciton bands: a segment count, then one line per vertex,
`kx ky kz n  # label`.

This page is maintained by hand. `tools/gen_input_reference.py` cannot evaluate
the imported defaults in `_DEFAULTS` and writes nothing (`tests/KNOWN_FAILURES.md`).

The last section documents a different input file, the `[downfold]` deck.

## System

| key | type | default | meaning |
|---|---|---|---|
| `nval` | int | `5` | Valence edge of the Σ window: b1 = nelec − nval. It sets the ISDF pair-density right window, not the bottom of the QP window. |
| `ncond` | int | `5` | Conduction bands in the Σ/QP window. Σ diagonals are computed for bands [0, nelec + ncond). |
| `number_bands` | int | `100` | Umbrella band count. It sets both the χ0/W band sum and the Σ band sum. |
| `number_bands_chi` | int | unset (= `number_bands`) | Band count of the χ0/W sum. Its edge takes a strict `band_degeneracy` check and refuses on a split multiplet (`LORRAX_BAND_DEGENERACY=snap` warns instead). Setting it and `number_bands` to different values refuses. Changing it on `restart = true` refuses. |
| `number_bands_sigma` | int | unset (= `number_bands`) | Band count of the Σ sum; the band-extrapolation brackets are fractions of this count. It has the same strict degeneracy check and the same conflict refusal as `number_bands_chi`. It may change on `restart = true`, because no restart tensor depends on it. |
| `zeta_nband` | int | unset | Narrows the ζ-fit band window without changing the χ0 or Σ sums. Allowed range: [1, max(chi, sigma)]. It is compared with the padded edge b4 (the fit window rounded up to the world size): a value equal to b4 follows the loaded window, and a smaller value narrows it. The edge takes the strict degeneracy check. The comparison is resolved once, in `gw.gw_init.resolve_zeta_fit_edge`. |
| `sys_dim` | int | required | `3` for bulk, `2` for a slab with vacuum along c; selects the Coulomb truncation of V_q, W and the Hartree term. A deck without it refuses (`GATE sys_dim_required`). |
| `ecutrho` | float | unset (= WFN ecutwfc) | Density-grid cutoff (Ry) for the kin_ion and dipole preprocessing tools. |
| `bispinor` | bool | `false` | Four-component run: four-channel ζ fit, transverse Σ channels, and a second centroid table (`centroids_file_current`). |
| `bispinor_gw` | str | `bare_transverse` | Four-current screening model under `bispinor = true`. `bare_transverse`: CC screening from the four-component charge carrier plus bare TT exchange, with no CT/TT Dyson solve; under `compute_mode = mpa` and `sigma_w_model = shared_pole`, the CC response is full-frequency. `full_shared_pole`: the full-frequency CC/CT/TC/TT sector bank and its four-current Σ. It requires `screening_diagrams = w_rpa` and `head_correction = no_local_fields` (or `off`). `full_static_cohsex`: packed static four-current COHSEX. Any other value refuses. See [four-current physics](theory/four-current-head-corrections.md#four-current-phase-status) and [wiring](architecture/four_current_wiring.md). |
| `vnl_velocity_sign` | str | `""` | Sign of the i[r, V_NL] commutator in the velocity that `psp.get_dipole_mtxels` assembles (the CLI `--vnl-velocity-sign` overrides it). `-1`/`shipped` is the shipped assembly; `+1`/`flipped` reproduces BerkeleyGW's q→0 head. Empty means undeclared and resolves to `+1` (flipped). The choice is stamped into `dipole.h5`. |
| `hubbard_input` | str | `""` | DFT+U mean field: the deck-relative pw.x input carrying the `HUBBARD` card (ortho-atomic). With `hubbard_occupations`, this adds i[r, V_U] to the velocity and stamps both inputs into `dipole.h5` (`prov_hubbard`) and the velocity artifact. Consumers recompute the stamp and refuse a mismatch. A WFN whose QE schema declares DFT+U refuses without both keys (`GATE dftu_velocity_input`), and the keys refuse on a WFN without DFT+U. Refused: non-ortho-atomic projectors, J0/V/α/β, collinear nspinor = 1, US/PAW. |
| `hubbard_occupations` | str | `""` | That run's `prefix.save/occup.txt` (QE `rho%ns_nc`), deck-relative. Its SHA-256 is part of the DFT+U stamp. U and J are read from the card, never from the XML. |

The ISDF fit is sized by max(`number_bands_chi`, `number_bands_sigma`), rounded
up to the world size, and the pad bands are zero. The startup band-count banner
names which count sets it.

## Occupations and Fermi level

The material class is inferred from the WFN occupations; no deck key selects it.

| key | type | default | meaning |
|---|---|---|---|
| `fermi_reference` | str | `midgap` | `vbm` or `midgap` on an insulator. On a metal, `mp1_fixed_n` (the fixed-N chemical potential) is required, and it is refused on an insulator. The value is resolved once, in `gw.efermi.resolve_sigma_efermi_ry`, and stamped as `omega_reference_provenance` in `sigma_mnk.h5`. |
| `occ_smearing_width_ry` | float | unset | The Fermi–Dirac kBT (Ry) of a metal's occupations, f = 1/(1 + exp((E − μ)/kBT)); QE `smearing = 'fd'` matches it at `degauss = kBT`. It is required when the WFN occupations identify a metal and refused when they identify an insulator. It is the one width every metallic occupation solve uses. |
| `occ_broadening` | float (eV) | `0.0` | First-order Methfessel–Paxton broadening for the smeared QSGW head on an insulator, in BerkeleyGW's convention z = (E − E_F)/(2·occ_broadening); `0` gives step occupations. With `qp_solver = self_consistent` and a head mode in `sc_head_update`, each map solves its own fixed-N table at entry. A value > 0 beside `occ_smearing_width_ry` refuses (`GATE metal_sc_head_update_disabled`). |
| `occupation_clamp_tol` | float | `1e-8` | Snaps an MP1 occupation within `tol` of 0 or 1 to exactly 0 or 1. It is applied inside the fixed-N root, so μ is solved for the clamped table. Unclamped, the table's support edge is where `exp(−x²)` underflows, `x = 27.2971`, a float64 property; at `1e-8` it is `\|x\| < 4.30834`. The overshoot lobe beyond [0, 1] is never touched. Range [0, 1e-3]; `0` disables it. Fermi–Dirac tables ignore it. It decides which values exist in the table; `occupation_window_threshold` decides which bands enter a branch. |
| `occupation_window_threshold` | float | `0.995` | Drops a band from a Green's-function branch when \|weight\| ≤ 1 - threshold, where the weight is `f` on the hole side and `1 − f` on the electron side. It is a MAGNITUDE cut, so negative MP1 weights are kept. Consumers share `gw.efermi.occupation_weight_floor` and `band_in_occupation_window`: the Σ branch supports (`gw.ppm_windows.branches_for_omega_grid`), the MPA pane geometry (`gw.mpa.sigma_windows`), and the χ0 fractional-occupation supports (`gw.w_isdf`). |

## ISDF / zeta

| key | type | default | meaning |
|---|---|---|---|
| `centroids_file` | str | `centroids_frac.txt` | Charge-channel centroid table written by `centroid.kmeans_cli`. |
| `centroids_file_current` | str | `""` | Bispinor transverse-channel centroid table, selected from the Dirac-current feature norm. |
| `zeta_rcond` | float | `1e-8` | Rank-truncation cutoff of the charge CCT, relative to λ_max. A cut that discards directions while the achieved κ_eff exceeds the certified 1e8 refuses ([rank-truncation policy](dev/rank_truncation_policy.md)). |
| `zeta_ridge` | float | `0.0` | Tikhonov ridge on the charge CCT, as a fraction of the mean diagonal; `0` means no ridge. Only the `cholesky` charge family reads it; the `rank_truncate` factor that both `linalg` layouts resolve does not. |
| `zeta_cutoff` | float | unset (= ecutwfc) | G-sphere cutoff (Ry) of the per-q ζ_q(G) writes. It must be ≥ `bare_coulomb_cutoff`. |
| `linalg` | str | `local` | Dense linear-algebra layout. `local`: each task factors ⌈N_q,irr/P⌉ whole N_μ×N_μ matrices (the startup report prints the complex128 GiB per task). `distributed`: 2-D block-distributed matrices through the `distrib_la` providers (cuSOLVERMp/cuBLASMp on CUDA, ScaLAPACK on CPU) for the W Dyson solve, the transverse ζ LU and the eigensolvers; the charge ζ solve stays whole-tile (route G). The value does not invalidate a restart. |
| `memory_per_device_gb` | float | `0.0` | Per-device budget for the chunk planners; `0` auto-detects it. |
| `vq_g_chunk_size` | int | `0` | G-axis tile of the V_q GEMM. `0` lets `v_q_g_flat._plan_vq_tiles` choose. |
| `gamma_contract_mode` | str | `take` | HLO variant of the γ̃ double contraction: `take`, `einsum` or `scan`. All three are mathematically identical. |

**Raw-parent GW.** ψ is stored band-distributed, bands on one mesh axis and
centroids on the other ([memory model](architecture/memory-model.md)). The
Green's-function contraction consumes diagonal band occupations
(`occupation_state`); an explicit dense `Gij` passed to `compute_sigma_xc`
refuses (`GATE explicit_gij_unported`).

A centroid set that is not closed under the symmetry orbits runs unreduced: a
WARNING names the set, `SymMaps.trivial_view()` selects loader-unfolded full-k
parents and the full q grid, and the centroid file is unchanged. Orbit-closed
k-means avoids this.

## Screening

There is no deck key for time reversal. `SymMaps.trs_allowed` is measured from
the QE-schema receipt and the occupied two-component DFT states
([symmetry-service contract](services/symmetry_maps.md#contract)).

| key | type | default | meaning |
|---|---|---|---|
| `screening_diagrams` | str | `"w_rpa"` | Which diagrams W sums; ORTHOGONAL to `screening_method` (the frequency treatment), `head_correction` and the Σ `compute_mode`. `w_rpa`: the RPA series through the Dyson solve. `w_bse` and `w_rpa_resolvent` both evaluate the non-TDA resolvent W(z) − v = v(z − H)⁻¹v with one matvec builder: `w_bse` includes the static direct rung −W(0) (the ladder), and `w_rpa_resolvent` omits it (H = H_RPA). `w_bse` supports COHSEX, GN-PPM and MPA; `w_rpa_resolvent` supports COHSEX and GN-PPM. `w_bse` refuses by name: `x_only` (`w_bse_needs_a_screened_mode`), `hl_ppm` (`w_bse_hl_ppm_broadening_unimplemented`), `qp_solver = self_consistent` (`w_bse_self_consistency_unimplemented`), a non-`off` `mc_average_placement` (`w_bse_head_placement_unimplemented`), fractional WFN occupations (`w_bse_insulators_only`), and a WFN without measured time reversal (`w_bse_requires_measured_trs`). `w_rpa_resolvent` refuses the same cases under its own prefix, and `compute_mode = mpa` (`w_rpa_resolvent_mpa_unimplemented`). |
| `ladder_probe_chunk` | int | `0` | `w_bse`/`w_rpa_resolvent` only: probe columns of the μ² resolvent tile solved per block. `0` solves the whole padded basis in one block. A positive value bounds the per-block memory and is rounded up to a multiple of the mesh `y` axis; the values are unchanged. |
| `head_correction` | str | `full` | Finite-grid Γ policy. `full`: scalar routes build and Schur-fold the charge head; packed routes take their charge/current Γ blocks from the coupled Γ-cell completion, so the bispinor transverse head is always on and has no dial. `no_local_fields`: the direct frequency-dependent charge head S(ω) without a wing/body fold (shared-pole `bare_transverse`), or the direct CC/CT/TC/TT head (shared-pole `full_shared_pole`); other bispinor routes refuse it. On an ordered (time-reversal-broken) shared-pole store, `full` refuses (`GATE shared_pole_head_ordered`). `off`: no special Γ contribution, for brute-grid convergence only, announced as DEBUG. |
| `wcoul0_source` | str | `s_tensor` | Source of the direct no-local-field head: `s_tensor` (from `dipole.h5`) or `epshead`. Under `head_correction = full` it is completed by the microscopic body and wings. A missing source file refuses. |
| `wcoul0_eta` | float | `0.0` | Broadening η of the W-head evaluation. |
| `vhead` | float | unset | Override of the bare Coulomb head; unset = analytic. |
| `whead_0freq` | float | unset | Override of the static W head. |
| `whead_imfreq` | float | unset | Override of the imaginary-frequency (GN probe) W head. |
| `head_minibz_average` | bool | `false` | Coulomb head as a mini-BZ cell average rather than a point value. BSE densification (`bse_k_grid`) must use the GW run's value. |
| `mc_average_vcoul_body` | bool | `true` | Monte-Carlo mini-BZ average of the Coulomb body at every q ≠ 0, matching BerkeleyGW's default `cell_average_cutoff`. `false` matches a BerkeleyGW run with `cell_average_cutoff 1d-12`, which averages only q+G = 0. |
| `mc_average_placement` | str | `off` | Where the q ≠ 0 mini-BZ average is applied. `off`: the average replaces the argmin \|q+G\| slots of the V tile that serves as both the Dyson operator and its right-hand side. `bgw`: BerkeleyGW parity; the average is applied to W's head channel after the Dyson solve, W_head = ε_c⁻¹⟨v⟩, at the cost of a second Dyson solve per q. `schur_avg` refuses. Non-`off` values refuse on `restart = true` and on the bispinor V_q builder. |
| `mc_average_placement_vcoul` | str | `""` | A BerkeleyGW `write_vcoul` dump that supplies ⟨v⟩/v_c under `mc_average_placement = bgw`, matched to LORRAX head slots by \|q+G\|² shell. |
| `bgw_metal_q0_treatment` | str | `exact` | Metallic q = 0 convention. `exact`: LORRAX's exact metallic q→0 treatment. `bgw_q0shift`: the BerkeleyGW full-frequency convention as one mode. It requires `compute_mode = mpa` and `sys_dim = 3`. It sets `mc_average_vcoul_body = false` (an explicit `true` refuses), builds the q = 0 bare head from the analytic inscribed sphere plus the outer MC estimator, and takes the W head from the finite shifted-q0 Dyson row. |
| `bgw_metal_q0_vector` | str | `0 0 0.125` | Reduced coordinates of the finite q0 used by `bgw_q0shift`. It must be nonzero, lie on the deck's reciprocal grid and select a unique G = 0-like head slot. |
| `w_av_first_neighbors` | bool | `false` | Preprocessing: write the symmetry-reduced finite-q density vertices for the first ± reciprocal-grid neighbours (`get_dipole_mtxels --w-av-only --parallel-transport-out FILE`). |
| `w_av_second_neighbors` | bool | `false` | Preprocessing: also write the second axial and mixed neighbours of the 3-D quadratic W-av stencil. It requires at least 5 grid points on each active axis. |
| `bare_coulomb_cutoff` | float | unset (= ecutwfc) | G cutoff (Ry) of the bare V_q build. |
| `screened_coulomb_cutoff` | float | unset (= ecutwfc) | Real-space (non-ISDF) GW: the cutoff (Ry) of the χ_q(G,G′)/W_q(G,G′) sphere, \|q+G\|² ≤ cutoff, on the WFN's FFT box. It refuses at or above the box's alias cap, the smallest \|q+G\|² whose Miller index lies outside the window that products of two ψ spheres leave alias-free (`gw.mixed_basis_pair_convolution.screened_coulomb_cutoff_cap`). On a density box it sits below 4·ecutwfc, the reach of a pair product: on Fe's 25³ box at 70 Ry it is 237.03 Ry (3.386·ecutwfc). No driver reads the key yet. |
| `use_bgw_vcoul` | bool | `false` | Read v(q, G) from a BerkeleyGW vcoul file instead of building it. |
| `bgw_vcoul_file` | str | `""` | The BerkeleyGW vcoul file for `use_bgw_vcoul`. |
| `bgw_vcoul_sym_wfn` | str | `""` | Auxiliary WFN supplying the full symmetry group that folds LORRAX q-points onto BerkeleyGW's IBZ q list. |
| `screening_method` | str | `minimax` | χ0 frequency treatment. `minimax` is the only value. |
| `minimax_target_error` | float | `1e-6` | Target error of the χ0 time/frequency quadrature. MPA applies it to the static, imaginary-axis and damped-line samples. |
| `minimax_max_nodes` | int | `64` | Node cap of one interval minimax solve, and the per-panel Gauss-order cap on an MPA damped line. |
| `minimax_energy_reference` | str | `midgap` | Energy reference of the minimax transition range: `midgap` or `vbm`. |
| `mpa_n_poles` | int | `8` | MPA pole count, 1–16; each sampling line carries this many points. |
| `mpa_sampling_alpha` | int | derived | Exponent of the nested partition ω_n = ω_m·s_n^α: `1` or `2`. When omitted it is `1` for an insulator and `2` for a metal; the startup report prints the value and its provenance. |
| `mpa_sampling_schedule` | str | `nested` | Continuation above eight poles: `nested`, or `leon` (the Leon/Yambo qPPS schedule). |
| `mpa_pole_solver` | str | `loewner` | Pole identification: `loewner` (normalized pencil), `companion` (Yambo `LA`) or `thiele` (Yambo `PT`). All three share the pole guards and the all-sample residue refit. |
| `mpa_varpi_near_ry` | float | `0.2` | Height (Ry) of the near complex-frequency sampling line. |
| `mpa_varpi_far_ry` | float | `2.0` | Height (Ry) of the far line; it must exceed `mpa_varpi_near_ry`. |
| `mpa_metal_origin_shift_ry` | float | unset (= 2e-5 Ry) | Metal only: height (Ry) of the near line's first sample, z = i·shift, which avoids the zero-energy intraband pile-up. It must satisfy 0 < shift < `mpa_varpi_near_ry`, and it refuses on an insulator. Quoted Hartree values double in Ry. |
| `mpa_pole_batch_size` | int | `4` | Fitted-pole slabs resident during MPA Σ, 1–8. It sets the HBM schedule, not a spectral grouping. |
| `mpa_fit_reuse_file` | str | unset | A finalized MPA body/head fit, read by a `qp_solver = one_shot_dft` MPA run (a relative path resolves beside the deck). The loader certifies the grid, pole count, q table, centroids, screening-diagram provenance, occupations, source-WFN fingerprint and charge-ζ identity; a store missing either source identity refuses. |
| `mpa_overwrite_completed_artifacts` | bool | `false` | Permits destructive regeneration of an incompatible or completed MPA artifact. Without it, a compatible interrupted run resumes its ready slabs and committed fit ranges, a completed fit is write-once, and an incompatible artifact refuses by path. |
| `ppm_omega_p` | float | `2.0` | Second PPM probe frequency (Ry): iω_p for GN, real ω_p for HL. |
| `ppm_fallback_omega` | float | `2.0` | Real fallback pole (Ry) used by `ppm_invalid_mode = 2ry`. |
| `ppm_head_omega_h_ry` | float | unset | Direct override of the q→0 head pole Ω_h (Ry). |
| `ppm_invalid_mode` | str | `static_limit` | Treatment of a mode with no valid Ω² fit: `static_limit` (BerkeleyGW mode 3; alias `infinity`), `zero` (mode 0; alias `skip`) or `2ry` (mode 2). `imaginary` refuses. |
| `sigma_w_model` | str | `mpa` | Body model under `compute_mode = mpa`: `mpa` or `shared_pole` ([shared-pole model](architecture/shared_pole_model.md)). Under self-consistency, shared poles rebuild W from the current wavefunctions and retain the certified quadrature rules. |
| `sigma_w_accuracy` | str | `production` | Shared-pole recipe tier: `production` or `relaxed`. It requires `sigma_w_model = shared_pole`. |
| `sigma_w_support_sites_ev` | str | `""` | Shared-pole ladder override: `"<line eV list> \| <imaginary eV list>"`, each strictly increasing. Empty keeps the resolver's line rule and the Zolotarev imaginary ladder. The sites enter the recipe hash, so a model built on another ladder refuses. It requires `sigma_w_model = shared_pole`. |

`bispinor_tt_head_correction` is not a deck key: the transverse Γ head comes
with `head_correction`, and a deck that names the key refuses at parse. A
hand-built config with `head.bispinor_tt_head_correction = true` refuses with
`GATE bispinor_tt_head_unsupported` (`bispinor = false`, or `sys_dim` not 2
or 3) or `GATE packed_bare_transverse_tt_head_double_count` (a packed
static-photon route, which already inserts the head). Fix: leave the field
at its default, `false`.

## Sigma

| key | type | default | meaning |
|---|---|---|---|
| `compute_mode` | str | `auto` | Self-energy ansatz: `x_only`, `cohsex`, `gn_ppm`, `hl_ppm` or `mpa`. `auto` resolves from the deprecated aliases below and never selects `mpa`. On a metal, only `mpa` (screened) and `x_only` are admitted; `gn_ppm` refuses (`GATE gn_ppm_refuses_metals`). The production metal model is `mpa` with `sigma_w_model = shared_pole`. See [multipole integration](theory/THEORY_mpa_implementation.md) and [metallic MPA](theory/metallic-mpa-screening.md). |
| `sigma_quadrature_eps` | float | `1e-4` | Per-window sup-norm certificate of every denominator-box rule (MPA and GN/HL-PPM), in (0, 1): \|Q(d) − 1/d\| ≤ eps/η. The product windows partition the causal (state, pole, ω-sign) tuples, so \|δΣ_n\| ≤ (Σ_p \|M_np\|)·eps/η. A one-pole GN/HL model concentrates that mass: on Si, GN-PPM needs `1e-5` for 0.1 meV, where eight-pole MPA reaches it at `1e-4`. |
| `sigma_quadrature_cache_dir` | str | `auto` | Immutable rule cache. `auto` resolves to `tmp/sigma_quadrature_rules` beside the deck; `off` disables it; a relative path resolves beside the deck. A cache hit is rechecked against the request. |
| `sigma_regularization_ev` | float | `0.25` | Retarded broadening η (eV) of Σ(ω) for every ansatz, inserted once as exp(−ηt). It must be finite and positive and is stamped in `sigma_mnk.h5`. |
| `sigma_out_of_grid` | str | `cover` | Where a QSGW Σ(E) evaluation off the sampled grid reads. `cover`: the SC grid grows over every protected identity the W model treats as active (max_k E_DFT ≥ E_F − 15 eV) and not in `sc_frozen_core_bands`; deeper identities read Σ(ω = 0). `clamp`: the nearest grid edge. `static`: ω = 0. [Self-consistency §4](self_consistency.md) compares them. |
| `sigma_window_edge_factor` | float | `1.5` | MPA product-window margin: state edge = factor·η. It moves tuples among the three boxed products and adds no broadening. |
| `sigma_omega_min_ev` | float | `-5.0` | Lower edge of the Σ(ω) grid (eV, relative to E_DFT). |
| `sigma_omega_max_ev` | float | `5.0` | Upper edge; it must be ≥ `sigma_omega_min_ev`. |
| `sigma_omega_step_ev` | float | `0.25` | Grid step (eV), > 0. |
| `sigma_omega_patches_ev` | str | `""` | `"lo:hi, lo:hi, …"` (eV): ascending uniform patches at `sigma_omega_step_ev`, separated by at least one step, that replace the contiguous grid. A solved QP energy inside a hole refuses (`gw.qsgw_utils.assert_omega_grid_covers`). |
| `use_band_extrapolation` | bool | `true` | Extrapolates the Σ_c band sum from three bracket sums in one pass: disjoint band-bracket Green's functions against one W(τ), at about 3× the Σ τ-loop wall time. It refuses at startup, before the ζ fit, when `number_bands_sigma` < 2·n_occ, and at the Σ stage when the brackets do not resolve to three distinct counts; raising `number_bands_chi` clears neither refusal. On a non-PPM `compute_mode` it disables itself with a log note, but naming the key there refuses. Under self-consistency Σ is extrapolated and then diagonalized, so the extrapolated Σ is Hermitian. |
| `band_extrapolation_bracket_scheme` | str | `total_fractions` | Which three band sums are computed. `total_fractions`: 80, 90 and 100 % of `number_bands_sigma`. `conduction_fractions`: the same fractions of the conduction range, N_i = n_occ + round(f_i·(N3 − n_occ)). `conduction_energy_midpoint`: N1 at the conduction midpoint, snapped to a clean multiplet boundary, and N2 at the clean boundary nearest the k-mean energy midpoint. Naming it with extrapolation off, or on a run with no PPM stage, refuses. |
| `band_extrapolation_estimator` | str | `spectral_shell` | Which estimator consumes the three sums; it changes no compute. `spectral_shell`: a per-state decay exponent β from the two shell increments against spectral moments of the DFT eigenvalues, with the tail integrated to the plane-wave basis; a state whose two increments give no exponent in (0.05, 40) keeps its computed N3 sum, and the log counts those states. `band_index_only`: the two-parameter S∞ + A/N least squares. |
| `no_degen_averaging` | bool | `false` | Disables BerkeleyGW-style averaging of diagonal Σ within degenerate sets in the terminal output. |
| `degen_avg_tol_ry` | float | `1e-6` | Degeneracy tolerance (Ry) of that terminal averaging. It is not an SC tolerance. |
| `sigma_at_dft_extrapolate` | bool | `false` | Extrapolates Σ to E_DFT outside the ω grid instead of clamping. |
| `sigma_freq_debug_output` | bool | `false` | Writes the per-branch Σ(ω) debug table. |
| `sigma_freq_debug_file` | str | `sigma_freq_debug.dat` | Path of that table. |
| `sigma_lorentz_debug_output` | bool | `false` | Writes each four-current SC map's on-shell (CC, CT+TC, TT) matrices to `sigma_lorentz_iterNNNN.h5`. Under `full_shared_pole` it also evaluates the mixed and transverse sectors on shell. |

## QP solver and self-consistency

| key | type | default | meaning |
|---|---|---|---|
| `qp_solver` | str | `auto` | `one_shot_dft` (the `auto` default): one full-matrix effective Hamiltonian with Σ evaluated at E_DFT under QSGW Hermitian symmetrization. `fixed_point`: a diagonal on-shell solve, then the full-matrix Hamiltonian. `self_consistent`: the QSGW loop, which writes one energy table and one Z table per map ([self-consistency](self_consistency.md)). `fixed_point` with a static `compute_mode` refuses. |
| `sc_max_iter` | int | `30` | SC map cap, ≥ 1. Reaching it without meeting `sc_tol_ev` refuses (`GATE sc_fixed_point_not_converged`) and keeps the per-map tables. `1` runs a labelled one-map diagnostic. |
| `sc_tol_ev` | float | `1e-4` | SC convergence tolerance (eV), > 0. |
| `sc_accelerator` | str | `anderson` | `anderson` is the only value: one-evaluation Anderson type II with history `sc_history_depth`. Any other value refuses (`GATE sc_accelerator_anderson_only`); the fix is to delete the key. |
| `sc_history_depth` | int | `20` | Anderson history depth, ≥ 1. |
| `sc_mixing` | float | `1.0` | Linear-mixing α in (0, 1]. Only the in-process linear diagnostic reads it; no deck selects that path. |
| `density_self_consistent` | bool | `true` when omitted under `self_consistent` | Rebuilds V_H (and the bispinor current field) from the current orbitals every map, instead of rotating the DFT V_H. An explicit scalar `false` runs as an announced comparison mode; a bispinor `false` refuses. |
| `sc_on_ibz` | bool | `true` | Runs the loop's H/E/U and carried state on the star wedge and broadcasts at the boundary; Σ stays on the full BZ. It has no effect when every star is a singleton. |
| `sc_head_update` | str | `off` | `off` keeps the fixed DFT direct response. On an insulator, `dft_velocity` and `parallel_transport` rebuild the head each map from `parallel_transport_file`. On a metal, only `dft_velocity` with `sigma_w_model = shared_pole` and `head_correction = no_local_fields` is admitted: a direct Drude head from the authenticated `dipole.h5` velocity, tetrahedron Fermi-surface weights and a static Thomas–Fermi limit. Other metallic combinations refuse (`GATE metal_sc_head_update_disabled`). |
| `parallel_transport_file` | str | `parallel_transport.h5` | SlabIO velocity/link artifact for the insulating head modes. `parallel_transport` needs a derivative rule on every reduced k axis (`common.parallel_transport.link_stencil_orders`): the ±2 link stencil at ≥ 5 points, ±1 at 3–4, and on a one-point axis (a slab normal, a wire's transverse axes) the position operator along it, with its branch cut at the centre of the largest vacuum gap; a two-point axis is refused. |
| `static_gauge_hall_file` | str | `""` | Hall pseudovector written by `get_dipole_mtxels --static-gauge-hall-only --static-gauge-hall-out`, read only by the packed Γ-cell completion. Empty means σ_H = 0, which is exact for a Chern-trivial insulator. A named path must exist (`GATE static_gauge_hall_file_missing`) and authenticate against the run's WFN, band manifold and k count. On packed `bare_transverse`, a nonzero σ_H refuses (`GATE packed_bare_transverse_hall_unavailable`). |
| `sc_initial_qp_rotations_file` | str | `""` | A `qp_wfn_rotations.h5` that seeds a new SC run (a warm start, not a restart). The source WFN fingerprint, k table, band range, finite E/U and unitarity are validated. The initial carry is U diag(E) U^H in the DFT basis, and the accelerator history starts empty. |
| `sc_exact_degeneracy_tol_ev` | float | `1e-4` | Largest splitting (eV) averaged as an accidental degeneracy inside a map, in (0, 1e-4]. |
| `sc_tail_fit` | str | `conduction_mean` | Sum-band tail law. `conduction_mean`: one rigid shift, the k-weighted mean QP correction over the trusted conduction states. `frontier`: a rigid shift from the lowest conduction manifold. `all_conduction`: an affine fit over every in-grid conduction state. `buffer_edges`: a fit to the adjacent buffer; it requires `sc_buffer_nbands > 0`. |
| `sc_buffer_nbands` | int | `0` | Extra valence and conduction states evaluated around the `nval`/`ncond` window under self-consistency. They lie outside the full-matrix window. |
| `sc_buffer_mode` | str | `diagonal` | Treatment of the buffer. `diagonal`: its own Σ diagonal, with no off-diagonals. `one_sided`: keeps the cross-edge couplings at the in-window energy. `carry`: drops the couplings and carries the previous map's buffer energies. |
| `sc_frozen_core_bands` | int | `0` | The lowest N bands keep their DFT energies each map (no Σ, no scissor) and stay in every band sum. |
| `sc_dump_dir` | str | `""` | Directory for the output-energy history and each map's input rotation `rotation_iterNNNN.npy`. These are diagnostics, not checkpoints. |
| `write_eqp2` | bool | `false` | Writes `eqp2_file` beside eqp0/eqp1 by iterating the stored full-matrix Σ(ω) to an eigenvalue fixed point. It rebuilds no G, χ0, W or Σ. It requires a dynamic `compute_mode` and `qp_solver = one_shot_dft`, and diagonalizes with the `linalg` layout. |
| `eqp2_tol_ev` | float | `1e-3` | eqp2 tolerance (eV) on the largest eigenvalue change over the protected states. |
| `eqp2_max_iter` | int | `20` | eqp2 iteration cap. Failure refuses and writes no eqp2 file. |
| `eqp2_accelerator` | str | `rcrop` | eqp2 fixed-Σ map accelerator in the DFT basis: `rcrop` or `linear` (Picard). |
| `eqp2_history_depth` | int | `5` | eqp2 rCROP history depth, ≥ 1. |

## Output / IO / restart

| key | type | default | meaning |
|---|---|---|---|
| `wfn_file` | str | `WFN.h5` | BerkeleyGW-format wavefunction input. |
| `kin_ion_file` | str | `kin_ion.h5` | T + V_loc + V_NL matrix. Hartree is built live ([contract](theory/hartree.md)). |
| `report_file` | str | `gwjax.out` | Rank-0 human-readable report: architecture, backends, symmetry and IBZ, band windows, QP gap, paths, warnings and stage timings. A QSGW run adds one row per map and the terminal verdict. |
| `sigma_diag_file` | str | `sigma_diag.dat` | Per-(k, n) Σ decomposition, LORRAX text format. |
| `eqp0_file` | str | `eqp0.dat` | BerkeleyGW-format zeroth-order QP energies. |
| `eqp1_file` | str | `eqp1.dat` | BerkeleyGW-format Z-linearized QP energies (Z = 1 in static COHSEX). |
| `eqp2_file` | str | `eqp2.dat` | Fixed-Σ eigenvalue-self-consistent QP energies, written only under `write_eqp2 = true`. |
| `sigma_omega_h5_file` | str | `sigma_mnk.h5` | Σ_c(ω, k, m, n) HDF5 output. |
| `restart` | bool | `false` | `true` reads the copied restart stores instead of refitting and rebuilding V/W. It authenticates the WFN/QP-state source, bands, centroids, q tables and finite values, and it requires raw-parent face pairs; an older full-k file refuses, and the fix is a run with `restart = false`. Packed bispinor restart also requires `photon_g0_vectors_0`–`_3` in `v_q_bispinor.h5`. |
| `write_restart_tensors` | bool | `true` | Persists the restart tensors and raw-parent face pairs to `tmp/isdf_tensors_{n_rmu}.h5`. With `false`, consumers that need the file refuse. |
| `restart_q_storage` | str | `auto` | q set of the restart store. `auto`: the q-IBZ producer block when the q grid reduces, otherwise the full q axis. `ibz`: additionally requires a reduced wedge. |
| `qp_rotations_k_storage` | str | `auto` | k set of `qp_wfn_rotations.h5`. `auto`: the WFN wedge when the reader's own round trip reproduces the arrays exactly, otherwise full BZ, naming the array that failed. `ibz`: the wedge, refusing instead of falling back. `full`: the full BZ. A dataset without a `k_storage` attribute reads as full BZ. |
| `write_wfn_h5` | bool | `true` | Writes `WFN_qp.h5` at the end of the run (ψ rotated by the final U, QP energies). `qp_wfn_rotations.h5` is written regardless. |
| `write_qsgw_datasets` | bool | `false` | Adds `sigma_xc_qsgw_kij_ev`, `qp_omega0_ev`, `qp_diag_self_consistent_ev` and (only under `compute_mode = cohsex`) `qp_static_cohsex_ev` to `sigma_mnk.h5` on its own k set. The cost is one (nk, nb, nb) eigh. |
| `write_poles` | bool | `false` | Shared-pole MPA only: exports `tmp/mpa/<map>_poles.h5` with `b[q, μ, spin, j]`, Λ = Ω² (`poles2_ry2[q, j]`) and the active columns `K[q]`, where W_c(z) = b(z² − Λ)⁻¹b†. It is complete for a static-W consumer: W(0) = v − bΛ⁻¹b†. Existing outputs are immutable. |
| `write_w` | bool | `false` | Debug, shared-pole MPA only: writes the full W_c sample bank to `tmp/mpa/<map>_w.h5`, about 16× the pole export. BSE does not need it; use `write_poles`. |

## BSE and band interpolation

| key | type | default | meaning |
|---|---|---|---|
| `bse_k_grid` | str | `""` | `"NX NY NZ"`: densifies the BSE bundle (ψ/ε, W) from the coarse restart grid before any solve. Each extent must be at least the coarse one; integer nesting is not required. Empty keeps the coarse grid. |
| `w_head_densify` | str | `c1` | Coarse-to-fine W-head treatment under `bse_k_grid`. `c1`: splits off the singular Γ head and reattaches it analytically. `legacy`: trigonometric interpolation, as an A/B control. |
| `htransform_rank_multiplier` | float | `20.0` | Search ceiling ⌈multiplier·N_band⌉ of the whole-state randomized QRCP basis. `htransform_qr_eps` sets the delivered rank. |
| `htransform_qr_eps` | float | `1e-3` | Relative QR-diagonal rank threshold of the whole-state sketch (the pivoted-Cholesky form uses qr_eps²). |
| `htransform_qrcp_seed` | int | `0` | Seed of the candidate shuffle and the Gaussian sketch; the candidate and pivot hashes are printed. |
| `get_centroids_fi` | bool | `false` | htransform → BSE handoff: also computes fine-grid ψ at the coarse centroids. |
| `wfn_fi_min` | int | `0` | Lower edge of the htransform band sub-window (0-based). |
| `wfn_fi_max` | int | `0` | Exclusive upper edge; `0` selects the full window. |
| `kgrid_fi` | str | `""` | `"nx ny nz"` fine k-grid for the wavefunction recovery; empty = none. |
| `wfn_fi_q_chunk` | int | `0` | Fine-grid q-points per f(H(q)) build; `0` = N_q,coarse. It is rounded to the device count. |

htransform keeps its Galerkin basis in `galerkin_dft.h5` beside the deck. A file
whose provenance matches (WFN fingerprint, centroid SHA-256, band window, grids,
spinor mode, QRCP controls) is reused. A file with other provenance is left
unchanged and the basis is refit in memory. With no file, the basis is fitted
and published there. The log prints `REUSED`, `REFIT` or `FITTED and published`.

## Deprecated aliases

These keys are honoured; new decks write the replacement. The self-energy-axis
aliases (`gw_config.LEGACY_SIGMA_AXIS_KEYS`) print a deprecation note quoting the
resolved `compute_mode` / `qp_solver`. `nband`, `do_screened`,
`sigma_band_extrapolation` and `do_G0` refuse when they contradict an explicitly
named replacement; `use_ppm_sigma`, `ppm_model` and `self_consistent` are read
only when the replacement is `auto`.

| key | default | write instead |
|---|---|---|
| `nband` | unset | alias of `number_bands` |
| `do_screened` | `true` | `compute_mode` (`false` = `x_only`) |
| `use_ppm_sigma` | `false` | `compute_mode = gn_ppm` or `hl_ppm` |
| `ppm_model` | `gn` | `compute_mode = gn_ppm` or `hl_ppm` |
| `self_consistent` | `false` | `qp_solver = self_consistent` |
| `sigma_at_dft_energies` | `false` | `qp_solver = one_shot_dft` |
| `sigma_band_extrapolation` | unset | `use_band_extrapolation` |
| `do_G0` | `true` | `head_correction` (`false` = `off`) |

## Downfold — the `[downfold]` input file

This is not a GW deck. `gw.downfold_cli` compresses a finished GW calculation
onto a smaller ISDF basis for BSE and exciton-band work. The section header must
be `[downfold]`; a `[cohsex]` section refuses, and an unknown key refuses. The
keys are `downfold_config.DOWNFOLD_DEFAULTS`, and `--print-schema` prints them.
[downfold.md](downfold.md) argues for each default.

| key | default | meaning |
|---|---|---|
| `source_restart` | required | The finished GW run: its directory, or its `tmp/isdf_tensors_<mu>.h5`. A directory holding more than one bundle refuses. Relative paths resolve against the input file's directory. |
| `output_restart` | required | Output directory; the driver writes `<dir>/tmp/isdf_tensors_<mu_S>.h5`. It may not be the source directory. |
| `parent_centroids_file` | `""` | The parent's centroid table. When given, the kept rows are written as a sibling centroid file and its md5 is stamped on the small bundle; without it the bundle carries no centroid hash. |
| `parent_input_file` | `""` | The producer GW deck, which authenticates the raw-parent symmetry action. |
| `n_val` | unset | Shorthand for the retained window: left = (0, n_val). Only one of the two window spellings may be used. |
| `n_cond` | unset | right = (n_val, n_val + n_cond). |
| `band_range_left` | `""` | Left leg of the retained window, `lo:hi`, half-open, in absolute band indices. |
| `band_range_right` | `""` | Right leg. The legs are equal for BSE; an asymmetric window (the Σ shape) is accepted as unvalidated at about 2× the μ_S. |
| `mu_small` | required | Centroid count of the small basis, in points. When the parent stores a centroid source map, selection is by whole symmetry orbits and the count is floored, so realized ≤ requested ≤ ceiling; both numbers are printed. `auto` uses the eigenvalue rank at `downfold_rcond` and warns. Asking for more than the window's rank ceiling refuses before any expensive stage. |
| `downfold_rcond` | `1.1e-6` | Relative eigenvalue cut on the small basis's Gram, which caps the pseudo-inverse amplification at 1/rcond. |
| `downfold_select_tol` | unset (= √eps) | Stopping tolerance of the pivoted Cholesky. It is a different knob from `downfold_rcond` and gives a different rank; both are printed. |
| `mode` | `cur` | `cur` takes a subset of the parent's centroids, so no second ζ fit exists. `refit` refuses. |
| `plan` | `auto` | `auto`/`local`: the local plan, independent of the process grid. `distributed` refuses. |
| `report_residual` | `true` | Prints the per-q Pythagorean error bar ε_W (two GEMMs at μ_L per q). |
| `residual_refuse_above` | unset | Refuses to write the bundle when the worst-q ε_W exceeds this value. It is a tripwire, not a meV guarantee. |
