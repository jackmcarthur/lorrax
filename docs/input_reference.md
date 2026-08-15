# Input reference

Every deck key of the `[cohsex]` section: name, default, one-line meaning.

**This page is hand-maintained, and `tools/gen_input_reference.py` must not be
run against it.** It began as that script's output, but many rows have since
grown into paragraphs the generator's one-line model cannot express, and the
generator writes the whole file when it succeeds — so regenerating would flatten
`restart_q_storage`, `mc_average_placement` and a dozen others back to a
sentence. Verified 2026-08-10: the script's `KEYS` table is missing entries for
`mc_average_placement`, `mc_average_placement_vcoul`, `restart_q_storage`,
`sc_eigh` and `write_restart_tensors`, and still carries the deleted `slab_io`
and `use_ffi_io`, so today it refuses before writing — which is the only reason
this page survived. Use it as a **drift checker** (it prints exactly which keys
have appeared in or vanished from `_DEFAULTS`) and edit this page by hand.
`gw_config._DEFAULTS` remains the single source of truth for deck keys.
Unknown keys warn: a key not in `_DEFAULTS` and not covered by a legacy branch
is reported in one aggregated rank-0 warning and ignored; `strict_keys = true`
upgrades this to a refusal naming every unknown key.
Longer discussions of the load-bearing keys are in [drivers.md](drivers.md).

The last section of this page is a **different input file**. The downfold
driver takes its own `[downfold]` section rather than a GW deck, its keys are
`downfold_config.DOWNFOLD_DEFAULTS` and share nothing with `_DEFAULTS`, and an
unknown key there is refused rather than warned about. That table is
transcribed from `DOWNFOLD_DEFAULTS`, which is where anything regenerating this
page has to read it from.


## System

| key | default | meaning |
|---|---|---|
| `nval` | `5` | Interior valence edge of the sigma window: b1 = nelec - nval (gates the ISDF pair-density right window, not the QP window bottom). |
| `ncond` | `5` | Conduction bands in the sigma/QP window; sigma diagonals are computed for [0, nelec+ncond). |
| `nband` | `100` | chi0/Sigma band-sum top b4, rounded up to the world size (pads are zeroed). Sets the zeta-fit window top too unless `zeta_nband` narrows it. |
| `zeta_nband` | None | Top of the band window the ISDF zeta fit runs on, DECOUPLED from the chi0/Sigma band-sum top. Unset (the default) = follow `nband`, which is bit-identical to every deck written before 2026-08-11. An integer ≤ `nband` narrows the fit's left/right band ranges only: chi0 and Sigma keep summing [b0, b4). It exists because the two want opposite windows — the per-Q zeta refit behind a dense exciton band path needs `n_mu*n_s >= nk*nb` for its htransform Galerkin leg (Si 4×4×4 / 2628 centroids: nb ≈ 52) while the band sum wants every band it can get, and narrowing `nband` to serve the fit cost 222 meV per quasiparticle level for reasons unrelated to the zeta basis. The edge takes a STRICT `band_degeneracy` check and REFUSES on a split multiplet — on a spin-orbit deck every odd edge splits a Kramers pair. |
| `sys_dim` | `2` | System dimensionality (2 = slab): selects the Coulomb truncation. |
| `ecutrho` | None | Density-grid cutoff (Ry) for the psp tools (kin_ion/dipole); None = the WFN's own ecutwfc. |
| `bispinor` | false | Bispinor (4-spinor) run: 4-channel zeta-fit, Sigma^B transverse channels, two centroid files. |
| `vnl_velocity_sign` | `""` | Relative sign of the `i[r, V_NL]` commutator in the assembled velocity, read by `psp.get_dipole_mtxels` and passed to `common.mtxel_sweep.dipole_operator`; the CLI `--vnl-velocity-sign` overrides it. `-1` (equivalently `shipped`) is the shipped assembly; `+1` (`flipped`) is the arm that reproduces BerkeleyGW's q→0 head. Empty means NOT DECLARED and resolves to the shipped sign, which is why the default is a string and not a float: a float default would make "unset" and an explicit `-1` indistinguishable, and the point of the stamp this feeds is to record which arm a given `dipole.h5` was built with. This key was absent from this page until 2026-08-10 while being present in `_DEFAULTS`. |
| `fermi_reference` | `"midgap"` | Where E_F sits for the G/W time kernels: `vbm`/`midgap` inside a gap; `mp1_fixed_n` is the fixed-N MP1 chemical potential and is REQUIRED (and only legal) when `mpa_material_class = metal`. |
| `occ_smearing_family` | absent | Metal-only, required with `occ_smearing_width_ry` when `mpa_material_class = metal`; refused under insulator. Only `mp1` (Methfessel-Paxton order 1) is certified. Deck keys because WFN.h5 carries no smearing metadata. |
| `occ_smearing_width_ry` | absent | The occupation-smearing width in Ry, in **BerkeleyGW's convention** — the same quantity as `occ_broadening` (whose MP1 argument is `(E-E_F)/(2*width)`), just in Ry instead of eV. It is therefore **half the QE `degauss`**: a deck run at QE `degauss = 0.02` Ry sets `occ_smearing_width_ry = 0.01`. This is the single width the metal path's MP1 solve consumes (`LorraxConfig.occ_broadening_ry`); a deck that sets it and `occ_broadening` to inconsistent values is REFUSED at parse with the conversion spelled out. Metal-only; see `occ_smearing_family`. |

## ISDF / zeta

| key | default | meaning |
|---|---|---|
| `centroids_file` | `"centroids_frac.txt"` | Charge-channel ISDF centroid table written by centroid.kmeans_cli. |
| `centroids_file_current` | `""` | Second centroid table (Gordon-current weight) for the bispinor transverse channels; empty = not set. |
| `gflat_chunk_size` | `0` | Flat-axis chunk of the r-chunk G-accumulation; 0 = planner-chosen, explicit > 0 wins. |
| `vq_g_chunk_size` | `0` | V_q inner G-axis GEMM chunk; 0 = auto (largest divisor of ngkmax <= 4096). |
| `zeta_ridge` | `0.0` | Opt-in Tikhonov ridge epsilon on the charge CCT (fraction of mean diagonal); 0 = bit-identical historical factor. Env LORRAX_ZETA_RIDGE. |
| `charge_zeta_solve` | `"rank_truncate"` | Charge zeta conditioner: rank_truncate (default; rank-revealing eigh pseudo-inverse) or the historical cholesky. |
| `distributed_zeta_solve` | `"auto"` | Zeta back-solve tier: replicated | per_q | distributed (nothing O(mu^2) replicated); auto = replicated under the 4 GiB gather cap, else per_q. |
| `zeta_rcond` | `1e-08` | Rank-truncation cutoff relative to lambda_max (default 1e-8, low end of the recovery plateau). Env LORRAX_ZETA_RCOND. |
| `transverse_zeta_solve` | `"ridge"` | Transverse (bispinor) zeta-solve family: ridge (default; hoisted pivoted LU + 1e-12 ridge, byte-identical historical path) or rank_truncate (per-q eigh pseudo-inverse with an |lambda| cut; distributed plan via distributed_zeta_solve=distributed runs pzheevd at the padded extent, so any centroid count fits any square mesh). |
| `transverse_zeta_rcond` | `1e-10` | Transverse rank-truncation cutoff tau relative to |lambda|_max (rank_truncate family only; no env twin). |
| `gamma_contract_mode` | `"take"` | HLO variant of the gamma-tilde double contraction: take (default) | einsum | scan; math-identical. |
| `memory_per_device_gb` | `0.0` | Per-device memory budget for the chunk planners; 0 = auto-detect. |
| `band_chunk_size` | `16` | Bands per chunk in the band-chunked FFT/pair-density loops. |
| `r_chunk_size` | `0` | Real-space columns per zeta-fit chunk; 0 = auto from the memory model. |
| `zeta_cutoff` | None | Zeta-sphere G-cutoff (Ry) for per-q zeta_q_G writes; None = ecutwfc; must be >= bare_coulomb_cutoff. |

## Screening

| key | default | meaning |
|---|---|---|
| `do_screened` | true | Legacy mode flag: build W and the screened Sigma terms (false = bare exchange only); compute_mode=auto reads it. |
| `use_ppm_sigma` | false | Legacy flag for the dynamic PPM Sigma^c(omega); compute_mode=auto reads it. |
| `w_dyson_solver` | `"auto"` | W Dyson plan: local (per-q pivoted LU; auto alias) | distributed (2-D block-cyclic backsolve; refuses loudly, never downgrades). |
| `mc_average_vcoul_body` | true | Monte-Carlo mini-BZ average of the Coulomb body at every q!=0. Matches BGW's **default** `cell_average_cutoff` (1e12, average everywhere). Set **false** to match a BGW run with `cell_average_cutoff 1d-12` ("noavg"), which averages ONLY the literal q+G=0 element (vcoul_generator.f90:101-103). Mismatching it cost 136 meV MAE in bare Sigma_X on the Si 4x4x4 anchor; see tests/regression/si_cohsex_debug/README.md. |
| `bgw_metal_q0_treatment` | `exact` | Supported metallic q=0 convention bundle: `exact` preserves LORRAX's exact metallic q→0 treatment. `bgw_q0shift` reproduces the BerkeleyGW full-frequency convention as one indivisible mode: it overrides an inherited `mc_average_vcoul_body=true` to false (and prints that provenance), switches only the q=0 bare head to the analytic-inscribed-sphere plus outer-MC estimator, and obtains the W head including both wings/body Schur coupling from the finite shifted q0 Dyson row. An explicitly named `mc_average_vcoul_body=true` is contradictory and refuses naming both keys; explicit false is accepted. The mode does not change occupation/spectral broadening or any MPA quadrature rule. |
| `bgw_metal_q0_vector` | `0 0 0.125` | Reduced reciprocal coordinates of BerkeleyGW's finite epsilon q0 sample, consumed only by `bgw_metal_q0_treatment=bgw_q0shift`. It must be nonzero, lie exactly on the deck's reciprocal grid, and select a unique G=0-like head slot. On the 8x8x8 comparison deck the default is one grid step along z. |
| `mc_average_placement` | off | WHERE the q!=0 mini-BZ average is applied. Orthogonal to `mc_average_vcoul_body`, which decides WHETHER one is computed. **off** (default) = today's placement: `<v>` is substituted into the argmin `\|q+G\|` slots of the one production V tile, which is then both the Dyson operator and the Dyson right-hand side (`w_isdf.py:382-384`). **bgw** = BerkeleyGW parity: the average is applied to W's HEAD CHANNEL as a real scalar per q-cell AFTER the Dyson solve, i.e. `W_head = eps_c^-1 <v>` with `eps_c` built from the bare `v` — the placement BGW's Sigma (`mtxel_cor.f90:1659-1662`) and BSE (`intkernel.f90:887`) both use, and the exact cell average of the screened head under the f-sum-rule scaling `chi ~ q^2`. **schur_avg** = the derivable target `<W>_C = <v eps^-1>_C`; wired and REFUSED (it needs `chi` inside the cell). q=0 is untouched in every mode. `off` reproduces every existing deck bit-identically; `bgw` moves W and therefore every frozen reference. Costs a second Dyson solve per q. Refuses on `restart = true` and on the bispinor V_q builder. See `src/gw/head_channel.py`. |
| `mc_average_placement_vcoul` | (empty) | Optional BerkeleyGW `write_vcoul` dump to source the mini-BZ enhancement `<v>/v_c` from byte-for-byte instead of from LORRAX's own estimator, when `mc_average_placement = bgw`. Matched to LORRAX's head slots by `\|q+G\|^2` shell. Same override pattern as `vhead` / `whead_0freq`: it pins a cross-code comparison to BGW's values so what is left over cannot be a difference of Monte-Carlo estimators. LORRAX's own `<v>` already agrees with BGW's to 7e-4 - 2e-3 relative on every shell, so this is a tightening, not a correction. |
| `head_minibz_average` | false | Coulomb head as mini-BZ cell average instead of point value (fixes 4-13% near-Gamma error at finite Q); false = bit-identical legacy. |
| `W_av_first_neighbors` | false | Write symmetry-reduced finite-q density vertices for the first ± reciprocal-grid neighbors. `get_dipole_mtxels --w-av-only --parallel-transport-out FILE` produces the standalone SlabIO artifact without the unrelated dipole/parallel-transport sweeps. Preprocessing only; does not activate metallic screening. |
| `W_av_second_neighbors` | false | Also write second axial and mixed non-collinear neighbors required by the 3D-metal quadratic W-av stencil. Symmetry closure/reduction uses `SymMaps`; requires a non-aliased grid (at least 5 points on active axes). |
| `bare_coulomb_cutoff` | None | G-cutoff (Ry) of the bare Coulomb V_q build; None = ecutwfc. |
| `use_bgw_vcoul` | false | Diagnostic: read v(q,G) from a BGW vcoul file instead of building it. |
| `bgw_vcoul_file` | `""` | Path of the BGW vcoul file for use_bgw_vcoul. |
| `bgw_vcoul_sym_wfn` | `""` | Aux WFN supplying the full symmetry group to fold LORRAX q's onto BGW's IBZ q-list. |
| `wcoul0_source` | `"s_tensor"` | Where W(q=0) head comes from (s_tensor = analytic head correction from dipole.h5). |
| `wcoul0_eta` | `0.0` | Broadening eta for the W head evaluation. |
| `vhead` | None | Override the bare Coulomb head value; None = analytic. |
| `whead_0freq` | None | Override the static W head; None = computed. |
| `whead_imfreq` | None | Override the imaginary-frequency W head (GN probe); None = computed. |
| `screening_method` | `"minimax"` | chi0 frequency treatment. `minimax` is the ONLY supported value and any other REFUSES at config construction. The legacy spelling `ctsp` was accepted until 2026-08-06 and silently ran minimax, so replacing it with `minimax` (or deleting the key) changes no result. |
| `screening_diagrams` | `"w_rpa"` | **Which diagrams W sums.** `w_rpa` (default) is `W = (1 - V chi0)^-1 V`, every deck written before 2026-08-15, bit-identical. `w_bse` replaces the BODY of W with the ladder-corrected `W(z) - v = v (z - H)^-1 v`, where `H` carries the statically screened direct rung `-W(0)` — so the run still computes the RPA `W(0)` first and persists it, because that IS the ladder kernel's `W_R`. **This key is ORTHOGONAL to `screening_method`** (which says how the chi0 frequency integral is taken — minimax, the only one) **and to `compute_mode`** (which says which Sigma ansatz asks for W, and therefore at which frequencies). The role plan is identical either way; only which W satisfies the request changes. Supported under `w_bse`: `cohsex` (ladder W(0)), `gn_ppm` (ladder W(0) + W(i*omega_p)), `mpa` (ladder Wc at the sample plan's z-list, through the same Pade fit and store). REFUSED AT PARSE TIME under `w_bse`, each naming a rule id: `x_only` (`w_bse_needs_a_screened_mode` — that mode builds no W, and a key that parses and changes nothing is the `ctsp` defect); `hl_ppm` (`w_bse_hl_ppm_broadening_unimplemented` — the HL probe is a real frequency and `(z-H)^-1` on the real axis needs a broadening policy this tree does not have one answer for, see KNOWN_LORRAX_ISSUES.md:131/:134); `qp_solver = self_consistent` (`w_bse_self_consistency_unimplemented` — the per-iteration persist/reload cycle is unbuilt); `mc_average_placement != off` (`w_bse_head_placement_unimplemented` — v1 keeps the q=0 head/wing channel RPA and lets the ladder replace the body only); and a declared metal, `mpa_material_class = metal` (`w_bse_insulators_only`). **`w_bse` is for INSULATING systems only** — the ladder, its TRS gauge and every certification behind it assume integer occupations and a gapped D — and because no deck key describes the WFN's occupations, the same `w_bse_insulators_only` id fires again at the stage, before any compute, if the mean field's own `occ` array is fractional (beyond 1e-6) anywhere in the active band window. `w_bse` additionally requires `write_restart_tensors = true` and `restart_q_storage = full` (the ladder reads its kernel back through the sharded BSE loader, whose hyperslab transport refuses a q wedge); both refuse before any compute. W0 and MPA fit artifacts carry the value in their attrs and the MPA Sigma consumer asserts it at load. |
| `occ_broadening` | `0.0` | First-order Methfessel-Paxton occupation broadening in eV, using BerkeleyGW's convention `z=(E-E_F)/(2*occ_broadening)`. Zero preserves step occupations. With `qp_solver=self_consistent` and `sc_head_update` set to a head mode (see that row for the two), every QSGW map call solves its own fixed-N occupation table **at entry**, from the spectrum of the Hamiltonian it was handed, and that one table is what the call's screening, head and Sigma consume. There is no end-of-iteration carry and no accelerator restriction: the refusal that required `sc_accelerator=linear` on a fractional-occupation deck guarded the old carry and was deleted with it, so `sc_accelerator=rcrop` now parses on a metal. See [Metallic MPA screening](theory/metallic-mpa-screening.md) §6.1 and §7.2. **Two roles, and only one of them is a width.** Whether it is zero is the dial that selects step occupations; how wide the smearing is comes from `LorraxConfig.occ_broadening_ry`, which prefers the Ry-valued `occ_smearing_width_ry` on a metal deck and falls back to this key elsewhere. The two are the same quantity and a disagreement is refused at parse. Under `mpa_material_class = metal` the fractional occupations now reach Sigma_x/Sigma_SX/V_H as well (`build_Gij`'s `diag(f)`), not only the QSGW head; the finite-q screening body has its own metallic formulation under `compute_mode = mpa`. |
| `minimax_target_error` | `1e-06` | Target scalar error of the chi0 time/frequency quadrature. MPA applies it to the static, imaginary-axis, and damped-line sample evaluators; keep this tighter than the Sigma budgets because the pole fit can amplify sample noise. |
| `minimax_max_nodes` | `64` | Node cap for interval minimax solves and per-panel Gauss-order cap for an MPA damped line. It is not a cap on the total nodes across all damped-line panels. |
| `regenerate_minimax_tables` | false | Force re-solving the minimax tables instead of reusing cached ones. |
| `minimax_energy_reference` | `"midgap"` | Energy reference for the minimax transition range (midgap default). |
| `mpa_n_poles` | `8` | Number of MPA poles and therefore the number of points on each double-parallel sampling line (`2*mpa_n_poles` samples total). |
| `mpa_material_class` | `"insulator"` | Sampling and evaluator formulation. `insulator` puts the first near-line point at zero and preserves the integer-occupation path. `metal` puts it at `i*2e-5 Ry` and enables the live fractional-occupation χ/W/Σ path; it requires the metal smearing/reference keys above. `screening_diagrams = w_bse` still refuses this value because that distinct ladder path is certified for insulators only; `w_rpa` is the supported metal choice. |
| `mpa_sampling_alpha` | `1` | Real-coordinate exponent for the nested partition, `omega_n = omega_m*s_n^alpha`; supported values are 1 and 2. |
| `mpa_varpi_near_ry` | `0.2` | Height in Ry of the near complex-frequency sampling line. |
| `mpa_varpi_far_ry` | `2.0` | Height in Ry of the far complex-frequency sampling line; must exceed `mpa_varpi_near_ry`. |
| `mpa_metal_origin_shift_ry` | None | Height in **Ry** of the metal near line's FIRST sample, `z_1^1 = i*shift` — the published stability dodge around zero-energy intraband transitions, **not** a broadening. Unset (the default) = the published constant `1e-5 Ha = 2e-5 Ry` (`gw.mpa.sampling._METAL_ORIGIN_SHIFT`), which is bit-for-bit every grid built before this key existed. **METAL-ONLY**: under `mpa_material_class = insulator` the first sample is `z = 0` exactly, so setting this key there is refused by name rather than ignored. Refused unless `0 < mpa_metal_origin_shift_ry < mpa_varpi_near_ry`. **MIND THE UNIT.** This key is Ry like every deck key, while the multipole papers and the fallback ladder are quoted in Hartree, and **Ry = 2 × Ha** for every energy here: the owner's `1e-4 / 3e-4 / 1e-3 Ha` conditioning ladder is typed into a deck as **`2e-4` / `6e-4` / `2e-3` Ry**. Halving or doubling it is invisible in the output, which is why the two values are written out here rather than left as a conversion. When (and only when) a deck sets it, the resolved value is stamped additively on the sample/Wc stores as `mpa_prov_metal_origin_shift_ry`; it is outside the ω-grid digest, so a deck that leaves it unset writes a byte-identical store. |
| `mpa_pole_batch_size` | `4` | Maximum fitted-pole slabs resident during MPA Sigma. Values 1–4 are accepted. This is an HBM schedule, not a spectral grouping. |
| `ppm_model` | `"gn"` | Plasmon-pole ansatz: gn (Godby-Needs, imaginary probe) | hl (Hybertsen-Louie, real probe). |
| `ppm_omega_p` | `2.0` | Second PPM probe frequency (Ry): i*omega_p for GN, real omega_p for HL. |
| `ppm_fallback_omega` | `2.0` | Positive real fallback pole (Ry) for elements with no valid Omega^2 fit. |
| `ppm_head_omega_h_ry` | None | Override the q->0 head pole Omega_h (Ry) directly; None = compute normally. BGW comparison aid. |
| `ppm_probe_chi_reuse` | `"off"` | off \| auto. auto (GN only): represent the probe integrand on the static tau nodes plus the minimal augmentation from the dedicated quadrature's node set, and accumulate the probe chi0 inside ONE fused tau sweep — shared nodes' G-build/FFT/contraction tensors are computed once, only the k extra nodes cost new compute. Error gated at max(dedicated err, target_error) with a guaranteed exact-dedicated fallback. Same quadrature-error contract, not bit-identical to off — keep off for pinned-baseline decks. |
| `ppm_invalid_mode` | `"static_limit"` | Invalid-pole treatment: static_limit (default, BGW mode 3) | zero (BGW 0) | 2ry (BGW 2). |

## Sigma

| key | default | meaning |
|---|---|---|
| `compute_mode` | `"auto"` | Self-energy ansatz: x_only | cohsex | gn_ppm | hl_ppm | mpa; auto never infers mpa. MPA is a live disk-backed screening/Sigma path for insulating and explicitly configured metallic decks. See [Multipole frequency integration](theory/THEORY_mpa_implementation.md) and [Metallic MPA screening](theory/metallic-mpa-screening.md). |
| `no_degen_averaging` | false | Disable BGW-style averaging of diagonal Sigma within degenerate sets. |
| `degen_avg_tol_ry` | `1e-06` | Degeneracy tolerance for the averaging (BGW TOL_Degeneracy = 1e-6 Ry). |
| `ppm_sigma_target_error` | `1e-06` | Target error of the PPM Sigma^c tau-quadrature. |
| `ppm_sigma_max_nodes` | `64` | Node-count cap for the PPM Sigma^c quadrature. |
| `mpa_sigma_sector_target_error` | `6.5e-4` | Sampled relative-residual target `|1-d Q(d)|` for sign-definite complex-sector MPA windows. Nearby values need not give nested supports, so converge the emitted plan at the QP level. |
| `mpa_sigma_crossing_target_error` | `2e-3` | Relative-residual target `|1-d Q(d)|` for the positive causal Gauss rule on the MPA crossing rectangle. It is the same metric as the sector target but has a separately validated observable budget. |
| `mpa_sigma_max_nodes` | `96` | Maximum rank of one sign-definite MPA Sigma rule; the positive crossing fallback retains its 500-node safety ceiling. |
| `sigma_omega_min_ev` | `-5.0` | Sigma(omega) grid lower edge (eV, relative to E_DFT). |
| `sigma_omega_max_ev` | `5.0` | Sigma(omega) grid upper edge (eV). |
| `sigma_omega_step_ev` | `0.25` | Sigma(omega) grid step (eV). |
| `sigma_regularization_ev` | `0.25` | Retarded broadening of Sigma(omega) in eV. In MPA this is a literal external `eta`, applied once in addition to each fitted pole width; it is not a pole-fit width or an HGL scale. Must be finite and positive. |
| `sigma_window_edge_factor` | `1.5` | MPA core/sector partition margin in `T = max(abs(omega_min),abs(omega_max)) + factor*eta`. It moves work between exact quadrature families and does not add a second broadening. |
| `sigma_band_extrapolation` | `false` | Extrapolate Sigma_c to an infinite band count. GN-PPM / HL-PPM only (any other `compute_mode` REFUSES rather than ignoring the key). ON evaluates the Sigma_c band sum at THREE band counts in a single pass — the occupied block plus ~50%, ~75% and 100% of the conduction range, with the two interior cuts snapped so neither splits a degenerate multiplet (`common.band_degeneracy`) — by building three DISJOINT band-bracket Green's functions per tau against ONE W(tau), then fits `S(N) = S_inf + A/N` (two parameters, three points) and logs S(N3), S_inf, A, the three pairwise intercepts, `Delta_tail`, `Delta_model` and a trustworthiness verdict. REFUSES BY NAME when the band sum has `n_cond <= n_occ` (raise `nband`). Costs about 3x the Sigma tau-loop wall: G(tau) lives in the ISDF centroid basis, so the FFT chain and the psi projection are paid once per bracket however the bands are split. Off, the path is bit-identical to a build without the feature. |
| `sigma_omega_layout` | `"replicated"` | Sigma_c(omega,k,m,n) cube layout: replicated (default) | sharded (stays mesh-tiled end-to-end; works for every qp_solver; refuses an indivisible window or h5py_allgather at P>1). Production-size MPA runs should select sharded explicitly. |
| `sigma_at_dft_extrapolate` | false | Extrapolate Sigma to E_DFT outside the omega grid instead of clamping. |
| `sigma_at_dft_energies` | false | DEPRECATED alias for qp_solver = one_shot_dft (now the default). |
| `sigma_freq_debug_output` | false | Dump the per-branch Sigma(omega) debug table. |
| `sigma_freq_debug_file` | `"sigma_freq_debug.dat"` | Path of the Sigma(omega) debug dump. |

## IO / restart

| key | default | meaning |
|---|---|---|
| `wfn_file` | `"WFN.h5"` | BGW-format wavefunction input (WFN.h5). |
| `kin_ion_file` | `"kin_ion.h5"` | kin_ion.h5 from gw.kin_ion_io: T+V_loc+V_NL matrix (+ stored v_hartree). |
| `hartree_source` | `"auto"` | The G-space vs ISDF V_H switch: auto (stored -> folded -> isdf) | stored | isdf | gspace. |
| `sigma_diag_file` | `"sigma_diag.dat"` | LORRAX-native per-(k,n) Sigma-decomposition text output. |
| `eqp0_file` | `"eqp0.dat"` | BGW-format zeroth-order QP energies output. |
| `eqp1_file` | `"eqp1.dat"` | BGW-format Z-linearized QP energies output (Z=1 in static COHSEX). |
| `sigma_omega_h5_file` | `"sigma_mnk.h5"` | Sigma_c(omega,k,m,n) HDF5 output. |
| `restart` | true | Reuse tmp/isdf_tensors_{n_rmu}.h5 (skip zeta-fit/V_q); guarded by band-window attrs + centroid-table md5 stamps. |
| `write_restart_tensors` | true | Persist tmp/isdf_tensors_{n_rmu}.h5 at all. false skips every dataset (V_qmunu, G0_mu_nu, enk_full, psi_full_y, W0_qmunu, head scalars) with one rank-0 line. A COMPLEMENT to q_irr storage, not an alternative: this is for runs that DISCARD the artifact (nothing in gw_jax reads it back — measured 4.5 s of a ~21 s Si warm wall and 2.01 GB), while q_irr is for runs that KEEP it and want it 8x smaller. A BSE run against a directory written this way refuses on the missing file. |
| `restart_q_storage` | `"full"` | On WHICH q-set `V_qmunu` / `W0_qmunu` are STORED in `tmp/isdf_tensors_*.h5`. `full` (default) writes every full-BZ q, which is what every restart file has always held — a deck that omits this key keeps its bytes. `auto` writes the pre-unfold IBZ wedge **when the deck's centroid set is orbit-closed and this run's q path actually reduced**, and the full BZ otherwise, so on a non-closed set it is byte-for-byte `full`; on a closed set the two restart tensors shrink ~8x (measured 4.155x on the whole file, the difference being `psi_full_y`, which lives on a band/k index the wedge never touches). `ibz` is `auto` that REFUSES instead of falling back, naming which of the two conditions failed — for a deck that believes it is closed and wants to be told the day it stops being. Case- and whitespace-insensitive; an unrecognised value dies at PARSE time, not after the compute. **THE WEDGE IS OPT-IN PER DECK BECAUSE ONE READER STILL REFUSES IT.** The GW restart reader now ALWAYS UNFOLDS: it reads the wedge through the same transport it reads a full-BZ tensor with and applies the producer's own `unfold_isdf_operator` to the sharded result, so a wedge file and a full-BZ file are indistinguishable to `gw_init` (measured bit-identical, element-wise on the off-diagonals). `bse_io._MunuSlabPlan` still refuses a wedge at every process count. Its stated reason — that the unfold is a double gather across the very μ/ν axes it shards on — is a fact about SlabIO's hyperslab transport and NOT about the unfold, which is a `shard_map` over four `lax.all_to_all` collectives that redistribute those axes volume-preservingly and never exceed one tile per rank. The BSE unfold is designed (`DESIGN_restart_consolidation.md` §4) and is held only until its redistribution is timed on a real interconnect. So `auto`/`ibz` are for runs whose restart file is read back by the GW leg or through the serial h5py path, but not by a sharded BSE leg. Note the serial path is a stopgap and not a scalable answer: it materialises the full-BZ tensor on the host, once per caller, per rank. The default was briefly `auto` and the 2026-08-08 landing census measured the cost — nine red cells on the two in-tree decks that are already orbit-closed — which is why it is `full` here, per `DESIGN_symmetry_restart_followup.md`. Independent of `write_restart_tensors`, which decides *whether* the file is written at all. **⚠ TRANSITIONAL KEY — DO NOT BUILD ON IT (owner ruling 2026-08-08 ~13:20).** This knob is scheduled for DELETION, not for a better default. The ruling: symmetry should never have needed a mode switch, because the wavefunction file already answers the question — "if symmetries are not to be used, the wavefunction file should've been generated with no symmetries." The end state is restart storage that simply FOLLOWS the WFN's own symmetry (the wedge whenever the deck carries symmetries, readers that always unfold), reached by consolidating the GW and BSE restart-from-GW-tensors paths into a few dozen lines in the core drivers, after which `restart_q_storage` goes away entirely — one fewer feature to track. Set it only to pin a deck against the transition; see `tests/KNOWN_FAILURES.md` for the registered work. |
| `gspace_mode` | `"host_cache"` | psi(G) host lifecycle: host_cache (resident, default) | file_reread (rebuild per r-chunk; zero persistent residency). |
| `write_wfn_h5` | true | End-of-run WFN_qp.h5 write (BGW format, psi rotated by the final U, E_QP energies). |
| `write_qsgw_datasets` | false | Add the QSGW plotting appendix to sigma_mnk.h5: `sigma_xc_qsgw_kij_ev` (the static Hermitian Sigma_xc the QSGW ansatz builds) plus the QP energy ladders `qp_omega0_ev` (H0 + Sigma_x + Sigma_c(omega=0)), `qp_diag_self_consistent_ev` (the diagonal on-shell fixed point E = h0 + ReSigma(E)) and `qp_static_cohsex_ev` (H0 + Sigma_SX + Sigma_COH, written only by a run that BUILT those two channels, i.e. `compute_mode = cohsex`; a PPM run says so in one rank-0 line rather than putting a different operator under that name). Default false is byte-for-byte today's file: these four datasets had no producer between 2026-04-11, when the QP/output rewrite deleted the block that wrote them, and 2026-08-08, and survived only in the committed `cohsex_debug` fixture. They are written on the file's OWN k-set, carrying the same `k_storage` stamp and the same four star-spread numbers as every other cube in it, so a plotting script reads them through the reader path it already uses for Sigma_c. The cost is a write plus one (nk, nb, nb) eigh and one host-side (nk, nb) fixed point — never a Sigma or screening kernel. A quantity the run's compute mode did not build is omitted and named, not manufactured. |
| `strict_keys` | false | Unknown deck keys refuse (ValueError naming every one) instead of the default aggregated warning. |

## Solver

| key | default | meaning |
|---|---|---|
| `density_self_consistent` | false | Rebuild V_H from the current orbitals every SC iteration instead of rotating the fixed DFT V_H into the QP basis; off keeps QSGW fixed-density. |
| `sc_on_ibz` | false | Run the SC loop's H/E/U and carried state on the IBZ, broadcasting back at the boundary; Sigma stays on the full BZ. Ignored when every k-star is a singleton. |
| `sc_head_update` | `"off"` | `off` preserves the fixed DFT head; `parallel_transport` and `dft_velocity` each rebuild the occupation-aware head every QSGW iteration. They run the SAME chain — S(z) Kubo tensor, tetrahedron Drude term, ISDF wings, Schur fold, mini-BZ, Loewner fit — and differ only in the velocity operator: `parallel_transport` adds the covariant DeltaH correction built from the saved Berry connection and the current fixed-basis Hamiltonian, while `dft_velocity` uses the artifact's exact DFT p-matrix velocity alone, rotated into each iteration's QP basis by the same U. `dft_velocity` therefore needs only the velocity stage of the artifact and never opens the connection stage; it is the route every published sodium head number was measured through (`tools/qsgw_head_spectrum.py --dft-velocity-only`, claims 0180/0181/0189), and the covariant upgrade it forgoes is parked on claim 0183. COHSEX/PPM also form q-linear wings from the resident centroid wavefunctions and fold them through the matching screened body. MPA evaluates the head and direct centroid Y/Z wings on its exact complex-frequency plan, Schur-folds each resident body-W sample in the ISDF basis, and fits the resulting head with the body Loewner policy. No path reloads the WFN. At iteration zero DeltaH=0 and U=I, so under `parallel_transport` the same stationary map reconstructs the DFT head. |
| `parallel_transport_file` | `"parallel_transport.h5"` | SlabIO artifact written by `get_dipole_mtxels --parallel-transport`; required whenever `sc_head_update` names a head mode. Its WFN fingerprint, k grid, band manifold and reciprocal lattice must match the run on both modes; `parallel_transport` additionally requires the completed connection stage and its velocity-identity validation, which `dft_velocity` does not read. |
| `qp_solver` | `"auto"` | QP extraction: one_shot_dft (G0W0 at E_DFT; auto default) | fixed_point (on-shell) | self_consistent (QSGW loop). |
| `do_G0` | true | Compute the analytic q->0 static head terms (needs dipole.h5); part of every production run. |
| `self_consistent` | false | DEPRECATED alias for qp_solver = self_consistent. |
| `sc_max_iter` | `20` | Self-consistency iteration cap. |
| `sc_tol_ev` | `0.0001` | Self-consistency convergence tolerance (eV). |
| `sc_accelerator` | `"rcrop"` | SC mixing accelerator: rcrop | linear. |
| `sc_history_depth` | `5` | rCROP history depth. |
| `sc_mixing` | `1.0` | Linear-mixing alpha (accelerator = linear only). |
| `sc_dump_dir` | `""` | Directory for per-iteration E-history npy dumps; empty = off. |
| `sc_eigh` | `"auto"` | Eigh for the per-iteration (nk, nb, nb) carry: native = k-sharded batch, one whole (nb, nb) tile per device; distributed = one tile spread over the mesh; auto = distributed once a tile exceeds a fraction of `memory_per_device_gb` and the mesh divides nb, else native. Layout only — the eigenvalues agree and the physics does not change. |
| `distributed_cholesky` | `"auto"` | Charge-channel zeta-fit Cholesky backend: auto | off | cusolvermp | slate. |
| `distributed_lu` | `"auto"` | Transverse-channel LU backend: auto | off | cusolvermp | scalapack (host CPU; explicit only). |
| `distrib_la_batched_route` | `"auto"` | Schedule used by every array-returning `distrib_la.Plan.batched` call. `auto` preserves the plan's normal distributed scan or backend-batched implementation (the robust default when one matrix must stay mesh-sharded). `batch_reshard` explicitly moves `H_k(m_X,n_Y)` to `H_{k_XY}(m,n)`, runs the local JAX operation on each device's whole matrices, and reshards outputs back to the plan contract. Use only when one complete matrix plus its result/workspace fits on each device. Backend resolution and the Plan I/O contract stay unchanged, but this explicit call route executes the local JAX kernel rather than the resolved backend. It does not apply to opaque `factor()`/`solve()` token routes. The htransform and exciton-band CLIs expose the same spelling as `--distrib-la-batched-route`; CLI wins over the deck. **On the GW W-solve it is INERT unless `w_dyson_solver = distributed`** — the default `local` plan is a per-q dense LU inside a `shard_map` that never enters `distrib_la` at all, so the key parses, resolves and then changes nothing. Measured 2026-08-16 on the sodium metallic MPA deck: with `w_dyson_solver` at its default the two arms are byte-identical in `eqp0`/`eqp1` over all 1392 rows; with `w_dyson_solver = distributed` the announcement changes from `batched 'auto' -> backend_batched` to `batched 'batch_reshard' -> batch_reshard` and the arms agree to 0.001 ueV. |
| `eigh_backend` | `"auto"` | BSE/htransform distributed-eigh sites: auto|off = q-batched native eigh; distributed | cusolvermp | slate | scalapack spread ONE tile over the mesh. CLI --eigh-backend overrides. |
| `use_low_mem_eigh` | false | Same axis by intent: one (rank,rank) matrix does not fit a rank; true + auto => distributed; true + off refused at parse. |

`distrib_la_batched_route = batch_reshard` and
`use_low_mem_eigh = true` are contradictory and refuse during input
resolution: the former requires a complete matrix on a device while the
latter says that residency is unsafe.  GWJAX reads the deck key through its
shared `BackendConfig`; htransform and exciton-bands read the same deck key
and optionally override it with the identically-valued CLI flag.  The k-means
executable is intentionally not given this option: it is the centroid
preprocessing step and runs its own distributed greedy pivoted-Cholesky
selection, not an array-returning `distrib_la.Plan.batched` operation.  Its
~1000-centroid output is subsequently consumed by the GWJAX and htransform
paths where this route applies.  Restart tensors and ζ provenance describe
math/layout outputs, not an execution schedule, so selecting a different
route does not by itself invalidate an otherwise reusable restart; route is
included in compiled-kernel cache keys wherever it changes a closed-over
Plan.

## BSE

| key | default | meaning |
|---|---|---|
| `bse_k_grid` | `""` | "NX NY NZ" fine grid: densify the BSE bundle (psi/eps, W) from the coarse restart grid before any solve; the q=0 exchange tile is k-grid-invariant and is carried through unchanged unless head_minibz_average is set; empty = coarse, byte-identical. |
| `get_centroids_fi` | false | htransform -> BSE handoff: also compute fine-grid psi at the coarse centroids (bse_setup.compute_wfns_fi). |
| `wfn_fi_min` | `0` | Sub-window lower edge on the htransform band axis (0-based). |
| `wfn_fi_max` | `0` | Sub-window upper edge, exclusive; 0 = full window. |
| `kgrid_fi` | `""` | "nx ny nz" fine k-grid for the wfn recovery; empty = none. |
| `wfn_fi_q_chunk` | `0` | Fine-grid q-points per f(H(q)) build; 0 = N_q_coarse (same per-rank residency as fH_R); floor, rounded to device count. |

## Downfold — the `[downfold]` input file

Not a GW deck. These are the keys of `gw.downfold_cli`, the driver that
compresses a finished GW calculation onto a smaller ISDF basis so that BSE and
exciton-band work runs in a basis sized for its own band window; the section
header must be `[downfold]`, a `[cohsex]` section is refused by name, and an
unrecognised key is refused rather than ignored. `--print-schema` prints the
same list. Each key's default is argued for in prose, with the measurements
behind it, in [downfold.md](downfold.md).

| key | default | meaning |
|---|---|---|
| `source_restart` | None | REQUIRED. The finished GW run to compress: the run directory, or its `tmp/isdf_tensors_<mu>.h5` directly. REFUSES on more than one bundle in the directory rather than taking the newest. Relative paths resolve against the input file's own directory. |
| `output_restart` | None | REQUIRED. An output DIRECTORY; the driver writes `<dir>/tmp/isdf_tensors_<mu_S>.h5`, the layout every BSE consumer already looks for. May not be the source directory. |
| `parent_centroids_file` | `""` | The parent's centroid coordinate table. When given, the kept rows are written as a sibling centroid file and its md5 stamped onto the small bundle. Without it the bundle carries no centroid hash and says so, rather than inheriting the parent's, which would name the wrong points. |
| `n_val` | None | Valence/conduction shorthand for the retained window: left = (0, n_val). Exactly one of the two spellings may appear; giving both refuses. |
| `n_cond` | None | The other half of the shorthand: right = (n_val, n_val + n_cond). |
| `band_range_left` | `""` | Left leg of the retained band window, `lo:hi`, half-open, ABSOLUTE band indices into `psi_full_y`/`enk_full`. This is what the compression is faithful to, and it has no default because it is a physics choice. |
| `band_range_right` | `""` | Right leg of the same window — the two are the legs of the pair density. EQUAL for BSE work; an asymmetric window is the Sigma-serving shape, accepted and announced as unvalidated, at about 2x the mu_S. |
| `mu_small` | None | REQUIRED. Centroid count of the small basis, **stated in POINTS**. When the parent bundle carries its own centroid source map (it does whenever the parent stored its tensors on the q wedge), the selection is made in whole symmetry orbits and `mu_small` is FLOORED: the run realizes the largest union of whole orbits whose point total does not exceed your request, prints both numbers, and never overruns the budget — so a request of 185 on `si_bse_debug` comes back as 168 and says so. The direction is always INWARD: `mu_small` is a budget and may not be overrun, and the floor is taken against the rank ceiling as this run resolved it, so `realized <= requested <= ceiling` regardless of how that cut settles. Set an EXPLICIT integer and validate it by comparing the observable against the parent's; `auto` = the eigenvalue rank of the window's Gram at `downfold_rcond`, which is a rank CEILING and is no longer recommended — it sized a run to 189 that came out 2.09 eV wrong in the lowest BSE eigenvalue with `eps_W` silent (`PIPELINE_HEALTH.md`, 2026-08-10), and it now prints a loud warning. Asking for more directions than the window holds REFUSES before any expensive stage and prints the measured ceiling. |
| `downfold_rcond` | `1.1e-06` | Relative eigenvalue cut on the small basis's Gram: a cap on the truncated pseudo-inverse's amplification (1/rcond) by construction, NOT a gap-finder. The measured 20-band ceiling is ~190 directions on two decks, and loosening the cut to 1e-8 bought 3.2x more centroids and a slightly worse spectrum. |
| `downfold_select_tol` | None | Pivoted Cholesky's stopping tolerance; None = the kernel's own sqrt(eps). NOT the same knob as `downfold_rcond` and it does not give the same rank — the selection certificate runs about 3x the eigenvalue rank at the same nominal number, which is why both are printed, labelled differently, on every run. |
| `mode` | `"cur"` | How the small basis is chosen: `cur` takes a SUBSET of the parent's centroids, so both fit operands are submatrices of one object and no second zeta fit exists anywhere. `refit` (fresh narrow-window k-means plus a second zeta fit) is REFUSED rather than demoted, so that nobody reads a CUR result as a refit one. |
| `plan` | `"auto"` | `auto`/`local` = the local plan: no block-cyclic factorisation, no dependence on the process grid. `distributed` (mu tiled over a 2-D grid) is later work and REFUSES rather than demoting, because a different factorisation is a different numerical gauge. |
| `report_residual` | true | Compute and print the per-q Pythagorean error bar `eps_W`. Two GEMMs at mu_L per q; leave it on — it is the only answer to "did this work" that needs no reference calculation. |
| `residual_refuse_above` | None | Refuse to WRITE the small bundle when the worst-q `eps_W` exceeds this; empty = report and always write. A tripwire against a compression that has gone badly wrong, NOT a transferable statement about meV: the same 1% `eps_W` sat on 37 meV of exciton drift on one parent and 1.7 eV on another. |
