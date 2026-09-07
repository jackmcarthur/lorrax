| Lane | Branch | State | Evidence root |
|---|---|---|---|
| BISP-REVIEW | `refactor/bisp-compaction-2026-09-06` | ACTIVE; final gates pending | /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/117_bisp_review_codex_2026-09-06 |

| Item | Status |
|---|---|
| 1 — dead dispatch | Planner and direct Gram cuts pushed; live-consumer exceptions below |
| 2 — phase extraction | In progress; remaining functions above 120 lines are reported explicitly |
| 3 — duplication | Pending |
| 4 — docstrings | Prepared relocation; not yet applied broadly |
| 5 — configuration | Strict-key/legacy exemptions inspected; parser separation pending |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/gflat_memory_model.py` | `_persistent_bytes` | 63 | 60 | Existing statements moved to named stages or dead selector removed |
| `src/gw/gflat_memory_model.py` | `plan_gflat_chunks` | 731 | 689 | Existing statements moved to named stages or dead selector removed |
| `src/gw/gflat_memory_model.py` | `plan_gflat_chunks._floor_at` | 38 | 36 | Existing statements moved to named stages or dead selector removed |
| `src/gw/gflat_memory_model.py` | `plan_gflat_chunks._band_candidate_fits` | 31 | 25 | Existing statements moved to named stages or dead selector removed |
| `src/gw/gflat_memory_model.py` | **File total** | 1451 | 1406 | Net -45 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/gw_jax.py` | `main` | 1448 | 1021 | Existing statements moved to named stages or dead selector removed |
| `src/gw/gw_jax.py` | `main._config_print` | 5 | 0 | Deleted or renamed to direct owner |
| `src/gw/gw_jax.py` | `_open_production_report` | 0 | 38 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_open_production_report._config_print` | 0 | 5 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_report_head_and_photon_policy` | 0 | 64 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_load_system_inputs` | 0 | 46 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_prepare_band_metadata` | 0 | 63 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_report_sampling_and_bands` | 0 | 39 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_prepare_oneshot_response` | 0 | 94 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_install_oneshot_head` | 0 | 30 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_persist_screening` | 0 | 15 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_prepare_static_head` | 0 | 24 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_load_kinetic_ionic_hamiltonian` | 0 | 21 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_close_timing` | 0 | 23 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_report_file_rows` | 0 | 45 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | **File total** | 1712 | 1811 | Net +99 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/isdf_fitting.py` | `fit_zeta_to_h5` | 1366 | 1366 | Existing statements moved to named stages or dead selector removed |
| `src/gw/isdf_fitting.py` | **File total** | 1542 | 1542 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/sigma_dispatch.py` | `compute_sigma_xc` | 809 | 809 | Unchanged; target remains |
| `src/gw/sigma_dispatch.py` | **File total** | 1604 | 1604 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/gw_init.py` | `_plan_gflat_chunks_for_channel` | 136 | 135 | Existing statements moved to named stages or dead selector removed |
| `src/gw/gw_init.py` | `fit_zeta` | 576 | 575 | Existing statements moved to named stages or dead selector removed |
| `src/gw/gw_init.py` | `compute_V_q` | 398 | 398 | Unchanged; target remains |
| `src/gw/gw_init.py` | `prepare_isdf_and_wavefunctions` | 742 | 742 | Unchanged; target remains |
| `src/gw/gw_init.py` | **File total** | 3786 | 3784 | Net -2 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/isdf/core.py` | `c_q_from_psi_sm` | 42 | 95 | Existing statements moved to named stages or dead selector removed |
| `src/isdf/core.py` | `_c_q_legacy` | 122 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `_c_q_face_parent` | 95 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `_z_q_face_parent` | 327 | 327 | Unchanged; target remains |
| `src/isdf/core.py` | `factor_c_q` | 317 | 317 | Unchanged; target remains |
| `src/isdf/core.py` | `solve_zeta` | 699 | 699 | Unchanged; target remains |
| `src/isdf/core.py` | `c_q_downfold` | 0 | 122 | New explicit phase/direct owner |
| `src/isdf/core.py` | **File total** | 5831 | 5789 | Net -42 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/common/wfn_transforms.py` | `gflat_to_rmu` | 362 | 362 | Unchanged; target remains |
| `src/common/wfn_transforms.py` | `load_centroids_band_chunked` | 683 | 683 | Unchanged; target remains |
| `src/common/wfn_transforms.py` | **File total** | 3158 | 3158 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/gw_config.py` | `read_lorrax_input` | 378 | 378 | Unchanged; target remains |
| `src/gw/gw_config.py` | **File total** | 5586 | 5586 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `unfold_isdf_operator` | 412 | 412 | Unchanged; target remains |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps.__init__` | 488 | 41 | Existing statements moved to named stages or dead selector removed |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._initialize_symmetry_provenance` | 0 | 48 | New explicit phase/direct owner |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._initialize_active_operations` | 0 | 39 | New explicit phase/direct owner |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._validate_identity_grid` | 0 | 40 | New explicit phase/direct owner |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._initialize_identity_maps` | 0 | 104 | New explicit phase/direct owner |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._initialize_spatial_operators` | 0 | 98 | New explicit phase/direct owner |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._initialize_k_maps` | 0 | 96 | New explicit phase/direct owner |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._initialize_q_maps` | 0 | 41 | New explicit phase/direct owner |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | **File total** | 4190 | 4217 | Net +27 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/w_isdf.py` | `compute_static_photon_response` | 362 | 58 | Existing statements moved to named stages or dead selector removed |
| `src/gw/w_isdf.py` | `_resolve_static_photon_policy` | 0 | 92 | New explicit phase/direct owner |
| `src/gw/w_isdf.py` | `_read_static_photon_body` | 0 | 27 | New explicit phase/direct owner |
| `src/gw/w_isdf.py` | `_report_static_photon_body` | 0 | 26 | New explicit phase/direct owner |
| `src/gw/w_isdf.py` | `_screen_static_photon_body` | 0 | 85 | New explicit phase/direct owner |
| `src/gw/w_isdf.py` | `_complete_static_photon_head` | 0 | 60 | New explicit phase/direct owner |
| `src/gw/w_isdf.py` | **File total** | 3072 | 3068 | Net -4 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/head_correction.py` | **File total** | 2559 | 2559 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/downfold.py` | `pair_density_gram` | 45 | 45 | Existing statements moved to named stages or dead selector removed |
| `src/gw/downfold.py` | **File total** | 1636 | 1636 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/common/contract_bands.py` | **File total** | 799 | 799 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/multi_device/bispinor_physics_oracles.py` | `test_isdf_current_signed_normal_matrix_against_literal_pair_gram` | 30 | 30 | Existing statements moved to named stages or dead selector removed |
| `tests/multi_device/bispinor_physics_oracles.py` | **File total** | 892 | 892 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_centroid_fft_k_plan.py` | `_plan` | 22 | 21 | Existing statements moved to named stages or dead selector removed |
| `tests/test_centroid_fft_k_plan.py` | **File total** | 164 | 163 | Net -1 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_conv_kpair_plan.py` | `test_shared_downfold_cq_enters_the_conv_plan` | 5 | 5 | Existing statements moved to named stages or dead selector removed |
| `tests/test_conv_kpair_plan.py` | **File total** | 157 | 157 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_downfold.py` | `test_RED_TWIN_the_raw_kernel_labels_the_gram_by_MINUS_q` | 36 | 36 | Existing statements moved to named stages or dead selector removed |
| `tests/test_downfold.py` | **File total** | 1816 | 1816 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_gflat_planner_face_pairs.py` | `_synthetic_plan` | 11 | 11 | Existing statements moved to named stages or dead selector removed |
| `tests/test_gflat_planner_face_pairs.py` | `_run50_plan` | 16 | 16 | Existing statements moved to named stages or dead selector removed |
| `tests/test_gflat_planner_face_pairs.py` | `_profile_cliff_plan` | 11 | 11 | Existing statements moved to named stages or dead selector removed |
| `tests/test_gflat_planner_face_pairs.py` | **File total** | 260 | 260 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_gflat_planner_q_accounting.py` | `test_plan_receipt_distinguishes_full_K_from_selected_Q` | 20 | 20 | Existing statements moved to named stages or dead selector removed |
| `tests/test_gflat_planner_q_accounting.py` | **File total** | 55 | 55 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_isdf_cq_face_parity.py` | `check_shared_cq` | 31 | 31 | Existing statements moved to named stages or dead selector removed |
| `tests/test_isdf_cq_face_parity.py` | **File total** | 138 | 138 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_isdf_zq_parent_parity.py` | `_worker` | 259 | 259 | Existing statements moved to named stages or dead selector removed |
| `tests/test_isdf_zq_parent_parity.py` | **File total** | 408 | 408 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_wavefunction_bundle_face_carrier.py` | `test_memory_model_prices_resolved_layout` | 19 | 11 | Existing statements moved to named stages or dead selector removed |
| `tests/test_wavefunction_bundle_face_carrier.py` | **File total** | 204 | 196 | Net -8 lines |

| Tree scope | Before | After | Net |
|---|---|---|---|
| Requested production roots, tracked Python; baseline `00_inventory/before.json` | 125247 | 125280 | 33 |

| Pushed batch | Claim | CPU scope/result | P4 printed identity | Evidence |
|---|---|---|---|---|
| 1 — planner (25d2e75e) | 1395 | 473 passed,2 skipped,1 xfailed; centroid/parent/planner/physics/symmetry | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-174825-2182563-9864; 06_planner_cpu/cpu.xml; allocation58000949 lx-Xg4-172154-2062898-2839; 02_face_planner_cpu4/identity.json |
| 2 — direct Gram (9b41e422) | 1396 | 543 passed,2 skipped,1 xfailed; centroid/parent/Gram/downfold/physics/symmetry | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-175412-2208707-1724, lx-Xg4-175656-2221647-2270; 07_gram_cpu/cpu.xml; 07_gram_p4_ready/mos2/identity.json |
| 3 — symmetry stages (c604c3d6) | 1397 | 445 passed,2 skipped,1 xfailed; centroid/parent/physics/symmetry | Si SOC GN exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-180057-2240310-9849, lx-Xg4-180106-2241405-3156; 09_symmetry_stages/cpu/cpu.xml; 09_symmetry_stages/p4/si_soc/identity.json |
| 4 — driver/photon stages (this commit) | 1398 | 467 passed,2 skipped,1 xfailed; centroid/parent/photon/physics/symmetry | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-180746-2273070-8918, lx-Xg4-180754-2273585-4383; 10_photon_stages/cpu/cpu.xml; 10_photon_stages/p4/mos2/identity.json |

| Deviation / open gate | Reason / disposition |
|---|---|
| Historical MoS2 run73 reference superseded | Owner instructed fresh untouched-tip references; expected covariant Gamma price maximum4.647 microeV is documented in sandbox claim1201; no tolerance relaxed |
| Scalar reference incomplete | Two untouched-source attempts stopped after no post-screening output; first >16min, native stacks in CUDA/NCCL. `00_references/si_scalar_attempt01`, `si_scalar`, `stack_1606206.log`, `stop_retry.log`; allocation58001753. Source cause unisolated, registered in sandbox KNOWN_LORRAX_ISSUES.md |
| Additional launch attempts | 07_gram_p4 and 07_gram_p4_retry refused pool occupancy; no numerical execution. Completed accepted leg:07_gram_p4_ready/mos2 |
| Live rectangular Gram | Downfold uses this calculation; retain it under c_q_downfold and remove only the wrapper selector |
| Live legacy band projector | BSE ring consumers and common/zeta_projection still call it; no blanket removal |
| Live absent-plan Green seam | photon_sigma has already unfolded mixed endpoints before build_G; applying another plan would unfold twice |
| Captured symmetry data in identity caches | Shape equality alone does not authenticate captured action tables; no unsafe shape-only replacement |
| Strict-key legacy exemptions | read_lorrax_input includes _LEGACY_DECK_KEYS in _known; dropping only explicit refusals would silently accept retired keys |
| First push parent synchronization | Fetched parent immediately before first push; still c2f69987, so no rebase needed. No rebases after first push |
| Final CPU/core/P16/compile-event gates | Pending; no final-suite or performance claim |
| CPU skips/xfail | Host FFT FFI unavailable; not WSL; existing CPU partitioned-max NaN xfail. Explicit scoped CPU logs own details |
