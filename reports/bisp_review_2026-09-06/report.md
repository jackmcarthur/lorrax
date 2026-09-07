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
| `src/gw/gw_jax.py` | `main` | 1448 | 118 | Existing statements moved to named stages or dead selector removed |
| `src/gw/gw_jax.py` | `main._config_print` | 5 | 0 | Deleted or renamed to direct owner |
| `src/gw/gw_jax.py` | `_open_production_report` | 0 | 27 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_open_production_report._config_print` | 0 | 5 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_report_head_and_photon_policy` | 0 | 46 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_load_system_inputs` | 0 | 40 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_prepare_band_metadata` | 0 | 41 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_report_sampling_and_bands` | 0 | 29 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_prepare_isdf_carriers` | 0 | 54 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_prepare_oneshot_response` | 0 | 57 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_report_packed_screening` | 0 | 57 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_run_oneshot_screening` | 0 | 89 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_install_oneshot_head` | 0 | 26 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_persist_screening` | 0 | 15 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_prepare_static_head` | 0 | 15 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_run_oneshot_sigma` | 0 | 51 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_load_kinetic_ionic_hamiltonian` | 0 | 20 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_solve_qp_stage` | 0 | 56 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_sigma_output_fields` | 0 | 84 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_diagonalize_qp_hamiltonian` | 0 | 28 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_sigma_diagnostic_fields` | 0 | 84 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_assemble_gw_results` | 0 | 56 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_write_gw_results` | 0 | 43 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_close_timing` | 0 | 13 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_report_final_observables` | 0 | 72 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | `_report_file_rows` | 0 | 45 | New explicit phase/direct owner |
| `src/gw/gw_jax.py` | **File total** | 1712 | 1476 | Net -236 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/isdf_fitting.py` | `fit_zeta_to_h5` | 1366 | 124 | Existing statements moved to named stages or dead selector removed |
| `src/gw/isdf_fitting.py` | `_prepare_zeta_fit_geometry` | 0 | 94 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | `_build_zeta_fit_gram` | 0 | 60 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | `_report_zeta_factor_route` | 0 | 41 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | `_factor_zeta_fit_gram` | 0 | 94 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | `_prepare_zeta_output_sphere` | 0 | 51 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | `_open_zeta_fit_output` | 0 | 56 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | `_prepare_zeta_fit_wavefunctions` | 0 | 58 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | `_prepare_zeta_accumulator` | 0 | 55 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | `_prepare_coupled_zeta_tile` | 0 | 71 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | `_fit_zeta_tile` | 0 | 43 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | `_accumulate_zeta_tile` | 0 | 31 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | `_report_zeta_tile_memory` | 0 | 27 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | `_run_zeta_fit_tiles` | 0 | 95 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | `_write_zeta_fit_result` | 0 | 24 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | `_close_zeta_fit_output` | 0 | 32 | New explicit phase/direct owner |
| `src/gw/isdf_fitting.py` | **File total** | 1542 | 1162 | Net -380 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/sigma_dispatch.py` | `compute_sigma_xc` | 809 | 809 | Unchanged; target remains |
| `src/gw/sigma_dispatch.py` | **File total** | 1604 | 1604 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/gw_init.py` | `_plan_gflat_chunks_for_channel` | 136 | 135 | Existing statements moved to named stages or dead selector removed |
| `src/gw/gw_init.py` | `fit_zeta` | 576 | 56 | Existing statements moved to named stages or dead selector removed |
| `src/gw/gw_init.py` | `compute_V_q` | 398 | 22 | Existing statements moved to named stages or dead selector removed |
| `src/gw/gw_init.py` | `prepare_isdf_and_wavefunctions` | 742 | 742 | Unchanged; target remains |
| `src/gw/gw_init.py` | `_report_zeta_chunk_plan` | 0 | 16 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_reuse_zeta_faces` | 0 | 27 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_plan_transverse_zeta` | 0 | 28 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_plan_coupled_zeta_fit` | 0 | 108 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_fit_charge_zeta_channel` | 0 | 74 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_report_zeta_fit_peak` | 0 | 33 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_transverse_zeta_channel_runner` | 0 | 80 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_transverse_zeta_channel_runner._drop_traced_caches` | 0 | 3 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_transverse_zeta_channel_runner._drain_coupled_rank_findings` | 0 | 5 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_transverse_zeta_channel_runner._fit_transverse_channel` | 0 | 62 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_run_transverse_zeta_schedule` | 0 | 47 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_fit_transverse_zeta_channels` | 0 | 28 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_vcoul_geometry_and_budget` | 0 | 19 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_vcoul_transverse_inputs` | 0 | 25 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_compute_photon_vq` | 0 | 90 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_compute_scalar_vq` | 0 | 35 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_finalize_vq_views` | 0 | 23 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | **File total** | 3786 | 3550 | Net -236 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/isdf/core.py` | `c_q_from_psi_sm` | 42 | 95 | Existing statements moved to named stages or dead selector removed |
| `src/isdf/core.py` | `_c_q_legacy` | 122 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `_c_q_face_parent` | 95 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `_z_q_face_parent` | 327 | 327 | Unchanged; target remains |
| `src/isdf/core.py` | `factor_c_q` | 317 | 317 | Unchanged; target remains |
| `src/isdf/core.py` | `solve_zeta` | 699 | 101 | Existing statements moved to named stages or dead selector removed |
| `src/isdf/core.py` | `solve_zeta._ridge_indef_solve` | 7 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._ridge_indef_solve._ridged_lu` | 4 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._lu_apply_logical` | 16 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._tri_solve_logical` | 13 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._tri_solve_logical._chol_backsolve` | 4 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._pinv_matmul_logical` | 9 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._pinv_matmul_logical._mm` | 2 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._pinv_apply_T_logical` | 8 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._pinv_apply_T_logical._mm` | 2 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `c_q_downfold` | 0 | 122 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers` | 0 | 61 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._ridge_indef_solve` | 0 | 7 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._ridge_indef_solve._ridged_lu` | 0 | 4 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._lu_apply_logical` | 0 | 16 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._tri_solve_logical` | 0 | 13 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._tri_solve_logical._chol_backsolve` | 0 | 4 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._pinv_matmul_logical` | 0 | 9 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._pinv_matmul_logical._mm` | 0 | 2 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._pinv_apply_T_logical` | 0 | 8 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._pinv_apply_T_logical._mm` | 0 | 2 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_batched_kernels` | 0 | 68 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_batched_kernels._sharded_cho_solve` | 0 | 11 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_batched_kernels._sharded_cho_solve_batch` | 0 | 23 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_batched_kernels._solve_batch_and_update` | 0 | 8 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_batched_kernels._solve_all_at_once` | 0 | 5 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_per_q_kernel` | 0 | 88 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_per_q_kernel._per_q_block` | 0 | 25 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_per_q_kernel._solve_one_q_and_update` | 0 | 16 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_rhs_resharder` | 0 | 21 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_rhs_resharder._reshard_z` | 0 | 3 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_cache_zeta_solve_kernels` | 0 | 24 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_apply_replicated_zeta` | 0 | 103 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_solve_zeta_replicated` | 0 | 103 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_solve_zeta_token` | 0 | 33 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_solve_zeta_token._run_token` | 0 | 8 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_solve_zeta_fused_lu` | 0 | 70 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_solve_zeta_fused_lu._dist_ridged_lu` | 0 | 27 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_solve_zeta_fused_lu._run_lu` | 0 | 8 | New explicit phase/direct owner |
| `src/isdf/core.py` | **File total** | 5831 | 5780 | Net -51 lines |

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
| `tests/test_bispinor_zeta_reuse_ast.py` | `_fit_zeta_tree` | 2 | 8 | Existing statements moved to named stages or dead selector removed |
| `tests/test_bispinor_zeta_reuse_ast.py` | `test_reuse_contract_precedes_and_bypasses_fit_only_planners` | 32 | 36 | Existing statements moved to named stages or dead selector removed |
| `tests/test_bispinor_zeta_reuse_ast.py` | **File total** | 290 | 300 | Net +10 lines |

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
| `tests/test_coupled_mu123_gflat_host_spill.py` | `test_production_lifetime_spills_before_prepared_and_around_accumulate` | 18 | 22 | Existing statements moved to named stages or dead selector removed |
| `tests/test_coupled_mu123_gflat_host_spill.py` | `test_host_spill_is_automatic_and_only_threads_through_coupled_route` | 8 | 8 | Existing statements moved to named stages or dead selector removed |
| `tests/test_coupled_mu123_gflat_host_spill.py` | `test_automatic_policy_keeps_fragmentation_platform_and_host_gates` | 9 | 9 | Existing statements moved to named stages or dead selector removed |
| `tests/test_coupled_mu123_gflat_host_spill.py` | **File total** | 108 | 113 | Net +5 lines |

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

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_zeta_slice_before_commit.py` | `test_c_and_prebuilt_z_select_before_outer_block_until_ready` | 22 | 22 | Existing statements moved to named stages or dead selector removed |
| `tests/test_zeta_slice_before_commit.py` | **File total** | 59 | 59 | Net +0 lines |

| Tree scope | Before | After | Net |
|---|---|---|---|
| Requested production roots, tracked Python; baseline `00_inventory/before.json` | 125247 | 124322 | -925 |

| Pushed batch | Claim | CPU scope/result | P4 printed identity | Evidence |
|---|---|---|---|---|
| 1 — planner (25d2e75e) | 1395 | 473 passed,2 skipped,1 xfailed; centroid/parent/planner/physics/symmetry | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-174825-2182563-9864; 06_planner_cpu/cpu.xml; allocation58000949 lx-Xg4-172154-2062898-2839; 02_face_planner_cpu4/identity.json |
| 2 — direct Gram (9b41e422) | 1396 | 543 passed,2 skipped,1 xfailed; centroid/parent/Gram/downfold/physics/symmetry | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-175412-2208707-1724, lx-Xg4-175656-2221647-2270; 07_gram_cpu/cpu.xml; 07_gram_p4_ready/mos2/identity.json |
| 3 — symmetry stages (c604c3d6) | 1397 | 445 passed,2 skipped,1 xfailed; centroid/parent/physics/symmetry | Si SOC GN exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-180057-2240310-9849, lx-Xg4-180106-2241405-3156; 09_symmetry_stages/cpu/cpu.xml; 09_symmetry_stages/p4/si_soc/identity.json |
| 4 — driver/photon stages (76a1587d) | 1398 | 467 passed,2 skipped,1 xfailed; centroid/parent/photon/physics/symmetry | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-180746-2273070-8918, lx-Xg4-180754-2273585-4383; 10_photon_stages/cpu/cpu.xml; 10_photon_stages/p4/mos2/identity.json |
| 5 — zeta stages (b3487462) | 1399 | 485 passed,2 skipped,1 xfailed; parent/factor-hoist/mesh-invariance/charge/refit/physics/symmetry | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-181422-2302374-7638, lx-Xg4-181430-2302826-9451; 11_zeta_stages/cpu/cpu.xml; 11_zeta_stages/p4/mos2/identity.json |
| 6 — complete driver stages (208ed323) | 1401 | 456 passed,2 skipped,1 xfailed; centroid closure/parent/physics/symmetry | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-183807-44846-3326, lx-Xg4-183816-45219-2050; 14_driver_stages/cpu/cpu.xml; 14_driver_stages/p4/mos2/identity.json |
| 7 — Vq stages (f579b808) | 1403 | 445 passed, 2 skipped, 1 xfailed; centroid/parent/physics/symmetry | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-184246-65473-4941, lx-Xg4-184258-66104-4580; 15_vq_stages/cpu/cpu.xml; 15_vq_stages/p4/mos2/identity.json |
| 8 — extract charge and current fit stages (36b46cb0) | 1404 | 453 passed, 2 skipped, 1 xfailed, 102 warnings in 177.56s (0:02:57) | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-185204-112340-9401, lx-Xg4-185204-112565-5557; 16_fit_stages/cpu/cpu.xml; 16_fit_stages/p4/mos2/identity.json |
| 9 — extract zeta writer and tile stages (this commit) | 1405 | 465 passed, 2 skipped, 1 xfailed, 102 warnings in 247.62s (0:04:07) | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-190233-157458-4126, lx-Xg4-190235-157617-6713; 18_fit_writer_stages/cpu/cpu.xml; 18_fit_writer_stages/p4/mos2/identity.json |

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
| Reverted centroid-loader stage cut | 12_centroid_stages/verdict.txt: P4 exact; CPU1 failed/460 passed/2 skipped/1 xfailed because test_centroid_fft_k_plan requires in-entry calls. Source and contract relocation reverted |
| Reverted Sigma stage cut | 13_sigma_stages/verdict.txt: P4 exact; CPU9 failed/496 passed/2 skipped/3 xfailed: four nb=3 fixtures incompatible with CPU4 band sharding, four absent-host-FFI failures, one entry-body structural guard. Source/tests/contract relocation reverted |
| Reverted initializer stage cut | 17_prepare_stages/verdict.txt: CPU476 passed/2 skipped/1 xfailed; P4 failed because the extracted fresh-carrier stage omitted _parent_green_faces. Source/tests/contract relocation reverted |
| Final CPU/core/P16/compile-event gates | Pending; no final-suite or performance claim |
| CPU skips/xfail | Host FFT FFI unavailable; not WSL; existing CPU partitioned-max NaN xfail. Explicit scoped CPU logs own details |
