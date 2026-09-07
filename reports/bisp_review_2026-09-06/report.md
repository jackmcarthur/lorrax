| Lane | Branch | State | Evidence root |
|---|---|---|---|
| BISP-REVIEW | `refactor/bisp-compaction-2026-09-06` | REVIEW CUTS GATED; scalar/CPU environment and global length targets remain open | /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/117_bisp_review_codex_2026-09-06 |

| Item | Status |
|---|---|
| 1 — dead dispatch | Owner-required two-layout planner and caller restoration pushed; direct Gram cut retained; live absent-plan seams documented |
| 2 — phase extraction | Nine named entry stages extracted; writer reverted after fresh-fit failure; global 120-line target remains unmet (142 functions) |
| 3 — duplication | Sigma operand selector, shared Gram shape/refusal rules, first mesh-key cut and fit-price reuse pushed; broad cache cut reverted after missing-host-FFI gate failures |
| 4 — docstrings | Six modules below 3.2%; complete contracts relocated with executable AST identity |
| 5 — configuration | Table-driven parsing and typed envelope stages gated; legacy refusals retained because strict-key exemptions still apply |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/gflat_memory_model.py` | `plan_gflat_chunks` | 731 | 731 | Unchanged; target remains |
| `src/gw/gflat_memory_model.py` | **File total** | 1451 | 1451 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/gw_jax.py` | `main` | 1448 | 118 | Stages in this module: `_open_production_report`, `_report_head_and_photon_policy`, `_load_system_inputs`, `_prepare_band_metadata`, `_report_sampling_and_bands`, `_prepare_isdf_carriers`, `_prepare_oneshot_response`, `_run_oneshot_screening`, `_install_oneshot_head`, `_persist_screening`, `_prepare_static_head`, `_run_oneshot_sigma`, `_load_kinetic_ionic_hamiltonian`, `_solve_qp_stage`, `_sigma_output_fields`, `_diagonalize_qp_hamiltonian`, `_sigma_diagnostic_fields`, `_assemble_gw_results`, `_write_gw_results`, `_close_timing`, `_report_final_observables`, `_report_file_rows`; narrative: docs/architecture/decisions.md |
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
| `src/gw/gw_jax.py` | Docstring lines / share | 70 / 4.09% | 94 / 6.37% | Owner: docs/architecture/decisions.md |
| `src/gw/gw_jax.py` | **File total** | 1712 | 1476 | Net -236 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/isdf_fitting.py` | `fit_zeta_to_h5` | 1366 | 1366 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/gw/isdf_fitting.py` | Docstring lines / share | 29 / 1.88% | 29 / 1.88% | Owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/gw/isdf_fitting.py` | **File total** | 1542 | 1542 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/sigma_dispatch.py` | `compute_sigma_xc` | 809 | 63 | Stages in this module: `_validate_sigma_stage`, `_static_sigma_channels`, `_sigma_hartree_fields`, `_static_sigma_result`, `_compute_mpa_sigma`, `_compute_ppm_sigma`; narrative: docs/architecture/decisions.md |
| `src/gw/sigma_dispatch.py` | `_validate_sigma_stage` | 0 | 69 | New explicit phase/direct owner |
| `src/gw/sigma_dispatch.py` | `_packed_static_sigma_channels` | 0 | 47 | New explicit phase/direct owner |
| `src/gw/sigma_dispatch.py` | `_packed_dynamic_sigma_channels` | 0 | 68 | New explicit phase/direct owner |
| `src/gw/sigma_dispatch.py` | `_static_sigma_channels` | 0 | 79 | New explicit phase/direct owner |
| `src/gw/sigma_dispatch.py` | `_sigma_hartree_fields` | 0 | 26 | New explicit phase/direct owner |
| `src/gw/sigma_dispatch.py` | `_static_sigma_result` | 0 | 33 | New explicit phase/direct owner |
| `src/gw/sigma_dispatch.py` | `_compute_mpa_sigma` | 0 | 101 | New explicit phase/direct owner |
| `src/gw/sigma_dispatch.py` | `_compute_ppm_sigma` | 0 | 62 | New explicit phase/direct owner |
| `src/gw/sigma_dispatch.py` | **File total** | 1604 | 1359 | Net -245 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/gw_init.py` | `fit_zeta` | 576 | 56 | Stages in this module: `_report_zeta_chunk_plan`, `_reuse_zeta_faces`, `_plan_transverse_zeta`, `_plan_coupled_zeta_fit`, `_fit_charge_zeta_channel`, `_report_zeta_fit_peak`, `_fit_transverse_zeta_channels`; narrative: docs/architecture/zeta_fit_face_psi_cct.md / four_current_wiring.md |
| `src/gw/gw_init.py` | `compute_V_q` | 398 | 22 | Stages in this module: `_vcoul_geometry_and_budget`, `_vcoul_transverse_inputs`, `_compute_photon_vq`, `_compute_scalar_vq`, `_finalize_vq_views`; narrative: docs/architecture/zeta_fit_face_psi_cct.md / four_current_wiring.md |
| `src/gw/gw_init.py` | `prepare_isdf_and_wavefunctions` | 742 | 78 | Stages in this module: `_prepare_fresh_isdf`, `_prepare_restart_isdf`; narrative: docs/architecture/zeta_fit_face_psi_cct.md / four_current_wiring.md |
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
| `src/gw/gw_init.py` | `_prepare_fresh_parent_faces` | 0 | 46 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_prepare_fitted_zeta` | 0 | 44 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_prepare_fresh_carriers` | 0 | 57 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_prepare_fresh_coulomb` | 0 | 19 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_write_fresh_restart` | 0 | 70 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_prepare_fresh_isdf` | 0 | 43 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_read_authenticated_restart` | 0 | 34 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_restart_charge_basis` | 0 | 61 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_restart_charge_carrier` | 0 | 59 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_restart_current_carrier` | 0 | 106 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_restart_gamma_vectors` | 0 | 24 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | `_prepare_restart_isdf` | 0 | 45 | New explicit phase/direct owner |
| `src/gw/gw_init.py` | Docstring lines / share | 504 / 13.31% | 504 / 14.37% | Owner: docs/architecture/zeta_fit_face_psi_cct.md / four_current_wiring.md |
| `src/gw/gw_init.py` | **File total** | 3786 | 3507 | Net -279 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/isdf/core.py` | `host_rss_gb` | 17 | 10 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `complete_ordered_pair_normal_equations` | 30 | 19 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_conv_kpair_static_gamma` | 14 | 8 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_pair_density_kernel` | 24 | 24 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `pair_density` | 24 | 11 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `pair_density_aot_peak_bytes` | 33 | 26 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_gram_q0_kernel` | 43 | 43 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `gram_q0_from_pair` | 66 | 31 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `transverse_gram_q0_from_pair` | 39 | 24 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_gram_q0_from_psi_kernel` | 63 | 63 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `gram_q0_from_psi_sm` | 65 | 30 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_gram_q0_tiled_from_psi_kernel` | 154 | 146 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `gram_q0_tiled_from_psi_sm` | 88 | 50 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `gram_q0_tiled_from_psi_aot_resident_increment_bytes` | 72 | 58 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `gram_q0_from_psi_aot_peak_bytes` | 48 | 41 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `gram_q0_aot_peak_bytes` | 39 | 32 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `c_q_from_psi_sm` | 42 | 95 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_c_q_legacy` | 122 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `_c_q_face_parent` | 95 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `build_psi_r_cache_sm` | 70 | 59 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_band_chunk_compaction` | 30 | 24 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_z_q_face_parent` | 327 | 327 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_identity_pad_block_diagonal` | 66 | 23 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_replicate_charge_ok` | 13 | 5 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_replicate_rank_truncate_ok` | 36 | 8 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_rank_truncate_capacity_error` | 63 | 39 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_resolve_channel_ladder` | 51 | 27 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_resolve_solver_kind_charge` | 116 | 73 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_resolve_solver_kind_transverse` | 190 | 115 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_resolve_solver_kind` | 37 | 22 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_env_override_raw` | 12 | 6 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `deprecated_env_record` | 9 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_deprecated_env_float` | 21 | 16 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_resolve_zeta_gather` | 122 | 67 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_close_the_cut` | 73 | 36 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_certify_the_cut` | 117 | 58 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_close_the_cut_padded` | 35 | 8 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_withdraw_identity_pad` | 17 | 8 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_charge_factor_math` | 156 | 137 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `solve_zeta_charge_dense` | 66 | 24 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_factor_c_q_replicated` | 116 | 69 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `factor_c_q_replicated_batched` | 29 | 17 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_qparallel_factor_ok` | 19 | 11 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_factor_c_q_replicated_qparallel` | 121 | 80 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_certify_transverse_ridge` | 78 | 50 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_embed_lu_padded` | 14 | 10 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_factor_c_q_transverse_lu` | 127 | 107 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_factor_c_q_transverse_distributed_lu` | 59 | 36 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_distributed_q_batch` | 8 | 3 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_collective_chunk_bytes` | 16 | 10 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_chunk_q` | 11 | 5 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_chunk_log` | 63 | 57 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_factor_c_q_distributed_rank_truncate` | 246 | 195 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_distributed_pinv_apply` | 99 | 71 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `factor_c_q` | 317 | 229 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_reshard_zeta_mu_X_r_Y_to_mu_XY` | 23 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_distributed_backsolve` | 36 | 9 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_reshard_zeta_r_XY_to_mu_XY` | 18 | 6 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_factor_nbatch` | 25 | 5 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `solve_zeta` | 699 | 101 | Stages in this module: `_solve_zeta_token`, `_solve_zeta_fused_lu`, `_solve_zeta_replicated`; narrative: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `solve_zeta._ridge_indef_solve` | 7 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._ridge_indef_solve._ridged_lu` | 4 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._lu_apply_logical` | 16 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._tri_solve_logical` | 13 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._tri_solve_logical._chol_backsolve` | 4 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._pinv_matmul_logical` | 9 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._pinv_matmul_logical._mm` | 2 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._pinv_apply_T_logical` | 8 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `solve_zeta._pinv_apply_T_logical._mm` | 2 | 0 | Deleted or renamed to direct owner |
| `src/isdf/core.py` | `fit_one_rchunk` | 137 | 137 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_band_norms_slice` | 21 | 14 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | `_gram_gamma_mode` | 0 | 7 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_gram_planning_gamma_mode` | 0 | 8 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_validate_gram_face_shapes` | 0 | 27 | New explicit phase/direct owner |
| `src/isdf/core.py` | `c_q_downfold` | 0 | 122 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers` | 0 | 39 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._ridge_indef_solve` | 0 | 7 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._ridge_indef_solve._ridged_lu` | 0 | 4 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._lu_apply_logical` | 0 | 7 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._tri_solve_logical` | 0 | 7 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._tri_solve_logical._chol_backsolve` | 0 | 4 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._pinv_matmul_logical` | 0 | 5 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._pinv_matmul_logical._mm` | 0 | 2 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._pinv_apply_T_logical` | 0 | 5 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_logical_solvers._pinv_apply_T_logical._mm` | 0 | 2 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_batched_kernels` | 0 | 68 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_batched_kernels._sharded_cho_solve` | 0 | 11 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_batched_kernels._sharded_cho_solve_batch` | 0 | 23 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_batched_kernels._solve_batch_and_update` | 0 | 8 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_batched_kernels._solve_all_at_once` | 0 | 5 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_per_q_kernel` | 0 | 79 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_per_q_kernel._per_q_block` | 0 | 25 | New explicit phase/direct owner |
| `src/isdf/core.py` | `_zeta_per_q_kernel._solve_one_q_and_update` | 0 | 7 | New explicit phase/direct owner |
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
| `src/isdf/core.py` | Docstring lines / share | 1365 / 23.41% | 110 / 2.40% | Owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/isdf/core.py` | **File total** | 5831 | 4585 | Net -1246 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/common/wfn_transforms.py` | `_cached_gindex_dev` | 54 | 36 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `_resolve_gindex_dev` | 49 | 18 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `_box_kernel` | 29 | 22 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `_spec_of` | 16 | 9 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `_local_box_fft` | 15 | 8 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `_sharding_key` | 17 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `to_box` | 29 | 24 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `to_rbox` | 46 | 38 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `from_rbox` | 77 | 51 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `to_rmu` | 49 | 44 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `to_rchunk_inner` | 74 | 27 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `to_rpoints_inner` | 59 | 25 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `take_rchunk_padded` | 36 | 28 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `to_rchunk` | 102 | 86 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `gflat_to_rchunk_aot_memory` | 113 | 93 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `gflat_to_rchunk_aot_peak_bytes` | 22 | 17 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `to_rmu_inner` | 59 | 19 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `gflat_to_rmu` | 362 | 267 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `accumulate_rchunk_to_gflat` | 284 | 189 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `apply_bloch_phase` | 38 | 28 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `apply_bloch_phase_at` | 58 | 40 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `apply_bloch_phase_on_slice` | 35 | 14 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `_refuse_spinor_zero_fill` | 34 | 11 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `load_kpoint_fftbox_local` | 32 | 16 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `load_kpoint_fftbox` | 19 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `get_enk_bandrange` | 74 | 45 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `read_Gvecs_to_devices` | 45 | 30 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `load_psi_gflat_padded` | 52 | 32 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `prepare_rchunk_carrier` | 68 | 56 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `iter_psi_rchunk_bandwise` | 141 | 107 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `load_centroids_band_chunked` | 683 | 51 | Stages in this module: `_centroid_sampling_geometry`, `_centroid_sampling_shardings`, `_centroid_stream_geometry`, `_centroid_resident_bytes`, `_centroid_fft_scan_chunk`, `_centroid_sampling_indices`, `_centroid_face_kernels`, `_load_streamed_centroid_faces`, `_load_bulk_centroid_faces`; narrative: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | `load_centroids_band_chunked._reshard_centroid_tile` | 16 | 0 | Deleted or renamed to direct owner |
| `src/common/wfn_transforms.py` | `load_centroids_band_chunked._finish_faces` | 25 | 0 | Deleted or renamed to direct owner |
| `src/common/wfn_transforms.py` | `_centroid_sampling_geometry` | 0 | 44 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_centroid_sampling_shardings` | 0 | 42 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_centroid_stream_geometry` | 0 | 50 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_centroid_resident_bytes` | 0 | 102 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_centroid_fft_scan_chunk` | 0 | 54 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_centroid_sampling_indices` | 0 | 12 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_centroid_face_kernels` | 0 | 52 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_centroid_face_kernels._reshard_centroid_tile` | 0 | 16 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_centroid_face_kernels._finish_faces` | 0 | 25 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_centroid_stream_kernels` | 0 | 58 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_centroid_stream_kernels._zero_faces` | 0 | 9 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_centroid_stream_kernels._zero_parent_faces` | 0 | 9 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_centroid_stream_kernels._insert_tile` | 0 | 8 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_centroid_stream_kernels._sample_and_insert_one` | 0 | 19 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_sample_centroid_parent_groups` | 0 | 74 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_sample_centroid_domain_tiles` | 0 | 52 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_load_streamed_centroid_faces` | 0 | 32 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | `_load_bulk_centroid_faces` | 0 | 102 | New explicit phase/direct owner |
| `src/common/wfn_transforms.py` | Docstring lines / share | 833 / 26.38% | 48 / 1.93% | Owner: docs/architecture/zeta_fit_face_psi_cct.md |
| `src/common/wfn_transforms.py` | **File total** | 3158 | 2486 | Net -672 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/gw_config.py` | `env_float` | 35 | 22 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `active_zeta_truncating_knobs` | 12 | 8 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `ComputeMode.is_dynamic` | 12 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `ComputeMode.ppm_model` | 12 | 6 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `announce_legacy_sigma_axis_keys` | 22 | 16 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `SigmaChannel.label` | 14 | 8 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `coerce_compute_mode` | 24 | 12 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `coerce_screening_diagrams` | 20 | 13 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `refuse_unsupported_screening_diagrams` | 38 | 19 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `explain_missing_channels` | 15 | 10 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `refuse_unimplemented_compute_mode` | 18 | 10 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `normalize_w_dyson_solver` | 43 | 28 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `eigh_backend_choices` | 42 | 20 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `resolve_linalg` | 44 | 39 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `distrib_la_batched_route_choices` | 19 | 11 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `BandCounts.describe` | 34 | 14 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `resolve_band_counts` | 110 | 73 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `resolve_band_extrapolation` | 52 | 26 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `sigma_stage_modes` | 74 | 28 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `band_extrapolation_is_consumable` | 9 | 3 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `_deck_key_line` | 11 | 7 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `_print_deck_report` | 15 | 9 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `read_lorrax_input` | 378 | 29 | Stages in this module: `_locate_input_blocks`, `_read_input_section`, `_report_early_retired_keys`, `_report_remaining_retired_keys`, `_refuse_unknown_input_keys`, `_parse_input_keys`, `_parse_input_kpoints`; narrative: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `_normalize_placement` | 13 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `scalar_head_overrides_named` | 14 | 9 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `packed_static_envelope` | 76 | 47 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `packed_bare_transverse_route` | 52 | 26 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `packed_photon_screens_current` | 12 | 5 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `uses_static_photon_response` | 12 | 7 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `packed_photon_replaces_charge_sigma` | 19 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `uses_dynamic_packed_photon_route` | 12 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `uses_coupled_photon_head` | 14 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `incumbent_bispinor_head_record` | 39 | 24 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `refuse_unsupported_bispinor_gw` | 122 | 116 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `refuse_unsupported_bispinor_tt_head_correction` | 57 | 35 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `DynamicSigmaConfig.parsed_omega_patches_ev` | 35 | 29 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `MPAConfig.sample_plan` | 24 | 19 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `_validate_occupation_smearing` | 50 | 41 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `resolve_mpa_sampling_alpha` | 28 | 24 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `LorraxConfig.occ_broadening_ry` | 37 | 5 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `LorraxConfig.compute_mode` | 49 | 32 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `LorraxConfig.qp_solver` | 48 | 29 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `LorraxConfig.omega_grid_ev` | 31 | 21 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `LorraxConfig.from_input_file` | 584 | 41 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | `LorraxConfig.from_input_file._g` | 2 | 0 | Deleted or renamed to direct owner |
| `src/gw/gw_config.py` | `LorraxConfig.from_input_file._sc_env` | 9 | 0 | Deleted or renamed to direct owner |
| `src/gw/gw_config.py` | `_input_key_type` | 0 | 15 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_normalize_input_string` | 0 | 3 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_parse_input_keys` | 0 | 20 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_resolve_input_memory` | 0 | 15 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_resolve_input_metal_policy` | 0 | 59 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_input_paths` | 0 | 20 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_input_head` | 0 | 50 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_input_response` | 0 | 92 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_input_iteration` | 0 | 47 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_input_iteration._sc_env` | 0 | 9 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_input_memory_group` | 0 | 19 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_input_backend` | 0 | 44 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_input_storage` | 0 | 38 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_input_band_windows` | 0 | 24 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_assemble_input_config` | 0 | 51 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_apply_input_envelope` | 0 | 26 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_locate_input_blocks` | 0 | 27 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_read_input_section` | 0 | 15 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_report_early_retired_keys` | 0 | 91 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_report_remaining_retired_keys` | 0 | 68 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_refuse_unknown_input_keys` | 0 | 15 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | `_parse_input_kpoints` | 0 | 34 | New explicit phase/direct owner |
| `src/gw/gw_config.py` | Docstring lines / share | 1002 / 17.94% | 116 / 2.51% | Owner: docs/architecture/decisions.md |
| `src/gw/gw_config.py` | **File total** | 5586 | 4622 | Net -964 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `kgrid_shift_map` | 47 | 18 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `bgw_signed_q_representative` | 26 | 19 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `bgw_integer_q_to_fractional` | 28 | 22 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `q_negation_index` | 22 | 16 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `common_uniform_grid_indices` | 54 | 38 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `find_irreducible_bz_points` | 90 | 71 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `map_full_kpoints_to_irreducible` | 58 | 47 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `build_spatial_operator_tables` | 36 | 31 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `slice_q_full_to_ibz` | 49 | 7 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `unfold_isdf_operator` | 412 | 301 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `_apply_unfold_phase_and_trs_local` | 24 | 15 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `unfold_operator_local` | 72 | 46 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `unfold_wavefunction_local` | 79 | 51 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `open_spin_block_coefficient` | 13 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `_get_unfold_isdf_operator_jit` | 172 | 165 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `_rotate_open_spin_centroid_operator` | 48 | 13 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `unfold_spin_centroid_operator` | 137 | 143 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `unfold_isdf_one_leg` | 283 | 212 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `_get_unfold_isdf_one_leg_jit` | 83 | 76 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `apply_spinor_rotation` | 48 | 30 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `tau_phase_row` | 33 | 8 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `tau_phase_row_jax` | 17 | 11 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `unfold_reciprocal_carriers` | 14 | 9 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `unfold_psi` | 195 | 76 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps.__init__` | 488 | 10 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps.create_kpoint_symmetry_map` | 28 | 3 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps.syms_crystal_to_cartesian` | 72 | 13 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps.q_irr_is_full_identity` | 19 | 12 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps.operation_rows` | 22 | 17 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps.cartesian_action` | 24 | 18 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps.get_spinor_rotations` | 85 | 73 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps.get_kminusq_map` | 13 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._get_kminusq_index_map` | 55 | 47 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps.get_umklapp_vector` | 24 | 18 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps.find_qpoint_index` | 36 | 28 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `_jit_with` | 8 | 3 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `_row_out_sharding` | 15 | 7 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `_scalar_out_sharding` | 12 | 6 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `_star_row_order` | 19 | 7 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `_broadcast_rows` | 40 | 32 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `_star_conj_flags` | 26 | 8 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `_spread_tables` | 14 | 6 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `_star_stats` | 44 | 38 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `star_select` | 10 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `star_broadcast` | 98 | 39 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `star_tables_of` | 19 | 5 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `unfold_file_wedge_to_full_bz` | 43 | 14 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `unfold_file_wedge_band_operator` | 35 | 11 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `unfold_file_wedge_polar_matrix` | 38 | 21 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `reduce_full_bz_to_file_wedge` | 23 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `unfold_star_wedge_to_full_bz` | 14 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `star_spread` | 16 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `KStarMap.select` | 7 | 3 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `KStarMap.spread` | 9 | 4 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `KStarMap.spread_rel` | 11 | 5 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._initialize_symmetry_provenance` | 0 | 48 | New explicit phase/direct owner |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._initialize_active_operations` | 0 | 39 | New explicit phase/direct owner |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._validate_identity_grid` | 0 | 40 | New explicit phase/direct owner |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._initialize_identity_maps` | 0 | 104 | New explicit phase/direct owner |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._initialize_spatial_operators` | 0 | 98 | New explicit phase/direct owner |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._initialize_k_maps` | 0 | 96 | New explicit phase/direct owner |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | `SymMaps._initialize_q_maps` | 0 | 41 | New explicit phase/direct owner |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | Docstring lines / share | 1172 / 27.97% | 111 / 3.54% | Owner: docs/architecture/symmetry_register.md |
| `services/symmetry_maps/src/symmetry_maps/maps.py` | **File total** | 4190 | 3133 | Net -1057 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/w_isdf.py` | `_complete_static_vertex_orientations` | 24 | 6 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_get_chi_minimax_kernel` | 48 | 48 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_get_chi_minimax_kernel_face` | 223 | 223 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_get_chi_fractional_contour_kernel` | 28 | 28 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_get_w_solve_fn_local` | 142 | 123 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_get_w_solve_fn_distributed` | 201 | 151 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_w_residual_report` | 24 | 16 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_w_solve_pref_scalar` | 17 | 9 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_resolve_w_solve_fn` | 44 | 25 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_require_w_operand_geometry` | 32 | 26 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `solve_w` | 42 | 12 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `compute_chi0` | 51 | 33 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `compute_chi0_imag_ordered` | 45 | 18 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `compute_experimental_no_pair_photon_chi0` | 54 | 50 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_load_static_photon_hall` | 59 | 51 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `compute_static_photon_response` | 362 | 58 | Stages in this module: `_resolve_static_photon_policy`, `_read_static_photon_body`, `_report_static_photon_body`, `_screen_static_photon_body`, `_complete_static_photon_head`; narrative: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_chi0_multi_kernel_args` | 39 | 29 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_chi0_contour_alpha_rows` | 23 | 18 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `compute_chi0_contour` | 16 | 11 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `compute_chi0_contour_ordered` | 131 | 80 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_occupation_support_slices` | 49 | 25 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `compute_chi0_contour_fractional` | 39 | 28 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_fractional_pair_scan_face` | 147 | 139 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_fractional_pair_scan_face._gather_mun` | 14 | 11 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_fractional_pair_scan_face._gather_nmu` | 17 | 12 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_parent_face_unfold_operands` | 22 | 22 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `iter_parent_children_faces` | 56 | 56 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `iter_parent_children_faces._kernel` | 17 | 17 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_get_chi_static_fractional_gamma_kernel_face` | 52 | 52 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_get_chi_fractional_q_kernel_face` | 60 | 60 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `compute_chi0_static_fractional_gamma` | 78 | 63 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `occupation_support_bandwidth` | 17 | 8 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `compute_chi0_static_fractional` | 28 | 14 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `compute_chi0_direct_fractional` | 101 | 90 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `precompile_chi0` | 34 | 29 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `precompile_solve_w` | 23 | 19 | Existing statements moved to named stages or dead selector removed; narrative owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | `_resolve_static_photon_policy` | 0 | 92 | New explicit phase/direct owner |
| `src/gw/w_isdf.py` | `_read_static_photon_body` | 0 | 27 | New explicit phase/direct owner |
| `src/gw/w_isdf.py` | `_report_static_photon_body` | 0 | 26 | New explicit phase/direct owner |
| `src/gw/w_isdf.py` | `_screen_static_photon_body` | 0 | 85 | New explicit phase/direct owner |
| `src/gw/w_isdf.py` | `_complete_static_photon_head` | 0 | 60 | New explicit phase/direct owner |
| `src/gw/w_isdf.py` | Docstring lines / share | 512 / 16.67% | 73 / 2.72% | Owner: docs/architecture/four_current_wiring.md |
| `src/gw/w_isdf.py` | **File total** | 3072 | 2682 | Net -390 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/head_correction.py` | `static_hall_linear_response` | 37 | 28 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `canonicalize_static_gauge_q2_tensor` | 13 | 6 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `static_gauge_tensor_residuals` | 39 | 31 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `resolve_bgw_q0_channel` | 64 | 57 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `finite_q0_epsinv_head` | 41 | 29 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `_check_dipole_coverage` | 58 | 43 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `_dipole_window_from_params` | 43 | 18 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `_check_dipole_provenance` | 67 | 43 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `build_S_cart_omega` | 60 | 30 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `fold_small_head_wings_sharded` | 104 | 62 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `fold_cartesian_head_wings_sharded` | 17 | 12 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `small_head_wing_halves_sharded` | 45 | 34 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `_static_slab_photon_head_moment_chunk` | 79 | 61 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `static_slab_photon_head_moment_chunk` | 56 | 44 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `complete_static_slab_photon_q0` | 234 | 220 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `resolve_head_S_cart` | 57 | 30 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `fit_head_ppm` | 72 | 65 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `fit_head_ppm_from_samples` | 27 | 13 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `fit_head_hl_analytic` | 50 | 33 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `fit_head_with_fixed_omega` | 31 | 21 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `compute_static_head_terms` | 40 | 34 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `expand_band_diagonal_to_kij` | 19 | 14 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `static_head_terms_to_kij` | 30 | 13 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `compute_ppm_head_sigma_kij` | 68 | 26 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `compute_ppm_head_sigma_diag` | 46 | 37 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `on_shell_occupied_head_sigma_ry` | 39 | 19 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `compute_complex_pole_head_sigma_diag` | 62 | 44 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `apply_q0_head_rank1` | 34 | 22 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | `apply_q0_head_rank1_sharded` | 37 | 23 | Contract relocated to docs/architecture/four_current_wiring.md; executable AST unchanged |
| `src/gw/head_correction.py` | Docstring lines / share | 610 / 23.84% | 64 / 3.18% | Owner: docs/architecture/four_current_wiring.md |
| `src/gw/head_correction.py` | **File total** | 2559 | 2013 | Net -546 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/downfold.py` | `pair_density_gram` | 45 | 45 | Existing statements moved to named stages or dead selector removed |
| `src/gw/downfold.py` | **File total** | 1636 | 1636 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/common/contract_bands.py` | **File total** | 799 | 799 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/centroid_k_unfold.py` | `CentroidKUnfoldPlan.unfold_operator` | 21 | 24 | Existing statements moved to named stages or dead selector removed |
| `src/gw/centroid_k_unfold.py` | **File total** | 426 | 429 | Net +3 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/cohsex_sigma.py` | `_make_static_convolution` | 34 | 34 | Existing statements moved to named stages or dead selector removed |
| `src/gw/cohsex_sigma.py` | `_make_cohsex_kernels` | 28 | 28 | Existing statements moved to named stages or dead selector removed |
| `src/gw/cohsex_sigma.py` | `_make_cohsex_kernels_face` | 77 | 77 | Existing statements moved to named stages or dead selector removed |
| `src/gw/cohsex_sigma.py` | **File total** | 639 | 640 | Net +1 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/greens_function_kernel.py` | `_build_G_face` | 41 | 37 | Existing statements moved to named stages or dead selector removed |
| `src/gw/greens_function_kernel.py` | `build_G` | 23 | 22 | Existing statements moved to named stages or dead selector removed |
| `src/gw/greens_function_kernel.py` | `build_G_tau` | 30 | 30 | Existing statements moved to named stages or dead selector removed |
| `src/gw/greens_function_kernel.py` | **File total** | 181 | 180 | Net -1 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/photon_layout.py` | `_empty` | 12 | 12 | Existing statements moved to named stages or dead selector removed |
| `src/gw/photon_layout.py` | `_insert_program` | 37 | 37 | Existing statements moved to named stages or dead selector removed |
| `src/gw/photon_layout.py` | `_view_program` | 26 | 26 | Existing statements moved to named stages or dead selector removed |
| `src/gw/photon_layout.py` | `_vector_pack_program` | 47 | 47 | Existing statements moved to named stages or dead selector removed |
| `src/gw/photon_layout.py` | `_q0_update_program` | 52 | 52 | Existing statements moved to named stages or dead selector removed |
| `src/gw/photon_layout.py` | `_q0_block_program` | 55 | 55 | Existing statements moved to named stages or dead selector removed |
| `src/gw/photon_layout.py` | **File total** | 736 | 737 | Net +1 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/photon_sigma.py` | `_make_photon_static_class_kernel` | 51 | 44 | Existing statements moved to named stages or dead selector removed |
| `src/gw/photon_sigma.py` | `_make_photon_static_class_kernel.unfold` | 4 | 0 | Deleted or renamed to direct owner |
| `src/gw/photon_sigma.py` | `_make_photon_static_class_kernel.contract_class` | 10 | 12 | Existing statements moved to named stages or dead selector removed |
| `src/gw/photon_sigma.py` | **File total** | 302 | 296 | Net -6 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/ppm_sigma.py` | `_compute_invalid_static_sigma` | 107 | 107 | Existing statements moved to named stages or dead selector removed |
| `src/gw/ppm_sigma.py` | `_invalid_static_coh_by_bracket` | 83 | 83 | Existing statements moved to named stages or dead selector removed |
| `src/gw/ppm_sigma.py` | **File total** | 1005 | 1005 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/ppm_tau_kernel.py` | `_get_sigma_kij_kernel` | 159 | 138 | Existing statements moved to named stages or dead selector removed |
| `src/gw/ppm_tau_kernel.py` | `_get_sigma_kij_kernel.unfold` | 5 | 0 | Deleted or renamed to direct owner |
| `src/gw/ppm_tau_kernel.py` | `_get_sigma_kij_kernel._g_from_selector` | 19 | 15 | Existing statements moved to named stages or dead selector removed |
| `src/gw/ppm_tau_kernel.py` | **File total** | 561 | 540 | Net -21 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `src/gw/wavefunction_bundle.py` | `parent_sigma_operands` | 17 | 20 | Existing statements moved to named stages or dead selector removed |
| `src/gw/wavefunction_bundle.py` | **File total** | 886 | 889 | Net +3 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/multi_device/bispinor_physics_oracles.py` | `test_isdf_current_signed_normal_matrix_against_literal_pair_gram` | 30 | 30 | Existing statements moved to named stages or dead selector removed |
| `tests/multi_device/bispinor_physics_oracles.py` | **File total** | 892 | 892 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_bispinor_dynamic_packed_route.py` | `test_the_dispatch_asks_for_the_current_blocks_only` | 9 | 9 | Existing statements moved to named stages or dead selector removed |
| `tests/test_bispinor_dynamic_packed_route.py` | **File total** | 232 | 232 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_bispinor_zeta_reuse_ast.py` | `_fit_zeta_tree` | 2 | 8 | Existing statements moved to named stages or dead selector removed |
| `tests/test_bispinor_zeta_reuse_ast.py` | `test_reuse_contract_precedes_and_bypasses_fit_only_planners` | 32 | 36 | Existing statements moved to named stages or dead selector removed |
| `tests/test_bispinor_zeta_reuse_ast.py` | **File total** | 290 | 300 | Net +10 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_centroid_fft_k_plan.py` | `test_planned_k_tile_reaches_the_one_fixed_shape_padding_owner` | 86 | 94 | Existing statements moved to named stages or dead selector removed |
| `tests/test_centroid_fft_k_plan.py` | **File total** | 164 | 172 | Net +8 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_centroid_k_unfold.py` | `test_square_antiunitary_operator_uses_itself_as_transpose_partner` | 0 | 17 | New explicit phase/direct owner |
| `tests/test_centroid_k_unfold.py` | **File total** | 428 | 447 | Net +19 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_conv_kpair_plan.py` | `test_shared_downfold_cq_enters_the_conv_plan` | 5 | 5 | Existing statements moved to named stages or dead selector removed |
| `tests/test_conv_kpair_plan.py` | **File total** | 157 | 157 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_coupled_mu123_gflat_host_spill.py` | `test_host_spill_is_automatic_and_only_threads_through_coupled_route` | 8 | 8 | Existing statements moved to named stages or dead selector removed |
| `tests/test_coupled_mu123_gflat_host_spill.py` | `test_automatic_policy_keeps_fragmentation_platform_and_host_gates` | 9 | 9 | Existing statements moved to named stages or dead selector removed |
| `tests/test_coupled_mu123_gflat_host_spill.py` | **File total** | 108 | 109 | Net +1 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_downfold.py` | `test_RED_TWIN_the_raw_kernel_labels_the_gram_by_MINUS_q` | 36 | 36 | Existing statements moved to named stages or dead selector removed |
| `tests/test_downfold.py` | **File total** | 1816 | 1816 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_ff_compute_mode.py` | `test_the_naming_decision_and_its_rejected_alternative_are_written_down` | 9 | 10 | Existing statements moved to named stages or dead selector removed |
| `tests/test_ff_compute_mode.py` | `test_the_sigma_dispatch_no_longer_reaches_the_ppm_pipeline_by_else` | 22 | 25 | Existing statements moved to named stages or dead selector removed |
| `tests/test_ff_compute_mode.py` | **File total** | 509 | 513 | Net +4 lines |

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
| `tests/test_low_mem_bands_envelope.py` | `test_compute_sigma_xc_checks_the_gij_row_before_any_kernel` | 15 | 14 | Existing statements moved to named stages or dead selector removed |
| `tests/test_low_mem_bands_envelope.py` | **File total** | 320 | 320 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_mpa_sampling_config.py` | `test_explicit_mpa_fit_reuse_gates_the_fresh_head_allocation` | 7 | 8 | Existing statements moved to named stages or dead selector removed |
| `tests/test_mpa_sampling_config.py` | **File total** | 717 | 719 | Net +2 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_sc_four_current_hartree.py` | `test_density_sc_suppresses_both_frozen_direct_components` | 29 | 29 | Existing statements moved to named stages or dead selector removed |
| `tests/test_sc_four_current_hartree.py` | **File total** | 366 | 366 | Net +0 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_sigma_fermi_split.py` | `test_mpa_head_occupation_preflight_precedes_the_body_sweep` | 9 | 8 | Existing statements moved to named stages or dead selector removed |
| `tests/test_sigma_fermi_split.py` | `test_the_occupation_state_actually_reaches_all_three_build_Gij_sites` | 37 | 40 | Existing statements moved to named stages or dead selector removed |
| `tests/test_sigma_fermi_split.py` | **File total** | 714 | 716 | Net +2 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_wavefunction_basis_receipt.py` | `test_prepare_constructs_receipts_on_host_from_one_canonical_scan` | 34 | 39 | Existing statements moved to named stages or dead selector removed |
| `tests/test_wavefunction_basis_receipt.py` | **File total** | 326 | 331 | Net +5 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_windowed_exp_iEt.py` | `test_build_G_rejects_unknown_layout` | 4 | 4 | Existing statements moved to named stages or dead selector removed |
| `tests/test_windowed_exp_iEt.py` | `test_build_G_face_requires_a_gemm_plan` | 4 | 0 | Deleted or renamed to direct owner |
| `tests/test_windowed_exp_iEt.py` | `test_build_G_tau_forwards_layout_and_gemm_to_build_G` | 11 | 11 | Existing statements moved to named stages or dead selector removed |
| `tests/test_windowed_exp_iEt.py` | `test_build_G_and_tau_unfold_faces_before_the_only_contraction` | 27 | 0 | Deleted or renamed to direct owner |
| `tests/test_windowed_exp_iEt.py` | `test_build_G_and_tau_unfold_faces_before_the_only_contraction.ParentRows.unfold_face` | 2 | 0 | Deleted or renamed to direct owner |
| `tests/test_windowed_exp_iEt.py` | `test_build_G_face_requires_a_contraction_plan` | 0 | 4 | New explicit phase/direct owner |
| `tests/test_windowed_exp_iEt.py` | `test_build_G_and_tau_transport_parent_operators_after_contraction` | 0 | 28 | New explicit phase/direct owner |
| `tests/test_windowed_exp_iEt.py` | `test_build_G_and_tau_transport_parent_operators_after_contraction.ParentRows.unfold_operator` | 0 | 4 | New explicit phase/direct owner |
| `tests/test_windowed_exp_iEt.py` | **File total** | 361 | 362 | Net +1 lines |

| Module | Function | Lines before | Lines after | Deleted or moved; destination |
|---|---|---|---|---|
| `tests/test_zeta_nband_decoupling.py` | `test_the_fit_window_travels_into_the_provenance_stamp` | 25 | 29 | Existing statements moved to named stages or dead selector removed |
| `tests/test_zeta_nband_decoupling.py` | **File total** | 412 | 416 | Net +4 lines |

| Tree scope | Before | After | Net |
|---|---|---|---|
| Requested production roots, tracked Python; baseline `00_inventory/before.json` | 125247 | 119592 | -5655 |

| Whole tracked tree scope | Added | Removed | Net |
|---|---|---|---|
| Text diff against c2f69987, including relocated documentation, tests and this report | 14131 | 10849 | 3282 |

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
| 9 — extract zeta writer and tile stages (48194876) | 1405 | 465 passed, 2 skipped, 1 xfailed, 102 warnings in 247.62s (0:04:07) | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-190233-157458-4126, lx-Xg4-190235-157617-6713; 18_fit_writer_stages/cpu/cpu.xml; 18_fit_writer_stages/p4/mos2/identity.json |
| 10 — use canonical mesh cache keys (37aa7db2) | 1406 | 452 passed, 2 skipped, 1 xfailed, 102 warnings in 177.23s (0:02:57) | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-190804-191933-7526, lx-Xg4-190804-192105-9882; 19_owner_cache_keys/cpu/cpu.xml; 19_owner_cache_keys/p4/mos2/identity.json |
| 11 — relocate long source contracts to architecture owners (14b6fb1d) | 1408 | 450 passed, 2 skipped, 1 xfailed, 102 warnings in 178.98s (0:02:58) | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-191254-220724-5967, lx-Xg4-191254-220772-4165; 20_contract_docs/cpu/cpu.xml; 20_contract_docs/p4/mos2/identity.json |
| 12 — separate input parsing from configuration envelopes (35b45dc6) | 1409 | 751 passed, 2 skipped, 1 xfailed, 116 warnings in 184.45s (0:03:04) | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-192253-278495-2175, lx-Xg4-191743-249096-5555; 22_config_corrected/cpu/cpu.xml; 22_config_corrected/p4/mos2/identity.json |
| 13 — extract bounded centroid loading stages (7225ea43) | 1410 | 458 passed, 2 skipped, 1 xfailed, 102 warnings in 176.27s (0:02:56) | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58001753 lx-Xg0-193204-329368-3760, lx-Xg4-192729-303163-2911; 31_centroid_guard/cpu/cpu.xml; 31_centroid_guard/p4/mos2/identity.json |
| 14 — restore low_mem_bands and two-layout planner dispatch (a9560071) | 1413 | 474 passed, 2 skipped, 1 xfailed, 102 warnings in 187.73s (0:03:07) | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58006471 lx-Xg0-201643-544166-5522, lx-Xg4-201649-544831-8820; 34_restore_layout/cpu/cpu.xml; 34_restore_layout/p4/mos2/identity.json |
| 15 — revert writer extraction exposed by fresh fitting (fe51b102) | 1414 | 455 passed, 2 skipped, 1 xfailed, 102 warnings in 192.85s (0:03:12) | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58006471 lx-Xg0-202436-590723-1370, lx-Xg4-202604-591455-1706; 39_writer_revert_corrected/cpu/cpu.xml; 39_writer_revert_corrected/p4/mos2/identity.json |
| 16 — split Sigma dispatch into explicit physics stages (9f2c64d5) | 1416 | 478 passed, 2 skipped, 3 xfailed, 102 warnings in 196.16s (0:03:16); CPU1 13 passed, 1 warning in 3.65s | Si SOC GN exact eqp0/eqp1 and sigma_diag | allocation58006471 lx-Xg0-202851-614453-8799, lx-Xg4-202906-616745-3091, lx-Xg0-202904-616331-8607; 36_sigma_stages/cpu/cpu.xml; 36_sigma_stages/p4/si_soc/identity.json |
| 17 — split fresh and restart ISDF initialization stages (49d96f1b) | 1418 | 472 passed, 2 skipped, 1 xfailed, 102 warnings in 190.64s (0:03:10) | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58006471 lx-Xg0-203308-639938-5868, lx-Xg4-203317-641493-1330; 37_initializer_port/cpu/cpu.xml; 37_initializer_port/p4/mos2/identity.json |
| 18 — share Gram face shape and channel refusal rules (bb3a8568) | 1419 | 459 passed, 2 skipped, 1 xfailed, 102 warnings in 194.62s (0:03:14) | MoS2 exact eqp0/eqp1 and sigma_diag | allocation58006471 lx-Xg0-203735-696670-7606, lx-Xg4-203744-697852-1411; 41_gram_shape_rules/cpu/cpu.xml; 41_gram_shape_rules/p4/mos2/identity.json |
| 19 — align final guards with Sigma stages and parent Green transport (this commit) | 1423 | 61 passed, 1 warning in 4.12s | Si SOC GN exact eqp0/eqp1 and sigma_diag | allocation58006471 lx-Xg0-205528-792709-2931, lx-Xg4-205531-793059-9785; 43_final_guard_owners/cpu/cpu.xml; 43_final_guard_owners/p4/si_soc/identity.json |

| Deviation / open gate | Reason / disposition |
|---|---|
| Historical MoS2 run73 reference superseded | Owner instructed fresh untouched-tip references; expected covariant Gamma price maximum4.647 microeV is documented in sandbox claim1201; no tolerance relaxed |
| Scalar reference unresolved after Green fix | Untouched ebee1467 also stalled after screening, stopped at 4m36s total; 33_green_references/si_scalar and stop/kill receipts, pool58006471, step lx-Xg4-201057-511760-9255. Si SOC and MoS2 references completed. Two original untouched-source attempts stopped after no post-screening output; first >16min, native stacks in CUDA/NCCL. `00_references/si_scalar_attempt01`, `si_scalar`, `stack_1606206.log`, `stop_retry.log`; allocation58001753. Source cause unisolated, registered in sandbox KNOWN_LORRAX_ISSUES.md |
| Additional launch attempts | 07_gram_p4 and 07_gram_p4_retry refused pool occupancy; no numerical execution. Completed accepted leg:07_gram_p4_ready/mos2 |
| Live rectangular Gram | Downfold uses this calculation; retain it under c_q_downfold and remove only the wrapper selector |
| Live legacy band projector | BSE ring consumers and common/zeta_projection still call it; no blanket removal |
| Live absent-plan Green seam | photon_sigma has already unfolded mixed endpoints before build_G; applying another plan would unfold twice |
| Captured symmetry data in identity caches | Shape equality alone does not authenticate captured action tables; no unsafe shape-only replacement |
| Strict-key legacy exemptions | read_lorrax_input includes _LEGACY_DECK_KEYS in _known; dropping only explicit refusals would silently accept retired keys |
| Authorized Green-fix synchronization | Owner required one rebase onto ebee1467 on resume; completed, then cherry-picked aa0fdb6e as 7225ea43. Original-to-rebased commit mapping:32_green_rebase/commit_mapping.json. No further rebase permitted |
| Reverted centroid-loader stage cut | 12_centroid_stages/verdict.txt: P4 exact; CPU1 failed/460 passed/2 skipped/1 xfailed because test_centroid_fft_k_plan requires in-entry calls. Initial source/contract relocation reverted; corrected extraction accepted in31_centroid_guard |
| Reverted Sigma stage cut | 13_sigma_stages/verdict.txt: P4 exact; CPU9 failed/496 passed/2 skipped/3 xfailed: four nb=3 fixtures incompatible with CPU4 band sharding, four absent-host-FFI failures, one entry-body structural guard. Source/tests/contract relocation reverted |
| Reverted initializer stage cut | 17_prepare_stages/verdict.txt: CPU476 passed/2 skipped/1 xfailed; P4 failed because the extracted fresh-carrier stage omitted _parent_green_faces. Source/tests/contract relocation reverted |
| Configuration guard retry | 21_config_stages/verdict.txt: initial CPU750 pass/1 fail depended on a moved driver comment; reverted and reapplied with owner-based structural guard in22_config_corrected. Its P4 receipt is reused after exact production-diff comparison |
| Restored low_mem_bands/layout dispatch | Owner amendment applied: gflat_memory_model restored byte-for-byte to c2f69987; both gw_init caller guards and planner tests restored. Config and Sigma flagged lines remain in extracted owners; 34_restore_layout/dispatch_audit.json records the mapping |
| Writer extraction failed fresh-fit gate | 35_fresh_fit/mos2_6x6, step lx-Xg4-201745-548578-5869: _close_zeta_fit_output lacks nqx/nqy/nqz arguments; writer extraction reverted to its exact pre-extraction version (retaining the Gram API). Fresh retry39 passed: eqp0/eqp1 max0.018 microeV vs104, step lx-Xg4-202436-590632-3315, fresh_eqp0.txt and fresh_eqp1.txt. Original copied-zeta evidence did not cover this branch |
| Broad mesh-value cache cut reverted | 42_mesh_value_keys/verdict.txt: CPU463 passed,2 missing-host-FFT failures,6 skipped,3 xfailed; Si SOC P4 exact. All12 file changes reverted per owner instruction; initial19_owner_cache_keys cut remains |
| Global length target unmet | 142 functions exceed120 lines; 30_residual_inventory/remaining.json. The prioritized writer extraction was reverted after a real fresh-fit defect; many remaining SC, preprocessing, kernel and service functions were outside the named entry-stage cuts and were not covered by a matching supplied deck |
| Phase-specific banners retained | Unique phase announcements remain; no blanket replacement of diagnostics or changes to their synchronization/exception timing |
| Final CPU/core/P16/compile-event gates | Completed with recorded limits below: core57 passed/19 device-count skips; P16 all printed files exact and473 compile events versus674; full CPU has3 inherited host-FFI failures after6 stale guards were fixed |
| CPU skips/xfail | Host FFT FFI unavailable; not WSL; existing CPU partitioned-max NaN xfail. Explicit scoped CPU logs own details |

| Final gate | Result | Step / allocation | Artifact |
|---|---|---|---|
| Untouched Green references | Si SOC and MoS2 completed on ebee1467; scalar stalled after screening | allocation58006471 | 33_green_references/{si_soc,mos2,si_scalar}/source.txt and driver.rank0.log |
| Fresh MoS2 6x6 P4; no copied tmp | eqp0/eqp1 max0.018 microeV vs104 across210 rows; DFT energies identical | 58006471 / lx-Xg4-202436-590632-3315 | 39_writer_revert_corrected/{fresh_eqp0.txt,fresh_eqp1.txt} |
| Full CPU union and every bispinor pytest module | 1061 passed,9 failed,6 skipped,1 xfailed; six stale guards corrected by focused gate below | 58006471 / lx-Xg0-204800-756033-8929 | 23_final_gates/cpu_retry/cpu.xml; cpu_suites.txt |
| Focused final guard owners | 61 passed; resolves two extracted-Sigma guards and four pre-ebee Green-interface tests without production changes | 58006471 / lx-Xg0-205528-792709-2931 | 43_final_guard_owners/cpu/cpu.xml |
| Remaining CPU failures | Three inherited missing-host-FFI cases: DFT dipole provenance and two bispinor V_q SlabIO tests; no all-green CPU claim | Same full CPU step | 23_final_gates/cpu_retry/cpu.xml; baseline parent sub_14_final_audit/final_gate_table.md |
| Main core tier (lx test) | 57 passed,19 device-count skips | 58006471 / lx-Xg4-204635-749908-8956 | 23_final_gates/core/core.xml |
| MoS2 6x6 P16 copied-tensor identity | eqp0/eqp1 and sigma_diag identical to104;217/217/246 printed lines, zero differing rows | 58007765 / lx-Xg4-205616-796667-3750 | 44_p16_retry/identity.json |
| P16 compile-event count | 473 on every rank; baseline rank0 P4 count674; no increase (cross-geometry count check, not a speedup attribution) | Same P16 step | 44_p16_retry/compile_counts.json; donor104/driver.rank0.log:1313 |
| Four-node pool release | Allocation cancelled after successful step completion | 58007765 | 44_p16_retry/release_status.txt |
| Rejected first P16 launch | Allocation FAILED before scientific output; retried on a successfully returned allocation | 58007540 / lx-Xg4-204952-766253-9574 | 23_final_gates/p16/launch.log |
| Non-pytest launchers in initial CPU collection | Four standalone GPU scripts have no pytest test functions; removed from CPU payload after import-time host-FFI refusal | 58006471 / lx-Xg0-204635-749765-5736 | 23_final_gates/non_pytest_launchers.txt; cpu/collection.log |
