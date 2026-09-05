"""Exact existing cells retained in the two-minute core tier.

All tests physically under ``tests/core`` are selected automatically.  This
roster is only for established service/guard cells whose check bodies should
not be copied into a second test implementation.
"""

CORE_NODES = {
    # distrib_la: plans, padded algebra, batched route and reshape inverses.
    "services/distrib_la/tests/test_distrib_la_batch_reshard.py::test_public_route_grammar_and_plan_provenance",
    "services/distrib_la/tests/test_distrib_la_batch_reshard.py::test_eigh_route_handles_ragged_batches_and_restores_layout[3]",
    "services/distrib_la/tests/test_distrib_la_batch_reshard.py::test_cholesky_route_uses_safe_ragged_padding_and_ignores_block_size",
    "services/distrib_la/tests/test_distrib_la_batch_reshard.py::test_solve_lu_route_skips_ragged_padding_and_restores_layout",
    "services/distrib_la/tests/test_distrib_la_batch_reshard.py::test_staged_forward_then_inverse_is_bit_exact",
    "services/distrib_la/tests/test_distrib_la_batch_reshard.py::test_extent_refusals_fire_before_the_kernel",
    "services/distrib_la/tests/test_distrib_la_emulated_mesh.py::test_native2d_factorizes_like_numpy[8-complex128]",
    "services/distrib_la/tests/test_distrib_la_emulated_mesh.py::test_a_nondividing_extent_survives_identity_padding[prime-both-axes]",
    "services/distrib_la/tests/test_distrib_la_emulated_mesh.py::test_native2d_agrees_between_a_1x1_and_a_2x2_mesh",
    "services/distrib_la/tests/test_distrib_la_emulated_mesh.py::test_native_eigh_agrees_between_a_1x1_and_a_2x2_mesh",
    "services/distrib_la/tests/test_distrib_la_emulated_mesh.py::test_native_eigh_recovers_a_padded_spectrum_on_a_2x2",
    "services/distrib_la/tests/test_distrib_la_shape_algebra.py::test_the_block_size_satisfies_all_three_constraints[mesh_shape1]",
    "services/distrib_la/tests/test_distrib_la_shape_algebra.py::test_hostile_extents_are_refused_or_tiled_honestly[mesh_shape1]",
    "services/distrib_la/tests/test_distrib_la_shape_algebra.py::test_dense_to_tiles_round_trips_on_the_lower_triangle",
    "services/distrib_la/tests/test_distrib_la_shape_algebra.py::test_both_axes_can_have_a_remainder_at_once",
    "services/distrib_la/tests/test_distrib_la_contract.py::test_plan_resolution_is_identical_to_resolve_backend",
    "services/distrib_la/tests/test_distrib_la_contract.py::test_plan_native_contract",
    # SlabIO: host/emulated transport, logical extents, carriers and refusals.
    "tests/test_slab_io_emulated_mesh.py::test_a_two_axis_sharded_write_lands_where_h5py_reads_it",
    "tests/test_slab_io_emulated_mesh.py::test_a_replicated_write_lands_once_and_correctly",
    "tests/test_slab_io_emulated_mesh.py::test_a_host_numpy_operand_and_an_off_mesh_array_both_write",
    "tests/test_slab_io_emulated_mesh.py::test_pad_rows_past_the_dataset_are_dropped_with_no_argument",
    "tests/test_slab_io_emulated_mesh.py::test_write_then_read_round_trips_through_the_same_handle",
    "tests/test_slab_io_emulated_mesh.py::test_write_attr_and_dataset_attrs_land_at_close",
    "tests/test_slab_io_emulated_mesh.py::test_create_dataset_reuses_an_identical_one_and_refuses_a_clash",
    "tests/test_slab_io_emulated_mesh.py::test_read_slabs_packs_n_windows_exactly_as_n_read_slab_calls_would",
    "tests/test_slab_io_hostile_geometry.py::test_padded_slab_clips_to_the_logical_extent[logical1-padded1-expect_valid1]",
    "tests/test_slab_io_hostile_geometry.py::test_valid_shape_override_that_overruns_the_dataset_refuses",
    "tests/test_slab_io_hostile_geometry.py::test_nondivisible_sharded_extent_refuses_naming_the_padded_shape",
    "tests/test_slab_io_hostile_geometry.py::test_one_dim_sharded_by_both_mesh_axes_uses_the_axis_PRODUCT",
    # Strict deck vocabulary and routing.
    "tests/test_deck_dials.py::test_linalg_dial_resolves_one_complete_profile[local-local-auto-auto-auto-auto]",
    "tests/test_deck_dials.py::test_linalg_dial_resolves_one_complete_profile[distributed-distributed-distributed-distributed-distributed-distributed]",
    "tests/test_deck_dials.py::test_linalg_rejects_unknown_values[fast]",
    "tests/test_deck_dials.py::test_retired_linalg_keys_refuse_by_name_with_migration_hint[distributed_cholesky]",
    "tests/test_deck_dials.py::test_strict_keys_is_retired_and_unknown_keys_always_refuse",
    "tests/test_deck_doctor_config.py::test_hardware_free_config_keeps_auto_memory_and_gpu_request",
    "tests/test_deck_doctor_config.py::test_material_class_uses_the_driver_tolerance[occupations0-insulator]",
    # Runtime seals and rank-conditional evaluation lint.
    "tests/test_source_closure.py::test_root_metadata_declares_every_workspace_service_as_one_closure",
    "tests/test_source_closure.py::test_bootstrap_seals_source_before_existing_runtime_steps",
    "tests/test_source_closure.py::test_production_source_seal_does_not_import_jax",
    "tests/test_rank_conditional_evaluation.py::test_the_two_production_deadlock_shapes_are_detected",
    "tests/test_rank_conditional_evaluation.py::test_current_tree_has_only_reviewed_rank_conditional_evaluations",
    "tests/test_compile_agreement_p4.py::test_identical_modules_pass_and_report_per_module_overhead",
    "tests/test_compile_agreement_p4.py::test_rank_divergent_shape_refuses_before_the_deadline",
    "services/lxkit/tests/test_native_provider.py::test_one_manifest_binds_both_selected_legs",
    "services/lxkit/tests/test_native_provider.py::test_stale_bytes_refuse_before_dlopen",
    # Cheap htransform/BSE/exciton algorithm guards; fixture drivers are core.
    "tests/test_htransform_kpath_gates.py::test_active_character_follows_state_through_guard_energy_crossing",
    "tests/test_htransform_kpath_gates.py::test_degenerate_character_boundary_has_invariant_energy_multiset",
    "tests/test_bse_head_resolvent.py::test_ladder_head_matches_a_dense_solve_of_the_same_operator",
    "tests/test_exciton_gate_spectrum.py::test_gamma_gate_reports_the_overlap_spectrum",
}


CORE_EXTENDED_NODES = CORE_NODES | {
    "services/distrib_la/tests/test_distrib_la_contract.py::test_cusolvermp_batched_cholesky_potrs",
    "services/distrib_la/tests/test_distrib_la_contract.py::test_cusolvermp_solve_lu_general",
    "services/distrib_la/tests/test_distrib_la_contract.py::test_cusolvermp_eigh",
    "services/distrib_la/tests/test_distrib_la_contract.py::test_cublasmp_gemm",
    "services/wfn_loader/tests/test_wfn_loader_contract.py::test_parent_child_unfold_uses_typed_improper_nonsymmorphic_tr_action",
    "services/wfn_loader/tests/test_wfn_loader_contract.py::test_bispinor_lift_uses_cartesian_momentum_with_blat",
    "services/zeta_loader/tests/test_zeta_loader_contract.py::test_load_refuses_full_bz_on_an_ibz_file_naming_the_post_v_q_unfold",
    "services/symmetry_maps/tests/test_symmetry_maps_deck_tables.py::test_the_round_trip_is_exact_on_the_production_tables",
    "services/vcoul/tests/test_vcoul_door_smoke.py::test_box_0d_serves_q_equals_zero_and_refuses_finite_q",
    "services/minimax/tests/test_minimax_door.py::test_final_run33_two_pane_requests_are_publicly_certified",
}


def matches(nodeid: str, roster: set[str]) -> bool:
    """Match exact parameter IDs, or every case named by a base spelling."""
    return matching_entry(nodeid, roster) is not None


def matching_entry(nodeid: str, roster: set[str]) -> str | None:
    """Return the roster spelling that selected ``nodeid``, if any."""
    if nodeid in roster:
        return nodeid
    base = nodeid.split("[", 1)[0]
    return base if base in roster else None
