"""Focused lifetime gates for coupled transverse G-flat host spilling."""

import inspect

import numpy as np

from gw import isdf_fitting
from gw.gw_init import _select_coupled_mu123_route, fit_zeta


def test_spill_helpers_are_default_off_and_use_the_collectives_door(monkeypatch):
    value = object()
    events = []

    monkeypatch.setattr(
        isdf_fitting.collectives, "spill_to_host",
        lambda arr: events.append(("spill", arr)) or "host")
    monkeypatch.setattr(
        isdf_fitting.collectives, "restore_from_host",
        lambda spill: events.append(("restore", spill)) or "device")

    assert isdf_fitting._coupled_gflat_spill(
        value, enabled=False) is value
    assert isdf_fitting._coupled_gflat_restore(
        value, enabled=False) is value
    assert events == []
    assert isdf_fitting._coupled_gflat_spill(value, enabled=True) == "host"
    assert isdf_fitting._coupled_gflat_restore("host", enabled=True) == "device"
    assert events == [("spill", value), ("restore", "host")]


def test_host_byte_diagnostic_counts_only_addressable_local_shards():
    class Shard:
        def __init__(self, shape, dtype):
            self.data = np.empty(shape, dtype=dtype)

    arr = type("Array", (), {
        "addressable_shards": [
            Shard((2, 3), np.complex128),
            Shard((1, 4), np.float64),
        ]
    })()
    assert isdf_fitting._local_shard_nbytes(arr) == 128


def test_production_lifetime_spills_before_prepared_and_around_accumulate():
    source = inspect.getsource(isdf_fitting.fit_zeta_to_h5)
    initial = source.index('timing.section("zeta_fit.gflat_spill_initial")')
    prepared = source.index('_coupled_rank_gate("prepared")')
    restore = source.index(
        'timing.section("zeta_fit.gflat_restore_accumulate")')
    accumulate = source.index('gflat_acc = accumulate_rchunk_to_gflat(', restore)
    respill = source.index(
        'timing.section("zeta_fit.gflat_spill_accumulated")', accumulate)
    finish_chunk = source.index(
        '_coupled_mu123_coordinator.finish_chunk(', respill)
    wait_finalize = source.index(
        '_coupled_mu123_coordinator.wait_finalize(', finish_chunk)
    restore_final = source.index(
        'timing.section("zeta_fit.gflat_restore_final_write")', wait_finalize)
    final_write = source.index("zeta_io.write_slab('zeta_q_G'", restore_final)
    assert initial < prepared < restore < accumulate < respill < finish_chunk
    assert finish_chunk < wait_finalize < restore_final < final_write


def test_host_spill_is_automatic_and_only_threads_through_coupled_route():
    source = inspect.getsource(fit_zeta)
    assert "LORRAX_EXPERIMENTAL_COUPLED_ZQ" not in source
    assert "LORRAX_EXPERIMENTAL_COUPLED_ZQ_HOST_SPILL" not in source
    call = source.index("_spill_coupled_gflat_to_host=")
    assert "coordinator is not None" in source[call:call + 180]
    solve_call = source.index("_stack_coupled_solve_inputs=")
    assert "coordinator is not None" in source[solve_call:solve_call + 80]


def test_auto_route_prefers_local_then_distributed_then_sequential():
    assert _select_coupled_mu123_route(
        requested_route="auto", base_hwm_bytes=40, budget_bytes=100,
        local_delta_bytes=50, distributed_delta_bytes=20,
    ) == (True, "batch_reshard", 50.0)
    assert _select_coupled_mu123_route(
        requested_route="auto", base_hwm_bytes=40, budget_bytes=100,
        local_delta_bytes=50, distributed_delta_bytes=20,
        local_capacity_ok=False,
    ) == (True, "auto", 20.0)
    assert _select_coupled_mu123_route(
        requested_route="auto", base_hwm_bytes=90, budget_bytes=100,
        local_delta_bytes=50, distributed_delta_bytes=20,
    ) == (False, "auto", None)


def test_explicit_local_route_never_silently_changes_backend():
    assert _select_coupled_mu123_route(
        requested_route="batch_reshard", base_hwm_bytes=60,
        budget_bytes=100, local_delta_bytes=50,
        distributed_delta_bytes=20,
    ) == (False, "batch_reshard", None)


def test_automatic_policy_keeps_fragmentation_platform_and_host_gates():
    source = inspect.getsource(fit_zeta)
    assert "_gflat_plan_T.target_utilization" in source
    assert "'A100' in _device_kind" in source
    assert "_p_xy in (4, 16)" in source
    assert "three_host_gflat_outputs" in source
    assert "0.35 * _host_total_gb" in source
    assert "_gflat_plan_T.persistent_bytes" in source
    assert "_local_budget_delta" in source
