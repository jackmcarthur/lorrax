"""Focused lifetime gates for coupled transverse G-flat host spilling."""

import inspect

import numpy as np

from gw import isdf_fitting
from gw.gw_init import fit_zeta


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


def test_host_spill_env_requires_and_only_threads_through_coupled_route():
    source = inspect.getsource(fit_zeta)
    flag = "LORRAX_EXPERIMENTAL_COUPLED_ZQ_HOST_SPILL"
    assert flag in source
    assert f"{flag} requires " in source
    call = source.index("_spill_coupled_gflat_to_host=")
    assert "coordinator is not None" in source[call:call + 180]

