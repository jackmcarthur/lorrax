"""Stage-C memory accounting follows the selected conv_kpair route."""

from types import SimpleNamespace

import pytest


def _meta():
    return SimpleNamespace(
        nk_tot=64,
        nspinor=2,
        n_rmu=64,
        n_rmu_padded=64,
        n_rtot=24**3,
        ngkmax=800,
        fft_grid=(24, 24, 24),
        kgrid=(4, 4, 4),
    )


def _mesh():
    return SimpleNamespace(shape={"x": 1, "y": 1})


@pytest.mark.parametrize(
    "arm,want_pair,want_reduced",
    [("xla", 3, 0), ("resident", 2, 0), ("device", 2, 3)],
)
def test_stage_c_slots_follow_conv_route(
        monkeypatch, arm, want_pair, want_reduced):
    from ffi import fft
    from gw import gflat_memory_model as model

    monkeypatch.setattr(model, "_pair_density_slots", lambda: 3)
    monkeypatch.setattr(
        fft, "conv_kpair_plan",
        lambda mesh, kgrid, ns, trailing: (arm, "test route"))
    plan = model.plan_gflat_chunks(
        meta=_meta(), mesh_xy=_mesh(), nb_total=80, fit_nb_total=50,
        ngkmax=800, n_q_disk=8, budget_gb=1000.0,
        band_chunk_override=16, r_chunk_override=1024,
        slab_io_replicates=False,
    )
    assert plan.conv_kpair_arm == arm
    assert plan.pair_density_slots == want_pair
    assert plan.conv_reduced_slots == want_reduced


def test_explicit_slot_calibration_bypasses_route_inference(monkeypatch):
    from ffi import fft
    from gw import gflat_memory_model as model

    monkeypatch.setattr(
        fft, "conv_kpair_plan",
        lambda *args, **kwargs: pytest.fail("route inference must be bypassed"))
    plan = model.plan_gflat_chunks(
        meta=_meta(), mesh_xy=_mesh(), nb_total=80, fit_nb_total=50,
        ngkmax=800, n_q_disk=8, budget_gb=1000.0,
        band_chunk_override=16, r_chunk_override=1024,
        pair_density_slots=7, slab_io_replicates=False,
    )
    assert plan.conv_kpair_arm == "explicit"
    assert plan.pair_density_slots == 7
    assert plan.conv_reduced_slots == 0
