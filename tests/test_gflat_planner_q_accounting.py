"""Regression gates for selected-Q zeta planner accounting."""

from types import SimpleNamespace

import numpy as np
import pytest

from gw.gflat_memory_model import _persistent_bytes, plan_gflat_chunks


@pytest.mark.parametrize(
    "K,Q,M,G,R,expected_gb",
    [
        (64, 8, 836, 588, 4380, 1.907096576),
        (9, 5, 840, 1963, 46080, 1.276302720),
        (512, 29, 896, 1532, 984, 7.609792512),
    ],
)
def test_review_false_q_charge_exact(K, Q, M, G, R, expected_gb):
    """Reproduce claim 710's false K-minus-Q binding-stage bytes."""
    false_bytes = 16 * (K - Q) * (M * M + M * G + 2 * M * R) / 4
    assert false_bytes / 1e9 == pytest.approx(expected_gb, abs=1e-12)


def test_persistent_factor_and_gflat_use_selected_q_rows():
    common = dict(
        nk=64, ns=2, nq=64, mu=836, nb=80, ngkmax=588,
        n_rtot=13824, p_x=2, p_y=2,
    )
    full = _persistent_bytes(nq_disk=64, **common)
    selected = _persistent_bytes(nq_disk=8, **common)
    expected = 16 * (64 - 8) * (836 * 836 + 836 * 588) / 4
    assert sum(full.values()) - sum(selected.values()) == expected


def test_plan_receipt_distinguishes_full_K_from_selected_Q(monkeypatch):
    monkeypatch.setattr(
        "gw.gflat_memory_model._fft_box_bytes",
        lambda **kwargs: 1.0,
    )
    meta = SimpleNamespace(
        nk_tot=9, nspinor=2, n_rmu=840, n_rmu_padded=840,
        n_rtot=46080, fft_grid=(48, 40, 24),
    )
    mesh = SimpleNamespace(
        shape={"x": 2, "y": 2}, devices=np.empty(4, dtype=object))
    plan = plan_gflat_chunks(
        meta=meta, mesh_xy=mesh, nb_total=80, face_nb_total=80,
        fit_nb_total=80, ngkmax=1963, n_q_disk=5, n_q_ibz=5,
        budget_gb=80.0, target_utilization=0.8,
        r_chunk_override=46080, band_chunk_override=16,
        distributed_zeta_solve="distributed", low_mem_bands=True,
    )
    assert (plan.n_q_full, plan.n_q_selected) == (9, 5)
    assert "selected Q 5 / full-zone K 9" in plan.format()
