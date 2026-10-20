"""Independent-grid head algebra, without a material run or native library."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from gw.head_correction import (HeadSample, compute_static_head_terms,
                                compute_static_head_terms_from_sample)

jax.config.update("jax_enable_x64", True)


def test_bare_exchange_keeps_fine_weight_correlation_gets_coarse_weight():
    occ = np.array([[1.0, 0.2], [0.7, 0.0]])
    sample = HeadSample(vc0=4.0, wcoul0=1.5, source="coarse", omega=0j)
    result = compute_static_head_terms_from_sample(
        sample, occ=occ, cell_volume=10.0, nk_tot=576,
        bare_vc0=1.0, correlation_nq_tot=36)
    bare = -occ / (10.0 * 576)
    corr = 2.5 * occ / (10.0 * 36)
    np.testing.assert_allclose(result.sigma_x_diag, bare, atol=1e-17)
    np.testing.assert_allclose(result.sigma_sx_minus_x_diag, corr, atol=1e-17)
    np.testing.assert_allclose(result.sigma_sx_diag, bare + corr, atol=1e-17)
    np.testing.assert_allclose(result.sigma_coh_diag, np.full_like(occ, -1.25 / 360), atol=1e-17)
    assert result.vc0 == 1.0
    assert result.correlation_vc0 == 4.0
    assert result.correlation_nq_tot == 36
    assert result.wc_head_0 == -2.5


def test_unchanged_grid_head_has_original_arithmetic():
    occ = np.array([1.0, 0.3, 0.0])
    sample = HeadSample(vc0=4.0, wcoul0=1.5, source="one-grid", omega=0j)
    actual = compute_static_head_terms_from_sample(
        sample, occ=occ, cell_volume=10.0, nk_tot=576)
    old = compute_static_head_terms(vc0=4.0, wcoul0_static=1.5,
                                   occ=occ, cell_volume=10.0, nk_tot=576)
    for name in ("sigma_x_diag", "sigma_sx_diag", "sigma_sx_minus_x_diag", "sigma_coh_diag"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(old, name))
    assert actual.correlation_vc0 is None
    assert actual.correlation_nq_tot is None


@pytest.mark.parametrize("qgrid,expected", [((6, 6, 1), (6, 6, 1)),
                                           ((24, 24, 1), None)])
def test_optical_s_unchanged_only_cell_geometry_switches(monkeypatch, qgrid, expected):
    from gw import qsgw_head, vcoul
    recorded = []
    S = np.arange(18).reshape(2, 3, 3).astype(complex) * 0.001
    meta = SimpleNamespace(kgrid=(24, 24, 1), qgrid=qgrid, nk_tot=576)
    config = SimpleNamespace(head=SimpleNamespace(
        vhead=None, whead_0freq=None, whead_imfreq=None, head_minibz_average=False))

    def average(wfn, call_meta, S_carts, **kwargs):
        recorded.append((call_meta, np.asarray(S_carts), kwargs))
        return 4.0, [1.5] * len(S_carts)

    monkeypatch.setattr(vcoul, "compute_q0_averages_screened", average)
    samples = qsgw_head.head_samples_from_s(
        S, [0j, 0.5j], wfn=None, meta=meta, config=config)
    assert len(samples) == 2
    assert recorded[0][0] is meta
    assert recorded[0][2].get("qgrid") == expected
    np.testing.assert_array_equal(recorded[0][1], S)
    assert meta.nk_tot == 576
