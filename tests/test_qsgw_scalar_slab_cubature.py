"""Focused routing gates for scalar QSGW samples through photon cubature."""

from types import SimpleNamespace

import numpy as np

import gw.head_correction as head_correction
import gw.vcoul as legacy_vcoul
from gw.qsgw_head import head_samples_from_s
import vcoul


def _config():
    return SimpleNamespace(head=SimpleNamespace(
        vhead=None,
        whead_0freq=None,
        whead_imfreq=None,
        head_minibz_average=False,
    ))


def _wfn():
    return SimpleNamespace(
        blat=1.0,
        bvec=np.diag((1.0, 1.0, 0.25)),
        cell_volume=32.0,
    )


def test_slab_samples_cubature_each_frequency_as_a_cc_response(monkeypatch):
    receipt = object()
    issued = []
    responses = []

    def issue(kernel, geometry, kgrid):
        issued.append((kernel.sys_dim, geometry, kgrid))
        return receipt

    def solve(H_linear, S_effective, got_receipt):
        H = np.asarray(H_linear)
        S = np.asarray(S_effective)
        assert got_receipt is receipt
        np.testing.assert_array_equal(H, np.zeros((2, 4, 4)))
        np.testing.assert_array_equal(S[:, :, 1:, :], 0.0)
        np.testing.assert_array_equal(S[:, :, :, 1:], 0.0)
        responses.append(S.copy())
        marker = S[0, 0, 0, 0]
        D_mean = np.zeros((4, 4), dtype=np.complex128)
        moments = np.zeros((3, 3, 4, 4), dtype=np.complex128)
        D_mean[0, 0] = 17.0
        moments[0, 0, 0, 0] = 31.0 + marker
        return SimpleNamespace(
            bare_D_mean=D_mean, screened_moments=moments)

    monkeypatch.setattr(vcoul, "slab_minibz_photon_cubature", issue)
    monkeypatch.setattr(
        head_correction, "slab_photon_cubature_moments", solve)
    monkeypatch.setattr(
        legacy_vcoul, "compute_q0_averages",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("slab scalar sample used the legacy estimator")))

    S_samples = np.asarray([
        np.diag((0.2, 0.4, 0.7)),
        np.asarray(((0.6, 0.1, 0.3),
                    (0.1, 0.8, 0.2),
                    (0.3, 0.2, 1.1))),
    ], dtype=np.complex128)
    samples = head_samples_from_s(
        S_samples,
        (0.0, 0.37j),
        wfn=_wfn(),
        meta=SimpleNamespace(sys_dim=2, kgrid=(3, 5, 1)),
        config=_config(),
    )

    assert len(issued) == 1
    assert issued[0][0] == 2
    assert issued[0][2] == (3, 5, 1)
    assert len(responses) == 2
    for response, S in zip(responses, S_samples):
        np.testing.assert_array_equal(response[:, :, 0, 0], S[:2, :2])
    assert [sample.vc0 for sample in samples] == [17.0 + 0.0j] * 2
    assert [sample.wcoul0 for sample in samples] == [
        31.2 + 0.0j,
        31.6 + 0.0j,
    ]


def test_non_slab_sample_keeps_the_legacy_estimator(monkeypatch):
    calls = []

    def legacy(*args, **kwargs):
        calls.append((args, kwargs))
        return 5.0 + 0.0j, 3.0 + 0.25j

    monkeypatch.setattr(legacy_vcoul, "compute_q0_averages", legacy)
    monkeypatch.setattr(
        head_correction,
        "slab_photon_cubature_moments",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("non-slab sample entered slab cubature")),
    )

    sample, = head_samples_from_s(
        np.eye(3, dtype=np.complex128)[None],
        (0.41j,),
        wfn=object(),
        meta=SimpleNamespace(sys_dim=3),
        config=_config(),
    )

    assert len(calls) == 1
    assert sample.vc0 == 5.0 + 0.0j
    assert sample.wcoul0 == 3.0 + 0.25j
