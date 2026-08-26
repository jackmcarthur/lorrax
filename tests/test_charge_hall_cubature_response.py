"""Focused algebra gate for the truncated charge+Hall producer."""
from __future__ import annotations

from dataclasses import replace
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import common.parallel_transport as parallel_transport
import gw.qsgw_head as qsgw_head
from gw.head_correction import static_hall_linear_response
from gw.photon_layout import PhotonBasisLayout
from gw.static_gauge_response import (
    CHARGE_HALL_CUBATURE_AVAILABILITY,
    StaticGaugeResponseCapability,
    StaticGaugeTermStatus,
    build_charge_hall_cubature_response,
    require_full_static_gauge_availability,
)

jax.config.update("jax_enable_x64", True)


def _mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _placed(value, mesh, spec):
    return jax.device_put(np.asarray(value), NamedSharding(mesh, spec))


def test_charge_hall_builder_has_only_declared_support(
        monkeypatch):
    mesh = _mesh()
    layout = PhotonBasisLayout.from_centroid_extents(3, 2, mesh)
    wfn_sha = "1" * 64

    S = np.asarray([[[2.0, 0.25, 0.0],
                     [0.25, 1.0, 0.0],
                     [0.0, 0.0, 0.5]]], dtype=np.complex128)
    charge_y = np.asarray(
        [[1.0 + 2.0j, -0.5j, 0.75],
         [0.25 - 0.5j, -1.25, 0.125j]], dtype=np.complex128)
    Y = np.zeros((1, 3, 3), dtype=np.complex128)
    Z = np.zeros((1, 3, 3), dtype=np.complex128)
    Y[0, :2] = charge_y
    Z[0, :, :2] = np.conj(charge_y).T
    direct = SimpleNamespace(
        S_direct=_placed(S, mesh, P()),
        Y_x=_placed(Y, mesh, P(None, None, "x")),
        Z_y=_placed(Z, mesh, P(None, "y", None)),
    )
    monkeypatch.setattr(qsgw_head, "build_dft_head_response",
                        lambda *args, **kwargs: direct)
    monkeypatch.setattr(parallel_transport, "wfn_fingerprint",
                        lambda _wfn: wfn_sha)

    class Hall:
        pass

    monkeypatch.setattr(qsgw_head, "StaticGaugeHallTransaction", Hall)
    hall = Hall()
    hall.sigma_H = _placed([0.0, 0.0, -2.0], mesh, P())
    hall.wfn_fingerprint = wfn_sha
    hall.band_start, hall.band_stop = 0, 2
    hall.nk_tot = 6
    hall.producer_id = "lorrax.static_gauge_hall/full_bz_uniform_gauge_v1"

    wfn = SimpleNamespace(nspin=1)
    meta = SimpleNamespace(
        b_id_0=0, b_id_4_chi_user=2, nspinor=4,
        nspinor_wfnfile=2, cell_volume=9.0)
    config = SimpleNamespace(
        nval=1, ncond=1, nband=2, vnl_velocity_sign=1.0,
        head=SimpleNamespace(wcoul0_eta=0.0))
    kwargs = dict(
        input_dir="/bounded/not-read", mesh=mesh, wfn=wfn, meta=meta,
        config=config, layout=layout, hall_transaction=hall)
    response = build_charge_hall_cubature_response(object(), **kwargs)

    assert response.capability is (
        StaticGaugeResponseCapability.CHARGE_HALL_CUBATURE)
    assert response.availability == CHARGE_HALL_CUBATURE_AVAILABILITY
    assert response.ward_residual == 0.0
    assert response.hermiticity_residual == 0.0
    assert response.wing_reciprocity_residual == 0.0

    pi1 = np.asarray(static_hall_linear_response(response.sigma_H))
    np.testing.assert_array_equal(pi1[:, 0, 0], np.zeros(2))
    np.testing.assert_array_equal(pi1[:, 1:, 1:], np.zeros((2, 3, 3)))
    np.testing.assert_array_equal(pi1[:, 1:, 0], np.conj(pi1[:, 0, 1:]))
    assert pi1[0, 0, 2] == -2.0j
    assert pi1[1, 0, 1] == 2.0j

    S_got = np.asarray(response.S_direct)
    np.testing.assert_array_equal(S_got[:, :, 0, 0], S[0, :2, :2])
    np.testing.assert_array_equal(S_got[:, :, 0, 1:], 0.0)
    np.testing.assert_array_equal(S_got[:, :, 1:, :], 0.0)

    packed_y = np.asarray(response.Y_x)
    packed_z = np.asarray(response.Z_y)
    np.testing.assert_array_equal(packed_y[:, 0, :3], charge_y)
    np.testing.assert_array_equal(packed_y[:, 1:, :], 0.0)
    np.testing.assert_array_equal(packed_y[:, 0, 3:], 0.0)
    np.testing.assert_array_equal(packed_z[:, :3, 0], np.conj(charge_y))
    np.testing.assert_array_equal(packed_z[:, :, 1:], 0.0)
    np.testing.assert_array_equal(packed_z[:, 3:, 0], 0.0)

    with pytest.raises(ValueError, match="availability"):
        require_full_static_gauge_availability(response.availability)
    unavailable_charge = replace(
        response.availability, cc_q2=StaticGaugeTermStatus.UNAVAILABLE)
    assert not unavailable_charge.is_complete_for(
        StaticGaugeResponseCapability.CHARGE_HALL_CUBATURE)
