"""Focused algebra gate for the bounded packed static photon head producer.

``build_static_photon_head_response`` composes the charge CC head and
charge wings from the incumbent scalar producer with an OPTIONAL Hall term:
present and authenticated, or ``sigma_H = 0`` with a named source.  Nothing
else (current q^2, contact, current wings) may appear.
"""
from __future__ import annotations

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
    HALL_SOURCE_NONE,
    build_static_photon_head_response,
)

jax.config.update("jax_enable_x64", True)

_WFN_SHA = "1" * 64
_HALL_OPERATOR = "sha256:" + "c" * 64
_HALL_PRODUCER = "lorrax.dynamic_gauge_hall/full_bz_uniform_gauge_v2"


def _mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _placed(value, mesh, spec):
    return jax.device_put(np.asarray(value), NamedSharding(mesh, spec))


_S = np.asarray([[[2.0, 0.25 + 0.5j, 0.0],
                  [0.25 - 0.5j, 1.0, 0.0],
                  [0.0, 0.0, 0.5]]], dtype=np.complex128)
_CHARGE_Y = np.asarray(
    [[1.0 + 2.0j, -0.5j, 0.75],
     [0.25 - 0.5j, -1.25, 0.125j]], dtype=np.complex128)


def _patch_direct(monkeypatch, mesh):
    Y = np.zeros((1, 3, 3), dtype=np.complex128)
    Z = np.zeros((1, 3, 3), dtype=np.complex128)
    Y[0, :2] = _CHARGE_Y
    Z[0, :, :2] = np.conj(_CHARGE_Y).T
    direct = SimpleNamespace(
        S_direct=_placed(_S, mesh, P()),
        Y_x=_placed(Y, mesh, P(None, None, "x")),
        Z_y=_placed(Z, mesh, P(None, "y", None)),
    )
    monkeypatch.setattr(
        qsgw_head, "build_dft_head_response",
        lambda _wfns, omegas, **_kwargs: SimpleNamespace(
            omegas=tuple(complex(value) for value in omegas),
            S_direct=direct.S_direct, Y_x=direct.Y_x, Z_y=direct.Z_y))
    monkeypatch.setattr(parallel_transport, "wfn_fingerprint",
                        lambda _wfn: _WFN_SHA)


class _Hall:
    pass


def _hall(monkeypatch, mesh, sigma, *, probe_sigma=None, band_stop=2):
    monkeypatch.setattr(qsgw_head, "StaticGaugeHallTransaction", _Hall)
    hall = _Hall()
    hall.sigma_H = _placed(sigma, mesh, P())
    hall.sigma_H_at = lambda frequency: _placed(
        sigma if complex(frequency) == 0.0j else (
            probe_sigma if probe_sigma is not None else sigma), mesh, P())
    hall.wfn_fingerprint = _WFN_SHA
    hall.band_start, hall.band_stop = 0, band_stop
    hall.nk_tot = 6
    hall.producer_id = _HALL_PRODUCER
    hall.hamiltonian_config_operator_fingerprint = _HALL_OPERATOR
    return hall


def _kwargs(mesh):
    layout = PhotonBasisLayout.from_centroid_extents(3, 2, mesh)
    wfn = SimpleNamespace(nspin=1)
    meta = SimpleNamespace(
        b_id_0=0, b_id_4_chi_user=2, nspinor=4,
        nspinor_wfnfile=2, cell_volume=9.0)
    config = SimpleNamespace(
        sys_dim=2,
        nval=1, ncond=1, nband=2, vnl_velocity_sign=1.0,
        head=SimpleNamespace(wcoul0_eta=0.0))
    return dict(
        input_dir="/bounded/not-read", mesh=mesh, wfn=wfn, meta=meta,
        config=config, layout=layout)


def _check_charge_support(response):
    S_got = np.asarray(response.S_direct)
    expected_charge = 0.5 * (_S[0, :2, :2] + _S[0, :2, :2].T)
    np.testing.assert_array_equal(S_got[:, :, 0, 0], expected_charge)
    q = np.asarray([0.375, -0.625])
    np.testing.assert_allclose(
        q @ S_got[:, :, 0, 0] @ q,
        q @ _S[0, :2, :2] @ q,
        rtol=0.0, atol=1.0e-15,
    )
    np.testing.assert_array_equal(S_got[:, :, 0, 1:], 0.0)
    np.testing.assert_array_equal(S_got[:, :, 1:, :], 0.0)

    packed_y = np.asarray(response.Y_x)
    packed_z = np.asarray(response.Z_y)
    np.testing.assert_array_equal(packed_y[:, 0, :3], _CHARGE_Y)
    np.testing.assert_array_equal(packed_y[:, 1:, :], 0.0)
    np.testing.assert_array_equal(packed_y[:, 0, 3:], 0.0)
    np.testing.assert_array_equal(packed_z[:, :3, 0], np.conj(_CHARGE_Y))
    np.testing.assert_array_equal(packed_z[:, :, 1:], 0.0)
    np.testing.assert_array_equal(packed_z[:, 3:, 0], 0.0)
    assert response.ward_residual == 0.0
    assert response.hermiticity_residual == 0.0
    assert response.wing_reciprocity_residual == 0.0


def test_producer_with_hall_artifact_has_only_declared_support(monkeypatch):
    mesh = _mesh()
    _patch_direct(monkeypatch, mesh)
    hall = _hall(monkeypatch, mesh, [0.0, 0.0, -2.0])
    response = build_static_photon_head_response(
        object(), hall_transaction=hall, **_kwargs(mesh))

    _check_charge_support(response)
    assert _HALL_PRODUCER in response.hall_source
    assert _HALL_OPERATOR in response.hall_source
    np.testing.assert_array_equal(
        np.asarray(response.sigma_H), [0.0, 0.0, -2.0])

    pi1 = np.asarray(static_hall_linear_response(
        response.sigma_H, dimension=2))
    np.testing.assert_array_equal(pi1[:, 0, 0], np.zeros(2))
    np.testing.assert_array_equal(pi1[:, 1:, 1:], np.zeros((2, 3, 3)))
    np.testing.assert_array_equal(pi1[:, 1:, 0], np.conj(pi1[:, 0, 1:]))
    assert pi1[0, 0, 2] == 2.0j
    assert pi1[1, 0, 1] == -2.0j


def test_producer_selects_the_exact_dynamic_hall_sample(monkeypatch):
    mesh = _mesh()
    _patch_direct(monkeypatch, mesh)
    hall = _hall(
        monkeypatch, mesh, [0.0, 0.0, -2.0],
        probe_sigma=[0.25, -0.5, -0.75])
    response = build_static_photon_head_response(
        object(), hall_transaction=hall, frequency_ry=0.7j, **_kwargs(mesh))

    _check_charge_support(response)
    assert response.frequency_ry == 0.7j
    np.testing.assert_array_equal(
        np.asarray(response.sigma_H), [0.25, -0.5, -0.75])
    assert "z=0.7j Ry" in response.hall_source


def test_producer_without_hall_artifact_uses_sigma_h_zero(monkeypatch):
    mesh = _mesh()
    _patch_direct(monkeypatch, mesh)
    response = build_static_photon_head_response(
        object(), hall_transaction=None, **_kwargs(mesh))

    _check_charge_support(response)
    assert response.hall_source == HALL_SOURCE_NONE
    np.testing.assert_array_equal(np.asarray(response.sigma_H), 0.0)
    np.testing.assert_array_equal(
        np.asarray(static_hall_linear_response(
            response.sigma_H, dimension=2)), 0.0)


def test_bulk_producer_keeps_all_three_scalar_head_directions(monkeypatch):
    mesh = _mesh()
    charge_extent = 3
    Y = np.asarray([[[1.0, 0.0, 0.0],
                     [0.0, 2.0, 0.0],
                     [0.0, 0.0, 3.0]]], dtype=np.complex128)
    Z = np.conj(np.transpose(Y, (0, 2, 1)))
    direct = SimpleNamespace(
        omegas=(0.0j,),
        S_direct=_placed(_S, mesh, P()),
        Y_x=_placed(Y, mesh, P(None, None, "x")),
        Z_y=_placed(Z, mesh, P(None, "y", None)),
    )
    monkeypatch.setattr(qsgw_head, "build_dft_head_response",
                        lambda *args, **kwargs: direct)
    monkeypatch.setattr(parallel_transport, "wfn_fingerprint",
                        lambda _wfn: _WFN_SHA)
    kwargs = _kwargs(mesh)
    kwargs["config"].sys_dim = 3
    response = build_static_photon_head_response(
        object(), hall_transaction=None, **kwargs)
    assert response.dimension == 3
    assert response.S_direct.shape == (3, 3, 4, 4)
    assert response.Y_x.shape == (3, 4, response.layout.packed_extent)
    assert response.Z_y.shape == (3, response.layout.packed_extent, 4)
    np.testing.assert_array_equal(
        np.asarray(response.S_direct)[:, :, 0, 0],
        0.5 * (_S[0] + _S[0].T))
    np.testing.assert_array_equal(
        np.asarray(response.Y_x)[:, 0, :charge_extent], Y[0])
    np.testing.assert_array_equal(
        np.asarray(response.Z_y)[:, :charge_extent, 0], Z[0].T)


def test_producer_refuses_a_hall_term_from_another_band_manifold(monkeypatch):
    mesh = _mesh()
    _patch_direct(monkeypatch, mesh)
    hall = _hall(monkeypatch, mesh, [0.0, 0.0, -2.0], band_stop=3)
    with pytest.raises(ValueError, match="different band manifolds"):
        build_static_photon_head_response(
            object(), hall_transaction=hall, **_kwargs(mesh))


def test_producer_refuses_an_unsealed_hall_object(monkeypatch):
    mesh = _mesh()
    _patch_direct(monkeypatch, mesh)
    with pytest.raises(TypeError, match="sealed full-BZ Hall transaction"):
        build_static_photon_head_response(
            object(), hall_transaction=SimpleNamespace(sigma_H=np.zeros(3)),
            **_kwargs(mesh))
