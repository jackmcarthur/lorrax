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
    build_retained_alpha_first_order_head_wings,
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

    S = np.asarray([[[2.0, 0.25 + 0.5j, 0.0],
                     [0.25 - 0.5j, 1.0, 0.0],
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
    expected_charge = 0.5 * (S[0, :2, :2] + S[0, :2, :2].T)
    np.testing.assert_array_equal(S_got[:, :, 0, 0], expected_charge)
    q = np.asarray([0.375, -0.625])
    np.testing.assert_allclose(
        q @ S_got[:, :, 0, 0] @ q,
        q @ S[0, :2, :2] @ q,
        rtol=0.0, atol=1.0e-15,
    )
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


def _basis_wfn(shift=0.0):
    return SimpleNamespace(
        nbands=2, nelec=2, nspinor=2,
        energies=np.asarray([[[0.1 + shift, 0.7]]]),
        kpoints=np.asarray([[0.0, 0.0, 0.0]]),
    )


def _head_wing_binding(mesh, *, role, centroids, wfn):
    from file_io.wfn_basis import WavefunctionBasisReceipt
    from gw.wavefunction_bundle import (
        AuthenticatedWavefunctions, BandSlices, Wavefunctions)

    centroids = np.asarray(centroids, dtype=np.int32)
    nmu = int(centroids.shape[0])
    slices = BandSlices.from_band_edges(0, 0, 1, 2, 2)
    psi_nmu = _placed(
        np.zeros((1, 2, 4, nmu), dtype=np.complex128),
        mesh, P(None, "x", None, "y"))
    psi_mun = _placed(
        np.zeros((1, 4, nmu, 2), dtype=np.complex128),
        mesh, P(None, None, "x", "y"))
    carrier = Wavefunctions(
        enk=_placed([[0.1, 0.7]], mesh, P(None, None)),
        occ=_placed([[1.0, 0.0]], mesh, P(None, None)),
        slices=slices, psi_nmu=psi_nmu, psi_mun=psi_mun, layout="face")
    receipt = WavefunctionBasisReceipt.from_source(
        wfn=wfn, role=role, bispinor=True, band_interval=(0, 2),
        fft_grid=(4, 4, 4), centroid_fft_idx=centroids,
        n_rmu_logical=nmu, n_rmu_padded=nmu)
    return AuthenticatedWavefunctions(carrier, receipt)


def test_first_order_head_wings_pack_four_authenticated_body_channels(
        monkeypatch):
    """One contract cell: basis choice, (a,I) order, packing, refusal state."""
    from gw.qsgw_head import StaticGaugeFirstOrderComponent
    import gw.qsgw_head as qsgw_head

    mesh = _mesh()
    layout = PhotonBasisLayout.from_centroid_extents(3, 2, mesh)
    wfn = _basis_wfn()
    charge = _head_wing_binding(
        mesh, role="charge",
        centroids=((0, 0, 0), (1, 0, 0), (2, 0, 0)), wfn=wfn)
    transverse = _head_wing_binding(
        mesh, role="transverse",
        centroids=((0, 1, 0), (1, 1, 0)), wfn=wfn)
    first = StaticGaugeFirstOrderComponent(
        energy_scaled_d1_raw=_placed(
            np.zeros((2, 4, 1, 2, 2), dtype=np.complex128),
            mesh, P(None, None, None, "x", "y")),
        S_first_first=_placed(
            np.zeros((1, 2, 2, 4, 4), dtype=np.complex128), mesh, P()),
        bra_energy_dq_ry=_placed(np.zeros((2, 1, 2)), mesh, P()),
        occupation_difference_dq=_placed(np.zeros((2, 1, 2)), mesh, P()),
        charge_ward_residual=_placed(0.0, mesh, P()),
        retained_connection_cart=_placed(
            np.zeros((2, 1, 2, 2), dtype=np.complex128), mesh, P()),
        nb_logical=2,
        omegas_ry=_placed([0.0j], mesh, P(None)),
        nk_tot=1, nspin=1, normalization_nspinor=2, eta_ry=0.0,
    )

    calls = []

    def fake_wings(_p, wfns, _e, _f, omega, *, body_lorentz_channel,
                   **_kwargs):
        body = int(body_lorentz_channel)
        calls.append((body, wfns))
        extent = layout.padded_extent(body)
        y = np.empty((len(omega), 8, extent), dtype=np.complex128)
        z = np.empty((len(omega), extent, 8), dtype=np.complex128)
        for vertex in range(8):
            for mu in range(extent):
                y[:, vertex, mu] = 1000 * body + 10 * vertex + mu + 1j
                z[:, mu, vertex] = -1000 * body - 10 * vertex - mu + 2j
        return (
            _placed(y, mesh, P(None, None, "x")),
            _placed(z, mesh, P(None, "y", None)),
        )

    monkeypatch.setattr(qsgw_head, "head_wings_sharded", fake_wings)
    carriers = build_retained_alpha_first_order_head_wings(
        first, charge, transverse, layout=layout, mesh=mesh)

    assert [item[0] for item in calls] == [0, 1, 2, 3]
    assert calls[0][1] is charge.wavefunctions
    assert all(item[1] is transverse.wavefunctions for item in calls[1:])
    Y, Z = np.asarray(carriers.Y_x), np.asarray(carriers.Z_y)
    offset = 0
    for body in range(4):
        extent = layout.padded_extent(body)
        for a in range(2):
            for head in range(4):
                vertex = 4 * a + head
                mu = np.arange(extent)
                np.testing.assert_array_equal(
                    Y[0, a, head, offset:offset + extent],
                    1000 * body + 10 * vertex + mu + 1j)
                np.testing.assert_array_equal(
                    Z[0, a, offset:offset + extent, head],
                    -1000 * body - 10 * vertex - mu + 2j)
        offset += extent

    # This slice must not turn the existing restricted model into FULL.
    assert (CHARGE_HALL_CUBATURE_AVAILABILITY.y_current
            is StaticGaugeTermStatus.OMITTED_BY_MODEL)
    with pytest.raises(ValueError, match="availability"):
        require_full_static_gauge_availability(
            CHARGE_HALL_CUBATURE_AVAILABILITY)

    stale_transverse = _head_wing_binding(
        mesh, role="transverse",
        centroids=((0, 1, 0), (1, 1, 0)), wfn=_basis_wfn(1.0e-3))
    calls.clear()
    with pytest.raises(ValueError, match="wfn_fingerprint"):
        build_retained_alpha_first_order_head_wings(
            first, charge, stale_transverse, layout=layout, mesh=mesh)
    assert calls == []
