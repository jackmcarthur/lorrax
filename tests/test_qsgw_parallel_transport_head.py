"""Small independent oracles for the no-wavefunction QSGW head kernels."""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

import common.parallel_transport as parallel_transport_module
from common.chi_from_dipole import compute_S_omega
from common.bispinor_init import HALFALPHA
from common.mtxel_sweep import UniformGaugeCurrentMatrixElements
from common.parallel_transport import build_forward_neighbor_table
from gw.qsgw_head import (
    StaticGaugeHallTransaction,
    assemble_head_manifold,
    covariant_link_derivative,
    head_wings_sharded,
    head_s_tensor_sharded,
    raw_hall_pseudovector_sharded,
    static_gauge_hall_transaction,
    load_parallel_transport_head,
    reduced_covector_to_cartesian,
    rotate_velocity_active_to_qp,
    rotate_velocity_to_qp,
    static_gauge_first_order_component_sharded,
    static_gauge_second_order_component_sharded,
)

jax.config.update("jax_enable_x64", True)


def _mesh():
    devices = np.asarray(jax.devices())
    if devices.size >= 4:
        devices = devices[:4].reshape(2, 2)
    else:
        devices = devices[:1].reshape(1, 1)
    return Mesh(devices, ("x", "y"))


def _haar(rng, n):
    z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(z)
    phase = np.diag(r)
    return q * (phase / np.abs(phase))[None, :]


def test_static_gauge_first_order_component_reuses_charge_head_and_state_jet():
    mesh = _mesh()
    kgrid = (8, 5, 1)
    nk, nb = int(np.prod(kgrid)), 2
    kints = np.stack(np.unravel_index(
        np.arange(nk), kgrid), axis=1).astype(np.int32)
    plus = build_forward_neighbor_table(kints, kgrid)

    # A periodic two-band frame with a non-diagonal x connection.  Its
    # eigenvalues +/-2*pi make the constant link close exactly on the torus.
    A_continuum = 2.0 * np.pi * np.asarray([[0.0, 1.0], [1.0, 0.0]])
    evals, evecs = np.linalg.eigh(A_continuum)
    h = 1.0 / kgrid[0]
    link_x = (
        evecs @ np.diag(np.exp(-1.0j * h * evals)) @ evecs.conj().T)
    eye = np.eye(nb, dtype=np.complex128)
    links = np.broadcast_to(eye, (3, nk, nb, nb)).copy()
    links[0] = link_x
    link_x2 = link_x @ link_x
    A_x = (1.0j / (12.0 * h)) * (
        -link_x2 + 8.0 * link_x - 8.0 * link_x.conj().T
        + link_x2.conj().T)
    A_x = 0.5 * (A_x + A_x.conj().T)

    energies = np.broadcast_to(np.asarray([1.0, 0.0]), (nk, nb)).copy()
    occupations = np.broadcast_to(np.asarray([0.0, 1.0]), (nk, nb)).copy()
    H = np.diag(energies[0])
    velocity = np.zeros((3, nk, nb, nb), dtype=np.complex128)
    velocity[0] = (
        -1.0j * (A_x @ H - H @ A_x)
        + np.diag(np.asarray([0.31, -0.17])))
    velocity[1] = np.diag(np.asarray([-0.23, 0.29]))
    velocity[2] = np.asarray([[0.13, 0.07j], [-0.07j, -0.11]])
    gamma = np.moveaxis(HALFALPHA * velocity, 0, 1)
    q1 = np.zeros((nk, 3, 3, nb, nb), dtype=np.complex128)
    q1[:, 1, 0] = np.asarray([[0.02, 0.03j], [-0.03j, -0.01]])
    q1[:, 2, 1] = np.asarray([[-0.04, 0.01], [0.01, 0.05]])
    omegas = jnp.asarray([0.37j], dtype=jnp.complex128)

    got = static_gauge_first_order_component_sharded(
        jnp.asarray(gamma),
        jnp.asarray(q1),
        jnp.asarray(links),
        plus,
        jnp.asarray(energies),
        jnp.asarray(occupations),
        omegas,
        mesh=mesh,
        kgrid=kgrid,
        bvec_cart=np.eye(3),
        nb_logical=nb,
        cell_volume=7.0,
        nk_tot=nk,
        nspin=1,
        nspinor=2,
    )

    A = np.zeros((2, nk, nb, nb), dtype=np.complex128)
    A[0] = A_x
    gamma_dir = np.moveaxis(gamma, 1, 0)
    q_current = np.transpose(q1, (2, 1, 0, 3, 4))[:2]
    d_current = (
        -1.0j * np.einsum("akml,ikln->aikmn", A, gamma_dir)
        + q_current)
    delta = energies[:, :, None] - energies[:, None, :]
    expected_p_current = -delta[None, None] * d_current
    np.testing.assert_allclose(
        got.energy_scaled_d1_raw[:, 0], velocity[:2],
        rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        got.energy_scaled_d1_raw[:, 1:], expected_p_current,
        rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        got.bra_energy_dq_ry,
        -np.real(np.diagonal(velocity[:2], axis1=-2, axis2=-1)),
        rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(
        got.occupation_difference_dq,
        np.zeros_like(np.asarray(got.occupation_difference_dq)))
    assert float(got.charge_ward_residual) < 3.0e-15

    incumbent = head_s_tensor_sharded(
        jnp.asarray(velocity),
        jnp.asarray(energies),
        jnp.asarray(occupations),
        omegas,
        mesh=mesh,
        nb_logical=nb,
        cell_volume=7.0,
        nk_tot=nk,
        nspin=1,
        nspinor=2,
    )
    np.testing.assert_allclose(
        got.S_first_first[:, :, :, 0, 0], incumbent[:, :2, :2],
        rtol=3e-14, atol=3e-14)

    with pytest.raises(ValueError, match="insulating-only"):
        static_gauge_first_order_component_sharded(
            jnp.asarray(gamma), jnp.asarray(q1), jnp.asarray(links), plus,
            jnp.asarray(energies), jnp.asarray(occupations), omegas,
            mesh=mesh, kgrid=kgrid, bvec_cart=np.eye(3), nb_logical=nb,
            cell_volume=7.0, nk_tot=nk, nspin=1, nspinor=2,
            surface_weight_kn=jnp.ones_like(jnp.asarray(energies)))


def test_static_gauge_second_order_retained_jet_and_weight_hessian(
        monkeypatch):
    mesh = _mesh()
    kgrid = (8, 5, 1)
    nk, nb = int(np.prod(kgrid)), 2
    kints = np.stack(np.unravel_index(
        np.arange(nk), kgrid), axis=1).astype(np.int32)
    plus = build_forward_neighbor_table(kints, kgrid)

    sigma_x = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    connections_exact = (2.0 * np.pi * sigma_x, 2.0 * np.pi * sigma_x)
    eye = np.eye(nb, dtype=np.complex128)
    links = np.broadcast_to(eye, (3, nk, nb, nb)).copy()
    connections = []
    for axis, (grid_n, a_exact) in enumerate(zip(kgrid[:2], connections_exact)):
        h = 1.0 / grid_n
        evals, evecs = np.linalg.eigh(a_exact)
        link = evecs @ np.diag(np.exp(-1.0j * h * evals)) @ evecs.conj().T
        links[axis] = link
        link2 = link @ link
        a_discrete = (1.0j / (12.0 * h)) * (
            -link2 + 8.0 * link - 8.0 * link.conj().T
            + link2.conj().T)
        connections.append(0.5 * (a_discrete + a_discrete.conj().T))
    connections = np.asarray(connections)

    energies = np.broadcast_to(np.asarray([1.0, 0.0]), (nk, nb)).copy()
    occupations = np.broadcast_to(np.asarray([0.0, 1.0]), (nk, nb)).copy()
    hamiltonian = np.diag(energies[0])
    velocity = np.zeros((3, nk, nb, nb), dtype=np.complex128)
    velocity[0] = (
        -1.0j * (connections[0] @ hamiltonian
                  - hamiltonian @ connections[0])
        + np.diag(np.asarray([0.31, -0.17])))
    velocity[1] = (
        -1.0j * (connections[1] @ hamiltonian
                  - hamiltonian @ connections[1])
        + np.diag(np.asarray([-0.23, 0.29])))
    velocity[2] = np.asarray([[0.13, 0.07j], [-0.07j, -0.11]])
    gamma = np.moveaxis(HALFALPHA * velocity, 0, 1)

    q1 = np.zeros((nk, 3, 3, nb, nb), dtype=np.complex128)
    q1[:, 0, 0] = np.asarray([[0.02, 0.03j], [-0.03j, -0.01]])
    q1[:, 0, 1] = np.asarray([[-0.04, 0.01], [0.01, 0.05]])
    q1[:, 1, 0] = np.asarray([[0.01, -0.02], [-0.02, -0.03]])
    q1[:, 1, 1] = np.asarray([[0.015, 0.025j], [-0.025j, -0.02]])
    q1[:, 2, 0] = np.asarray([[-0.01, 0.007], [0.007, 0.012]])
    q1[:, 2, 1] = np.asarray([[0.019, -0.011j], [0.011j, -0.014]])
    q2 = np.zeros((nk, 3, 3, 3, nb, nb), dtype=np.complex128)
    q2[:, :, 0, 0] = np.asarray([
        [[0.004, 0.002j], [-0.002j, -0.003]],
        [[-0.002, 0.001], [0.001, 0.005]],
        [[0.003, -0.001j], [0.001j, -0.004]],
    ])
    q2[:, :, 0, 1] = np.asarray([
        [[-0.001, 0.003], [0.003, 0.002]],
        [[0.002, -0.001j], [0.001j, -0.004]],
        [[0.005, 0.002j], [-0.002j, -0.003]],
    ])
    q2[:, :, 1, 0] = q2[:, :, 0, 1]
    q2[:, :, 1, 1] = np.asarray([
        [[0.006, -0.002j], [0.002j, -0.001]],
        [[-0.003, 0.004], [0.004, 0.002]],
        [[0.001, 0.003j], [-0.003j, -0.005]],
    ])
    omegas = jnp.asarray([0.37j], dtype=jnp.complex128)

    matmul_widths = []
    original_make_matmul = (
        parallel_transport_module.make_distributed_band_matmul)

    def tracked_make_matmul(mesh_arg, *, n_batch_axes):
        multiply = original_make_matmul(mesh_arg, n_batch_axes=n_batch_axes)
        if n_batch_axes != 2:
            return multiply

        def tracked_multiply(left, right):
            matmul_widths.append((int(left.shape[0]), int(right.shape[0])))
            return multiply(left, right)

        return tracked_multiply

    monkeypatch.setattr(
        parallel_transport_module,
        "make_distributed_band_matmul",
        tracked_make_matmul,
    )
    got = static_gauge_second_order_component_sharded(
        jnp.asarray(gamma), jnp.asarray(q1), jnp.asarray(q2),
        jnp.asarray(links), plus, jnp.asarray(energies),
        jnp.asarray(occupations), omegas,
        mesh=mesh, kgrid=kgrid, bvec_cart=np.eye(3), nb_logical=nb,
        cell_volume=7.0, nk_tot=nk, nspin=1, nspinor=2)
    assert matmul_widths == [(6, 6)] * 6

    def covariant_derivative_const(operator, axis):
        h = 1.0 / kgrid[axis]
        link = links[axis, 0]
        link2 = link @ link
        tp1 = link @ operator @ link.conj().T
        tp2 = link2 @ operator @ link2.conj().T
        tm1 = link.conj().T @ operator @ link
        tm2 = link2.conj().T @ operator @ link2
        return (-tp2 + 8.0 * tp1 - 8.0 * tm1 + tm2) / (12.0 * h)

    connection_derivative = np.asarray([
        [covariant_derivative_const(connections[a], b) for b in range(2)]
        for a in range(2)])
    ordered_t = np.asarray([
        [1.0j * connection_derivative[a, b]
         - connections[b] @ connections[a] for b in range(2)]
        for a in range(2)])
    retained_t = 0.5 * (ordered_t + np.swapaxes(ordered_t, 0, 1))
    pairs = ((0, 0), (0, 1), (1, 1))
    t_pair = np.asarray([retained_t[a, b] for a, b in pairs])
    gamma_dir = np.moveaxis(gamma, 1, 0)
    q_current = np.transpose(q1, (2, 1, 0, 3, 4))[:2]
    expected_e2 = np.empty((3, 4, nk, nb, nb), np.complex128)
    for p, (a, b) in enumerate(pairs):
        expected_e2[p, 0] = t_pair[p]
        for i in range(3):
            expected_e2[p, i + 1] = (
                t_pair[p] @ gamma_dir[i, 0]
                - 1.0j * (
                    connections[a] @ q_current[b, i, 0]
                    + connections[b] @ q_current[a, i, 0])
                + q2[0, i, a, b])
    np.testing.assert_allclose(
        got.transition_d2_raw, expected_e2, rtol=3e-13, atol=3e-13)

    velocity_hessian = np.asarray([
        [0.5 * (
            covariant_derivative_const(velocity[a, 0], b)
            + covariant_derivative_const(velocity[b, 0], a))
         for b in range(2)] for a in range(2)])
    expected_energy_d2 = np.broadcast_to(np.asarray([
        np.real(np.diag(velocity_hessian[a, b])) for a, b in pairs
    ])[:, None, :], (3, nk, nb))
    np.testing.assert_allclose(
        got.bra_energy_dq2_ry, expected_energy_d2,
        rtol=3e-13, atol=3e-13)
    np.testing.assert_array_equal(
        got.occupation_difference_dq2,
        np.zeros_like(np.asarray(got.occupation_difference_dq2)))
    assert float(got.q2_symmetry_residual) == 0.0
    assert float(got.ordered_curvature_residual) < 2.0e-14

    delta = energies[:, :, None] - energies[:, None, :]
    d1 = np.where(
        delta[None, None] != 0.0,
        -np.asarray(got.first_order.energy_scaled_d1_raw)
        / np.where(delta != 0.0, delta, 1.0)[None, None],
        0.0)
    m0 = np.concatenate((
        np.broadcast_to(eye, (1, nk, nb, nb)), gamma_dir), axis=0)
    f_diff = occupations[:, None, :] - occupations[:, :, None]
    mask = delta > 0.0
    prefactor = 4.0 / (7.0 * nk * 2.0)
    z = complex(np.asarray(omegas)[0])
    g = delta / (z * z - delta * delta)
    gp = (z * z + delta * delta) / (z * z - delta * delta) ** 2
    gpp = (
        2.0 * delta * (3.0 * z * z + delta * delta)
        / (z * z - delta * delta) ** 3)
    phi = f_diff * g
    energy_d1 = np.asarray(got.first_order.bra_energy_dq_ry)
    expected_full = np.zeros((1, 3, 4, 4), np.complex128)
    expected_second_zero = np.zeros_like(expected_full)
    expected_weight = np.zeros_like(expected_full)
    for p, (a, b) in enumerate(pairs):
        phi_a = f_diff * gp * energy_d1[a, :, :, None]
        phi_b = f_diff * gp * energy_d1[b, :, :, None]
        phi_ab = f_diff * (
            gpp * energy_d1[a, :, :, None] * energy_d1[b, :, :, None]
            + gp * expected_energy_d2[p, :, :, None])
        for i in range(4):
            for j in range(4):
                r_a = np.conj(d1[a, i]) * m0[j] + np.conj(m0[i]) * d1[a, j]
                r_b = np.conj(d1[b, i]) * m0[j] + np.conj(m0[i]) * d1[b, j]
                r0 = np.conj(m0[i]) * m0[j]
                first_first = (
                    np.conj(d1[a, i]) * d1[b, j]
                    + np.conj(d1[b, i]) * d1[a, j])
                second_zero = (
                    np.conj(expected_e2[p, i]) * m0[j]
                    + np.conj(m0[i]) * expected_e2[p, j])
                second_term = prefactor * np.sum(
                    np.where(mask, phi * second_zero, 0.0))
                weight_term = prefactor * np.sum(np.where(
                    mask, phi_a * r_b + phi_b * r_a + phi_ab * r0, 0.0))
                first_term = prefactor * np.sum(
                    np.where(mask, phi * first_first, 0.0))
                expected_second_zero[0, p, i, j] = second_term
                expected_weight[0, p, i, j] = weight_term
                expected_full[0, p, i, j] = (
                    first_term + second_term + weight_term)
    np.testing.assert_allclose(
        got.S_second_zero_zero_second, expected_second_zero,
        rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        got.S_response_weight, expected_weight, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        got.S_bubble_second_derivative, expected_full,
        rtol=2e-12, atol=2e-12)
    expected_coefficient_pairs = 0.5 * expected_full
    expected_coefficient_cart = np.stack((
        np.stack((expected_coefficient_pairs[:, 0],
                  expected_coefficient_pairs[:, 1]), axis=1),
        np.stack((expected_coefficient_pairs[:, 1],
                  expected_coefficient_pairs[:, 2]), axis=1),
    ), axis=1)
    np.testing.assert_allclose(
        got.S_bubble_q2_coefficient_cart, expected_coefficient_cart,
        rtol=2e-12, atol=2e-12)
    q_probe = np.asarray([0.17, -0.11])
    expected_quadratic = 0.5 * (
        q_probe[0] ** 2 * expected_full[:, 0]
        + 2.0 * q_probe[0] * q_probe[1] * expected_full[:, 1]
        + q_probe[1] ** 2 * expected_full[:, 2])
    assert np.max(np.abs(expected_quadratic)) > 1.0e-8
    np.testing.assert_allclose(
        np.einsum(
            "a,wabij,b->wij", q_probe,
            np.asarray(got.S_bubble_q2_coefficient_cart), q_probe),
        expected_quadratic,
        rtol=2e-12, atol=2e-12)
    missing_half_quadratic = np.einsum(
        "a,wabij,b->wij", q_probe, 2.0 * expected_coefficient_cart, q_probe)
    assert np.max(np.abs(
        missing_half_quadratic - expected_quadratic)) > 1.0e-8

    malformed_carrier = replace(
        got,
        S_bubble_second_derivative=jnp.zeros(
            (1, 3, 5, 4), dtype=jnp.complex128),
    )
    with pytest.raises(
            ValueError,
            match=r"must have shape \(n_omega,3,4,4\)"):
        _ = malformed_carrier.S_bubble_q2_coefficient_cart

    bad_q2 = q2.copy()
    bad_q2[:, 0, 0, 1, 0, 1] += 1.0e-5
    with pytest.raises(ValueError, match="Q2 transfer indices are not symmetric"):
        static_gauge_second_order_component_sharded(
            jnp.asarray(gamma), jnp.asarray(q1), jnp.asarray(bad_q2),
            jnp.asarray(links), plus, jnp.asarray(energies),
            jnp.asarray(occupations), omegas,
            mesh=mesh, kgrid=kgrid, bvec_cart=np.eye(3), nb_logical=nb,
            cell_volume=7.0, nk_tot=nk, nspin=1, nspinor=2)


def test_pt_loader_reads_stamps_through_read_small_and_never_h5py(monkeypatch):
    """The stamps come through ONE handle, and it is not a second HDF5 stack.

    THIS TEST HAS BEEN WRONG TWICE, in opposite directions, and both
    spellings are worth naming because the property is easy to state and
    easy to pin badly.

    * It first asserted the stamps do NOT go through ``SlabIO.read_slab``.
      True, and for a reason the assertion could not see: a scalar HDF5
      dataspace has no hyperslab, so ``read_slab(shape=())`` refuses at
      ``_normalize_slab_request`` before a byte moves — which is why this
      loader could not read ANY ``parallel_transport.h5``.
    * It then pinned the repair of the day, a short-lived serial-h5py
      owner opened and closed ahead of SlabIO, by faking ``sys.modules
      ["h5py"]``.  Correct about ordering, and still a SECOND HDF5 library
      instance on a file the FFI wrote — the cohabitation class audit A1
      exists to retire, and the route the PHDF5-only ruling forbids.

    What it pins now is the property rather than either implementation:
    every stamp arrives through ``SlabIO.read_small`` on the SAME
    read-only handle the payload uses, ``read_slab`` is never handed an
    empty shape, and ``h5py`` is never opened at all.  The fixture makes
    the last one checkable by installing an h5py whose ``File`` raises.

    UPDATED 2026-08-23 (audit finding, D2 schema-break fix): the
    ``velocity_validation_{atol,rtol,max_abs,...}`` FLOAT diagnostics are
    now read only AFTER the refusal check, not before — a velocity-only
    artifact (``--parallel-transport-velocity-only``) never writes them,
    so reading them unconditionally crashed with a bare HDF5 ``KeyError``
    on that artifact class instead of the named ``ValueError`` refusal
    this test drives.  Every stamp the REFUSAL DECISION itself needs
    (the ints, kgrid, reciprocal lattice, fingerprint) still comes
    through ``read_small`` before any refusal, and is still checked
    below; the floats are deliberately no longer among them.
    """
    import sys

    from common import parallel_transport as pt_common
    import file_io.slab_io as slab_io_mod
    from file_io.parallel_transport import SCHEMA_VERSION

    fingerprint = "a" * 64
    raw = {
        name: np.asarray(value, dtype=np.int32)
        for name, value in {
            "schema_version": SCHEMA_VERSION,
            "connection_complete": 1,
            "velocity_validation_complete": 1,
            # The refusal this fixture drives: validation did not pass, so
            # the loader must raise BEFORE reading the (3, nk, nb, nb)
            # payload.  That ordering is the second thing under test.
            "velocity_validation_passed": 0,
            "band_start": 0,
            "band_stop": 8,
            "effective_nspinor": 1,
            "bispinor": 0,
        }.items()
    }
    raw["kgrid"] = np.asarray([2, 2, 2], dtype=np.int32)
    raw["reciprocal_lattice_cart"] = np.eye(3)
    raw["wfn_fingerprint_utf8"] = np.frombuffer(
        fingerprint.encode("ascii"), dtype=np.uint8
    )
    raw.update(
        {
            f"velocity_validation_{key}": np.asarray(value, dtype=np.float64)
            for key, value in {
                "atol": 5.0e-4,
                "rtol": 5.0e-3,
                "max_abs": 1.0,
                "max_rel": 2.0,
                "max_abs_diagonal": 0.5,
                "max_abs_offdiagonal": 0.75,
                "transition_relative_l2": 0.1,
                "transition_overlap_real": 0.99,
                "transition_overlap_imag": 0.0,
                "head_response_relative_frobenius": 1.0e-3,
                "head_response_trace_ratio": 1.0,
            }.items()
        }
    )

    opens = []
    small_reads = []

    class _FakeSlabIO:
        def __init__(self, path, *, mode="w", mesh=None):
            assert path == "parallel_transport.h5"
            assert mode == "r", (
                "the head loader must open the artifact READ-ONLY; a "
                "writable handle is what made the introspect refuse")
            assert mesh is not None
            opens.append((path, mode))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read_small(self, name, *, dtype=None):
            small_reads.append(name)
            value = raw[name]
            return value if dtype is None else np.asarray(value, dtype=dtype)

        def read_slab(self, name, *, shape=None, **kw):
            raise AssertionError(
                f"read_slab({name!r}, shape={shape!r}) reached the payload "
                f"path, but this fixture refuses on velocity_validation_"
                f"passed=0 and must do so BEFORE any O(nk*nb^2) read")

    class _RefusingH5py:
        @staticmethod
        def File(*a, **k):                       # noqa: N802 - h5py's name
            raise AssertionError(
                "gw.qsgw_head opened h5py.  parallel_transport.h5 is written "
                "by the phdf5 transport; a second HDF5 library instance on "
                "it is audit A1's hazard and the PHDF5-only ruling forbids "
                "it.  Stamps go through SlabIO.read_small.")

    monkeypatch.setattr(slab_io_mod, "SlabIO", _FakeSlabIO)
    monkeypatch.setitem(sys.modules, "h5py", SimpleNamespace(File=_RefusingH5py.File))
    monkeypatch.setattr(pt_common, "wfn_fingerprint", lambda _wfn: fingerprint)

    wfn = SimpleNamespace(kgrid=(2, 2, 2), bvec=np.eye(3), blat=1.0)
    meta = SimpleNamespace(b_id_4_user=8, nspinor=1, nk_tot=8)
    with pytest.raises(ValueError, match="mandatory finite-link DFT head"):
        load_parallel_transport_head(
            "parallel_transport.h5", mesh=_mesh(), wfn=wfn, meta=meta
        )

    assert opens == [("parallel_transport.h5", "r")], (
        f"expected exactly ONE read-only SlabIO open; got {opens}.  Two "
        f"opens means the stamps and the payload are on different handles, "
        f"which is the shape the h5py-then-SlabIO version had.")
    # Every stamp the REFUSAL DECISION needs, and the uint8 provenance
    # stamp, came through the scalar door.
    for name in ("schema_version", "band_stop", "kgrid",
                 "reciprocal_lattice_cart", "wfn_fingerprint_utf8"):
        assert name in small_reads, f"{name} was not read through read_small"
    # The validation FLOATS are read only past the refusal check (see the
    # docstring's 2026-08-23 update) -- this fixture refuses, so none of
    # them were ever read at all.
    assert "velocity_validation_max_abs" not in small_reads, (
        "a validation float was read before the refusal check fired; a "
        "velocity-only artifact does not have these datasets and this "
        "read ordering is what makes loading one crash instead of refuse")


def test_pt_loader_refuses_a_velocity_only_artifact_instead_of_crashing(
    monkeypatch,
):
    """Red twin, audit finding 2026-08-23 (D2 schema-break hunt).

    A ``--parallel-transport-velocity-only`` artifact
    (``initialize_parallel_transport_artifact`` alone, never followed by
    ``write_parallel_transport_artifact``) writes ``connection_complete``
    and ``velocity_validation_{complete,passed}`` as ``0``, but NEVER
    writes the ``velocity_validation_{atol,rtol,max_abs,...}`` float
    datasets at all -- those are only written by
    ``complete_velocity_validation``, at the end of the link/connection
    stage this artifact class skips entirely.

    Reproduced live against a real artifact through this exact loader
    before this fix (``runs/Na/02_soc48b_qsgw_mpa/
    09_dft_velocity_headgate_p16_20260823/veloc_build/
    parallel_transport_velocity_only.h5``): the unconditional read of
    those floats raised a bare ``KeyError: "...doesn't exist"`` instead
    of the named ``ValueError`` this test now pins.  This fixture omits
    those keys from ``raw`` entirely (a plain ``dict`` lookup, so a stray
    read reproduces the same ``KeyError`` shape the real HDF5 backend
    gave) to keep the red twin honest about WHY it would have failed.
    """
    from common import parallel_transport as pt_common
    import file_io.slab_io as slab_io_mod
    from file_io.parallel_transport import SCHEMA_VERSION

    fingerprint = "b" * 64
    raw = {
        name: np.asarray(value, dtype=np.int32)
        for name, value in {
            "schema_version": SCHEMA_VERSION,
            "connection_complete": 0,
            "velocity_validation_complete": 0,
            "velocity_validation_passed": 0,
            "band_start": 0,
            "band_stop": 8,
            "effective_nspinor": 1,
            "bispinor": 0,
        }.items()
    }
    raw["kgrid"] = np.asarray([2, 2, 2], dtype=np.int32)
    raw["reciprocal_lattice_cart"] = np.eye(3)
    raw["wfn_fingerprint_utf8"] = np.frombuffer(
        fingerprint.encode("ascii"), dtype=np.uint8
    )
    # Deliberately NO "velocity_validation_{atol,rtol,max_abs,...}" keys --
    # exactly what a real velocity-only artifact never writes.

    class _FakeSlabIO:
        def __init__(self, path, *, mode="w", mesh=None):
            assert mode == "r"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read_small(self, name, *, dtype=None):
            value = raw[name]
            return value if dtype is None else np.asarray(value, dtype=dtype)

        def read_slab(self, name, *, shape=None, **kw):
            raise AssertionError(
                f"read_slab({name!r}) reached the payload path on a "
                f"velocity-only artifact")

    monkeypatch.setattr(slab_io_mod, "SlabIO", _FakeSlabIO)
    monkeypatch.setattr(pt_common, "wfn_fingerprint", lambda _wfn: fingerprint)

    wfn = SimpleNamespace(kgrid=(2, 2, 2), bvec=np.eye(3), blat=1.0)
    meta = SimpleNamespace(b_id_4_user=8, nspinor=1, nk_tot=8)
    with pytest.raises(ValueError, match="connection_complete is not 1"):
        load_parallel_transport_head(
            "parallel_transport_velocity_only.h5", mesh=_mesh(), wfn=wfn,
            meta=meta,
        )


def test_reduced_covector_conversion_uses_row_basis_convention():
    B = np.asarray(
        [
            [1.8, 0.2, -0.1],
            [0.4, 1.3, 0.3],
            [-0.2, 0.1, 0.9],
        ]
    )
    reduced = np.arange(3 * 2 * 2).reshape(3, 2, 2) / 7.0
    got = np.asarray(reduced_covector_to_cartesian(reduced, B))
    ref = np.einsum("ij,jab->iab", np.linalg.inv(B), reduced)
    wrong_transpose = np.einsum("ij,jab->iab", np.linalg.inv(B).T, reduced)
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=2e-15)
    assert np.max(np.abs(got - wrong_transpose)) > 1e-2


def test_distributed_velocity_rotation_matches_explicit_u_dagger_v_u():
    rng = np.random.default_rng(551)
    nk, nb = 3, 8
    v = rng.standard_normal((3, nk, nb, nb)) + 1j * rng.standard_normal((3, nk, nb, nb))
    U = np.stack([_haar(rng, nb) for _ in range(nk)])
    got = np.asarray(
        rotate_velocity_to_qp(jnp.asarray(v), jnp.asarray(U), mesh=_mesh())
    )
    ref = np.einsum("kmp,akmn,knq->akpq", np.conj(U), v, U, optimize=True)
    np.testing.assert_allclose(got, ref, rtol=3e-15, atol=3e-15)
    # Negative control: U v U-dagger shares norms but is not this basis map.
    wrong = np.einsum("kpm,akmn,kqn->akpq", U, v, np.conj(U), optimize=True)
    assert np.max(np.abs(got - wrong)) > 1e-2


def test_sharded_s_tensor_pads_unaligned_manifold_and_matches_dft_formula():
    rng = np.random.default_rng(813)
    nk, nb, nocc = 4, 7, 3
    energies = np.sort(rng.uniform(-0.8, 1.4, (nk, nb)), axis=1)
    occ = np.zeros((nk, nb))
    occ[:, :nocc] = 1.0
    velocity = rng.standard_normal((3, nk, nb, nb)) + 1j * rng.standard_normal(
        (3, nk, nb, nb)
    )
    omegas = np.asarray([0.0 + 0.0j, 0.0 + 0.61j])
    volume = 83.5
    got = np.asarray(
        head_s_tensor_sharded(
            jnp.asarray(velocity),
            jnp.asarray(energies),
            jnp.asarray(occ),
            jnp.asarray(omegas),
            mesh=_mesh(),
            nb_logical=nb,
            cell_volume=volume,
            nk_tot=nk,
            nspin=1,
            nspinor=1,
        )
    )
    delta_e = energies[:, :, None] - energies[:, None, :]
    ref = np.asarray(
        compute_S_omega(
            jnp.asarray(velocity),
            jnp.asarray(delta_e),
            jnp.asarray(occ),
            volume,
            nk,
            1,
            1,
            jnp.asarray(omegas),
        )
    )
    np.testing.assert_allclose(got, ref, rtol=4e-15, atol=4e-15)


def test_packed_vertex_s_tensor_preserves_incumbent_three_axis_bits():
    """Width three stays exact; packed width eight has one direct oracle."""
    rng = np.random.default_rng(20260825)
    nk, nb = 2, 5
    energies = np.sort(rng.normal(size=(nk, nb)), axis=1)
    occupations = np.zeros((nk, nb), dtype=np.float64)
    occupations[:, :2] = 1.0
    v3 = (rng.normal(size=(3, nk, nb, nb))
          + 1j * rng.normal(size=(3, nk, nb, nb)))
    v8 = np.zeros((8, nk, nb, nb), dtype=np.complex128)
    v8[:3] = v3
    kwargs = dict(
        mesh=_mesh(), nb_logical=nb, cell_volume=41.0, nk_tot=nk,
        nspin=1, nspinor=2)
    s3 = np.asarray(head_s_tensor_sharded(
        v3, energies, occupations, [0.37j], **kwargs))
    s3_from_packed = np.asarray(head_s_tensor_sharded(
        v8[:3], energies, occupations, [0.37j], **kwargs))
    s8 = np.asarray(head_s_tensor_sharded(
        v8, energies, occupations, [0.37j], **kwargs))
    np.testing.assert_array_equal(s3_from_packed, s3)

    delta_e = energies[:, :, None] - energies[:, None, :]
    s8_oracle = np.asarray(compute_S_omega(
        v8, delta_e, occupations, 41.0, nk, 1, 2,
        np.asarray([0.37j])))
    scale = max(1.0, float(np.max(np.abs(s8_oracle))))
    assert np.max(np.abs(s8 - s8_oracle)) <= 32.0 * np.finfo(float).eps * scale
    np.testing.assert_array_equal(s8[:, 3:, :], 0.0)
    np.testing.assert_array_equal(s8[:, :, 3:], 0.0)
    with pytest.raises(ValueError, match="canonical n_vertex"):
        head_s_tensor_sharded(
            v8[:4], energies, occupations, [0.37j], **kwargs)


def test_raw_hall_matches_orbital_cB_owner_and_documented_sign():
    """Distributed raw Hall is the incumbent orbital cB with one rescale."""
    from common.bispinor_init import HALFALPHA
    from psp.orbital_magnetization import orbital_pieces_at_k

    rng = np.random.default_rng(82531)
    nk, nb, nocc = 3, 6, 3
    energies = np.sort(rng.normal(size=(nk, nb)), axis=1)
    # A same-occupation degeneracy exercises the incumbent denominator mask
    # without making the insulating occupied/unoccupied separation invalid.
    energies[:, 1] = energies[:, 0]
    raw = (rng.normal(size=(3, nk, nb, nb))
           + 1j * rng.normal(size=(3, nk, nb, nb)))
    velocity = 0.5 * (raw + np.conj(np.swapaxes(raw, -1, -2)))
    occupations = np.zeros((nk, nb), dtype=np.float64)
    occupations[:, :nocc] = 1.0
    gamma_raw = HALFALPHA * np.transpose(velocity, (1, 0, 2, 3))
    volume = 37.25
    nspin, nspinor_wfn = 1, 2

    got = np.asarray(raw_hall_pseudovector_sharded(
        gamma_raw, energies, occupations,
        mesh=_mesh(), nb_logical=nb, cell_volume=volume, nk_tot=nk,
        nspin=nspin, nspinor_wfn=nspinor_wfn,
        degeneracy_tolerance_ry=1.0e-10))
    cB = np.zeros(3, dtype=np.complex128)
    for k in range(nk):
        _pa, pb = orbital_pieces_at_k(
            velocity[:, k], energies[k], nocc, 1.0e-10)
        cB += pb.sum(axis=(1, 2)) / nk
    capacity = 2.0 / (nspin * nspinor_wfn)
    expected = -(HALFALPHA * capacity / volume) * np.imag(cB)
    np.testing.assert_allclose(got, expected, rtol=3e-14, atol=3e-14)
    # Red twin: omitting the occupied-Berry minus is an observable sign flip.
    assert np.max(np.abs(got - (-expected))) > 1.0e-10


def test_raw_hall_fractional_occupations_and_degeneracy_refusal():
    from common.bispinor_init import HALFALPHA

    rng = np.random.default_rng(82532)
    nk, nb = 2, 5
    energies = np.asarray([
        [-1.0, -0.2, 0.4, 0.9, 1.6],
        [-0.8, -0.1, 0.5, 1.1, 1.7],
    ])
    occupations = np.asarray([
        [1.0, 0.73, 0.21, 0.0, 0.0],
        [1.0, 0.61, 0.18, 0.0, 0.0],
    ])
    raw = (rng.normal(size=(3, nk, nb, nb))
           + 1j * rng.normal(size=(3, nk, nb, nb)))
    velocity = 0.5 * (raw + np.conj(np.swapaxes(raw, -1, -2)))
    gamma_raw = HALFALPHA * np.transpose(velocity, (1, 0, 2, 3))
    got = np.asarray(raw_hall_pseudovector_sharded(
        gamma_raw, energies, occupations,
        mesh=_mesh(), nb_logical=nb, cell_volume=29.0, nk_tot=nk,
        nspin=1, nspinor_wfn=2))
    with pytest.raises(ValueError, match="one Gamma_raw row per full-BZ"):
        raw_hall_pseudovector_sharded(
            gamma_raw, energies, occupations,
            mesh=_mesh(), nb_logical=nb, cell_volume=29.0, nk_tot=nk + 1,
            nspin=1, nspinor_wfn=2)

    cB = np.zeros(3, dtype=np.complex128)
    for k in range(nk):
        for n in range(nb):
            for m in range(nb):
                de = energies[k, n] - energies[k, m]
                if abs(de) <= 1.0e-10:
                    continue
                vn, vm = velocity[:, k, n, m], velocity[:, k, m, n]
                cB += (occupations[k, n] / (nk * de * de)) * np.asarray((
                    vn[1] * vm[2] - vn[2] * vm[1],
                    vn[2] * vm[0] - vn[0] * vm[2],
                    vn[0] * vm[1] - vn[1] * vm[0],
                ))
    expected = -(HALFALPHA / 29.0) * np.imag(cB)
    np.testing.assert_allclose(got, expected, rtol=3e-14, atol=3e-14)

    bad_energies = energies.copy()
    bad_energies[0, 2] = bad_energies[0, 1]
    with pytest.raises(ValueError, match="differently occupied states"):
        raw_hall_pseudovector_sharded(
            gamma_raw, bad_energies, occupations,
            mesh=_mesh(), nb_logical=nb, cell_volume=29.0, nk_tot=nk,
            nspin=1, nspinor_wfn=2)


def test_static_gauge_hall_transaction_uses_file_wedge_service_and_fingerprint():
    """One full-BZ uniform transaction owns Hall values and provenance."""
    rng = np.random.default_rng(82601)
    nk, logical, storage = 2, 3, 4
    raw = (rng.normal(size=(nk, 3, storage, storage))
           + 1j * rng.normal(size=(nk, 3, storage, storage)))
    gamma = 0.5 * (raw + np.conj(np.swapaxes(raw, -1, -2)))
    fingerprint = "sha256:" + "c" * 64
    uniform = UniformGaugeCurrentMatrixElements(
        gamma_raw=jnp.asarray(gamma),
        hamiltonian_config_operator_fingerprint=fingerprint,
    )
    energies_file = np.asarray([[[-1.1, -0.3, 0.8]]], dtype=np.float64)
    occupations_file = np.asarray([[[1.0, 1.0, 0.0]]], dtype=np.float64)
    wfn = SimpleNamespace(
        energies=energies_file,
        occs=occupations_file,
        nbands=logical,
        nspin=1,
        nspinor=2,
        cell_volume=31.0,
    )
    sym = SimpleNamespace(
        nk_tot=nk,
        nk_red=1,
        irr_idx_k=np.asarray([0, 0], dtype=np.int32),
        sym_idx_k=np.asarray([0, 1], dtype=np.int32),
        sym_mats_k=np.stack((np.eye(3), -np.eye(3))),
    )

    got = static_gauge_hall_transaction(
        uniform,
        wfn=wfn,
        sym=sym,
        band_start=0,
        band_stop=logical,
        mesh=_mesh(),
    )
    expected = raw_hall_pseudovector_sharded(
        uniform.gamma_raw,
        np.repeat(energies_file[0], nk, axis=0),
        np.repeat(occupations_file[0], nk, axis=0),
        mesh=_mesh(),
        nb_logical=logical,
        cell_volume=31.0,
        nk_tot=nk,
        nspin=1,
        nspinor_wfn=2,
    )
    assert isinstance(got, StaticGaugeHallTransaction)
    np.testing.assert_allclose(got.sigma_H, expected, rtol=0.0, atol=0.0)
    assert got.hamiltonian_config_operator_fingerprint == fingerprint
    assert (got.band_start, got.band_stop, got.nk_tot) == (0, logical, nk)
    assert got.producer_id == "lorrax.static_gauge_hall/full_bz_uniform_gauge_v1"
    with pytest.raises(TypeError, match="issued only"):
        replace(got, _producer_token=object())


def test_uniform_gauge_fingerprint_is_contact_capability_only():
    """Ordinary VNL setup does not hash tables it cannot authenticate."""
    from psp.vnl_ops import build_vnl_setup

    wfn = SimpleNamespace(
        nspinor=1,
        blat=1.0,
        bvec=np.eye(3),
        cell_volume=1.0,
        atom_types=np.zeros(0, dtype=np.int32),
        atom_crys=np.zeros((0, 3), dtype=np.float64),
    )
    ordinary = build_vnl_setup(
        wfn, pseudos={}, n_q=2, q_max=1.0, soc=False,
        compute_contact=False, print_fn=lambda *_: None)
    contact = build_vnl_setup(
        wfn, pseudos={}, n_q=2, q_max=1.0, soc=False,
        compute_contact=True, print_fn=lambda *_: None)
    assert (
        ordinary.uniform_gauge_fingerprint,
        contact.uniform_gauge_fingerprint.startswith("sha256:"),
    ) == ("", True)


def test_head_manifold_embedding_preserves_inactive_identity_and_cross_blocks():
    rng = np.random.default_rng(19)
    nk, na, nf = 2, 3, 6
    d = rng.standard_normal((nk, na, na)) + 1j * rng.standard_normal((nk, na, na))
    U = np.stack([_haar(rng, na) for _ in range(nk)])
    d_full, U_full = assemble_head_manifold(
        jnp.asarray(d), jnp.asarray(U), nb_storage=nf, mesh=_mesh()
    )
    d_full, U_full = np.asarray(d_full), np.asarray(U_full)
    np.testing.assert_array_equal(d_full[:, :na, :na], d)
    np.testing.assert_array_equal(d_full[:, na:, :], 0.0)
    np.testing.assert_array_equal(d_full[:, :, na:], 0.0)
    np.testing.assert_array_equal(U_full[:, :na, :na], U)
    np.testing.assert_array_equal(
        U_full[:, na:, na:], np.broadcast_to(np.eye(nf - na), (nk, nf - na, nf - na))
    )
    np.testing.assert_array_equal(U_full[:, :na, na:], 0.0)
    np.testing.assert_array_equal(U_full[:, na:, :na], 0.0)


def test_s_tensor_ignores_mesh_padding_bands():
    rng = np.random.default_rng(71)
    nk, nb, nb_pad, nocc = 2, 5, 8, 2
    e = np.sort(rng.normal(size=(nk, nb_pad)), axis=1)
    f = np.zeros_like(e)
    f[:, :nocc] = 1.0
    v = rng.normal(size=(3, nk, nb_pad, nb_pad)) + 1j * rng.normal(
        size=(3, nk, nb_pad, nb_pad)
    )
    # Deliberately enormous pad entries: a logical-manifold mask must make
    # them exactly inert rather than merely small.
    v[:, :, nb:, :] *= 1e9
    v[:, :, :, nb:] *= 1e9
    kw = dict(
        mesh=_mesh(),
        nb_logical=nb,
        cell_volume=31.0,
        nk_tot=nk,
        nspin=1,
        nspinor=1,
    )
    got = np.asarray(head_s_tensor_sharded(v, e, f, [0.2j], **kw))
    delta_e = e[:, :nb, None] - e[:, None, :nb]
    ref = np.asarray(
        compute_S_omega(
            jnp.asarray(v[:, :, :nb, :nb]),
            jnp.asarray(delta_e),
            jnp.asarray(f[:, :nb]),
            31.0,
            nk,
            1,
            1,
            jnp.asarray([0.2j]),
        )
    )
    # The padded contraction contains explicit zero terms and XLA may group
    # its reduction differently; inert means roundoff parity, not byte parity.
    np.testing.assert_allclose(got, ref, rtol=3e-15, atol=3e-15)


def test_frequency_blocked_isdf_wings_match_direct_transition_sum():
    """Y/Z retain all-band and surface terms across a block boundary."""
    rng = np.random.default_rng(714)
    nk, nb, ns, nmu, nw = 3, 5, 2, 4, 11
    energies = np.sort(rng.uniform(-0.8, 1.2, (nk, nb)), axis=1)
    occupations = np.asarray([
        [1.0, 0.91, 0.23, 0.0, 0.0],
        [1.0, 0.77, 0.12, -0.01, 0.0],
        [1.0, 0.84, 0.31, 0.02, 0.0],
    ])
    velocity = (
        rng.normal(size=(3, nk, nb, nb))
        + 1j * rng.normal(size=(3, nk, nb, nb)))
    psi = (
        rng.normal(size=(nk, ns, nmu, nb))
        + 1j * rng.normal(size=(nk, ns, nmu, nb)))
    surface = rng.uniform(0.0, 0.4, (nk, nb))
    omegas = np.linspace(0.07, 0.93, nw) + 0.11j
    Y, Z = head_wings_sharded(
        velocity,
        SimpleNamespace(psi_xn=jnp.asarray(psi), psi_yn=jnp.asarray(psi)),
        energies,
        occupations,
        omegas,
        mesh=_mesh(),
        nb_logical=nb,
        nk_tot=nk,
        nspin=1,
        nspinor=2,
        surface_weight_kn=surface,
    )

    pref_inter = 4.0 / (nk * 2.0)
    pref_surface = 2.0 / (nk * 2.0)
    Y_ref = np.zeros((nw, 3, nmu), dtype=np.complex128)
    Z_ref = np.zeros((nw, nmu, 3), dtype=np.complex128)
    for k in range(nk):
        for i in range(nb):
            density = np.einsum(
                "sm,sm->m", np.conj(psi[k, :, :, i]), psi[k, :, :, i])
            surface_weight = pref_surface * surface[k, i] / omegas
            Y_ref += (
                surface_weight[:, None, None]
                * np.conj(velocity[:, k, i, i])[None, :, None]
                * density[None, None, :])
            Z_ref += (
                surface_weight[:, None, None]
                * np.conj(density)[None, :, None]
                * velocity[:, k, i, i][None, None, :])
            for j in range(nb):
                delta = energies[k, i] - energies[k, j]
                if delta <= 0.0:
                    continue
                weight = (
                    pref_inter * (occupations[k, j] - occupations[k, i])
                    / (omegas**2 - delta**2))
                b_ij = np.einsum(
                    "sm,sm->m", np.conj(psi[k, :, :, i]), psi[k, :, :, j])
                Y_ref += (
                    weight[:, None, None]
                    * np.conj(velocity[:, k, i, j])[None, :, None]
                    * b_ij[None, None, :])
                Z_ref += (
                    weight[:, None, None]
                    * np.conj(b_ij)[None, :, None]
                    * velocity[:, k, i, j][None, None, :])
    np.testing.assert_allclose(np.asarray(Y), Y_ref, rtol=8e-14, atol=8e-14)
    np.testing.assert_allclose(np.asarray(Z), Z_ref, rtol=8e-14, atol=8e-14)


def test_finite_link_derivative_is_exactly_gauge_covariant():
    rng = np.random.default_rng(410)
    # A two-point periodic axis makes the +1 and -1 neighbours identical,
    # so every centred derivative (and the red twin below) is exactly zero.
    # Five points exercise the actual fourth-order covariant stencil.
    grid = (5, 5, 5)
    nk, nb = int(np.prod(grid)), 8
    coords = np.stack(
        np.meshgrid(*(np.arange(n) for n in grid), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    plus = build_forward_neighbor_table(coords, grid)
    links = np.broadcast_to(
        np.eye(nb, dtype=np.complex128), (3, nk, nb, nb)
    ).copy()
    raw = rng.normal(size=(nk, nb, nb)) + 1j * rng.normal(size=(nk, nb, nb))
    operator = raw + np.swapaxes(raw.conj(), -1, -2)
    gauge = np.stack([_haar(rng, nb) for _ in range(nk)])
    operator_gauge = np.einsum(
        "kmi,kmn,knj->kij", gauge.conj(), operator, gauge, optimize=True
    )
    links_gauge = np.empty_like(links)
    for idir in range(3):
        links_gauge[idir] = np.einsum(
            "kmi,kmn,knj->kij",
            gauge.conj(), links[idir], gauge[plus[:, idir]], optimize=True
        )

    kwargs = dict(
        mesh=_mesh(), kgrid=grid,
        bvec_cart=np.asarray([[1.7, 0.1, 0.0], [0.0, 1.2, 0.2], [0.1, 0.0, 0.9]])
    )
    reference = np.asarray(covariant_link_derivative(
        jnp.asarray(operator), jnp.asarray(links), plus, **kwargs))
    got = np.asarray(covariant_link_derivative(
        jnp.asarray(operator_gauge), jnp.asarray(links_gauge), plus, **kwargs))
    expected = np.einsum(
        "kmi,akmn,knj->akij", gauge.conj(), reference, gauge, optimize=True
    )
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=2e-13)

    # Red twin: rotating Delta H without co-rotating the links is not gauge
    # covariant and must remain observably wrong.
    wrong = np.asarray(covariant_link_derivative(
        jnp.asarray(operator_gauge), jnp.asarray(links), plus, **kwargs))
    assert np.max(np.abs(wrong - expected)) > 1.0e-2


def test_active_rotation_matches_dense_block_diagonal_rotation():
    rng = np.random.default_rng(411)
    nk, nb, na = 2, 8, 4
    velocity = rng.normal(size=(3, nk, nb, nb)) + 1j * rng.normal(size=(3, nk, nb, nb))
    U = np.stack([_haar(rng, na) for _ in range(nk)])

    got = np.asarray(
        rotate_velocity_active_to_qp(
            jnp.asarray(velocity), jnp.asarray(U), mesh=_mesh()
        )
    )
    U_full = np.broadcast_to(np.eye(nb, dtype=np.complex128), (nk, nb, nb)).copy()
    U_full[:, :na, :na] = U
    ref = np.einsum(
        "kmi,akmn,knj->akij", U_full.conj(), velocity, U_full, optimize=True
    )
    np.testing.assert_allclose(got, ref, rtol=3e-14, atol=3e-14)


def test_s_tensor_uses_k_dependent_occupations_not_band_cut():
    rng = np.random.default_rng(412)
    nk, nb, logical = 2, 8, 5
    energies = np.broadcast_to(np.arange(nb, dtype=np.float64), (nk, nb)).copy()
    energies[1] += 0.17
    occupations = np.zeros((nk, nb), dtype=np.float64)
    occupations[0, [0, 2]] = 1.0
    occupations[1, [1, 3]] = 1.0
    velocity = rng.normal(size=(3, nk, nb, nb)) + 1j * rng.normal(size=(3, nk, nb, nb))
    omega = 0.31j
    volume = 29.0

    got = np.asarray(
        head_s_tensor_sharded(
            jnp.asarray(velocity),
            jnp.asarray(energies),
            jnp.asarray(occupations),
            [omega],
            mesh=_mesh(),
            nb_logical=logical,
            cell_volume=volume,
            nk_tot=nk,
            nspin=1,
            nspinor=1,
        )
    )[0]
    ref = np.zeros((3, 3), dtype=np.complex128)
    prefactor = 4.0 / (volume * nk)
    for k in range(nk):
        for c in range(logical):
            for v in range(logical):
                de = energies[k, c] - energies[k, v]
                fd = occupations[k, v] - occupations[k, c]
                if de <= 0.0:
                    continue
                weight = prefactor * fd / (de * (omega * omega - de * de))
                ref += weight * np.outer(
                    velocity[:, k, c, v].conj(), velocity[:, k, c, v]
                )
    np.testing.assert_allclose(got, ref, rtol=5e-14, atol=5e-14)
