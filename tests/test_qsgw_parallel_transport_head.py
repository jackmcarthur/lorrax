"""Small independent oracles for the no-wavefunction QSGW head kernels."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from common.chi_from_dipole import compute_S_omega
from common.parallel_transport import build_forward_neighbor_table
from gw.qsgw_head import (
    assemble_head_manifold,
    covariant_link_derivative,
    head_wings_sharded,
    head_s_tensor_sharded,
    load_parallel_transport_head,
    reduced_covector_to_cartesian,
    rotate_velocity_active_to_qp,
    rotate_velocity_to_qp,
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
    # Every stamp the refusal list needs, and the uint8 provenance stamp,
    # came through the scalar door.
    for name in ("schema_version", "band_stop", "kgrid",
                 "reciprocal_lattice_cart", "wfn_fingerprint_utf8",
                 "velocity_validation_max_abs"):
        assert name in small_reads, f"{name} was not read through read_small"


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
    grid = (2, 2, 2)
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
