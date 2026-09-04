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
)
from gw.head_correction import static_hall_linear_response

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
            "parallel_transport.h5", mesh=_mesh(),
            sym=SimpleNamespace(trs_allowed=True), wfn=wfn, meta=meta
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
            sym=SimpleNamespace(trs_allowed=True), meta=meta,
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

    # Name both orientations explicitly.  Hall sums occupied bra (v,c),
    # whereas the live AW kernel energy-orders bra (c,v) and conjugates that
    # row.  Hermiticity makes the products equal, but the persisted Hall
    # tensor carries the occupied-Berry minus.  Its CT insertion therefore
    # needs the second minus.
    delta = energies[:, :, None] - energies[:, None, :]
    separated = np.abs(delta) > 1.0e-10
    inv_delta2 = np.where(
        separated, 1.0 / np.square(np.where(separated, delta, 1.0)), 0.0)
    gamma_dir = np.transpose(gamma_raw, (1, 0, 2, 3))
    p_charge = gamma_dir[:2] / HALFALPHA
    F = 2.0 * capacity / (volume * nk)
    hall_occupied_bra = -F * np.imag(np.einsum(
        "aknm,iknm,knm->ai", p_charge, np.conj(gamma_dir),
        occupations[:, :, None] * inv_delta2, optimize=True))
    axes = np.eye(3, dtype=np.float64)[:2]
    hall_from_sigma = np.stack(
        [np.cross(got, axis) for axis in axes], axis=0)
    np.testing.assert_allclose(
        hall_from_sigma, hall_occupied_bra, rtol=3e-14, atol=3e-14)

    f_diff = occupations[:, None, :] - occupations[:, :, None]
    energy_ordered = delta > 0.0
    direct_aw = F * np.imag(np.einsum(
        "aknm,iknm,knm->ai", np.conj(p_charge), gamma_dir,
        np.where(energy_ordered, f_diff * inv_delta2, 0.0),
        optimize=True))
    np.testing.assert_allclose(
        direct_aw, -hall_occupied_bra, rtol=3e-14, atol=3e-14)
    inserted = np.asarray(static_hall_linear_response(got))[:, 0, 1:].imag
    np.testing.assert_allclose(
        inserted, direct_aw, rtol=3e-14, atol=3e-14)
    assert np.max(np.abs((-inserted) - direct_aw)) > 1.0e-10


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
        kpoints=np.zeros((1, 1, 3), dtype=np.float64),
        nelec=2,
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
        wfn, pseudos={}, n_q=2, q_max=1.0,
        compute_contact=False, print_fn=lambda *_: None)
    contact = build_vnl_setup(
        wfn, pseudos={}, n_q=2, q_max=1.0,
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

    pref_inter = -4.0 / (nk * 2.0)
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


def test_head_wing_interband_sign_matches_direct_adler_wiser_density_jet():
    """Two-level direct-AW oracle for the one-leg ``D -> P`` sign.

    Use the energy-ordered pair ``(c,v)`` with ``Delta=2``, ``P_cv=b_cv=1``
    and therefore ``D_cv=-P_cv/Delta=-1/2``.  At ``z=0`` the mixed
    density-jet Adler--Wiser coefficient is

        F * Delta * conj(D) * b / (z**2 - Delta**2) = +F/4.

    The old positive interband kernel prefactor returned ``-F/4``.  This
    oracle names both representations so neither band orientation nor the
    defining ``P=-Delta*D`` minus can be hidden in a copied kernel formula.
    """
    delta = 2.0
    energies = np.asarray([[delta, 0.0]])
    occupations = np.asarray([[0.0, 1.0]])
    P_head = np.zeros((3, 1, 2, 2), dtype=np.complex128)
    P_head[0, 0, 0, 1] = 1.0
    body = np.ones((1, 1, 1, 2), dtype=np.complex128)
    omega = np.asarray([0.0 + 0.0j])

    Y, Z = head_wings_sharded(
        P_head,
        SimpleNamespace(psi_xn=jnp.asarray(body), psi_yn=jnp.asarray(body)),
        energies,
        occupations,
        omega,
        mesh=_mesh(),
        nb_logical=2,
        nk_tot=1,
        nspin=1,
        nspinor=1,
    )

    prefactor = 4.0
    p_cv = P_head[0, 0, 0, 1]
    d_cv = -p_cv / delta
    b_cv = 1.0 + 0.0j
    denom = omega[0] ** 2 - delta ** 2
    direct_aw = prefactor * delta * np.conj(d_cv) * b_cv / denom
    assert direct_aw == prefactor / 4.0
    np.testing.assert_allclose(np.asarray(Y)[0, 0, 0], direct_aw,
                               rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(Z)[0, 0, 0], direct_aw,
                               rtol=0.0, atol=1.0e-14)


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
