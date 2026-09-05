"""CPU contract and red-twin gates for parallel-transport preprocessing."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from common.parallel_transport import (
    band_storage_extent,
    build_forward_neighbor_table,
    build_g_wrap_lookup,
    fourth_order_connection,
    g_wrap_for_forward_step,
    make_distributed_band_matmul,
    undersampled_link_axes,
    wfn_fingerprint,
)
import file_io.parallel_transport as pt_io
from file_io.parallel_transport import (
    CONNECTION_CART_DATASET,
    ENERGIES_DATASET,
    LINKS_DATASET,
    OCCUPATIONS_DATASET,
    SINGULAR_VALUES_DATASET,
    VELOCITY_DFT_DATASET,
    complete_velocity_validation,
    initialize_parallel_transport_artifact,
    write_parallel_transport_artifact,
)
from gw.gw_config import read_lorrax_input


def test_w_av_neighbor_flags_are_input_controls(tmp_path):
    deck = tmp_path / "cohsex.in"
    deck.write_text(
        "[cohsex]\n"
        "W_av_first_neighbors = true\nW_av_second_neighbors = false\n")
    params = read_lorrax_input(str(deck))
    assert params["w_av_first_neighbors"] is True
    assert params["w_av_second_neighbors"] is False


def test_common_storage_extent_handles_rectangular_meshes():
    mesh = types.SimpleNamespace(
        axis_names=("x", "y"), shape={"x": 2, "y": 4})
    assert band_storage_extent(mesh, 5) == 8
    assert band_storage_extent(mesh, 8) == 8


def test_wfn_fingerprint_refuses_sampled_coefficient_mismatch(
        tmp_path, monkeypatch):
    """Equal headers must not hide a genuinely different fixed-gauge WFN."""
    import h5py

    path_a = tmp_path / "WFN_a.h5"
    path_b = tmp_path / "WFN_b.h5"
    coeffs = np.arange(24, dtype=np.float64).reshape(3, 1, 4, 2)
    for path in (path_a, path_b):
        with h5py.File(path, "w") as h5:
            h5.create_dataset("wfns/gvecs", data=np.arange(
                12, dtype=np.int32).reshape(4, 3))
            h5.create_dataset("wfns/coeffs", data=coeffs)

    # Change a sampled coefficient while every in-memory header field stays
    # identical.  A path/inode fingerprint passes this test for the wrong
    # reason; the red phase removes that identity and must then expose the
    # collision until bounded coefficient content is included.
    with h5py.File(path_b, "r+") as h5:
        h5["wfns/coeffs"][0, 0, 0, 0] += 1.0e-12

    common = dict(
        energies=np.asarray([[0.1, 0.2]]),
        kpoints=np.asarray([[0.0, 0.0, 0.0]]),
        nelec=1, nspinor=1, nbands=2)
    wfn_a = types.SimpleNamespace(path=str(path_a), **common)
    wfn_b = types.SimpleNamespace(path=str(path_b), **common)
    assert wfn_fingerprint(wfn_a) != wfn_fingerprint(wfn_b), (
        "a coefficient mismatch inside the bounded sample must change the "
        "fingerprint, or removing path/inode identity disables the guard")

    from common import sanity
    from psp.get_dipole_mtxels import (
        check_dipole_provenance, stamp_dipole_provenance)

    monkeypatch.setattr(sanity, "sanity_strict", lambda: False)
    dipole = tmp_path / "dipole.h5"
    with h5py.File(dipole, "w") as h5:
        stamp_dipole_provenance(
            h5, wfn=wfn_a, wfn_path=path_a,
            nval=8, ncond=32, nband=40, nb_written=40,
            bispinor=False, skip_vnl=False)
    lines = []
    assert check_dipole_provenance(
        dipole, wfn=wfn_b, nval=8, ncond=32, nband=40,
        print_fn=lines.append) is False
    assert any("prov_wfn_sha256" in line for line in lines)


def test_wfn_fingerprint_accepts_byte_identical_copy(tmp_path):
    """Filesystem identity is not part of the mean-field identity."""
    import h5py

    path_a = tmp_path / "first" / "WFN.h5"
    path_b = tmp_path / "second" / "WFN.h5"
    path_a.parent.mkdir()
    path_b.parent.mkdir()
    with h5py.File(path_a, "w") as h5:
        h5.create_dataset("mf_header/versionnumber", data=np.asarray([1, 0]))
        h5.create_dataset(
            "wfns/gvecs", data=np.arange(12, dtype=np.int32).reshape(4, 3))
        h5.create_dataset(
            "wfns/coeffs",
            data=np.arange(24, dtype=np.float64).reshape(3, 1, 4, 2))
    shutil.copyfile(path_a, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()
    assert path_a.stat().st_ino != path_b.stat().st_ino

    common = dict(
        energies=np.asarray([[0.1, 0.2]]),
        kpoints=np.asarray([[0.0, 0.0, 0.0]]),
        nelec=1, nspinor=1, nbands=2)
    wfn_a = types.SimpleNamespace(path=str(path_a), **common)
    wfn_b = types.SimpleNamespace(path=str(path_b), **common)
    assert wfn_fingerprint(wfn_a) == wfn_fingerprint(wfn_b)

    from psp.get_dipole_mtxels import (
        check_dipole_provenance, stamp_dipole_provenance)

    dipole = tmp_path / "dipole.h5"
    with h5py.File(dipole, "w") as h5:
        stamp_dipole_provenance(
            h5, wfn=wfn_a, wfn_path=path_a,
            nval=8, ncond=32, nband=40, nb_written=40,
            bispinor=False, skip_vnl=False)
    lines = []
    assert check_dipole_provenance(
        dipole, wfn=wfn_b, nval=8, ncond=32, nband=40,
        print_fn=lines.append) is True
    assert any("window nval=8 ncond=32 nband=40" in line for line in lines)


def test_undersampled_link_axes_names_only_two_point_axes():
    """Per-axis rule (2026-09-05): >= 5 points fourth order, 3-4 second
    order, 1 point collapsed (position operator), 2 points refused."""
    from common.parallel_transport import COLLAPSED_AXIS, link_stencil_orders
    assert undersampled_link_axes((8, 8, 8)) == []
    assert undersampled_link_axes((5, 5, 5)) == []          # exactly at floor
    assert undersampled_link_axes((4, 6, 6)) == []          # second order now
    assert undersampled_link_axes((9, 9, 1)) == []          # 2D slab: collapsed
    assert undersampled_link_axes((3, 1, 2)) == ["z"]       # only the 2-point axis
    assert link_stencil_orders((8, 8, 8)) == (4, 4, 4)
    assert link_stencil_orders((6, 6, 1)) == (4, 4, COLLAPSED_AXIS)
    assert link_stencil_orders((3, 3, 1)) == (2, 2, COLLAPSED_AXIS)
    assert link_stencil_orders((4, 6, 6)) == (2, 4, 4)
    assert link_stencil_orders((5, 1, 1)) == (4, COLLAPSED_AXIS, COLLAPSED_AXIS)
    with pytest.raises(ValueError, match="pt_two_point_axis"):
        link_stencil_orders((2, 6, 6))


def test_link_stage_refuses_a_two_point_axis_before_io():
    """D2/D3(c): the stencil gate lives on the LINK remainder, not the
    velocity-writing initializer, and fires before any I/O.  Only a
    two-point axis is unsupported now (no finite difference, not a vacuum
    direction)."""
    wfn = types.SimpleNamespace(kgrid=np.asarray([2, 6, 6]))
    try:
        write_parallel_transport_artifact(
            "must-not-open.h5", wfn=wfn, sym=None, mesh=None, nbands=1,
            bispinor=False)
    except ValueError as exc:
        msg = str(exc)
        assert "PT-LINK-STENCIL-UNSUPPORTED" in msg
        assert "kgrid=(2, 6, 6)" in msg and "two-point axes x" in msg
    else:
        raise AssertionError("two-point mesh was accepted")


def test_link_stage_accepts_collapsed_and_second_order_axes_at_the_gate():
    """A collapsed slab axis (kgrid[i]=1) and a 3- or 4-point axis pass the
    gate (2026-09-05): the collapsed axis takes the real-space position
    operator, the short axes the second-order stencil.  The probe fails
    downstream on the ``sym=None`` it supplies, past the removed refusal."""
    for grid in ((9, 9, 1), (4, 6, 6), (3, 3, 1), (7, 1, 1)):
        wfn = types.SimpleNamespace(kgrid=np.asarray(grid))
        with pytest.raises((AttributeError, TypeError, RuntimeError)):
            write_parallel_transport_artifact(
                "must-not-open.h5", wfn=wfn, sym=None, mesh=None, nbands=1,
                bispinor=False)


def test_initialize_no_longer_gates_the_velocity_write_on_kgrid():
    """D2: the exact-DFT-velocity write is unconditional on kgrid.

    Before this fix ``initialize_parallel_transport_artifact`` raised its
    OWN ``ValueError`` naming "at least five" on this exact 2D kgrid, before
    ``sym``/``mesh``/``velocity_dft_kmajor`` were ever touched (the test this
    replaces).  Now the same undersampled/collapsed kgrid must run PAST that
    removed check and fail downstream instead, on the ``sym=None`` this call
    intentionally supplies as a probe -- proving the gate is gone rather than
    merely relocated to a spot this call happens not to reach.
    """
    wfn = types.SimpleNamespace(kgrid=np.asarray([9, 9, 1]))
    with pytest.raises(AttributeError):
        initialize_parallel_transport_artifact(
            "must-not-open.h5", wfn=wfn, sym=None, mesh=None, nbands=1,
            effective_nspinor=1, bispinor=False,
            velocity_dft_kmajor=None, wfn_path="WFN.h5",
            wfn_fingerprint="0" * 64)


def test_forward_neighbors_do_not_assume_flattening_order():
    """A permuted full-grid table still produces the exact three neighbours."""
    grid = np.array([2, 2, 1])
    coords = np.array([
        [1, 1, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ])
    got = build_forward_neighbor_table(coords, grid)
    row = {tuple(v): i for i, v in enumerate(coords)}
    for ik, k in enumerate(coords):
        for idir in range(3):
            target = k.copy()
            target[idir] = (target[idir] + 1) % grid[idir]
            assert got[ik, idir] == row[tuple(target)]


def test_full_energy_rows_follow_symmaps_full_order_not_ibz_row_order():
    """irr_idx_k is the exact full-row source; kirr_fullids is inverse idiom."""
    wfn = types.SimpleNamespace(
        energies=np.array([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]]),
        occs=np.array([[[2.0, 0.0], [1.0, 0.0], [0.5, 0.0]]]),
    )
    sym = types.SimpleNamespace(
        irr_idx_k=np.array([2, 0, 2, 1], dtype=np.int32),
        kirr_fullids=np.array([1, 3, 0], dtype=np.int32),
    )
    energies, occupations = pt_io._full_band_tables(wfn, sym, 2)
    np.testing.assert_array_equal(
        energies, [[30, 31], [10, 11], [30, 31], [20, 21]])
    np.testing.assert_array_equal(
        occupations, [[0.5, 0], [2, 0], [0.5, 0], [1, 0]])
    np.testing.assert_array_equal(
        energies[sym.kirr_fullids], np.asarray(wfn.energies)[0])


def test_g_wrap_lookup_boundary_and_wrong_sign_red_twin():
    """The +G_wrap convention crosses the zone edge; its sign twin does not."""
    kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    wrap = g_wrap_for_forward_step(kpts, 1, 0, 0, (2, 1, 1))
    np.testing.assert_array_equal(wrap, [1, 0, 0])

    sentinel = [99, 99, 99]
    g_center = np.array([
        [-1, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        sentinel,
        sentinel,
    ])
    g_neighbor = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        sentinel,
        sentinel,
    ])
    index, valid = build_g_wrap_lookup(
        g_neighbor, g_center, wrap, ngk_neighbor=3, ngk_center=3)
    np.testing.assert_array_equal(index[:3], [0, 1, 2])
    np.testing.assert_array_equal(valid, [True, True, True, False, False])

    _wrong_index, wrong_valid = build_g_wrap_lookup(
        g_neighbor, g_center, -wrap, ngk_neighbor=3, ngk_center=3)
    # Negative control: the real boundary data lose two of three PW rows.
    assert int(np.count_nonzero(wrong_valid)) == 1


def test_cross_k_link_fuses_overlap_and_polar_on_hostile_2x2_mesh_subprocess():
    """The nb=5/8 overlap stays tiled through its planned polar factor."""
    code = r"""
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from common.parallel_transport import make_cross_k_link, make_cross_k_overlap
from distrib_la import plan_polar_factor

assert len(jax.devices()) == 4, len(jax.devices())
mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
rng = np.random.default_rng(8)
center = rng.normal(size=(1, 8, 1, 6)) + 1j*rng.normal(size=(1, 8, 1, 6))
neighbor = rng.normal(size=(1, 8, 1, 6)) + 1j*rng.normal(size=(1, 8, 1, 6))
center[:, 5:] = 0
neighbor[:, 5:] = 0
index = np.array([2, 0, 4, 1, 3, 5], dtype=np.int32)
valid = np.ones(6, dtype=bool)
xy = NamedSharding(mesh, P(None, ("x", "y"), None, None))
center_xy = jax.device_put(center, xy)
neighbor_xy = jax.device_put(neighbor, xy)
polar = plan_polar_factor(mesh, n=8, backend="off", rcond=1.0e-12)
to_x, make_link = make_cross_k_link(mesh, polar)
got, singular = make_link(to_x(center_xy), neighbor_xy, index, valid)
aligned = neighbor[..., index]
raw = np.einsum("kbsg,knsg->kbn", np.conj(center), aligned)[0]
to_x_raw, make_raw = make_cross_k_overlap(mesh)
got_raw = make_raw(to_x_raw(center_xy), neighbor_xy, index, valid)
np.testing.assert_allclose(
    np.asarray(got_raw), raw, rtol=2e-12, atol=2e-12)
assert tuple(got_raw.sharding.spec) == ("x", "y")
centers = np.concatenate([center, (0.2 - 0.4j) * center], axis=0)
neighbors = np.concatenate([
    neighbor, (0.5 + 0.25j) * neighbor, (-0.3 + 0.1j) * neighbor], axis=0)
indexes = np.stack([
    index, np.roll(index, 1), np.roll(index, -2)], axis=0)
valids = np.ones_like(indexes, dtype=bool)
valids[1, -1] = False
neighbors_kq = np.stack([neighbors, (0.7 + 0.1j) * neighbors], axis=0)
indexes_kq = np.stack([indexes, np.flip(indexes, axis=0)], axis=0)
valids_kq = np.stack([valids, np.roll(valids, 1, axis=1)], axis=0)
centers_xy = jax.device_put(
    centers, NamedSharding(mesh, P(None, ("x", "y"), None, None)))
neighbors_kq_xy = jax.device_put(
    neighbors_kq,
    NamedSharding(mesh, P(None, None, ("x", "y"), None, None)))
got_kq = make_raw(
    to_x_raw(centers_xy), neighbors_kq_xy, indexes_kq, valids_kq)
raw_kq = np.empty((3, 2, 8, 8), dtype=np.complex128)
for ik in range(2):
    for iq in range(3):
        aligned_kq = np.take(
            neighbors_kq[ik:ik + 1, iq], indexes_kq[ik, iq], axis=-1)
        aligned_kq = np.where(
            valids_kq[ik, iq][None, None, None, :], aligned_kq, 0.0)
        raw_kq[iq, ik] = np.einsum(
            "kbsg,knsg->bn", np.conj(centers[ik:ik + 1]), aligned_kq)
np.testing.assert_allclose(
    np.asarray(got_kq), raw_kq, rtol=2e-12, atol=2e-12)
assert tuple(got_kq.sharding.spec) == (None, None, "x", "y")
u, s, vh = np.linalg.svd(raw[:5, :5], full_matrices=False)
got_host = np.asarray(got)
singular_host = np.asarray(singular)
np.testing.assert_allclose(
    got_host[:5, :5], u @ vh, rtol=2e-12, atol=2e-12)
np.testing.assert_allclose(
    singular_host[:5], s, rtol=2e-12, atol=2e-12)
np.testing.assert_allclose(
    singular_host[5:], 0.0, rtol=0.0, atol=2e-12)
assert tuple(got.sharding.spec) == ("x", "y")
assert tuple(singular.sharding.spec) == ()
assert {part.data.shape for part in got.addressable_shards} == {(4, 4)}
np.testing.assert_allclose(
    got_host[5:, :], 0.0, rtol=0.0, atol=2e-12)
np.testing.assert_allclose(
    got_host[:, 5:], 0.0, rtol=0.0, atol=2e-12)
"""
    env = os.environ.copy()
    flags = env.get("XLA_FLAGS", "")
    env["XLA_FLAGS"] = (
        flags + " --xla_force_host_platform_device_count=4").strip()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    repo = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(repo / "src"),
        str(repo / "services" / "distrib_la" / "src"),
        str(repo / "services" / "lxkit" / "src"),
        env.get("PYTHONPATH", ""),
    )))
    run = subprocess.run(
        [sys.executable, "-c", code], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert run.returncode == 0, run.stdout


def test_distributed_band_matmul_hostile_2x2_mesh_subprocess():
    """One-axis gathers retain a P(x,y) result for nb=8 / logical nb=5."""
    code = r"""
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from common.parallel_transport import make_distributed_band_matmul

assert len(jax.devices()) == 4, len(jax.devices())
mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
rng = np.random.default_rng(9)
A = rng.normal(size=(3, 8, 8)) + 1j*rng.normal(size=(3, 8, 8))
B = rng.normal(size=(3, 8, 8)) + 1j*rng.normal(size=(3, 8, 8))
A[:, 5:] = 0
A[:, :, 5:] = 0
B[:, 5:] = 0
B[:, :, 5:] = 0
xy = NamedSharding(mesh, P(None, "x", "y"))
multiply = make_distributed_band_matmul(mesh, n_batch_axes=1)
got = multiply(jax.device_put(A, xy), jax.device_put(B, xy))
np.testing.assert_allclose(np.asarray(got), A @ B, rtol=2e-13, atol=2e-13)
assert tuple(got.sharding.spec) == (None, "x", "y")
assert {s.data.shape for s in got.addressable_shards} == {(3, 4, 4)}
assert np.count_nonzero(np.asarray(got)[:, 5:, :]) == 0
assert np.count_nonzero(np.asarray(got)[:, :, 5:]) == 0
"""
    env = os.environ.copy()
    flags = env.get("XLA_FLAGS", "")
    env["XLA_FLAGS"] = (
        flags + " --xla_force_host_platform_device_count=4").strip()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    run = subprocess.run(
        [sys.executable, "-c", code], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert run.returncode == 0, run.stdout


def test_fourth_order_connection_and_orientation_red_twin():
    """L=exp(-iAh) reconstructs +A; storing L dagger reconstructs -A."""
    grid = (8, 1, 1)
    coords = np.stack(np.meshgrid(
        np.arange(8), np.arange(1), np.arange(1), indexing="ij"),
        axis=-1).reshape(-1, 3)
    plus = build_forward_neighbor_table(coords, grid)
    h = 1.0 / grid[0]
    expected = 0.7
    links = np.ones((3, 8, 1, 1), dtype=np.complex128)
    links[0, :, 0, 0] = np.exp(-1.0j * expected * h)
    got = np.asarray(fourth_order_connection(
        jnp.asarray(links), plus, (h, 1.0, 1.0),
        band_matmul=lambda left, right: left @ right))
    np.testing.assert_allclose(got[0, :, 0, 0], expected, atol=2.0e-6)
    np.testing.assert_array_equal(got[1:], 0.0)
    np.testing.assert_array_equal(got, np.swapaxes(got.conj(), -1, -2))

    wrong = np.asarray(fourth_order_connection(
        jnp.asarray(links.conj()), plus, (h, 1.0, 1.0),
        band_matmul=lambda left, right: left @ right))
    np.testing.assert_allclose(wrong[0, :, 0, 0], -expected, atol=2.0e-6)
    assert np.max(np.abs(wrong - got)) > 1.0


def _identity_links(nk, nb):
    return np.broadcast_to(
        np.eye(nb, dtype=np.complex128), (3, nk, nb, nb)).copy()


def test_completion_runs_shared_spectral_service_and_stamps(monkeypatch):
    """Producer completion calls the shared finite-link service and passes."""
    energies = np.array([[0.2, 0.8], [0.3, 0.9], [0.25, 0.85]])
    occupations = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    exact = np.zeros((3, 3, 2, 2), dtype=np.complex128)
    exact[:, :, 1, 0] = np.array(
        [1.0 + 0.2j, 0.4 - 0.3j, -0.7 + 0.1j])[:, None]
    exact[:, :, 0, 1] = np.conj(exact[:, :, 1, 0])
    # (3, 1, 1): a second-order axis and two collapsed axes (2026-09-05); a
    # two-point axis has no derivative rule and is refused by name.
    plus = build_forward_neighbor_table(
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), (3, 1, 1))
    zero_position = np.zeros((3, 3, 2, 2), dtype=np.complex128)
    calls = []
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))

    def covariant(H, links, neighbors, spacing, *, band_matmul,
                  stencil_orders=None, collapsed_position=None):
        del band_matmul, collapsed_position
        calls.append((tuple(H.shape), tuple(links.shape),
                      np.asarray(neighbors), np.asarray(spacing),
                      tuple(stencil_orders)))
        return jnp.asarray(exact)

    monkeypatch.setattr(
        pt_io, "fourth_order_covariant_derivative", covariant)
    monkeypatch.setattr(
        pt_io, "write_velocity_validation",
        lambda path, *, mesh, metrics: calls.append(("stamp", metrics.copy())))
    metrics = complete_velocity_validation(
        "not-opened.h5", mesh=mesh, kgrid=(3, 1, 1),
        bvec_cart=np.eye(3), energies_full=energies,
        occupations_full=occupations, velocity_exact_cart=exact,
        links_full=_identity_links(3, 2), forward_neighbors=plus,
        atol=1.0e-12, rtol=1.0e-12, collapsed_position=zero_position)
    assert metrics["passed"]
    derivative_calls = [c for c in calls if c[0] != "stamp"]
    assert len(derivative_calls) == 1
    shapes = derivative_calls[0]
    assert shapes[0] == (3, 2, 2)
    assert shapes[1] == (3, 3, 2, 2)
    np.testing.assert_array_equal(shapes[2], plus)
    np.testing.assert_allclose(shapes[3], [1.0 / 3.0, 1.0, 1.0])
    assert shapes[4] == (2, 0, 0)          # second order, collapsed, collapsed
    assert calls[-1][0] == "stamp"
    assert calls[-1][1]["passed"]
    assert calls[-1][1]["transition_overlap_real"] == pytest.approx(1.0)
    assert calls[-1][1]["head_response_trace_ratio"] == pytest.approx(1.0)


def test_completion_sign_red_twin_stamps_failure_then_refuses(monkeypatch):
    """A phase-reversed derivative fails even though its response is equal."""
    energies = np.array([[0.2, 0.8], [0.3, 0.9], [0.25, 0.85]])
    occupations = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    exact = np.zeros((3, 3, 2, 2), dtype=np.complex128)
    exact[:, :, 1, 0] = np.array(
        [1.0 + 0.2j, 0.4 - 0.3j, -0.7 + 0.1j])[:, None]
    exact[:, :, 0, 1] = np.conj(exact[:, :, 1, 0])
    # (3, 1, 1): a second-order axis and two collapsed axes (2026-09-05); a
    # two-point axis has no derivative rule and is refused by name.
    plus = build_forward_neighbor_table(
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), (3, 1, 1))
    zero_position = np.zeros((3, 3, 2, 2), dtype=np.complex128)
    stamps = []
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))

    def wrong_covariant(H, links, neighbors, spacing, *, band_matmul,
                        stencil_orders=None, collapsed_position=None):
        del H, links, neighbors, spacing, band_matmul
        del stencil_orders, collapsed_position
        return -jnp.asarray(exact)

    monkeypatch.setattr(
        pt_io, "fourth_order_covariant_derivative", wrong_covariant)
    monkeypatch.setattr(
        pt_io, "write_velocity_validation",
        lambda path, *, mesh, metrics: stamps.append(metrics.copy()))
    with np.testing.assert_raises_regex(
            RuntimeError,
            "parallel-transport finite-link DFT head validation failed"):
        complete_velocity_validation(
            "not-opened.h5", mesh=mesh, kgrid=(3, 1, 1),
            bvec_cart=np.eye(3), energies_full=energies,
            occupations_full=occupations, velocity_exact_cart=exact,
            links_full=_identity_links(3, 2), forward_neighbors=plus,
            atol=1.0e-12, rtol=1.0e-12, collapsed_position=zero_position)
    assert len(stamps) == 1
    assert not stamps[0]["passed"]
    assert stamps[0]["head_response_relative_frobenius"] < 1.0e-12
    assert stamps[0]["transition_overlap_real"] == pytest.approx(-1.0)


def test_artifact_schema_is_slabio_only_and_names_the_head_manifold():
    """The new artifact has no raw h5py escape hatch and stable reader keys."""
    source = Path(
        sys.modules["file_io.parallel_transport"].__file__).read_text()
    assert "import h5py" not in source
    assert "h5py.File" not in source
    assert "SlabIO(" in source
    assert LINKS_DATASET == "links_ibz"
    assert SINGULAR_VALUES_DATASET == "singular_values_ibz"
    assert CONNECTION_CART_DATASET == "berry_connection_cart"
    assert VELOCITY_DFT_DATASET == "velocity_dft_cart"
    assert ENERGIES_DATASET == "dft_energies_ry_full"
    assert OCCUPATIONS_DATASET == "dft_occupations_full"



_CROSSING_KGRID = (8, 8, 8)
_CROSSING_BVEC = np.array([[1.1, 0.05, 0.0],
                           [0.0, 0.9, 0.03],
                           [0.02, 0.0, 1.3]])


def _two_band_crossing_fixture(crossing: bool, kgrid=_CROSSING_KGRID):
    """A four-band band-limited model whose lowest TWO bands are the window.

    ``crossing=True`` puts a genuine band crossing inside that window: the
    two window bands disperse oppositely along kappa_1 and are separated
    only by a small off-diagonal, so the band-index SORTED energies are
    kinked and the sorted eigenvectors swap character at the crossing.
    ``crossing=False`` pulls them apart by a constant 5 Ry gap and changes
    nothing else, which is the insulating control.

    Bands 3 and 4 sit ~6 Ry above and are coupled weakly to the window, so
    the window is a TRUNCATED manifold exactly as a real band window is:
    the link singular values are below one and the transported connection
    ``A_t`` is genuinely non-zero, not a degenerate zero.

    Everything is an explicit Fourier series, so ``dH/dkappa`` — and
    therefore the exact DFT velocity — is analytic, and the model is band
    limited on this mesh.  Eigenvectors come from ``eigh``, i.e. with the
    arbitrary per-k phase a real WFN also has.
    """
    grid = tuple(int(n) for n in kgrid)
    coords = np.stack(np.meshgrid(*[np.arange(n) for n in grid],
                                  indexing="ij"), axis=-1).reshape(-1, 3)
    kappa = coords / np.asarray(grid, dtype=np.float64)[None, :]
    nk = coords.shape[0]

    d0 = np.diag([0.0, 0.0, 6.0, 8.0]).astype(np.complex128)
    if not crossing:
        d0[0, 0], d0[1, 1] = -2.5, 2.5
    harmonics = {}
    c100 = np.zeros((4, 4), dtype=np.complex128)
    # exp(i pi/8) moves the two zeros of the window dispersion OFF the mesh
    c100[0, 0] = 0.5 * np.exp(1j * np.pi / 8.0)
    c100[1, 1] = -0.5 * np.exp(1j * np.pi / 8.0)
    c100[0, 2] = 0.25
    harmonics[(1, 0, 0)] = c100
    c010 = np.zeros((4, 4), dtype=np.complex128)
    c010[0, 1] = 0.010j
    c010[3, 3] = 0.15
    c010[1, 3] = 0.25
    harmonics[(0, 1, 0)] = c010
    c001 = np.zeros((4, 4), dtype=np.complex128)
    c001[0, 0] = c001[1, 1] = 0.075
    c001[2, 3] = 0.10
    c001[0, 1] = 0.005
    harmonics[(0, 0, 1)] = c001

    H = np.broadcast_to(d0, (nk, 4, 4)).copy()
    dH = np.zeros((3, nk, 4, 4), dtype=np.complex128)
    for R, C in harmonics.items():
        phase = np.exp(2j * np.pi * (kappa @ np.asarray(R, dtype=np.float64)))
        Ch = np.conj(C.T)
        H += phase[:, None, None] * C + np.conj(phase)[:, None, None] * Ch
        for j in range(3):
            f = 2j * np.pi * R[j]
            dH[j] += (f * phase)[:, None, None] * C \
                + np.conj(f * phase)[:, None, None] * Ch

    energies = np.empty((nk, 4))
    vectors = np.empty((nk, 4, 4), dtype=np.complex128)
    for k in range(nk):
        energies[k], vectors[k] = np.linalg.eigh(H[k])
    window = vectors[:, :, :2]
    dH_cart = np.einsum("ij,j...->i...", np.linalg.inv(_CROSSING_BVEC), dH)
    velocity = np.einsum("knm,dknp,kpq->dkmq",
                         np.conj(window), dH_cart, window)

    plus = build_forward_neighbor_table(coords, grid)
    links = np.empty((3, nk, 2, 2), dtype=np.complex128)
    smallest = 1.0
    for d in range(3):
        overlap = np.einsum("knm,knp->kmp", np.conj(window),
                            window[plus[:, d]])
        u, s, vh = np.linalg.svd(overlap)
        smallest = min(smallest, float(s.min()))
        links[d] = u @ vh
    return {
        "kgrid": grid, "plus": plus, "energies": energies[:, :2],
        "velocity": velocity, "links": links, "nk": nk,
        "min_singular_value": smallest,
        # |L_nn| collapses exactly where the sorted band index swaps
        # character between neighbouring k: the band-crossing fingerprint.
        "min_band_diagonal": float(np.min(np.abs(
            np.einsum("dknn->dkn", links)))),
        "window_gap": float(np.min(energies[:, 1] - energies[:, 0])),
        "outer_gap": float(np.min(energies[:, 2] - energies[:, 1])),
    }


def _run_fixture_gate(fixture, monkeypatch, atol=5.0e-4, rtol=5.0e-3):
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    occupations = np.zeros_like(fixture["energies"])
    occupations[:, 0] = 1.0
    stamped = []
    monkeypatch.setattr(
        pt_io, "write_velocity_validation",
        lambda path, *, mesh, metrics: stamped.append(metrics.copy()))
    metrics = complete_velocity_validation(
        "not-opened.h5", mesh=mesh, kgrid=fixture["kgrid"],
        bvec_cart=_CROSSING_BVEC, energies_full=fixture["energies"],
        occupations_full=occupations,
        velocity_exact_cart=fixture["velocity"],
        links_full=fixture["links"], forward_neighbors=fixture["plus"],
        atol=atol, rtol=rtol)
    assert len(stamped) == 1
    return metrics


def test_finite_link_gate_passes_a_band_crossing_and_refuses_wrong_orientation(
        monkeypatch):
    """The head-observable gate accepts a crossing and rejects bad links."""
    fixture = _two_band_crossing_fixture(crossing=True)
    assert fixture["min_band_diagonal"] < 0.1, "the window bands must cross"
    assert fixture["outer_gap"] > 2.0, "bands 3-4 must stay out of the window"
    assert 1.0 - fixture["min_singular_value"] > 1.0e-4, (
        "the window must be a truncated manifold, not an isolated one")

    metrics = _run_fixture_gate(fixture, monkeypatch)

    assert metrics["passed"]
    assert metrics["head_response_relative_frobenius"] < 5.0e-3
    assert metrics["transition_overlap_real"] > 0.995

    # Orientation red twin: conjugating the stored links flips the transport
    # AND the sign of the connection it builds, and must be refused.
    twin = dict(fixture)
    twin["links"] = np.conj(fixture["links"])
    with np.testing.assert_raises_regex(
            RuntimeError, "finite-link DFT head"):
        _run_fixture_gate(twin, monkeypatch)


def test_finite_link_gate_passes_a_non_crossing_window(monkeypatch):
    """The same production gate accepts the separated insulating control."""
    fixture = _two_band_crossing_fixture(crossing=False)
    assert fixture["window_gap"] > 2.0, "the control must not cross"
    assert fixture["min_band_diagonal"] > 0.9, "no band swaps in the control"
    assert 1.0 - fixture["min_singular_value"] > 1.0e-4, (
        "the window must be a truncated manifold, not an isolated one")

    metrics = _run_fixture_gate(fixture, monkeypatch)

    assert metrics["passed"]
    assert metrics["head_response_relative_frobenius"] < 5.0e-3
    assert metrics["transition_overlap_real"] > 0.995


def test_transported_frame_closes_every_line_on_the_torus():
    """The frame is unitary, and twisting makes the transported link constant.

    Untwisted, the frame jumps by the Wilson loop at the zone boundary and
    H_t is not periodic; the FFT derivative then rings on that step.  With
    the twist the transported link is the per-line constant S^(-1/N), which
    is what makes the fourth-order connection of it exact.
    """
    fixture = _two_band_crossing_fixture(crossing=True)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    band_matmul = make_distributed_band_matmul(mesh, n_batch_axes=1)
    links = jnp.asarray(fixture["links"])
    plus = fixture["plus"]
    for idir in range(3):
        U, defect = pt_io.transported_frame(
            links[idir], plus, idir, fixture["kgrid"][idir],
            mesh=mesh, band_matmul=band_matmul)
        U = np.asarray(U)
        assert defect > 1.0e-8, "a truncated window must have a holonomy"
        np.testing.assert_allclose(
            np.einsum("kji,kjl->kil", np.conj(U), U),
            np.broadcast_to(np.eye(2), U.shape), atol=2.0e-12)
        transported = np.einsum(
            "kji,kjl,klm->kim", np.conj(U), np.asarray(links[idir]),
            U[plus[:, idir]])
        lines = pt_io.line_index_table(plus, idir, fixture["kgrid"][idir])
        for line in lines:
            first = transported[line[0]]
            for node in line[1:]:
                np.testing.assert_allclose(
                    transported[node], first, atol=2.0e-12)


def test_fourth_order_negative_hop_keeps_noncommuting_product_order():
    rng = np.random.default_rng(921)
    nk, nb = 8, 2
    kints = np.stack((np.arange(nk), np.zeros(nk), np.zeros(nk)), axis=1)
    plus = build_forward_neighbor_table(kints, (nk, 1, 1))
    links = np.broadcast_to(
        np.eye(nb, dtype=np.complex128), (3, nk, nb, nb)).copy()
    for k in range(nk):
        raw = (rng.normal(size=(nb, nb))
               + 1j * rng.normal(size=(nb, nb)))
        q, r = np.linalg.qr(raw)
        phase = np.diag(r)
        links[0, k] = q * (phase / np.abs(phase))[None, :]

    def multiply(left, right):
        return jnp.einsum("...ij,...jk->...ik", left, right)

    got = np.asarray(fourth_order_connection(
        jnp.asarray(links), plus, (1.0 / nk, 1.0, 1.0),
        band_matmul=multiply))[0]
    minus = np.empty(nk, dtype=np.int32)
    minus[plus[:, 0]] = np.arange(nk)
    ref = np.empty_like(got)
    wrong = np.empty_like(got)
    for k in range(nk):
        km1 = minus[k]
        km2 = minus[km1]
        lp1 = links[0, k]
        lp2 = lp1 @ links[0, plus[k, 0]]
        lm1 = links[0, km1].conj().T
        second = links[0, km2].conj().T
        raw_ref = 1j * (-lp2 + 8 * lp1 - 8 * lm1 + lm1 @ second) \
            / (12.0 / nk)
        raw_wrong = 1j * (-lp2 + 8 * lp1 - 8 * lm1 + second @ lm1) \
            / (12.0 / nk)
        ref[k] = 0.5 * (raw_ref + raw_ref.conj().T)
        wrong[k] = 0.5 * (raw_wrong + raw_wrong.conj().T)
    np.testing.assert_allclose(got, ref, rtol=2e-14, atol=2e-14)
    assert np.max(np.abs(got - wrong)) > 1.0e-3
