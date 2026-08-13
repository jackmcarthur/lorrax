"""CPU contract and red-twin gates for parallel-transport preprocessing."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from common.parallel_transport import (
    build_forward_neighbor_table,
    build_g_wrap_lookup,
    fourth_order_connection,
    g_wrap_for_forward_step,
    make_distributed_band_matmul,
)
import file_io.parallel_transport as pt_io
from file_io.parallel_transport import (
    CONNECTION_CART_DATASET,
    ENERGIES_DATASET,
    LINKS_DATASET,
    OCCUPATIONS_DATASET,
    VELOCITY_DFT_DATASET,
    complete_velocity_validation,
    validate_covariant_velocity,
)


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


def test_cross_k_overlap_hostile_2x2_mesh_subprocess():
    """nb=5 is padded to 8 on a 2x2 mesh and no shard owns the full matrix."""
    code = r"""
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from common.parallel_transport import make_cross_k_overlap

assert len(jax.devices()) == 4, len(jax.devices())
mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
rng = np.random.default_rng(8)
center = rng.normal(size=(1, 8, 1, 6)) + 1j*rng.normal(size=(1, 8, 1, 6))
neighbor = rng.normal(size=(1, 8, 1, 6)) + 1j*rng.normal(size=(1, 8, 1, 6))
center[:, 5:] = 0
neighbor[:, 5:] = 0
index = np.array([2, 0, 4, 1, 3, 0], dtype=np.int32)
valid = np.array([1, 1, 0, 1, 1, 0], dtype=bool)
xy = NamedSharding(mesh, P(None, ("x", "y"), None, None))
center_xy = jax.device_put(center, xy)
neighbor_xy = jax.device_put(neighbor, xy)
to_x, overlap = make_cross_k_overlap(mesh)
got = overlap(to_x(center_xy), neighbor_xy, index, valid)
aligned = neighbor[..., index]
aligned = np.where(valid[None, None, None, :], aligned, 0)
want = np.einsum("kbsg,knsg->kbn", np.conj(center), aligned)
np.testing.assert_allclose(np.asarray(got), want, rtol=2e-13, atol=2e-13)
assert tuple(got.sharding.spec) == (None, "x", "y")
assert {s.data.shape for s in got.addressable_shards} == {(1, 4, 4)}
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


def test_velocity_validation_covers_complex_diagonal_and_offdiagonal():
    """The pass arm is exact and a commutator-sign red twin is rejected."""
    rng = np.random.default_rng(11)
    nk, nb = 2, 3
    A0 = rng.normal(size=(3, nk, nb, nb)) + 1j * rng.normal(
        size=(3, nk, nb, nb))
    A = 0.5 * (A0 + np.swapaxes(A0.conj(), -1, -2))
    H0 = rng.normal(size=(nk, nb, nb)) + 1j * rng.normal(
        size=(nk, nb, nb))
    H = 0.5 * (H0 + np.swapaxes(H0.conj(), -1, -2))
    exact0 = rng.normal(size=A.shape) + 1j * rng.normal(size=A.shape)
    exact = 0.5 * (exact0 + np.swapaxes(exact0.conj(), -1, -2))
    comm = A @ H[None] - H[None] @ A
    dH = exact + 1.0j * comm
    metrics = validate_covariant_velocity(
        dH, A, H, exact, band_matmul=np.matmul,
        atol=1.0e-12, rtol=1.0e-12)
    assert metrics["passed"]
    assert metrics["max_abs_diagonal"] < 1.0e-12
    assert metrics["max_abs_offdiagonal"] < 1.0e-12

    red = validate_covariant_velocity(
        exact - 1.0j * comm, A, H, exact, band_matmul=np.matmul,
        atol=1.0e-12, rtol=1.0e-12)
    assert not red["passed"]
    assert red["max_abs_offdiagonal"] > 1.0e-3


def test_completion_runs_shared_spectral_service_and_stamps(monkeypatch):
    """Production completion calls the one shared derivative and stamps pass."""
    energies = np.array([[0.2, 0.8], [0.3, 0.9]])
    A = np.zeros((3, 2, 2, 2), dtype=np.complex128)
    exact = np.zeros_like(A)
    calls = []
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))

    def covariant(H, A_arg, *, mesh, kgrid, bvec_cart):
        calls.append((
            tuple(H.shape), tuple(A_arg.shape), tuple(kgrid),
            np.asarray(bvec_cart)))
        return jnp.zeros_like(A)

    monkeypatch.setitem(
        sys.modules, "gw.qsgw_head",
        types.SimpleNamespace(covariant_cartesian_derivative=covariant))
    monkeypatch.setattr(
        pt_io, "write_velocity_validation",
        lambda path, *, mesh, metrics: calls.append(("stamp", metrics.copy())))
    metrics = complete_velocity_validation(
        "not-opened.h5", mesh=mesh, kgrid=(2, 1, 1),
        bvec_cart=np.eye(3), energies_full=energies,
        connection_cart=A, velocity_exact_cart=exact,
        atol=1.0e-12, rtol=1.0e-12)
    assert metrics["passed"]
    assert calls[0][0] == (2, 2, 2)
    assert calls[0][1] == (3, 2, 2, 2)
    assert calls[0][2] == (2, 1, 1)
    assert calls[1][0] == "stamp"
    assert calls[1][1]["passed"]


def test_completion_sign_red_twin_stamps_failure_then_refuses(monkeypatch):
    """A wrong shared derivative cannot leave a passing artifact."""
    energies = np.array([[0.2, 0.8], [0.3, 0.9]])
    A = np.zeros((3, 2, 2, 2), dtype=np.complex128)
    exact = np.zeros_like(A)
    stamps = []
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))

    def wrong_covariant(H, A_arg, *, mesh, kgrid, bvec_cart):
        del H, A_arg, mesh, kgrid, bvec_cart
        return jnp.ones_like(A)

    monkeypatch.setitem(
        sys.modules, "gw.qsgw_head",
        types.SimpleNamespace(
            covariant_cartesian_derivative=wrong_covariant))
    monkeypatch.setattr(
        pt_io, "write_velocity_validation",
        lambda path, *, mesh, metrics: stamps.append(metrics.copy()))
    with np.testing.assert_raises_regex(
            RuntimeError, "DFT velocity validation failed"):
        complete_velocity_validation(
            "not-opened.h5", mesh=mesh, kgrid=(2, 1, 1),
            bvec_cart=np.eye(3), energies_full=energies,
            connection_cart=A, velocity_exact_cart=exact,
            atol=1.0e-12, rtol=1.0e-12)
    assert len(stamps) == 1
    assert not stamps[0]["passed"]


def test_artifact_schema_is_slabio_only_and_names_the_head_manifold():
    """The new artifact has no raw h5py escape hatch and stable reader keys."""
    source = Path(
        sys.modules["file_io.parallel_transport"].__file__).read_text()
    assert "import h5py" not in source
    assert "h5py.File" not in source
    assert "SlabIO(" in source
    assert LINKS_DATASET == "links_ibz"
    assert CONNECTION_CART_DATASET == "berry_connection_cart"
    assert VELOCITY_DFT_DATASET == "velocity_dft_cart"
    assert ENERGIES_DATASET == "dft_energies_ry_full"
    assert OCCUPATIONS_DATASET == "dft_occupations_full"
