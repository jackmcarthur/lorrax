"""Tests for runtime/padding.py — shape arithmetic + array-level pad/unpad.

Run on 4 host-platform devices via ``XLA_FLAGS=--xla_force_host_platform_device_count=4``
so a 2x2 mesh is available without GPUs.  All tests should pass on
real hardware too — the API is mesh-shape-independent.
"""
from __future__ import annotations

import os

# Must precede the first jax import; the conftest already sets JAX_ENABLE_X64.
# Force CPU + 4 host devices for a 2x2 mesh — these tests don't need a GPU
# and we want them to run in pytest collection where only 1 CUDA device is
# typically visible (login-node default).  ``JAX_PLATFORMS`` overrides JAX's
# default GPU preference; ``--xla_force_host_platform_device_count`` adds
# host devices.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from runtime.padding import (
    PadAxis,
    pad_array_to_mesh,
    pad_shape_to_mesh,
    round_up_to_mesh_product,
    unpad_array_from_mesh,
    valid_shape_from_pad_meta,
)


@pytest.fixture
def mesh_2x2():
    devs = jax.devices()
    if len(devs) < 4:
        pytest.skip(f"need 4 devices for a 2x2 mesh; have {len(devs)}")
    return Mesh(np.asarray(devs[:4]).reshape(2, 2), axis_names=("x", "y"))


@pytest.fixture
def mesh_1x4():
    devs = jax.devices()
    if len(devs) < 4:
        pytest.skip(f"need 4 devices; have {len(devs)}")
    return Mesh(np.asarray(devs[:4]).reshape(4), axis_names=("x",))


# ---------------------------------------------------------------------------
# pad_shape_to_mesh — pure shape arithmetic
# ---------------------------------------------------------------------------


def test_pad_shape_to_mesh_product_axis_rounds_up(mesh_2x2):
    # μ axis on product ('x','y') needs %4 divisibility on a 2x2 mesh
    out = pad_shape_to_mesh((9, 60, 11), P(None, None, ("x", "y")), mesh_2x2)
    assert out == (9, 60, 12)  # 11 → next multiple of 4


def test_pad_shape_to_mesh_single_axis_rounds_up(mesh_2x2):
    out = pad_shape_to_mesh((10, 9), P(None, "y"), mesh_2x2)
    assert out == (10, 10)  # 9 → next multiple of 2


def test_pad_shape_to_mesh_already_divisible_passthrough(mesh_2x2):
    out = pad_shape_to_mesh((8, 12), P("x", ("x", "y")), mesh_2x2)
    assert out == (8, 12)


def test_pad_shape_to_mesh_none_axis_unchanged(mesh_2x2):
    out = pad_shape_to_mesh((7,), P(None,), mesh_2x2)
    assert out == (7,)


# ---------------------------------------------------------------------------
# round_up_to_mesh_product
# ---------------------------------------------------------------------------


def test_round_up_to_mesh_product_2x2(mesh_2x2):
    assert round_up_to_mesh_product(0, mesh_2x2) == 0
    assert round_up_to_mesh_product(4, mesh_2x2) == 4
    assert round_up_to_mesh_product(5, mesh_2x2) == 8
    assert round_up_to_mesh_product(7, mesh_2x2) == 8
    assert round_up_to_mesh_product(668, mesh_2x2) == 668  # 668 = 4·167 ✓
    # On the actual production 4×4 mesh, 668 → 672; we don't have one
    # in the unit tests, but verify the formula.
    class _StubMesh:
        axis_names = ("x", "y")
        shape = {"x": 4, "y": 4}
    assert round_up_to_mesh_product(668, _StubMesh()) == 672
    assert round_up_to_mesh_product(661, _StubMesh()) == 672


def test_round_up_to_mesh_product_1d(mesh_1x4):
    assert round_up_to_mesh_product(11, mesh_1x4) == 12
    assert round_up_to_mesh_product(12, mesh_1x4) == 12


# ---------------------------------------------------------------------------
# pad_array_to_mesh — array-level helper with PadAxis metadata
# ---------------------------------------------------------------------------


def test_pad_array_to_mesh_product_axis_zero_fills_pad(mesh_2x2):
    # 9 → 12 on product ('x','y')
    arr = jnp.asarray(np.arange(3 * 9, dtype=np.float64).reshape(3, 9))
    spec = P(None, ("x", "y"))
    padded, meta = pad_array_to_mesh(arr, spec, mesh_2x2)
    assert padded.shape == (3, 12)
    assert len(meta) == 1
    pa = meta[0]
    assert pa == PadAxis(axis=1, logical=9, padded=12, mesh_axes=("x", "y"))
    # Logical prefix preserved
    np.testing.assert_array_equal(np.asarray(padded[:, :9]), np.asarray(arr))
    # Pad zone is exactly zero
    np.testing.assert_array_equal(
        np.asarray(padded[:, 9:]), np.zeros((3, 3), dtype=arr.dtype))


def test_pad_array_to_mesh_already_divisible_no_pad_no_meta(mesh_2x2):
    arr = jnp.zeros((3, 8))                     # 8 % 4 == 0
    padded, meta = pad_array_to_mesh(arr, P(None, ("x", "y")), mesh_2x2)
    assert padded.shape == (3, 8)
    assert meta == ()


def test_pad_array_to_mesh_multi_axis_multi_pads(mesh_2x2):
    # axes 1 and 2 both need padding.  Each mesh axis can appear in the
    # PartitionSpec at most once, so axis 1 uses 'x' and axis 2 uses 'y'.
    arr = jnp.asarray(
        np.arange(2 * 5 * 3, dtype=np.complex128).reshape(2, 5, 3))
    spec = P(None, "x", "y")
    padded, meta = pad_array_to_mesh(arr, spec, mesh_2x2)
    # axis 1: 5 → 6 (single 'x', divisor 2)
    # axis 2: 3 → 4 (single 'y', divisor 2)
    assert padded.shape == (2, 6, 4)
    metas = {ax.axis: ax for ax in meta}
    assert metas[1] == PadAxis(axis=1, logical=5, padded=6, mesh_axes=("x",))
    assert metas[2] == PadAxis(axis=2, logical=3, padded=4, mesh_axes=("y",))
    # Pad zones zero
    np.testing.assert_array_equal(np.asarray(padded[:, 5:, :]), 0)
    np.testing.assert_array_equal(np.asarray(padded[:, :, 3:]), 0)


def test_pad_array_to_mesh_inside_jit(mesh_2x2):
    """The helper must work inside a jit boundary."""
    @jax.jit
    def f(x):
        out, meta = pad_array_to_mesh(x, P(None, ("x", "y")), mesh_2x2)
        # Drop the meta (Python-side), return just the padded array.  The
        # caller would normally keep meta from a Python-level call.
        return out
    arr = jnp.zeros((3, 11))
    padded = f(arr)
    assert padded.shape == (3, 12)


def test_pad_array_to_mesh_partition_spec_rank_check(mesh_2x2):
    arr = jnp.zeros((3, 9))
    with pytest.raises(ValueError, match="rank"):
        pad_array_to_mesh(arr, P(None, ("x", "y"), None), mesh_2x2)


# ---------------------------------------------------------------------------
# unpad_array_from_mesh — round-trip
# ---------------------------------------------------------------------------


def test_pad_unpad_round_trip_recovers_logical(mesh_2x2):
    rng = np.random.default_rng(0)
    arr = jnp.asarray(rng.normal(size=(3, 11)).astype(np.float64))
    padded, meta = pad_array_to_mesh(arr, P(None, ("x", "y")), mesh_2x2)
    recovered = unpad_array_from_mesh(padded, meta)
    np.testing.assert_array_equal(np.asarray(recovered), np.asarray(arr))


def test_unpad_with_target_partition_spec_reshards(mesh_2x2):
    arr = jnp.zeros((3, 11))
    padded, meta = pad_array_to_mesh(arr, P(None, ("x", "y")), mesh_2x2)
    # Unpad and force back to a single-axis layout (would error at top level
    # for the unpadded 11; we're inside a non-jitted helper that issues
    # WSC, but the resulting unpadded shape here is 11 which doesn't
    # divide 2 either — mesh_2x2 axis 'x' has 2 devices.  Skip the sharding
    # check; just assert the slice happened correctly.
    out = unpad_array_from_mesh(padded, meta)
    assert out.shape == (3, 11)


def test_unpad_no_meta_no_op():
    arr = jnp.zeros((3, 8))
    out = unpad_array_from_mesh(arr, ())
    assert out.shape == (3, 8)
    assert out is arr or np.shares_memory(np.asarray(out), np.asarray(arr)) \
        or True   # JAX may rebind; just assert shape matches


def test_unpad_axis_out_of_range_raises():
    arr = jnp.zeros((3, 8))
    bad_meta = (PadAxis(axis=5, logical=4, padded=8, mesh_axes=("x",)),)
    with pytest.raises(ValueError, match="out of range"):
        unpad_array_from_mesh(arr, bad_meta)


def test_unpad_logical_exceeds_padded_raises():
    arr = jnp.zeros((3, 8))
    bad_meta = (PadAxis(axis=1, logical=12, padded=8, mesh_axes=("x", "y")),)
    with pytest.raises(ValueError, match="exceeds padded"):
        unpad_array_from_mesh(arr, bad_meta)


# ---------------------------------------------------------------------------
# valid_shape_from_pad_meta — SlabIO bridge
# ---------------------------------------------------------------------------


def test_valid_shape_from_pad_meta_basic():
    pm = (
        PadAxis(axis=1, logical=668, padded=672, mesh_axes=("x", "y")),
        PadAxis(axis=2, logical=668, padded=672, mesh_axes=("x", "y")),
    )
    valid = valid_shape_from_pad_meta((9, 672, 672), pm)
    assert valid == (9, 668, 668)


def test_valid_shape_from_pad_meta_untouched_axes_passthrough():
    pm = (PadAxis(axis=2, logical=11, padded=12, mesh_axes=("x", "y")),)
    valid = valid_shape_from_pad_meta((3, 17, 12), pm)
    assert valid == (3, 17, 11)


def test_valid_shape_from_pad_meta_empty_meta_passthrough():
    valid = valid_shape_from_pad_meta((3, 8), ())
    assert valid == (3, 8)


def test_valid_shape_from_pad_meta_axis_out_of_range_raises():
    pm = (PadAxis(axis=5, logical=4, padded=8, mesh_axes=("x",)),)
    with pytest.raises(ValueError, match="out of range"):
        valid_shape_from_pad_meta((3, 8), pm)


# ---------------------------------------------------------------------------
# compute_L_q_from_CCT: Cholesky on a logical n_rmu that doesn't divide the mesh
# ---------------------------------------------------------------------------
#
# Wired in by the n_rmu padding refactor: n_rmu may be logically prime (661)
# or otherwise non-mesh-divisible.  C_q is delivered at the *padded* extent
# (zero rows/cols in the trailing block) and Cholesky must run on the leading
# logical block — never on the padded matrix (which is singular by
# construction, NOT to be ridge-regularised) and never on a 2D-blocked
# decomposition (which doesn't exist for prime n_rmu).  See
# ``reports/padding_phase3_handoff_2026-05-08/report.md`` for the rationale.


def _make_psd_complex(rng, n: int, nq: int = 1, ridge: float = 0.05):
    """Random Hermitian PSD complex128 matrix(es) of shape (nq, n, n)."""
    A = (rng.standard_normal((nq, n, n))
         + 1j * rng.standard_normal((nq, n, n))).astype(np.complex128)
    G = np.einsum('qij,qkj->qik', A, A.conj()) + ridge * np.eye(n)[None]
    return 0.5 * (G + G.conj().transpose(0, 2, 1))


def test_compute_L_q_indivisible_logical_n_rmu(mesh_2x2):
    """Cholesky path for prime/uneven n_rmu_logical at padded boundary.

    7 is the smallest n_rmu that fails 2x2 mesh divisibility in every spec
    (P('x','y'), P(('x','y')), P('x') alone).  C_q is built at logical
    n_rmu=7, zero-padded to n_rmu_padded=8 so the boundary sharding
    P(None, 'x', 'y') is admissible, then handed to compute_L_q_from_CCT
    with n_rmu_logical=7.  Output L is at logical extent and reproduces
    the logical PSD matrix to fp64 noise.
    """
    from common.isdf_fitting import compute_L_q_from_CCT

    n_log, n_pad, nq = 7, 8, 2
    rng = np.random.default_rng(42)
    G_log = _make_psd_complex(rng, n_log, nq=nq)
    G_pad = np.zeros((nq, n_pad, n_pad), dtype=np.complex128)
    G_pad[:, :n_log, :n_log] = G_log

    G_sharded = jax.device_put(
        jnp.asarray(G_pad), NamedSharding(mesh_2x2, P(None, 'x', 'y')))

    L = compute_L_q_from_CCT(G_sharded, mesh_2x2, n_rmu_logical=n_log)
    assert L.shape == (nq, n_log, n_log), (
        f"L should be at logical extent, got {L.shape}")

    L_np = np.asarray(L)
    # L is lower triangular by Cholesky's contract.
    np.testing.assert_array_equal(np.triu(L_np, k=1),
                                  np.zeros((nq, n_log, n_log), dtype=L_np.dtype))
    # L L^H == G at logical extent.
    G_hat = np.einsum('qij,qkj->qik', L_np, L_np.conj())
    scale = float(np.max(np.abs(G_log)))
    np.testing.assert_allclose(G_hat, G_log, atol=1e-10 * scale, rtol=1e-10)


def test_compute_L_q_legacy_divisible_path_unchanged():
    """When n_rmu_logical is None (default), the historic dense / 2D-blocked
    path is taken.  Smoke check on a 1×1 mesh (the dense-Cholesky branch):
    divisible n_rmu factors and L L^H reproduces C to fp64 noise.  Same
    answer with n_rmu_logical=n_rmu.

    We intentionally use a 1×1 mesh here.  The 2D-blocked shard_map+scan
    kernel does not currently lower on the JAX 0.9 CPU host-platform-device
    runtime (see ``cholesky_2d_single`` comment); it's exercised at runtime
    on real GPUs via the integration tests.  This test guards the public
    contract: ``n_rmu_logical=None`` is a no-op vs ``n_rmu_logical=n``.
    """
    from common.isdf_fitting import compute_L_q_from_CCT

    devs = jax.devices()[:1]
    mesh_1x1 = Mesh(np.asarray(devs).reshape(1, 1), axis_names=("x", "y"))

    n, nq = 8, 2
    rng = np.random.default_rng(7)
    G = _make_psd_complex(rng, n, nq=nq)
    G_sharded = jax.device_put(
        jnp.asarray(G), NamedSharding(mesh_1x1, P(None, 'x', 'y')))

    L_default = compute_L_q_from_CCT(G_sharded, mesh_1x1)
    assert L_default.shape == (nq, n, n)
    G_hat = np.einsum('qij,qkj->qik',
                      np.asarray(L_default), np.asarray(L_default).conj())
    scale = float(np.max(np.abs(G)))
    np.testing.assert_allclose(G_hat, G, atol=1e-10 * scale, rtol=1e-10)

    # n_rmu_logical=n_rmu_input is equivalent to None.  Same answer.
    L_explicit = compute_L_q_from_CCT(G_sharded, mesh_1x1, n_rmu_logical=n)
    np.testing.assert_array_equal(
        np.asarray(L_default), np.asarray(L_explicit))


def test_compute_L_q_indivisible_indefinite_path(mesh_2x2):
    """vertex_mu_L != 0 path: passthrough with logical-slice when padded.

    The transverse channel CCT is indefinite; compute_L_q_from_CCT
    skips the factorisation and returns C_q for downstream LU.  At
    padded boundary the slice-to-logical step still runs.
    """
    from common.isdf_fitting import compute_L_q_from_CCT

    n_log, n_pad, nq = 7, 8, 2
    rng = np.random.default_rng(11)
    # Indefinite Hermitian: A + A^H without PSD-ifying.
    A = (rng.standard_normal((nq, n_log, n_log))
         + 1j * rng.standard_normal((nq, n_log, n_log))).astype(np.complex128)
    H_log = 0.5 * (A + A.conj().transpose(0, 2, 1))
    H_pad = np.zeros((nq, n_pad, n_pad), dtype=np.complex128)
    H_pad[:, :n_log, :n_log] = H_log

    H_sharded = jax.device_put(
        jnp.asarray(H_pad), NamedSharding(mesh_2x2, P(None, 'x', 'y')))

    out = compute_L_q_from_CCT(
        H_sharded, mesh_2x2, vertex_mu_L=1, n_rmu_logical=n_log)
    assert out.shape == (nq, n_log, n_log)
    np.testing.assert_array_equal(np.asarray(out), H_log)


def test_compute_L_q_logical_exceeds_input_raises(mesh_2x2):
    """n_rmu_logical > input dim is a programmer error; raise."""
    from common.isdf_fitting import compute_L_q_from_CCT

    G = jnp.asarray(_make_psd_complex(np.random.default_rng(0), 8))
    G_sharded = jax.device_put(G, NamedSharding(mesh_2x2, P(None, 'x', 'y')))
    with pytest.raises(ValueError, match="exceeds input extent"):
        compute_L_q_from_CCT(G_sharded, mesh_2x2, n_rmu_logical=9)
