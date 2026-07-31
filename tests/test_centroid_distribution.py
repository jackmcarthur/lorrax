"""Gates for the centroid package's use of the device layer.

``common.collectives`` is THE device layer; ``centroid.distribution``
holds only centroid-specific policy.  Four things are pinned here, each
with a negative control in the same test so a vacuously-green assertion is
detectable:

1. :func:`common.collectives.single_device_mesh` holds a device THIS
   process can address, and ``centroid.distribution`` does not define a
   second mesh constructor.  The idiom both replaced,
   ``Mesh(jax.devices()[:1])``, holds process 0's device on every rank.
2. The sharded pivoted-Cholesky select reproduces the single-device
   reference on a 1x1 mesh, pivot for pivot.
3. The orbit-aware Lloyd step reduces to the plain one when the symmetry
   group is trivial — the invariant that keeps the two branches of
   ``kmeans_isdf.make_lloyd_loop`` honest.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────
# 1. Meshes
# ─────────────────────────────────────────────────────────────────────────

def test_process_local_mesh_is_addressable():
    """Every device in the process-local mesh belongs to THIS process.

    The mesh comes from ``common.collectives.single_device_mesh`` — the one
    device layer.  ``centroid.distribution`` deliberately does not define a
    second one; this test pins the property the centroid code depends on.
    """
    from src.common.collectives import single_device_mesh

    mesh = single_device_mesh()
    local = set(jax.local_devices())
    assert mesh.devices.size == 1
    assert set(mesh.devices.flat) <= local, (
        f"process_local_mesh holds {list(mesh.devices.flat)}, which this "
        f"process ({jax.process_index()}) cannot address; local devices are "
        f"{jax.local_devices()}")

    # Negative control: the SAME membership test applied to the idiom this
    # function replaced must be able to come out false.  At P=1 the global
    # and local device lists coincide, so the control can only be shown to
    # discriminate past P=1 -- say so rather than claim a pass.
    global_first = jax.devices()[:1]
    if jax.process_count() == 1:
        assert set(global_first) <= local          # P=1: they agree, as expected
        pytest.skip("P=1: jax.devices()[0] IS this process's device, so the "
                    "negative control cannot fire here. Run under P>=2 "
                    "(wk_REL/wave1_kmeans/kmmesh.sbatch) to exercise it.")
    else:
        assert not (set(global_first) <= local) or jax.process_index() == 0


def test_process_local_mesh_is_cached():
    """Same object every call — the to_box/to_rmu jit caches key on it."""
    from src.common.collectives import single_device_mesh
    assert single_device_mesh() is single_device_mesh()


def test_centroid_layer_defines_no_second_mesh_constructor():
    """``centroid.distribution`` must delegate, not reimplement.

    The orchestrator's 2026-07-30 ruling: ``common.collectives`` is THE
    device layer.  This test fails if the centroid layer ever grows its own
    ``single_device_mesh`` / ``process_local_mesh`` again — the exact name
    collision that made one spelling mean 'this rank's device' and the
    other 'global device 0, refuse at P>1'.
    """
    from src.centroid import distribution as dist
    for name in ("single_device_mesh", "process_local_mesh"):
        owner = getattr(getattr(dist, name, None), "__module__", None)
        assert owner in (None, "common.collectives", "src.common.collectives"), (
            f"centroid.distribution.{name} is defined in {owner}; it must be "
            f"common.collectives' or absent")


def test_build_mesh_single_device_is_2d():
    """A one-device run still gets a 2-D ('x','y') mesh, so the downstream
    pipeline has exactly one code path."""
    from src.centroid import distribution as dist
    if jax.process_count() > 1:
        pytest.skip("--no-shard is refused past one process")
    mesh = dist.build_mesh(10 ** 6, shard=False)
    assert tuple(mesh.axis_names) == ("x", "y")
    assert mesh.devices.shape == (1, 1)


# ─────────────────────────────────────────────────────────────────────────
# 2. Sharded pivoted-Cholesky select == single-device reference
# ─────────────────────────────────────────────────────────────────────────

def _psd_gram(M, rank, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((M, rank)) + 1j * rng.standard_normal((M, rank)))
    G = A @ A.conj().T
    return jnp.asarray(0.5 * (G + G.conj().T), dtype=jnp.complex128)


def test_sharded_select_matches_single_device_reference():
    from src.centroid import distribution as dist
    from src.centroid.pivoted_cholesky import (
        pivoted_cholesky_select, make_sharded_pivoted_cholesky_select)

    if jax.process_count() > 1:
        pytest.skip("reference comparison is a one-process gate")

    M, k_keep = 32, 12
    G = _psd_gram(M, rank=M, seed=3)
    mesh = dist.build_mesh(10 ** 6, shard=False)

    piv_ref, _, rank_ref, _, d_ref, _ = pivoted_cholesky_select(G, k_keep)
    step = make_sharded_pivoted_cholesky_select(
        mesh, M, k_keep, mesh_axis=dist.MESH_AXES)
    piv_sh, _, rank_sh, _, d_sh, _ = step(G)

    np.testing.assert_array_equal(np.asarray(piv_ref), np.asarray(piv_sh))
    assert int(rank_ref) == int(rank_sh)
    np.testing.assert_allclose(np.asarray(d_ref), np.asarray(d_sh),
                               rtol=1e-10, atol=0.0)

    # Negative control: the comparison must be able to FAIL.  A Gram with a
    # different pivot order has to produce different pivots; if it does not,
    # the assertion above proved nothing.
    piv_other, *_ = pivoted_cholesky_select(_psd_gram(M, rank=M, seed=4),
                                            k_keep)
    assert not np.array_equal(np.asarray(piv_ref), np.asarray(piv_other)), (
        "negative control did not fire: two different Grams gave the same "
        "pivot sequence, so the equality test above is vacuous")


# ─────────────────────────────────────────────────────────────────────────
# 3. Orbit-aware Lloyd reduces to plain Lloyd on the trivial group
# ─────────────────────────────────────────────────────────────────────────

def _bumpy_density(N=10, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 1, N, endpoint=False)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    rho = 1e-3 * np.ones((N, N, N))
    for c in rng.random((3, 3)):
        d = np.stack([X - c[0], Y - c[1], Z - c[2]], axis=-1)
        d -= np.round(d)
        rho += np.exp(-np.sum(d ** 2, axis=-1) / 0.02)
    return rho


def test_orbit_path_with_trivial_group_matches_plain_path():
    """With the identity as the only symmetry op, orbit representatives ARE
    literal points and the two branches of ``make_lloyd_loop`` must agree."""
    from src.centroid import distribution as dist
    from src.centroid.kmeans_isdf import weighted_kmeans_jax

    if jax.process_count() > 1:
        pytest.skip("branch-equality gate is a one-process test")

    avec = np.diag([4.0, 4.5, 5.0])
    rho = _bumpy_density()
    mesh = dist.build_mesh(rho.size, shard=False)
    eye = np.eye(3, dtype=np.int32)[None]
    zero = np.zeros((1, 3), dtype=np.float64)
    kw = dict(N_c=8, max_steps=40, tolerance=1e-4, seed=0, mesh=mesh,
              init_method="kpp")

    _, c_plain, s_plain, _ = weighted_kmeans_jax(avec, rho, **kw)
    _, c_orbit, s_orbit, _ = weighted_kmeans_jax(
        avec, rho, R=eye, Rinv=eye, tau=zero, **kw)

    assert s_plain == s_orbit, (
        f"trivial-group orbit path took {s_orbit} Lloyd steps vs "
        f"{s_plain} for the plain path")
    np.testing.assert_allclose(np.asarray(c_plain), np.asarray(c_orbit),
                               rtol=0, atol=1e-12)

    # Negative control: a NON-trivial group must move the answer, otherwise
    # the agreement above would be explained by the orbit machinery being
    # inert rather than by it being correct.
    inv = -np.eye(3, dtype=np.int32)
    grp = np.stack([np.eye(3, dtype=np.int32), inv])
    _, c_inv, _, _ = weighted_kmeans_jax(
        avec, rho, R=grp, Rinv=grp, tau=np.zeros((2, 3)), **kw)
    assert not np.allclose(np.asarray(c_plain), np.asarray(c_inv), atol=1e-8), (
        "negative control did not fire: adding inversion to the group left "
        "the centroids unchanged, so the trivial-group agreement is vacuous")
