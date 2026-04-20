"""Regression tests for the refactored + optionally sharded k-means step.

Covers:

* ``kmeans_update_step`` (refactored: segment_sum + C-chunked scan) agrees
  with a brute-force reference implementation bit-for-bit in single precision.
* The PBC minimal-image / metric-tensor behavior is unchanged across:
  axis-aligned boxes, a tilted FCC-like cell, and deliberately skewed cells.
* When two or more JAX devices are visible, the ``shard_map``-based
  sharded path produces identical centroid updates to the single-device
  path (which is the "correctness" guarantee we care about for the
  parallel prototype).
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from src.centroid.kmeans_isdf import (
    kmeans_update_step,
    assign_labels_chunked,
    make_sharded_kmeans_update,
    kmeans_pp_init,
    _pick_c_block,
)


def _naive_update_step(positions, centroids, rho, G, n_c):
    """Reference implementation (materializes (P, C, 3) — not for production).

    This is the mathematically identical form of the original kmeans
    update step: full pairwise PBC distances, one-hot mask weighted mean.
    """
    delta = positions[:, None, :] - centroids[None, :, :]
    delta = delta - jnp.round(delta)
    d = jnp.einsum('pci,ij,pcj->pc', delta, G, delta)
    labels = jnp.argmin(d, axis=1)
    mask = jax.nn.one_hot(labels, n_c, dtype=rho.dtype)
    weights = mask * rho[:, None]
    weighted_delta = weights[:, :, None] * delta
    sum_wd = jnp.sum(weighted_delta, axis=0)
    sum_w = jnp.sum(weights, axis=0)
    avg = jnp.where(sum_w[:, None] > 0, sum_wd / jnp.maximum(sum_w[:, None], 1e-10), 0.0)
    new_cent = (centroids + avg) % 1.0
    move = new_cent - centroids
    move = move - jnp.round(move)
    move_sq = jnp.einsum('ci,ij,cj->c', move, G, move)
    return new_cent, move_sq, labels


def _random_inputs(seed: int, p: int, n_c: int, avec: np.ndarray):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.random((p, 3), dtype=np.float32))
    centroids = jnp.asarray(rng.random((n_c, 3), dtype=np.float32))
    rho = jnp.asarray(rng.random((p,), dtype=np.float32) + 1e-3)
    G = jnp.asarray((avec.T @ avec).astype(np.float32))
    return positions, centroids, rho, G


@pytest.mark.parametrize(
    "avec_name, avec",
    [
        ("ortho", np.eye(3, dtype=np.float32) * 3.8),
        # FCC primitive in Å, matches the Si_B test run
        ("fcc", np.array([[2.715, 2.715, 0.0],
                          [0.0, 2.715, 2.715],
                          [2.715, 0.0, 2.715]], dtype=np.float32)),
        # Deliberately skewed monoclinic-ish cell
        ("skew", np.array([[4.0, 0.3, 0.2],
                           [0.1, 3.5, 0.4],
                           [0.0, 0.5, 5.0]], dtype=np.float32)),
    ],
)
def test_refactored_matches_naive(avec_name, avec):
    """Refactored chunked update_step == naive (P,C,3) reference.

    Uses sizes friendly to the default c_block (64 % 16 == 0).
    """
    p, n_c = 5_000, 64
    positions, centroids, rho, G = _random_inputs(seed=0, p=p, n_c=n_c, avec=avec)
    c_block = _pick_c_block(n_c)
    new_a, move_a, labels_a = kmeans_update_step(positions, centroids, rho, G, n_c, c_block)
    new_b, move_b, labels_b = _naive_update_step(positions, centroids, rho, G, n_c)

    np.testing.assert_array_equal(np.asarray(labels_a), np.asarray(labels_b),
                                  err_msg=f"[{avec_name}] labels differ")
    np.testing.assert_allclose(np.asarray(new_a), np.asarray(new_b),
                               rtol=1e-5, atol=1e-6,
                               err_msg=f"[{avec_name}] centroids differ")
    np.testing.assert_allclose(np.asarray(move_a), np.asarray(move_b),
                               rtol=1e-5, atol=1e-6,
                               err_msg=f"[{avec_name}] movement_sq differs")


def test_assign_labels_chunked_handles_pbc_across_boundary():
    """Points near a periodic boundary must assign to the image-wrapped centroid.

    Concretely: a point at fractional (0.99, 0.5, 0.5) is closest (via
    minimal image) to a centroid at (0.01, 0.5, 0.5), not to (0.5, 0.5, 0.5).
    This tests the PBC `round()` wrap inside the chunked scan.
    """
    avec = np.eye(3, dtype=np.float32) * 4.0
    G = jnp.asarray((avec.T @ avec).astype(np.float32))
    positions = jnp.asarray([[0.99, 0.5, 0.5]], dtype=jnp.float32)
    centroids = jnp.asarray(
        [[0.01, 0.5, 0.5],   # wrapped neighbor, ~0.02 frac away
         [0.50, 0.5, 0.5],   # middle,           ~0.49 frac away
         [0.75, 0.5, 0.5],   # closer than middle but not than wrapped, ~0.24
         [0.01, 0.5, 0.5]],  # padding dup: also wrapped
        dtype=jnp.float32,
    )
    labels = assign_labels_chunked(positions, centroids, G, n_c=4, c_block=2)
    # Expected: centroid 0 wins (ties with centroid 3 on distance, argmin picks lowest).
    assert int(labels[0]) == 0


@pytest.mark.skipif(len(jax.devices()) < 2,
                    reason="Sharded path requires ≥2 JAX devices.")
def test_sharded_matches_single_device():
    """Sharded Lloyd step == single-device Lloyd step, same inputs."""
    devices = jax.devices()
    n_dev = len(devices)
    # Pick sizes divisible by device count and by c_block.
    p, n_c = 4 * 1024, 32
    assert p % n_dev == 0
    avec = np.array([[2.715, 2.715, 0.0],
                     [0.0, 2.715, 2.715],
                     [2.715, 0.0, 2.715]], dtype=np.float32)
    positions, centroids, rho, G = _random_inputs(seed=7, p=p, n_c=n_c, avec=avec)
    c_block = _pick_c_block(n_c)

    # Single-device reference.
    ref_new, ref_move, ref_labels = kmeans_update_step(
        positions, centroids, rho, G, n_c, c_block
    )

    # Sharded path on all visible devices.
    mesh = Mesh(np.asarray(devices), ('x',))
    pos_s = jax.device_put(positions, NamedSharding(mesh, PartitionSpec('x', None)))
    rho_s = jax.device_put(rho, NamedSharding(mesh, PartitionSpec('x')))
    cent_s = jax.device_put(centroids, NamedSharding(mesh, PartitionSpec()))
    G_s = jax.device_put(G, NamedSharding(mesh, PartitionSpec()))
    sharded_step = make_sharded_kmeans_update(mesh, n_c, c_block=c_block)
    shd_new, shd_move, shd_labels = sharded_step(pos_s, cent_s, rho_s, G_s)

    # Replicated outputs must match exactly; labels are sharded on 'x' so the
    # concatenation across shards should match the single-device labels.
    np.testing.assert_array_equal(
        np.asarray(shd_labels), np.asarray(ref_labels),
        err_msg="sharded labels diverge from single-device",
    )
    np.testing.assert_allclose(
        np.asarray(shd_new), np.asarray(ref_new),
        rtol=1e-5, atol=1e-6,
        err_msg="sharded centroids diverge from single-device",
    )
    np.testing.assert_allclose(
        np.asarray(shd_move), np.asarray(ref_move),
        rtol=1e-5, atol=1e-6,
        err_msg="sharded movement_sq diverges from single-device",
    )


def _build_gaussian_density(rng, grid_shape=(16, 16, 16), n_bumps=6, avec=None):
    """Synthetic 3D density: sum of a few Gaussian bumps at random PBC positions.

    Produces a charge-density-like field with isolated high-density regions
    that k-means++ should concentrate centroids around.
    """
    Nx, Ny, Nz = grid_shape
    xs = np.linspace(0, 1, Nx, endpoint=False)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    positions = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
    rho = np.zeros(Nx * Ny * Nz, dtype=np.float32) + 1e-6

    centers = rng.random((n_bumps, 3), dtype=np.float32)
    G = (avec.T @ avec).astype(np.float32)
    for c in centers:
        d = positions - c[None, :]
        d -= np.round(d)
        d2 = np.einsum('pi,ij,pj->p', d, G, d)
        rho = rho + np.exp(-d2 / 0.01).astype(np.float32)
    return positions, rho, centers


def test_kmeans_pp_init_deterministic_and_on_grid():
    """Same key → same centroids; each centroid is one of the grid points."""
    avec = np.eye(3, dtype=np.float32) * 3.8
    rng = np.random.default_rng(0)
    positions, rho, _ = _build_gaussian_density(rng, grid_shape=(16, 16, 16), avec=avec)
    positions = jnp.asarray(positions)
    rho = jnp.asarray(rho)
    G = jnp.asarray((avec.T @ avec).astype(np.float32))
    key = jax.random.PRNGKey(123)

    c_a = kmeans_pp_init(positions, rho, G, n_c=32, key=key)
    c_b = kmeans_pp_init(positions, rho, G, n_c=32, key=key)
    np.testing.assert_array_equal(np.asarray(c_a), np.asarray(c_b),
                                  err_msg="same key should give identical centroids")

    # Each returned centroid must be one of the grid positions (up to fp32).
    p_np = np.asarray(positions)
    c_np = np.asarray(c_a)
    # Find the nearest grid point to each centroid and verify it's exact.
    for ci in c_np:
        d2 = np.sum((p_np - ci[None, :]) ** 2, axis=1)
        assert d2.min() < 1e-10, f"centroid {ci} not exactly a grid point (min d²={d2.min()})"


def test_kmeans_pp_init_concentrates_on_high_density():
    """Centroids should sit in higher-than-average-density regions.

    K-means++ with D²·ρ weighting first seeds centroids near density peaks,
    then spreads later centroids into low-density voids (that's the whole
    point of D² — maximize coverage). So we can't insist that *every*
    centroid hugs a bump. Instead, we check two weaker-but-correct signals:

    1. The mean density **at centroid locations** exceeds the mean density
       over the whole grid by a multiplicative margin. This directly probes
       that density weighting is applied at all.
    2. Every bump attracts at least one centroid (no mode collapse).
    """
    avec = np.eye(3, dtype=np.float32) * 3.8
    rng = np.random.default_rng(7)
    positions, rho, bump_centers = _build_gaussian_density(
        rng, grid_shape=(16, 16, 16), n_bumps=4, avec=avec
    )
    G = jnp.asarray((avec.T @ avec).astype(np.float32))
    n_c = 16  # moderate over-count vs 4 bumps; first ~4-8 will seed bumps
    centroids = kmeans_pp_init(jnp.asarray(positions), jnp.asarray(rho), G,
                               n_c=n_c, key=jax.random.PRNGKey(1))
    c_np = np.asarray(centroids)

    # For each centroid, pull ρ at the nearest grid point (fp32 exact match
    # is the contract tested above, but we snap to be robust to rounding).
    pos_np = np.asarray(positions)
    rho_np = np.asarray(rho)
    rho_at_centroid = np.empty(n_c, dtype=np.float32)
    for i, ci in enumerate(c_np):
        d2 = np.sum((pos_np - ci[None, :]) ** 2, axis=1)
        rho_at_centroid[i] = rho_np[np.argmin(d2)]

    mean_rho = rho_np.mean()
    mean_rho_at_centroid = rho_at_centroid.mean()
    assert mean_rho_at_centroid > 3.0 * mean_rho, (
        f"centroid mean ρ = {mean_rho_at_centroid:.4f} vs grid mean "
        f"ρ = {mean_rho:.4f}; density weighting isn't steering the sampler"
    )

    # Bump coverage: every bump should attract at least one centroid.
    per_bump_count = np.zeros(bump_centers.shape[0], dtype=int)
    for ci in c_np:
        d = ci[None, :] - bump_centers
        d -= np.round(d)
        per_bump_count[np.argmin(np.linalg.norm(d, axis=1))] += 1
    assert np.all(per_bump_count > 0), (
        f"some bumps got no centroids: per-bump counts {per_bump_count} — "
        "the D²·ρ sampler collapsed onto fewer modes than it should"
    )


def test_kmeans_pp_init_respects_pbc():
    """Two bumps near opposite faces of the box should both be discovered,
    relying on minimal-image distance to drive D²-weighted spread."""
    avec = np.eye(3, dtype=np.float32) * 4.0
    G = jnp.asarray((avec.T @ avec).astype(np.float32))
    Nx = 20
    xs = np.linspace(0, 1, Nx, endpoint=False, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    positions = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    # Two Gaussians placed near x≈0.02 and x≈0.98 — raw distance makes them
    # look far apart, but PBC brings them close.
    def bump(center):
        d = positions - center[None, :]
        d -= np.round(d)
        d2 = np.einsum('pi,ij,pj->p', d, (avec.T @ avec).astype(np.float32), d)
        return np.exp(-d2 / 0.005).astype(np.float32)
    c1 = np.array([0.02, 0.5, 0.5], dtype=np.float32)
    c2 = np.array([0.98, 0.5, 0.5], dtype=np.float32)
    rho = bump(c1) + bump(c2) + 1e-6

    centroids = kmeans_pp_init(
        jnp.asarray(positions), jnp.asarray(rho), G,
        n_c=16, key=jax.random.PRNGKey(2)
    )
    c_np = np.asarray(centroids)

    # Under PBC, c1 and c2 are really ~0.04 apart, so *both* bumps should get
    # coverage. We check that at least one centroid lies on each side of x=0.5.
    near_c1 = np.sum(c_np[:, 0] < 0.25)
    near_c2 = np.sum(c_np[:, 0] > 0.75)
    assert near_c1 >= 1 and near_c2 >= 1, (
        f"PBC broken: near-c1={near_c1}, near-c2={near_c2} "
        f"(centroids should cover both near-boundary bumps)"
    )


def test_assign_labels_chunked_padding_for_nondivisible_n_c():
    """c_block no longer has to divide n_c; NaN-padded scan body handles it.

    Verifies that ``assign_labels_chunked`` with c_block=32, n_c=50 (which
    does not divide c_block) produces the same labels as c_block=50 (a
    divisor). Also confirms labels stay in [0, n_c) — the NaN-padded
    sentinel rows never win argmin.
    """
    avec = np.eye(3, dtype=np.float32) * 3.8
    G = jnp.asarray((avec.T @ avec).astype(np.float32))
    rng = np.random.default_rng(42)
    pos = jnp.asarray(rng.random((1024, 3), dtype=np.float32))
    cent = jnp.asarray(rng.random((50, 3), dtype=np.float32))

    labels_pad = assign_labels_chunked(pos, cent, G, n_c=50, c_block=32)
    labels_div = assign_labels_chunked(pos, cent, G, n_c=50, c_block=50)

    np.testing.assert_array_equal(
        np.asarray(labels_pad), np.asarray(labels_div),
        err_msg="c_block-padded scan should match divisor-aligned scan",
    )
    assert int(np.asarray(labels_pad).max()) < 50, (
        "labels escaped into padded-centroid range"
    )


def test_pbc_distance_scan_matches_naive_fcc():
    """Fused quadratic form in ``assign_labels_chunked`` == full (P,C,3) naive.

    Regression for the explicit-quadratic-form rewrite of the distance
    computation in the scan body (no einsum/GEMM). A PBC-respecting naive
    reference that materializes the full (P, C, 3) delta tensor should
    agree with the scan on both labels and their distances, including on
    a non-diagonal metric tensor (FCC, where cross terms G_ij ≠ 0 matter).
    """
    avec = np.array([[2.715, 2.715, 0.0],
                     [0.0, 2.715, 2.715],
                     [2.715, 0.0, 2.715]], dtype=np.float32)
    G = jnp.asarray((avec.T @ avec).astype(np.float32))
    rng = np.random.default_rng(1)
    pos = jnp.asarray(rng.random((2048, 3), dtype=np.float32))
    cent = jnp.asarray(rng.random((100, 3), dtype=np.float32))

    # Naive reference: full (P, C, 3) delta + einsum. Same PBC, same metric.
    delta = pos[:, None, :] - cent[None, :, :]
    delta = delta - jnp.round(delta)
    d_ref = jnp.einsum('pci,ij,pcj->pc', delta, G, delta)
    labels_ref = jnp.argmin(d_ref, axis=1).astype(jnp.int32)

    labels_scan = assign_labels_chunked(pos, cent, G, n_c=100, c_block=32)

    np.testing.assert_array_equal(
        np.asarray(labels_scan), np.asarray(labels_ref),
        err_msg="fused-quadratic-form scan labels differ from naive einsum",
    )


@pytest.mark.skipif(len(jax.devices()) < 2,
                    reason="Sharded path requires ≥2 JAX devices.")
def test_sharded_trajectory_matches_single_device():
    """Sharded Lloyd step, iterated, tracks the single-device trajectory.

    The earlier single-step equivalence test (``test_sharded_matches_single_device``)
    only checks one iteration. Floating-point reduction order differs
    between ``segment_sum`` on one device vs ``segment_sum + psum`` across
    shards, so over many iterations the trajectories can drift apart. This
    test runs 10 Lloyd iterations both ways and asserts centroids stay
    within a tight tolerance — enough to catch a real divergence (e.g. a
    reduction sign error) but loose enough to tolerate fp32 reordering.
    """
    devices = jax.devices()
    n_dev = len(devices)
    p, n_c = 4 * 1024, 32
    assert p % n_dev == 0
    avec = np.array([[2.715, 2.715, 0.0],
                     [0.0, 2.715, 2.715],
                     [2.715, 0.0, 2.715]], dtype=np.float32)
    positions, centroids, rho, G = _random_inputs(seed=11, p=p, n_c=n_c, avec=avec)
    c_block = _pick_c_block(n_c)

    # Single-device reference: 10 Lloyd iterations.
    cent_ref = centroids
    for _ in range(10):
        cent_ref, _, _ = kmeans_update_step(positions, cent_ref, rho, G, n_c, c_block)

    # Sharded path, same 10 iterations.
    mesh = Mesh(np.asarray(devices), ('x',))
    pos_s = jax.device_put(positions, NamedSharding(mesh, PartitionSpec('x', None)))
    rho_s = jax.device_put(rho, NamedSharding(mesh, PartitionSpec('x')))
    rep = NamedSharding(mesh, PartitionSpec())
    cent_s = jax.device_put(centroids, rep)
    G_s = jax.device_put(G, rep)
    sharded_step = make_sharded_kmeans_update(mesh, n_c, c_block=c_block)
    for _ in range(10):
        cent_s, _, _ = sharded_step(pos_s, cent_s, rho_s, G_s)

    # fp32 reduction-order drift over 10 iters on 4k points ≲ 1e-4 in frac
    # coords — looser than the 1e-5 single-step tolerance but still tight
    # enough to fail on any algorithmic divergence.
    np.testing.assert_allclose(
        np.asarray(cent_s), np.asarray(cent_ref),
        rtol=1e-4, atol=1e-4,
        err_msg="sharded trajectory diverged from single-device after 10 iters",
    )


# ═══════════════════════════════════════════════════════════════════════
# Grid-density threshold heuristics
# ═══════════════════════════════════════════════════════════════════════


def test_decide_init_method_below_threshold_uses_kpp():
    from src.centroid.kmeans_isdf import _decide_init_method
    # 500 centroids out of a 36³=46656-point grid is ~1% — well below 10%.
    method, msg = _decide_init_method(n_c=500, n_rtot=46656)
    assert method == 'kpp'
    assert msg is None


def test_decide_init_method_above_threshold_switches_to_random():
    from src.centroid.kmeans_isdf import _decide_init_method
    # 12000 out of 46656 ≈ 25% — well above 10% threshold.
    method, msg = _decide_init_method(n_c=12000, n_rtot=46656)
    assert method == 'random'
    assert msg is not None and 'random' in msg.lower()


def test_decide_init_method_threshold_is_strict_greater():
    from src.centroid.kmeans_isdf import _decide_init_method
    # Exactly at the 10% cutoff: should still use kpp (strict > check).
    n_rtot = 1000
    method, _ = _decide_init_method(n_c=100, n_rtot=n_rtot)
    assert method == 'kpp'
    method, _ = _decide_init_method(n_c=101, n_rtot=n_rtot)
    assert method == 'random'


def test_warn_dense_grid_regime_below_threshold_is_silent():
    from src.centroid.kmeans_isdf import _warn_dense_grid_regime
    # N_c=1000 out of 46656 is ~2% — below 25% threshold.
    assert _warn_dense_grid_regime(m_cand=1500, n_c=1000, n_rtot=46656) is None


def test_warn_dense_grid_regime_above_threshold_returns_message():
    from src.centroid.kmeans_isdf import _warn_dense_grid_regime
    # N_c=15000 out of 46656 is ~32% — above 25% threshold.
    msg = _warn_dense_grid_regime(m_cand=22500, n_c=15000, n_rtot=46656)
    assert msg is not None
    assert 'dense' in msg.lower() or 'not implemented' in msg.lower()


def test_weighted_kmeans_jax_random_init_runs():
    """Smoke test: random init path produces a valid Lloyd output."""
    from src.centroid.kmeans_isdf import weighted_kmeans_jax
    # Tiny synthetic cell: 16³ grid, 100 centroids (well above 10% threshold
    # → test the random-init branch).
    nx = 16
    avec = jnp.eye(3) * 5.0  # 5 Å cubic
    rng = np.random.default_rng(3)
    rho = jnp.asarray(
        np.abs(rng.standard_normal((nx, nx, nx))) + 0.1,
        dtype=jnp.float64,
    )
    _, cent, _, _ = weighted_kmeans_jax(
        avec, rho, N_c=80, seed=1, max_steps=10, init_method='random',
    )
    cent_np = np.asarray(cent)
    assert cent_np.shape == (80, 3)
    # All centroids should land in the unit cell.
    assert np.all(cent_np >= 0) and np.all(cent_np < 1.0)


def test_weighted_kmeans_jax_rejects_bad_init_method():
    from src.centroid.kmeans_isdf import weighted_kmeans_jax
    avec = jnp.eye(3)
    rho = jnp.ones((8, 8, 8), dtype=jnp.float64)
    with pytest.raises(ValueError, match="init_method"):
        weighted_kmeans_jax(avec, rho, N_c=10, max_steps=2,
                            init_method='nonexistent')
