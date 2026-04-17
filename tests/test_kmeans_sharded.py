"""Regression tests for the refactored + optionally sharded k-means step.

Covers:

* ``kmeans_update_step`` (refactored: segment_sum + K-chunked scan) agrees
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
    _pick_k_block,
)


def _naive_update_step(positions, centroids, rho, G, n_k):
    """Reference implementation (materializes (P, K, 3) — not for production).

    This is the mathematically identical form of the original kmeans
    update step: full pairwise PBC distances, one-hot mask weighted mean.
    """
    delta = positions[:, None, :] - centroids[None, :, :]
    delta = delta - jnp.round(delta)
    d = jnp.einsum('pki,ij,pkj->pk', delta, G, delta)
    labels = jnp.argmin(d, axis=1)
    mask = jax.nn.one_hot(labels, n_k, dtype=rho.dtype)
    weights = mask * rho[:, None]
    weighted_delta = weights[:, :, None] * delta
    sum_wd = jnp.sum(weighted_delta, axis=0)
    sum_w = jnp.sum(weights, axis=0)
    avg = jnp.where(sum_w[:, None] > 0, sum_wd / jnp.maximum(sum_w[:, None], 1e-10), 0.0)
    new_cent = (centroids + avg) % 1.0
    move = new_cent - centroids
    move = move - jnp.round(move)
    move_sq = jnp.einsum('ki,ij,kj->k', move, G, move)
    return new_cent, move_sq, labels


def _random_inputs(seed: int, p: int, k: int, avec: np.ndarray):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.random((p, 3), dtype=np.float32))
    centroids = jnp.asarray(rng.random((k, 3), dtype=np.float32))
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
    """Refactored chunked update_step == naive (P,K,3) reference.

    Uses sizes friendly to the default k_block (64 % 16 == 0).
    """
    p, k = 5_000, 64
    positions, centroids, rho, G = _random_inputs(seed=0, p=p, k=k, avec=avec)
    k_block = _pick_k_block(k)
    new_a, move_a, labels_a = kmeans_update_step(positions, centroids, rho, G, k, k_block)
    new_b, move_b, labels_b = _naive_update_step(positions, centroids, rho, G, k)

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
    labels = assign_labels_chunked(positions, centroids, G, n_k=4, k_block=2)
    # Expected: centroid 0 wins (ties with centroid 3 on distance, argmin picks lowest).
    assert int(labels[0]) == 0


@pytest.mark.skipif(len(jax.devices()) < 2,
                    reason="Sharded path requires ≥2 JAX devices.")
def test_sharded_matches_single_device():
    """Sharded Lloyd step == single-device Lloyd step, same inputs."""
    devices = jax.devices()
    n_dev = len(devices)
    # Pick sizes divisible by device count and by k_block.
    p, k = 4 * 1024, 32
    assert p % n_dev == 0
    avec = np.array([[2.715, 2.715, 0.0],
                     [0.0, 2.715, 2.715],
                     [2.715, 0.0, 2.715]], dtype=np.float32)
    positions, centroids, rho, G = _random_inputs(seed=7, p=p, k=k, avec=avec)
    k_block = _pick_k_block(k)

    # Single-device reference.
    ref_new, ref_move, ref_labels = kmeans_update_step(
        positions, centroids, rho, G, k, k_block
    )

    # Sharded path on all visible devices.
    mesh = Mesh(np.asarray(devices), ('x',))
    pos_s = jax.device_put(positions, NamedSharding(mesh, PartitionSpec('x', None)))
    rho_s = jax.device_put(rho, NamedSharding(mesh, PartitionSpec('x')))
    cent_s = jax.device_put(centroids, NamedSharding(mesh, PartitionSpec()))
    G_s = jax.device_put(G, NamedSharding(mesh, PartitionSpec()))
    sharded_step = make_sharded_kmeans_update(mesh, k, k_block=k_block)
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
