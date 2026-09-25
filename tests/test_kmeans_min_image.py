"""Closest-image checks for skew real-space cells."""

from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest

from centroid.kmeans_isdf import (assign_labels_orbit_chunked,
                                  build_min_image_offsets, _min_image_delta,
                                  pbc_distance_sq_single)


_RADIUS_WITNESS = np.array([
    [0.6289290615318847, 0.5947537016920307, 0.6941186367981261],
    [0.5947537016920307, 0.9094453419831847, 1.5169258866223838],
    [0.6941186367981261, 1.5169258866223838, 5.640741076303628],
])
_SAMPLE_WITNESS = np.array([
    [1.0201772770761608, -1.2181411407733531, -1.4193764623236627],
    [-1.2181411407733531, 3.2381853306748294, 3.793780401280498],
    [-1.4193764623236627, 3.793780401280498, 12.625158137444112],
])


def _brute_force_distance(delta, metric):
    """Independent, deliberately wide integer search for these two cells."""
    translations = np.asarray(list(product(range(-4, 5), repeat=3)))
    images = delta[:, None, :] + translations[None, :, :]
    distances = np.einsum("pni,ij,pnj->pn", images, metric, images)
    best = distances.argmin(axis=1)
    return distances[np.arange(len(delta)), best], images[np.arange(len(delta)), best]


@pytest.mark.parametrize("metric,witness,winning_offset", [
    (_RADIUS_WITNESS, [0.03206861863504351, 0.47157804499564293,
                       0.4727241896744758], [1, -2, 0]),
    (_SAMPLE_WITNESS, [-0.46449446185729193, 0.4842814950148272,
                       0.4801543025990419], [1, -1, 0]),
])
def test_skew_cell_distance_and_lloyd_displacement(metric, witness,
                                                    winning_offset):
    """Cover both the former radius cutoff and the sampled-cover omission."""
    rng = np.random.default_rng(2509)
    delta = np.vstack((witness, rng.uniform(-0.5, 0.5, size=(128, 3))))
    expected_d2, expected_delta = _brute_force_distance(delta, metric)
    np.testing.assert_allclose(expected_delta[0] - delta[0], winning_offset)

    offsets = build_min_image_offsets(metric)
    got_d2 = np.asarray(pbc_distance_sq_single(
        jnp.asarray(delta), jnp.zeros(3), jnp.asarray(metric),
        jnp.asarray(offsets)))
    got_delta = np.asarray(_min_image_delta(
        jnp.asarray(delta), jnp.asarray(metric), jnp.asarray(offsets)))
    np.testing.assert_allclose(got_d2, expected_d2, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        np.einsum("pi,ij,pj->p", got_delta, metric, got_delta),
        expected_d2, rtol=2e-12, atol=2e-12)


def test_small_standard_cells_keep_short_tables():
    """Exact construction still has the useful 1/5/7 translation counts."""
    metrics = (
        np.eye(3),
        np.array([[1.0, -0.5, 0], [-0.5, 1.0, 0], [0, 0, 1.0]]),
        np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25],
                  [0.25, 0.25, 0.5]]),
    )
    for metric, count in zip(metrics, (1, 5, 7)):
        offsets = build_min_image_offsets(metric)
        assert offsets.shape == (count, 3)
        np.testing.assert_array_equal(offsets[0], [0, 0, 0])


def test_invalid_metric_refuses_before_gpu_distance():
    with pytest.raises(ValueError, match="positive definite"):
        build_min_image_offsets(np.diag([1.0, 1.0, -1.0]))


def test_orbit_assignment_ties_follow_final_winning_rep():
    """An independent all-rep/image oracle covers chunk changes and ties."""
    metric = np.array([[1.0, -0.5, 0], [-0.5, 1.0, 0], [0, 0, 3.0]])
    reps = np.array([[0, 0, 0], [0.2, 0.4, 0.25], [0.31, 0.6, 0.5],
                     [0.75, 0.1, 0.2], [0.4, 0.8, 0.4]])
    positions = np.vstack((np.zeros(3),
                           np.random.default_rng(413).random((40, 3))))
    Rinv = np.array([np.eye(3, dtype=np.int32),
                     -np.eye(3, dtype=np.int32)])
    images = np.stack((reps, -reps), axis=1)
    oracle = np.empty((len(positions), len(reps), 2))
    for i in range(len(reps)):
        for s in range(2):
            delta = positions - images[i, s]
            delta -= np.round(delta)
            oracle[:, i, s] = _brute_force_distance(delta, metric)[0]
    orbit_d = oracle.min(axis=2)
    expected_labels = orbit_d.argmin(axis=1)
    expected_d2 = orbit_d[np.arange(len(positions)), expected_labels]
    expected_ties = (oracle[np.arange(len(positions)), expected_labels]
                     <= expected_d2[:, None] + 1e-10)
    assert expected_ties[0].tolist() == [True, True]

    got_labels, got_d2, got_ties = assign_labels_orbit_chunked(
        jnp.asarray(positions), jnp.asarray(reps), jnp.asarray(metric),
        len(reps), c_block=2,
        offsets=jnp.asarray(build_min_image_offsets(metric)),
        Rinv=jnp.asarray(Rinv), tau=jnp.zeros((2, 3)))
    np.testing.assert_array_equal(np.asarray(got_labels), expected_labels)
    np.testing.assert_allclose(np.asarray(got_d2), expected_d2,
                               rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(np.asarray(got_ties), expected_ties)
