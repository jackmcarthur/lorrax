"""The deterministic candidate pool — pure host algebra, no device needed.

These are the properties the ``--candidate-pool full_grid`` selector rests
on: the pool is a pure function of the grid, the orbit labels are a pure
function of the grid and the symmetry table, and a symmetry table that does
not map the grid to itself is REFUSED rather than rounded onto it.  The
device-side gate (two full runs producing byte-identical centroid files)
lives with the run; this is the part that can be checked in a second.
"""
import numpy as np
import pytest

from centroid import grid_pool


GRID = (4, 4, 6)
IDENT = np.eye(3, dtype=np.int64)
INVERT = -np.eye(3, dtype=np.int64)


def test_pool_is_the_whole_grid_in_c_order():
    cand = grid_pool.full_grid_candidates(GRID)
    assert cand.shape == (int(np.prod(GRID)), 3)
    # Row m is the point whose flat C-order index is m — that is the whole
    # ordering contract, and everything downstream inherits it.
    flat = (cand[:, 0] * GRID[1] + cand[:, 1]) * GRID[2] + cand[:, 2]
    assert np.array_equal(flat, np.arange(flat.size))


def test_pool_and_labels_are_bit_reproducible():
    Rinv = np.stack([IDENT, INVERT])
    tau = np.zeros((2, 3))
    a = grid_pool.full_grid_candidates(GRID)
    b = grid_pool.full_grid_candidates(GRID)
    assert np.array_equal(a, b)
    ida, _ = grid_pool.grid_orbit_ids(a, Rinv, tau, GRID)
    idb, _ = grid_pool.grid_orbit_ids(b, Rinv, tau, GRID)
    assert np.array_equal(ida, idb)


def test_orbits_partition_the_grid_and_are_closed():
    Rinv = np.stack([IDENT, INVERT])
    tau = np.zeros((2, 3))
    cand = grid_pool.full_grid_candidates(GRID)
    orbit_id, sizes = grid_pool.grid_orbit_ids(cand, Rinv, tau, GRID)
    assert int(sizes.sum()) == cand.shape[0]
    assert np.array_equal(np.bincount(orbit_id), sizes)
    # Under inversion on an even grid the fixed points are the eight
    # points whose every coordinate is 0 or n/2; everything else pairs up.
    assert set(np.unique(sizes).tolist()) == {1, 2}
    assert int((sizes == 1).sum()) == 8
    # Closure: the image of a pool point carries the same label.
    images = grid_pool.grid_images(cand, Rinv, tau, GRID)
    flat_img = ((images[..., 0] * GRID[1] + images[..., 1]) * GRID[2]
                + images[..., 2])
    for s in range(images.shape[0]):
        assert np.array_equal(orbit_id[flat_img[s]], orbit_id)


def test_labels_are_assigned_in_canonical_index_order():
    """Label k belongs to the orbit whose lex-smallest member comes k-th.

    This is what makes the labelling independent of how the pool was
    enumerated, and therefore quotable as "the same set every run".
    """
    Rinv = np.stack([IDENT, INVERT])
    tau = np.zeros((2, 3))
    cand = grid_pool.full_grid_candidates(GRID)
    orbit_id, _ = grid_pool.grid_orbit_ids(cand, Rinv, tau, GRID)
    first_seen = [int(np.flatnonzero(orbit_id == k)[0])
                  for k in range(orbit_id.max() + 1)]
    assert first_seen == sorted(first_seen)


def test_a_grid_the_group_does_not_map_to_itself_is_refused():
    """An incommensurate fractional translation must not be rounded away.

    Rounding it produces a point set that is not orbit-closed while still
    having the right shape and count — the failure mode that a downstream
    shape check cannot see.
    """
    Rinv = np.stack([IDENT, INVERT])
    tau = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    cand = grid_pool.full_grid_candidates(GRID)
    with pytest.raises(ValueError, match="do not land on the FFT grid"):
        grid_pool.grid_orbit_ids(cand, Rinv, tau, GRID)


def test_a_commensurate_glide_is_accepted():
    tau = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.5]])  # x4, x4, x6 integral
    Rinv = np.stack([IDENT, INVERT])
    cand = grid_pool.full_grid_candidates(GRID)
    orbit_id, sizes = grid_pool.grid_orbit_ids(cand, Rinv, tau, GRID)
    assert int(sizes.sum()) == cand.shape[0]
