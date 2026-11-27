"""A symmetric centroid selection needs an FFT grid its operations map onto
itself, and a centroid table holds distinct rows.

The fixture is rock salt in the short-lattice-vector cell of the 2026-11-27
defect: two metal and two oxygen atoms, 24 decorated-atom operations, of which
only those that do not mix the long axis with the short ones map a
12 x 12 x 20 grid (QE's 48 x 48 x 80 scaled by four) onto itself.
"""
from math import lcm

import numpy as np
import pytest

from centroid.kmeans_isdf import (
    require_distinct_grid_points,
    require_grid_invariant_operations,
)
from symmetry_maps import (
    centroid_source_map_and_wrap,
    fft_grid_invariance,
    recover_atomic_space_group,
)


def _rock_salt_short_cell():
    """Rows a1 - a2, a2 - a3, a3 of the antiferromagnetic rhombohedral cell."""
    rhombohedral = np.array([[1., .5, .5], [.5, 1., .5], [.5, .5, 1.]])
    avec = np.array([rhombohedral[0] - rhombohedral[1],
                     rhombohedral[1] - rhombohedral[2], rhombohedral[2]])
    # (u1, u2, u3) -> (u1, u1 + u2, u1 + u2 + u3) from the rhombohedral cell.
    atoms = np.array([[0., 0., 0.], [.5, 0., .5], [.25, .5, .75], [.75, .5, .25]])
    return avec, atoms, np.array([1, 1, 2, 2])


def _maps_grid_onto_itself(Rinv, tau, grid):
    """Independent oracle: enumerate every grid point's image exactly.

    Coordinates are integers in units of 1/(2 L), L the least common multiple
    of the extents, which holds every image for translations that are
    multiples of 1/2 (asserted).
    """
    grid = np.asarray(grid, dtype=np.int64)
    unit = 2 * lcm(*(int(n) for n in grid))
    points = np.array(list(np.ndindex(*grid)), dtype=np.int64) * (unit // grid)
    shift = np.rint(np.asarray(tau) * unit).astype(np.int64)
    assert np.allclose(shift, np.asarray(tau) * unit, atol=1e-9)
    images = (points @ np.asarray(Rinv, dtype=np.int64).T + shift) % unit
    step = unit // grid
    if np.any(images % step):
        return False
    flat = np.ravel_multi_index((images // step).T, tuple(grid))
    return np.array_equal(np.sort(flat), np.arange(len(points)))


def test_grid_test_equals_exact_enumeration_and_the_selection_refuses():
    avec, atoms, species = _rock_salt_short_cell()
    _, Rinv, tau = recover_atomic_space_group(avec, atoms, species)
    assert Rinv.shape[0] == 24

    for grid, closes in (((12, 12, 20), False), ((12, 12, 12), True),
                         ((20, 20, 20), True), ((12, 20, 12), False)):
        rotation_fits, translation_fits = fft_grid_invariance(Rinv, tau, grid)
        fits = rotation_fits & translation_fits
        expected = [_maps_grid_onto_itself(R, t, grid) for R, t in zip(Rinv, tau)]
        assert fits.tolist() == expected
        assert bool(fits.all()) is closes
        # The operations that pass form a group: products stay in the set.
        kept = {(R.tobytes(), tuple(np.rint(2 * t).astype(int) % 2))
                for R, t in zip(Rinv[fits], tau[fits])}
        for Ra, ta in zip(Rinv[fits], tau[fits]):
            for Rb, tb in zip(Rinv[fits], tau[fits]):
                product = ((Ra @ Rb).astype(Rinv.dtype).tobytes(),
                           tuple(np.rint(2 * (Ra @ tb + ta)).astype(int) % 2))
                assert product in kept
        if closes:
            assert require_grid_invariant_operations(Rinv, tau, grid) is None
        else:
            assert 1 < np.count_nonzero(fits) < 24
            with pytest.raises(ValueError, match="GATE centroid_fft_grid_invariance") as refusal:
                require_grid_invariant_operations(Rinv, tau, grid)
            text = str(refusal.value)
            assert " x ".join(map(str, grid)) in text and "nr1 = nr2 = nr3" in text

    # A glide (x, y, z) -> (x + 1/2, -y, z): the rotation maps every grid, the
    # translation only an even first extent.
    glide_R = np.stack((np.eye(3), np.diag([1, -1, 1]))).astype(np.int64)
    glide_tau = np.array([[0., 0., 0.], [.5, 0., 0.]])
    for grid, closes in (((8, 4, 2), True), ((5, 4, 2), False)):
        rotation_fits, translation_fits = fft_grid_invariance(
            glide_R, glide_tau, grid)
        assert rotation_fits.tolist() == [True, True]
        assert translation_fits.tolist() == [True, closes]
        assert _maps_grid_onto_itself(glide_R[1], glide_tau[1], grid) is closes
    with pytest.raises(ValueError, match="GATE centroid_fft_grid_invariance") as refusal:
        require_grid_invariant_operations(glide_R, glide_tau, (5, 4, 2))
    assert "multiples of the translations' denominators" in str(refusal.value)


def test_repeated_centroid_rows_refuse_by_name():
    avec, atoms, species = _rock_salt_short_cell()
    R, Rinv, tau = recover_atomic_space_group(avec, atoms, species)
    grid = (12, 12, 12)

    # One closed orbit of a general point: distinct rows, every action a
    # permutation.
    seed = np.array([1, 2, 5])
    images = np.rint((seed / np.asarray(grid)) @ np.swapaxes(Rinv, 1, 2) * grid
                     + tau * grid).astype(np.int64) % grid
    orbit = np.unique(images, axis=0)
    assert require_distinct_grid_points(orbit, grid, "an orbit") == len(orbit)
    perm, _ = centroid_source_map_and_wrap(orbit, R, tau * (2 * np.pi), grid)
    assert np.array_equal(np.sort(perm, axis=1),
                          np.broadcast_to(np.arange(len(orbit)), perm.shape))

    # The same table with one row written twice.
    repeated = np.concatenate((orbit, orbit[3:4]))
    with pytest.raises(RuntimeError, match="GATE centroid_duplicate_rows"):
        require_distinct_grid_points(repeated, grid, "an orbit")
    for rows in (None, np.array([0])):
        with pytest.raises(RuntimeError, match="duplicate centroid rows") as refusal:
            centroid_source_map_and_wrap(
                repeated, R, tau * (2 * np.pi), grid, required_rows=rows)
        assert "tau" not in str(refusal.value).lower()

    # Distinct rows and a fourfold rotation between axes of unequal extent:
    # rounded images collide, and the refusal names the rotation, not a
    # fractional translation.  (The sixteen rock-salt operations that leave
    # 12 x 12 x 20 are shears whose rounded images are still a bijection of
    # the grid, which is why only the integer test sees them.)
    fourfold = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]])
    unequal = (4, 6, 2)
    every_point = np.array(list(np.ndindex(*unequal)))
    assert fft_grid_invariance(
        fourfold, np.zeros((2, 3)), unequal)[0].tolist() == [True, False]
    with pytest.raises(RuntimeError) as refusal:
        centroid_source_map_and_wrap(
            every_point, fourfold, np.zeros((2, 3)), unequal)
    text = str(refusal.value)
    assert "duplicate centroid rows" not in text
    assert "rotation does not map the FFT grid" in text
