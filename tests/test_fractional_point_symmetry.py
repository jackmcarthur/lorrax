"""Off-grid point transport: nonsymmorphic wraps, parity and red controls."""

import numpy as np
import pytest

from symmetry_maps import centroid_source_map_and_wrap, verify_centroid_orbit_closure


def glide_data():
    S = np.array([np.eye(3, dtype=int), np.diag([1, -1, -1])])
    tau = np.array([[0, 0, 0], [0.5, 0, 0]]) * (2*np.pi)
    x = np.array([[0.1234567, 0.2734567, 0.3134567],
                  [0.6234567, 0.7265433, 0.6865433]])
    return S, tau, x


def test_identity_closure_distinguishes_points_in_one_rounded_bucket():
    points = np.array([[.2000001, .3, .4], [.2000004, .3, .4]])
    verdict = verify_centroid_orbit_closure(
        points, np.eye(3, dtype=int)[None], tau=np.zeros((1, 3)), tol=2e-12)
    assert verdict.closed and verdict.worst_residual == 0.
    perm, _ = centroid_source_map_and_wrap(
        points, np.eye(3, dtype=int)[None], np.zeros((1, 3)), None,
        coordinate_kind='fractional', extend_trs=False)
    np.testing.assert_array_equal(perm, [[0, 1]])


def test_closure_nearest_neighbor_matches_independent_periodic_distances():
    from symmetry_maps.orbit_syms import _min_image_residual
    points = np.array([[.2000001, .3, .4], [.2000005001, .3, .4],
                       [.99, .01, .5]])
    images = np.array([[.2000004999, .3, .4], [1.01, -.01, .5],
                       [2.2000001, -.7, .4]])
    differences = images[:, None] - points[None]
    differences -= np.rint(differences)
    reference = np.linalg.norm(differences, axis=-1).min(axis=1)
    np.testing.assert_allclose(_min_image_residual(images, points), reference, atol=3e-16, rtol=0.)


def test_fractional_glide_wraps_and_antiunitary_rows():
    S, tau, x = glide_data()
    perm, wrap = centroid_source_map_and_wrap(
        x, S, tau, None, coordinate_kind='fractional', extend_trs=True)
    np.testing.assert_array_equal(perm, [[0, 1], [1, 0], [0, 1], [1, 0]])
    np.testing.assert_array_equal(wrap[1], [[-1, -1, -1], [0, -1, -1]])
    np.testing.assert_array_equal(wrap[3], wrap[1])
    # Independent Bloch phase check at a NON-TRIM k, sensitive to wrap signs.
    k = np.array([0.2, 0.17, -0.31])
    raw = (x - tau[1]/(2*np.pi)) @ S[1].T
    actual = np.exp(2j*np.pi * (x[perm[1]] @ k)) * np.exp(2j*np.pi * (wrap[1] @ k))
    np.testing.assert_allclose(actual, np.exp(2j*np.pi * (raw @ k)), atol=2e-15)


def test_fractional_mapping_is_exact_grid_specialization():
    S, tau, _ = glide_data()
    grid = np.array([8, 8, 8])
    indices = np.array([[1, 2, 3], [5, 6, 5]])
    exact = centroid_source_map_and_wrap(indices, S, tau, grid, extend_trs=True)
    fractional = centroid_source_map_and_wrap(indices/grid, S, tau, grid,
                                              coordinate_kind='fractional', extend_trs=True)
    for a, b in zip(exact, fractional):
        np.testing.assert_array_equal(a, b)


def test_unwrapped_physical_points_keep_their_bloch_wraps():
    S, tau, x = glide_data()
    x = x + np.array([[-1, 2, 0], [2, -1, 1]])
    perm, wrap = centroid_source_map_and_wrap(
        x, S, tau, None, coordinate_kind='fractional', extend_trs=True)
    np.testing.assert_array_equal(perm, [[0, 1], [1, 0], [0, 1], [1, 0]])
    k = np.array([.2, .17, -.31])
    for row in range(2):
        raw = (x-tau[row]/(2*np.pi)) @ S[row].T
        np.testing.assert_allclose(x[perm[row]]+wrap[row], raw, atol=2e-15)
        actual = np.exp(2j*np.pi*(x[perm[row]] @ k))*np.exp(2j*np.pi*(wrap[row] @ k))
        np.testing.assert_allclose(actual, np.exp(2j*np.pi*(raw @ k)), atol=5e-15)
    # A lattice-equivalent duplicate is ambiguous even at a different image.
    with pytest.raises(ValueError, match='ambiguous'):
        centroid_source_map_and_wrap(np.vstack((x, x[0]+[3, 0, 0])),
            S, tau, None, coordinate_kind='fractional')


def test_unwrapped_transition_density_fourier_phase_is_required():
    _, _, points = glide_data()
    points = points + np.array([[-1, 2, 0], [2, -1, 1]])
    wrapped = points % 1.
    image = np.rint(points-wrapped)
    q, G = np.array([1/3, 0., 0.]), np.array([2, -1, 3])
    periodic = lambda x: 1+.2*np.exp(2j*np.pi*(x @ np.array([1, 1, 0])))
    density = lambda x: np.exp(2j*np.pi*(x @ q))*periodic(x)
    physical = density(points)*np.exp(-2j*np.pi*(points @ (q+G)))
    restored = density(wrapped)*np.exp(2j*np.pi*(image @ q))*np.exp(-2j*np.pi*(points @ (q+G)))
    np.testing.assert_allclose(restored, physical, atol=5e-15)
    wrong = density(wrapped)*np.exp(-2j*np.pi*(points @ (q+G)))
    assert np.linalg.norm(wrong-physical) > 1.


def test_negative_roundoff_modulo_one_remains_a_valid_periodic_point():
    point = np.array([[-1e-18, .2, .3]])
    assert (point % 1.)[0, 0] == 1.
    perm, wrap = centroid_source_map_and_wrap(point, np.eye(3,dtype=int)[None],
        np.zeros((1,3)), None, coordinate_kind='fractional')
    np.testing.assert_array_equal(perm, [[0]])
    np.testing.assert_array_equal(wrap, 0)


def test_fractional_missing_image_refuses_without_nearest_fallback():
    S, tau, x = glide_data()
    with pytest.raises(RuntimeError, match='orbit closure'):
        centroid_source_map_and_wrap(x[:1], S, tau, None, coordinate_kind='fractional')
    # An unavailable unused row remains canonical -1; its ID is not compacted.
    perm, wrap = centroid_source_map_and_wrap(
        x[:1], S, tau, None, coordinate_kind='fractional', extend_trs=True,
        required_rows=np.array([2]))
    np.testing.assert_array_equal(perm, [[0], [-1], [0], [-1]])
    np.testing.assert_array_equal(wrap, 0)


def test_fractional_ambiguous_images_and_invalid_mode_refuse():
    S, tau, x = glide_data()
    duplicate = np.vstack([x, x[0]+1e-13])
    with pytest.raises(ValueError, match='ambiguous'):
        centroid_source_map_and_wrap(duplicate, S, tau, None, coordinate_kind='fractional')
    with pytest.raises(ValueError, match='coordinate_kind'):
        centroid_source_map_and_wrap(x, S, tau, None, coordinate_kind='rounded')
