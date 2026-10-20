"""Independent tiny transfer/column oracles for frequency equilibration.

No material replay, microscopic port truncation, or stability projection.
"""
import numpy as np
import pytest

from gw import shared_pole_constructor as owner
from tests._full_microscopic_oracle import retarded_reference_pairs
from tests.test_shared_pole_general_cpu import _case


def _transfer(model, b, c, z):
    left = c @ model["output_map"]
    right = b.conj().T @ model["input_map"]
    return (left / (z - model["poles_ry"])) @ right.conj().T


def _projected(e, a, b, c, spaces, z):
    x, y = spaces["right"], spaces["left"]
    return c @ x @ np.linalg.solve(y.conj().T @ (z*e-a) @ x, y.conj().T @ b)


def _diagonal_case():
    rng = np.random.default_rng(971)
    poles = np.array([-.8-.09j, -.2+.018j, .5-.07j, 1.1-.11j])
    descriptor = np.array([1e-12, 1e4, 1e-5, 1.])
    e = np.diag(descriptor).astype(complex)
    a = e @ np.diag(poles)
    physical_b = rng.normal(size=(4, 5)) + 1j*rng.normal(size=(4, 5))
    c = rng.normal(size=(5, 4)) + 1j*rng.normal(size=(5, 4))
    b = e @ physical_b
    return poles, e, a, b, c


def test_equilibration_resolves_live_channels_hidden_by_frequency_coordinates():
    poles, e, a, b, c = _diagonal_case()
    # All four channels have finite observable transfer; raw descriptor SVD
    # at this tolerance resolves only two. This is not a dark-state fixture.
    s = np.linalg.svd(e, compute_uv=False)
    assert np.count_nonzero(s > 1e-8*s[0]) == 2
    result = owner.reduce_general_pencil(e, a, max_order=4, rank_tolerance=1e-8)
    assert result["receipt"]["order"] == 4
    np.testing.assert_allclose(np.sort_complex(result["poles_ry"]),
                               np.sort_complex(poles), rtol=2e-11, atol=2e-11)
    assert result["poles_ry"].imag.max() > .017  # real gain survives
    for z in [-.3+.25j, .4+.2j, 1.8+.7j]:
        exact = c @ np.linalg.solve(z*e-a, b)
        np.testing.assert_allclose(_transfer(result, b, c, z), exact,
                                   rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(_transfer(result, b, c, z),
                                   _projected(e, a, b, c, result["spaces"], z),
                                   rtol=2e-10, atol=2e-10)


def test_mixed_finite_infinity_coordinates_are_frequency_unit_covariant():
    poles, _, e, a, b, c = _case()
    # Mixed role positions deliberately interleave finite and infinity
    # coordinates; no contiguous-prefix assumption is available here.
    row_inf = np.array([False, True, False, True])
    col_inf = np.array([True, False, True, False])
    scale = 3.7
    units = (np.where(row_inf, 1/scale, 1.), np.where(col_inf, 1/scale, 1.))
    original = owner.reduce_general_pencil(e, a, max_order=4, rank_tolerance=1e-10,
        frequency_scale_ry=scale, coordinate_units=units)
    conversion = 13.605693122994
    dl = np.where(row_inf, conversion, 1.)
    dr = np.where(col_inf, conversion, 1.)
    # These are the finite/infinity block dimensions when both frequency
    # and W units change: Eff stays fixed, Efi~u, Eii~u^2; A gains one u.
    eu = dl[:, None]*e*dr[None, :]
    au = conversion*dl[:, None]*a*dr[None, :]
    bu = conversion*dl[:, None]*b
    cu = conversion*c*dr[None, :]
    converted = owner.reduce_general_pencil(eu, au, max_order=4, rank_tolerance=1e-10,
        frequency_scale_ry=conversion*scale,
        coordinate_units=(np.where(row_inf, 1/(conversion*scale), 1.),
                          np.where(col_inf, 1/(conversion*scale), 1.)))
    for z in [-.3+.25j, .4+.2j, 1.8+.7j]:
        expected = _transfer(original, b, c, z)
        np.testing.assert_allclose(_transfer(converted, bu, cu, conversion*z)/conversion,
                                   expected, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(np.sort_complex(converted["poles_ry"])/conversion,
                               np.sort_complex(poles), rtol=2e-10, atol=2e-10)


def test_held_spaces_remain_original_coordinates_when_scale_and_units_change():
    _, _, e, a, b, c = _case()
    original = owner.reduce_general_pencil(e, a, max_order=4,
        frequency_scale_ry=2., coordinate_units=(np.array([1., .5, 1., .5]),
                                               np.array([.5, 1., .5, 1.])))
    shift = .006-.002j
    current_a = a + shift*e
    first = owner.reduce_general_pencil(e, current_a, max_order=4,
        fixed_spaces=original["spaces"], frequency_scale_ry=.03,
        coordinate_units=(np.array([1., 90., 1., 90.]), np.array([90., 1., 90., 1.])))
    second = owner.reduce_general_pencil(e, current_a, max_order=4,
        fixed_spaces=original["spaces"], frequency_scale_ry=70.,
        coordinate_units=(np.array([1., .01, 1., .01]), np.array([.01, 1., .01, 1.])))
    for key in ("left", "right"):
        np.testing.assert_array_equal(first["spaces"][key], original["spaces"][key])
        np.testing.assert_array_equal(second["spaces"][key], original["spaces"][key])
    for z in [-.3+.25j, .4+.2j, 1.8+.7j]:
        expected = _projected(e, current_a, b, c, original["spaces"], z)
        for result in (first, second):
            np.testing.assert_allclose(_transfer(result, b, c, z), expected,
                                       rtol=2e-10, atol=2e-10)


def test_equilibrated_full_complex_column_sigma_matches_literal_poles():
    poles, e, a, b, c = _diagonal_case()
    model = owner.reduce_general_pencil(e, a, max_order=4, rank_tolerance=1e-8)
    physical_b = np.linalg.solve(e, b)
    exact_residues = np.stack([np.outer(c[:, i], physical_b[i]) for i in range(4)])
    left, right = c @ model["output_map"], b.conj().T @ model["input_map"]
    got_residues = np.stack([np.outer(left[:, i], right[:, i].conjugate()) for i in range(4)])
    rng = np.random.default_rng(403)
    vertices = .1*(rng.normal(size=(2, 2, 3, 5))+1j*rng.normal(size=(2, 2, 3, 5)))
    energy = np.array([[-.7, -.1, .8], [-.6, .2, .9]])
    rho = np.broadcast_to(np.array([[.8, .09+.04j, .03j],
                                    [.09-.04j, .4, -.05], [-.03j, -.05, .1]]), (2, 3, 3))
    external = np.array([[-.3, .5], [-.2, .6]])
    expected = retarded_reference_pairs(vertices, energy, rho, external, poles, exact_residues, .3)
    got = retarded_reference_pairs(vertices, energy, rho, external,
                                   model["poles_ry"], got_residues, .3)
    for actual, exact in zip(got, expected):
        np.testing.assert_allclose(actual, exact, rtol=2e-10, atol=2e-10)
        assert np.linalg.norm(exact.imag) > 1e-3


def test_equilibration_exact_zero_rows_and_columns_do_not_add_modes():
    poles, e, a, b, c = _diagonal_case()
    ep, ap = np.pad(e, ((1, 1), (1, 1))), np.pad(a, ((1, 1), (1, 1)))
    result = owner.reduce_general_pencil(ep, ap, max_order=6, rank_tolerance=1e-8)
    assert result["receipt"]["order"] == len(poles)
    bp, cp = np.pad(b, ((1, 1), (0, 0))), np.pad(c, ((0, 0), (1, 1)))
    for z in [.3+.2j, -.9+.3j]:
        np.testing.assert_allclose(_transfer(result, bp, cp, z), c@np.linalg.solve(z*e-a, b),
                                   rtol=2e-10, atol=2e-10)


def test_finite_infinity_units_follow_actual_tiled_frequency_order():
    from gw.shared_pole_pencil import join_vectors
    # Two finite columns and one infinity column on each face tile. The
    # physical block join is NOT [all finite, all infinity] after gather.
    scale = 7.
    units = join_vectors((np.ones(4), np.full(2, 1/scale)), 2)
    np.testing.assert_array_equal(units, [1., 1., 1/scale, 1., 1., 1/scale])
    contiguous_ids = join_vectors((np.arange(4), np.arange(4, 6)), 2)
    np.testing.assert_array_equal(contiguous_ids, [0, 1, 4, 2, 3, 5])
    poles, e, a, b, c = _diagonal_case()
    # Insert exact empty coordinates to give the six-column tiled carrier.
    ep, ap = np.pad(e, ((0, 2), (0, 2))), np.pad(a, ((0, 2), (0, 2)))
    bp, cp = np.pad(b, ((0, 2), (0, 0))), np.pad(c, ((0, 0), (0, 2)))
    p = contiguous_ids.astype(int)
    model = owner.reduce_general_pencil(ep[np.ix_(p, p)], ap[np.ix_(p, p)],
        max_order=6, rank_tolerance=1e-8, frequency_scale_ry=scale,
        coordinate_units=(units, units))
    np.testing.assert_allclose(np.sort_complex(model["poles_ry"]),
                               np.sort_complex(poles), rtol=2e-10, atol=2e-10)
    z = .37+.25j
    np.testing.assert_allclose(_transfer(model, bp[p], cp[:, p], z),
                               c@np.linalg.solve(z*e-a, b), rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("scale,units", [
    (0., None), (np.inf, None),
    (1., (np.ones(3), np.ones(4))),
    (1., (np.array([1., 0., 1., 1.]), np.ones(4))),
    (1., (np.ones(4), np.array([1., -1., 1., 1.]))),
])
def test_invalid_equilibration_metadata_refuses_before_svd(monkeypatch, scale, units):
    import scipy.linalg as la
    _, e, a, _, _ = _diagonal_case()
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid coordinate metadata reached LAPACK")
    monkeypatch.setattr(la, "svd", forbidden)
    with pytest.raises(ValueError, match="shared_general_equilibration"):
        owner.reduce_general_pencil(e, a, max_order=4,
            frequency_scale_ry=scale, coordinate_units=units)
