"""External-energy slopes against independent ordered complex-pole models."""
import numpy as np
import pytest

from common.units import RYD_TO_EV
from gw import contour_reference as cd


def _integral(eta_ev, occupation, convention, *, derivative=True, shift=0.,
              masked=False, wrong_partner=False, omit_zero=False, n=384):
    eta, omega = eta_ev / RYD_TO_EV, .4
    rp = np.asarray([[.04, .006 + .009j], [.006 - .009j, .027]])[None, :, :, None]
    rm = np.asarray([[.025, -.008 + .004j], [-.008 - .004j, .035]])[None, :, :, None]
    q = np.asarray([-.6, -.4, -.1, -1e-9, 0., 1e-9, .1, .4, .6]) + shift
    x = np.broadcast_to(q[None, None, :, None], (1, 2, len(q), 1))
    f = np.full(x.shape, occupation)
    valid, n_active = None, None
    if masked:
        x = x + np.asarray([0., .03, .19])[None, None, None, :]
        f = np.broadcast_to(np.asarray([occupation, .7, .2]), x.shape)
        scale = np.asarray([1., 7., 1000.])[None, None, None, :]
        rp, rm = rp * scale, rm * scale
        valid = np.asarray([True, False, True])[None, None, None, :]
        n_active = 2
    physical = np.ones(x.shape, bool)
    if valid is not None:
        physical &= np.broadcast_to(valid, x.shape)
        physical &= np.arange(x.shape[-1])[None, None, None, :] < n_active

    def fields(z, partner=False):
        a, b = (rm, rp) if partner else (rp, rm)
        value = -a / (z - omega) + b / (z + omega)
        slope_s = (a / (z - omega)**2 - b / (z + omega)**2) / (2*z)
        return value, slope_s

    kwargs = dict(eta=eta, n_active=n_active, band_valid=valid,
                  analytic_convention=convention)
    value0, slope0 = fields(1j*eta)
    got, cp, cm, beta = cd.anchor_part(value0, slope0, x, f,
        external_derivative=derivative, **kwargs)
    u, weights = cd.imaginary_rule(n, eta, scale=omega)
    for ui, wi in zip(u, weights):
        got += cd.imag_remainder_node(fields(1j*ui)[0], ui, wi, x, f, cp, cm,
            beta, external_derivative=derivative, **kwargs)
    nonzero = np.where(x < 0, f, 1-f) != 0
    nodes = np.unique(np.abs(x)[physical & nonzero])
    for node in nodes:
        if omit_zero and node == 0:
            continue
        direct = fields(node + 1j*eta)
        partner = fields(node + 1j*eta, partner=not wrong_partner)
        if derivative:
            got += cd.real_residue_derivative_node(direct[1], partner[1],
                x, f, node, **kwargs)
        else:
            got += cd.real_residue_node(direct[0], partner[0], x, f, node, **kwargs)
    occupied_sheet = -1 if convention == "time_ordered_fractional" else 1
    dc, dv = x-omega+1j*eta, x+omega+occupied_sheet*1j*eta
    if derivative:
        left, right = -(1-f)/dc**2, -f/dv**2
    else:
        left, right = (1-f)/dc, f/dv
    expected = (np.einsum("kabl,kael->kabe", rp, np.where(physical, left, 0.))
                + np.einsum("kabl,kael->kabe", rm, np.where(physical, right, 0.)))
    return got, expected


def _relative_by_energy(got, expected):
    # Each read energy has its own scale; resonances cannot hide x=0 errors.
    norm = np.max(np.abs(expected), axis=(0, 1, 2))
    return np.max(np.max(np.abs(got-expected), axis=(0, 1, 2)) / norm)


@pytest.mark.parametrize("eta_ev", [.05, .25, 1.])
@pytest.mark.parametrize("occupation", [0., .25, .5, 1.])
@pytest.mark.parametrize("convention", ["time_ordered_fractional", "retarded"])
def test_exact_external_derivative_matches_independent_poles(eta_ev, occupation, convention):
    got, expected = _integral(eta_ev, occupation, convention)
    assert _relative_by_energy(got, expected) < 2e-7


@pytest.mark.parametrize("convention", ["time_ordered_fractional", "retarded"])
def test_external_derivative_masks_nonzero_invalid_coefficients(convention):
    got, expected = _integral(.25, .25, convention, masked=True)
    assert _relative_by_energy(got, expected) < 2e-7


@pytest.mark.parametrize("convention", ["time_ordered_fractional", "retarded"])
def test_crossing_derivative_matches_actual_two_sided_response_queries(convention):
    step = 1e-6 * (.25 / RYD_TO_EV)
    slope, _ = _integral(.25, .5, convention)
    plus, _ = _integral(.25, .5, convention, derivative=False, shift=step)
    minus, _ = _integral(.25, .5, convention, derivative=False, shift=-step)
    finite = (plus-minus) / (2*step)
    assert _relative_by_energy(slope, finite) < 2e-6


def test_wrong_partner_slope_and_missing_zero_crossing_are_discriminating():
    correct, expected = _integral(.25, .25, "retarded")
    wrong, _ = _integral(.25, .25, "retarded", wrong_partner=True)
    missing, _ = _integral(.25, .25, "retarded", omit_zero=True)
    assert _relative_by_energy(correct, expected) < 2e-7
    assert _relative_by_energy(wrong, expected) > 1e-3
    assert _relative_by_energy(missing, expected) > 1e-3


@pytest.mark.parametrize("bad", [1, "yes", None])
def test_external_derivative_requires_static_boolean(bad):
    x, f = np.zeros((1, 1, 1, 1)), np.ones((1, 1, 1, 1))
    value = np.ones((1, 1, 1, 1), complex)
    with pytest.raises(ValueError, match="static boolean"):
        cd.anchor_part(value, value, x, f, eta=.01, external_derivative=bad)
    with pytest.raises(ValueError, match="static boolean"):
        cd.imag_remainder_node(value, .2, .1, x, f, (value,value), (value,value),
                               (.1,.3), eta=.01, external_derivative=bad)
