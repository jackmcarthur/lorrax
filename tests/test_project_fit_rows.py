"""minimax.complex_response._project: the fit's Chebyshev rows scale with the node count.

ncheb = max(1400, 2 n).  (a) Every fit with n <= 700 nodes, and so every bank rule (capacity 192),
is bitwise the pre-change fit.  (b) The ZnO real:66 CD crossing window's closed-form imaginary-time
set (n = 1640 Gauss-Legendre nodes, 16 poles) now passes at 5e-9; on 1400 rows it does not.
"""
import numpy as np
import pytest
import scipy.linalg as la

from minimax import complex_response as cr


def _project_1400(lo, hi, pole, times, decay_rate=0.):
    """The pre-change ``_project`` (1400 Chebyshev fit rows at every node count), verbatim."""
    eta = abs(pole.imag)
    span, zp = (hi-lo)/eta, (pole-lo)/eta
    x = np.unique(np.r_[0., span,
        span/2*(1-np.cos(np.pi*(np.arange(1400)+.5)/1400)),
        zp.real+np.linspace(-5, 5, 500)])
    x = x[(x >= 0) & (x <= span)]
    f = 1/(x-zp)
    origin = 0. if decay_rate else lo
    def basis(xx):
        d = lo + eta*xx
        damp = np.minimum(decay_rate*d, 0.)
        a = np.empty((d.size, times.size), np.complex128)
        def fill(i, j):
            a[i:j] = np.exp(damp[i:j, None] - (d[i:j, None]-origin)*times)
        cr._map_rows(fill, d.size, times.size)
        return a, np.exp(damp)
    a, weight = basis(x)
    a = a/abs(f[:, None])
    scale = la.norm(a, axis=0)
    if np.any(scale == 0) or not np.isfinite(scale).all():
        return None
    coeff = la.lstsq(a/scale, weight[:, None]*np.column_stack((f, f*f))/abs(f[:, None]),
                    cond=1e-14, lapack_driver='gelsd', check_finite=False)[0]/scale[:, None]
    probe = np.unique(np.r_[x, np.linspace(0, span, 2501)])
    error = np.zeros(2)
    a, weight = basis(probe)
    start = 0
    for xx in np.array_split(probe, 16):
        stop = start + xx.size
        exact = 1/(xx-zp)
        error = np.maximum(error, np.max(abs(a[start:stop]@coeff
                       - weight[start:stop, None]*np.column_stack((exact, exact*exact))), axis=0))
        start = stop
    mass = np.sum(abs(coeff), axis=0)
    return coeff / np.array([eta, eta*eta]), error, mass


def _same(a, b):
    assert (a is None) == (b is None)
    if a is not None:
        for u, v in zip(a, b):
            assert np.array_equal(u, v)


LO, HI, ETA = 0.05, 4.0, 0.02


@pytest.mark.parametrize("n", [8, 96, 192, 700])
def test_fit_bitwise_unchanged_through_700_nodes(n):
    rng = np.random.default_rng(n)
    pole = 1.7 + 1j*ETA
    pencil_like = (rng.uniform(0.01, 2., n) + 1j*rng.uniform(-30., 30., n))/ETA
    u, _ = np.polynomial.legendre.leggauss(n)
    closed_form = 1j*20.*(u+1)/(2*ETA)
    for times in (pencil_like, closed_form):
        _same(cr._project(LO, HI, pole, times), _project_1400(LO, HI, pole, times))
        _same(cr._project(LO, HI, pole, times, .3), _project_1400(LO, HI, pole, times, .3))


def test_fit_rows_grow_above_700_nodes():
    u, _ = np.polynomial.legendre.leggauss(701)
    times = 1j*20.*(u+1)/(2*ETA)
    new, old = cr._project(LO, HI, 1.7 + 1j*ETA, times), _project_1400(LO, HI, 1.7 + 1j*ETA, times)
    assert not np.array_equal(new[0], old[0])


def test_bank_rules_bitwise_unchanged(monkeypatch):
    z = np.array([0.30 + 0.02j, 0.45 + 0.02j, 1.1 + 0.03j, 0.9j])
    rules = cr.response_group_rules(0.05, 1.6, z, rel_tol=1e-8)
    monkeypatch.setattr(cr, "_project", _project_1400)
    reference = cr.response_group_rules(0.05, 1.6, z, rel_tol=1e-8)
    assert _equal(rules, reference)


def _equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and sorted(a) == sorted(b) and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_equal(u, v) for u, v in zip(a, b))
    return a == b


# ZnO P64_k8 deck, CD knot window real:66 (8 knots 33.25-33.6875 eV + 0.25i eV), crossing pair
# valence (bands 9-26) x empty bands 27-55: the response_support interval, snapped outward [Ry].
ZNO_LO, ZNO_HI = 0.05603628147192522, 4.351090398023211
ZNO_Z = np.array([2.4438290368455675, 2.4484227004486607, 2.453016364051754, 2.457610027654847,
                  2.4622036912579404, 2.4667973548610336, 2.471391018464127, 2.47598468206722]) \
    + 1j*0.01837465441237269


def test_zno_crossing_closed_form_passes():
    tol = 1e-8/2
    gate = max(cr._RESPONSE_MAX_KAPPA, 0.01*1e-8/np.finfo(np.float64).eps)
    poles = cr._poles(ZNO_Z)
    got = cr._imaginary_times(ZNO_LO, ZNO_HI, poles, tol, 0., 2200, max_kappa=gate)
    assert got is not None
    times, fits = got
    assert len(times) == 1640 and len(fits) == len(poles) == 16
    for fit, pole in zip(fits, poles):
        assert np.max(fit[1]*[1., pole.imag/(2*abs(pole))]) <= tol
        assert fit[2][0] <= gate
    # Independent check of the value fit on a uniform grid 8x denser than the probe.
    d = np.linspace(ZNO_LO, ZNO_HI, 20001)
    for k in (0, 1):
        coeff, pole = fits[k][0][:, 0], poles[k]
        worst = max(np.max(abs(np.exp(-(dd[:, None]-ZNO_LO)*times) @ coeff - 1/(dd-pole)))
                    for dd in np.array_split(d, 20))
        assert worst*pole.imag <= tol
    # On the old 1400 fit rows the same set fails by orders of magnitude.
    old = _project_1400(ZNO_LO, ZNO_HI, poles[0], times)
    assert old[1][0] > 1e3*tol
