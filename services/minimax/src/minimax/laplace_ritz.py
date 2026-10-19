"""Elliptic time-Ritz placement and certified noncrossing response projection.

See docs/theory/response-laplace.md. All energies share one unit; times use
its inverse. Host scalar work only: no material data or spatial arrays.
"""
from functools import lru_cache
import math

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.special import ellipk, ellipkm1


def _elliptic_rates(n: int, a: float, b: float) -> NDArray:
    """b*dn((2j+1)K/(2n), k), k' = a/b; positive-nome evaluation.

    A product representation avoids forming 1-(a/b)**2 when a/b is small.
    SciPy's elliptic integrals use the parameter m, not the modulus k.
    """
    ratio = a/b
    if not 0 < ratio <= 1:
        raise ValueError('Require 0 < a <= b.')
    if ratio == 1:
        return np.full(n, a)
    ell = np.pi*ellipk(ratio*ratio)/ellipkm1(ratio*ratio)
    u = np.cos(np.pi*(np.arange(n)+0.5)/n)
    logphi = np.full(n, 0.5*np.log(ratio))
    cutoff = 1e-19*(-np.expm1(-2*ell))
    terms = max(1, math.ceil((-np.log(cutoff)/ell-1)/2)+2)
    for j in range(terms):
        qj = np.exp(-(2*j+1)*ell)
        logphi += np.log1p(2*qj*u+qj*qj)-np.log1p(-2*qj*u+qj*qj)
    return b*np.exp(logphi)


def place_times(lo: float, hi: float, max_abs_real_z: float, degree: int) -> NDArray:
    """Return a prescribed set of positive real times, with NO nonlinear fit.

    Only the transition bounds, real-frequency extent, and degree are used.
    Heights enter the coefficient projection and certificate/degree selection,
    not this fixed-degree node map. A Lyapunov solve and symmetric eigensolve
    are the only linear-algebra operations required.
    """
    lo, hi, reach = map(float, (lo, hi, max_abs_real_z))
    if not all(np.isfinite([lo, hi, reach])) or not 0 <= reach < lo <= hi:
        raise ValueError('Require finite 0 <= frequency reach < lo <= hi.')
    if isinstance(degree, bool) or int(degree) != degree or degree < 1:
        raise ValueError('degree must be a positive integer.')
    n = int(degree)
    a, b = (lo-reach)/2, (hi+reach)/2
    scale = np.sqrt(a)*np.sqrt(b)
    rates = _elliptic_rates(n, a/scale, b/scale)
    bb = np.sqrt(2*rates)
    generator = -np.diag(rates)-np.triu(np.outer(bb, bb), 1)
    moment = linalg.solve_continuous_lyapunov(generator.T, -np.eye(n))
    moment = (moment+moment.T)/2
    t = linalg.eigvalsh(moment)/scale
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ArithmeticError('Loss of positivity in the time Ritz spectrum.')
    return t


def _samples(lo: float, hi: float, reach: float, n: int):
    # Uniform Chebyshev angles in log(d-reach), clustering at both endpoints.
    q = np.cos(np.pi*(np.arange(n)+0.5)/n)
    la, lb = np.log(lo-reach), np.log(hi-reach)
    return reach+np.exp((la+lb)/2+(lb-la)*q/2)


def _project(t, lo, hi, z, ordered):
    """Linear projection of exact values AND ds targets onto fixed times."""
    d = _samples(lo, hi, float(np.abs(z.real).max()), max(400, 16*len(t)))
    basis = np.exp(-(d-lo)[:, None]*t)
    den = d[:, None]**2-z**2
    targets = [d[:, None]/den, d[:, None]/den**2]
    if ordered:
        targets += [1/den, 1/den**2]
    coefficients = np.empty((len(t), len(targets)*len(z)), dtype=np.complex128)
    condition = 0.
    for k, target in enumerate(np.concatenate(targets, axis=1).T):
        magnitude = np.abs(target)
        matrix = basis/magnitude[:, None]
        scale = np.linalg.norm(matrix, axis=0)
        if np.any(scale == 0):
            raise ArithmeticError('Numerically null response time column')
        rhs = np.c_[target.real/magnitude, target.imag/magnitude]
        c, _, _, sv = linalg.lstsq(matrix/scale, rhs, cond=1e-15,
                                   lapack_driver='gelsd', check_finite=False)
        coefficients[:, k] = (c[:, 0]+1j*c[:, 1])/scale
        condition = max(condition, float(sv[0]/sv[-1]))
    return coefficients, condition


@lru_cache(maxsize=1)
def _power_to_chebyshev(order):
    """Exact dyadic conversion entries; residuals accumulate in long double."""
    conv = np.zeros((order+1, order+1), dtype=np.longdouble)
    for k in range(order+1):
        e = np.zeros(order+1)
        e[k] = 1
        c = np.polynomial.chebyshev.poly2cheb(e)
        conv[:len(c), k] = c
    return conv


def _certificate(t, c, lo, hi, z, tol, ordered):
    """Continuum polynomial-denominator residuals, including squared kernels.

    On each d panel, expand the exponentials to degree20, multiply by the
    exact denominator polynomial (degree2 or4), and bound the residual by
    its Chebyshev coefficient l1 norm plus the explicit exponential remainder.
    The floating-point guard is conventional, not an interval-libm proof.
    """
    t, c, z = (np.asarray(x, dtype=dtype) for x, dtype in
               ((t, np.longdouble), (c, np.clongdouble), (z, np.clongdouble)))
    ns = len(z)
    names = ['value_bound', 'derivative_bound']
    if ordered:
        e, de, k, dk = np.split(c, 4, axis=1)
        # Literal reciprocal orientations and their derivatives, without
        # charging their cancellation through an absolute triangle bound.
        do = k/(2*z)+z*dk
        c = np.c_[c, e+z*k, e-z*k, 2*z*(de+do), -2*z*(de-do)]
        names += ['odd_value_bound', 'odd_derivative_bound',
                  'forward_value_bound', 'backward_value_bound',
                  'forward_derivative_bound', 'backward_derivative_bound']
    reach = float(np.abs(z.real).max())
    npan = max(1, math.ceil(math.log((hi-reach)/(lo-reach))/math.log(1.15)))
    edges = reach+np.geomspace(lo-reach, hi-reach, npan+1)
    edges[0], edges[-1] = lo, hi
    order = 20
    convert = _power_to_chebyshev(order+4)
    bound = np.zeros(c.shape[1], dtype=np.longdouble)
    amplification, arithmetic = 0., 0.
    fact = np.longdouble(math.factorial(order+1))
    unit = np.longdouble(np.finfo(np.float64).eps/2)
    for left, right in zip(edges[:-1], edges[1:]):
        left, right = np.longdouble(left), np.longdouble(right)
        centre, half = (left+right)/2, (right-left)/2
        ht = half*t
        power = np.ones((order+1, len(t)), dtype=np.longdouble)
        for k in range(1, order+1):
            power[k] = power[k-1]*(-ht)/k
        moments = power@(np.exp(-(centre-lo)*t)[:, None]*c)
        p0, p1, p2 = centre*centre-z*z, 2*centre*half, half*half
        den = np.zeros((5, c.shape[1]), dtype=np.clongdouble)
        quad = np.array([p0, np.full(ns, p1), np.full(ns, p2)])
        square = np.array([p0*p0, 2*p0*p1, p1*p1+2*p0*p2,
                           np.full(ns, 2*p1*p2), np.full(ns, p2*p2)])
        den[:3, :ns], den[:, ns:2*ns] = quad, square
        quadmax = np.maximum(abs(left*left-z*z), abs(right*right-z*z))
        dmax = [quadmax, quadmax**2]
        if ordered:
            den[:3, 2*ns:3*ns], den[:, 3*ns:4*ns] = quad, square
            den[0, 4*ns:6*ns] = np.r_[centre-z, centre+z]
            den[1, 4*ns:6*ns] = half
            den[0, 6*ns:] = np.r_[(centre-z)**2, (centre+z)**2]
            den[1, 6*ns:] = 2*half*np.r_[centre-z, centre+z]
            den[2, 6*ns:] = half*half
            minus = np.maximum(abs(left-z), abs(right-z))
            plus = np.maximum(abs(left+z), abs(right+z))
            dmax += [quadmax, quadmax**2, minus, plus, minus**2, plus**2]
        poly = np.zeros((order+5, c.shape[1]), dtype=np.clongdouble)
        for k in range(5):
            poly[k:k+order+1] += den[k]*moments
        poly[0, :2*ns] -= centre
        poly[1, :2*ns] -= half
        poly[0, 2*ns:] -= 1
        normalizer = np.r_[np.full(2*ns, left), np.ones(c.shape[1]-2*ns)]
        dmax = np.concatenate(dmax)/normalizer
        numer = np.sum(abs(convert@poly), axis=0)/normalizer
        remainder = (np.exp(-(left-lo)*t)*ht**(order+1)/fact)@abs(c)*dmax
        amp = (np.exp(-(left-lo)*t)@abs(c))*dmax
        guard = (16*len(t)+128)*unit*amp+32*unit
        bound = np.maximum(bound, numer+remainder+guard)
        amplification = max(amplification, float(amp.max()))
        arithmetic = max(arithmetic, float(guard.max()))
    cert = {name: np.asarray(values, dtype=float).tolist()
            for name, values in zip(names, np.split(bound, len(names)))}
    cert.update(status='PASS' if bound.max() <= tol else 'FAIL',
        scope='continuum delta interval; every supplied z; values and ds',
        norm='relative Ke, dsKe; ordered K, dsK and primitive reciprocals/squares',
        maximum_bound=float(bound.max()), amplification_bound=amplification,
        arithmetic_guard=arithmetic, panel_count=npan, polynomial_degree=order,
        rounding_scope='standard-rounding guard, not interval-libm',
        delta_ry=[float(lo), float(hi)], rel_tol=tol,
        z_ry=[[float(v.real), float(v.imag)] for v in z],
        ordered=ordered, owner='elliptic time Ritz / linear Hermite projection')
    return cert


def project_response(t, lo, hi, z, reference, tol, ordered):
    """Return and certify the rounded, reference-shifted production rows."""
    c, condition = _project(t, lo, hi, z, ordered)
    rows = (c*np.exp(-(reference-lo)*t)[:, None]).T
    ns = len(z)
    result = dict(t=t, reference_ry=reference,
                  projection_value=rows[:ns].copy(),
                  projection_derivative=rows[ns:2*ns].copy())
    if ordered:
        result.update(odd_projection_value=z[:, None]*rows[2*ns:3*ns],
            odd_projection_derivative=rows[2*ns:3*ns]/(2*z[:, None])+z[:, None]*rows[3*ns:])
    # Reconstruct K and dsK from ACTUAL rounded returned odd rows, including
    # the 1/(2z) chain term, before certifying the padded domain at anchor lo.
    actual = [result['projection_value'], result['projection_derivative']]
    if ordered:
        zz = z.astype(np.clongdouble)[:, None]
        odd = result['odd_projection_value'].astype(np.clongdouble)
        actual += [odd/zz,
                   result['odd_projection_derivative'].astype(np.clongdouble)/zz-odd/(2*zz**3)]
    shift = np.exp(np.longdouble(reference-lo)*t.astype(np.longdouble))
    actual = np.concatenate(actual).astype(np.clongdouble).T*shift[:, None]
    if not np.all(np.isfinite(actual)):
        raise ArithmeticError('Reference shift exceeds scalar certification range')
    result['certificate'] = _certificate(t, actual, lo, hi, z, tol, ordered)
    result['certificate']['maximum_scaled_basis_condition'] = condition
    return result
