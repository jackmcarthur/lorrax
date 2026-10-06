"""Normalized RKB atom corrections, before the augmented-sample ISDF fit.

The source radial correction is delta_R = R_AE - R_PS, not an independently
normalized Dirac small component.  Linearity gives ``U T psi = U psi +
sum_i c_i U delta_phi_i``.  This module builds the species tables of
``U delta_phi_i``, using the carrier owned by ``common.bispinor_init``.
All lengths are bohr, momenta bohr^-1, radial weights integrate dr or dK.

Tables are small host data.  Band contractions and distributed orbital
samples belong to the caller.  No band orthogonalization, core occupation,
or support cutoff is inferred here.  In particular the normalized RKB
kernel has short noncompact tails; evaluating beyond a cache refuses.
"""
from __future__ import annotations

import numpy as np


def _quadrature(nodes, weights, name):
    x = np.asarray(nodes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (x.ndim != 1 or len(x) < 2 or w.shape != x.shape
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(w))
            or np.any(x <= 0) or np.any(np.diff(x) <= 0) or np.any(w <= 0)):
        raise ValueError(f"{name} requires increasing positive nodes and positive integration weights")
    return x, w


def _labels(ell, kappa):
    l = np.asarray(ell)
    k = np.asarray(kappa)
    if (l.ndim != 1 or k.shape != l.shape or not len(l)
            or np.any(l != l.astype(int)) or np.any(k != k.astype(int))
            or np.any(l < 0) or np.any((k != l) & (k != -l-1))
            or np.any(k == 0)):
        raise ValueError("atomic spinors require integer l >= 0 and kappa = l or -(l+1), kappa != 0")
    return l.astype(int), k.astype(int)


def _vectors(values, name):
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a finite (n_point,3) Cartesian array")
    return v


def spinor_spherical_harmonic(kappa, two_mj, directions):
    r"""Condon--Shortley Omega_(kappa,mj), shape (n_point,2).

    ``two_mj`` is an odd integer from -(2|kappa|-1) to +(2|kappa|-1).
    The convention satisfies sigma.rhat Omega_kappa = -Omega_(-kappa).
    At a zero vector the direction is chosen as +z; only l=0 can contribute
    a nonzero regular radial function at the origin.
    """
    from scipy.special import sph_harm_y

    k = int(kappa)
    m2 = int(two_mj)
    if (k != kappa or k == 0 or m2 != two_mj or m2 % 2 != 1
            or abs(m2) > 2*abs(k)-1):
        raise ValueError("invalid kappa or half-integer magnetic quantum number")
    l = -k-1 if k < 0 else k
    v = _vectors(directions, "spinor directions")
    radius = np.linalg.norm(v, axis=1)
    cos_theta = np.divide(v[:, 2], radius, out=np.ones_like(radius), where=radius > 0)
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    phi = np.arctan2(v[:, 1], v[:, 0])
    mj = m2 / 2
    if k < 0:
        coefficients = (np.sqrt((l+mj+0.5)/(2*l+1)),
                        np.sqrt((l-mj+0.5)/(2*l+1)))
    else:
        coefficients = (-np.sqrt((l-mj+0.5)/(2*l+1)),
                        np.sqrt((l+mj+0.5)/(2*l+1)))
    result = np.zeros((len(v), 2), dtype=np.complex128)
    for spin, (m, coefficient) in enumerate(zip(((m2-1)//2, (m2+1)//2), coefficients)):
        if coefficient != 0:
            result[:, spin] = coefficient * sph_harm_y(l, m, theta, phi)
    return result


def spinor_function_labels(ell, kappa):
    """Enumerate flattened (radial OPF, two_mj) without adding radial states."""
    _, k = _labels(ell, kappa)
    return np.asarray([(i, m) for i, ki in enumerate(k)
                       for m in range(-2*abs(ki)+1, 2*abs(ki), 2)], dtype=np.int32)


def radial_fourier_bessel(delta_R, radius, weights_dr, ell, momentum):
    r"""A_l(K) = integral r^2 delta_R_l(r) j_l(Kr) dr, (n_K,n_OPF).

    The 4 pi (-i)^l angular Fourier factor is excluded.  Input radial
    columns have shape (n_r,n_OPF) and may be complex.  Source-grid and
    momentum-grid convergence remain independent obligations.
    """
    from scipy.special import spherical_jn

    r, w = _quadrature(radius, weights_dr, "radial quadrature")
    l = np.asarray(ell)
    values = np.asarray(delta_R, dtype=np.complex128)
    K = np.asarray(momentum, dtype=np.float64)
    if (l.ndim != 1 or np.any(l < 0) or np.any(l != l.astype(int))
            or values.shape != (len(r), len(l)) or not np.all(np.isfinite(values))
            or K.ndim != 1 or np.any(K < 0) or not np.all(np.isfinite(K))):
        raise ValueError("invalid radial correction, angular labels or momentum nodes")
    result = np.empty((len(K), len(l)), dtype=np.complex128)
    for degree in np.unique(l.astype(int)):
        columns = np.flatnonzero(l == degree)
        result[:, columns] = spherical_jn(degree, K[:, None]*r) @ (
            values[:, columns] * (w*r*r)[:, None])
    return result


def _lift_cartesian(pauli, K_cart):
    """Use the one carrier owner, treating Cartesian K as G with b=I,k=0."""
    from common.bispinor_init import NORMALIZED_RKB_LIFT, lift_to_4spinor

    lifted = np.asarray(lift_to_4spinor(
        pauli[None], K_cart[None], np.zeros((1, 3)), np.eye(3),
        representation=NORMALIZED_RKB_LIFT))[0]
    if lifted.dtype != np.complex128:
        raise RuntimeError("atomic normalized RKB tables require the runtime's JAX x64 initialization")
    return lifted


def _pauli_fourier_amplitudes(radial, ell, kappa, vectors, center_cart):
    """One angular/translation convention for direct and cached transforms."""
    l, k = _labels(ell, kappa)
    center = np.asarray(center_cart, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("atomic center must be a finite Cartesian three-vector")
    if radial.shape != (len(vectors), len(l)) or not np.all(np.isfinite(radial)):
        raise ValueError("atomic Fourier amplitudes disagree with their channel labels")
    labels = spinor_function_labels(l, k)
    pauli = np.empty((len(labels), 2, len(vectors)), dtype=np.complex128)
    phase = np.exp(-1j*(vectors @ center))
    angular = {}
    for row, (i, m2) in enumerate(labels):
        key = (int(k[i]), int(m2))
        if key not in angular:
            angular[key] = spinor_spherical_harmonic(*key, vectors).T
        pauli[row] = 4*np.pi*(-1j)**l[i] * angular[key] * radial[:, i] * phase
    return pauli


def atomic_pauli_fourier(radial_R, radius, weights_dr, ell, kappa,
                         K_cart, *, center_cart=(0, 0, 0)):
    r"""Continuous Fourier transform of R_l(r) Omega_kappa, (function,2,K).

    Uses F(K)=integral exp(-i K.r) f(r) d^3r.  There is no cell-volume
    normalization; a periodic plane-wave caller supplies its own convention.
    The same angular transform serves raw dual projector functions and
    the atomic differences later passed through the normalized RKB lift.
    """
    l, k = _labels(ell, kappa)
    vectors = _vectors(K_cart, "Fourier momentum")
    radial = radial_fourier_bessel(radial_R, radius, weights_dr, l,
                                  np.linalg.norm(vectors, axis=1))
    return _pauli_fourier_amplitudes(radial, l, k, vectors, center_cart)


def build_pauli_fourier_cache(radial_R, radius, weights_dr, ell, kappa, *,
                              momentum_max, momentum_points,
                              relative_tolerance=1e-10, absolute_tolerance=1e-12,
                              validation_points=64):
    """Cache species Fourier amplitudes and check independent midpoint values.

    Amplitudes contain no cell normalization, dual inversion or RKB factor.
    The caller owns source identity; the same cache serves duals and raw
    AE-minus-PS functions. This routine refuses interpolation errors larger
    than the declared mixed absolute/relative tolerance.
    """
    from scipy.interpolate import CubicSpline

    l, k = _labels(ell, kappa)
    maximum, count = float(momentum_max), int(momentum_points)
    if (not np.isfinite(maximum) or maximum <= 0 or count != momentum_points
            or count < 8 or int(validation_points) != validation_points
            or int(validation_points) < 4 or not np.isfinite(relative_tolerance)
            or not np.isfinite(absolute_tolerance)
            or relative_tolerance < 0 or absolute_tolerance < 0):
        raise ValueError("invalid atomic Pauli Fourier cache controls")
    momentum = np.linspace(0., maximum, count)
    radial = radial_fourier_bessel(radial_R, radius, weights_dr, l, momentum)
    spline = CubicSpline(momentum, radial, axis=0, extrapolate=False)
    cells = np.unique(np.linspace(0, count-2, int(validation_points)).astype(int))
    query = (momentum[cells]+momentum[cells+1])/2
    direct = radial_fourier_bessel(radial_R, radius, weights_dr, l, query)
    error = np.max(np.abs(spline(query)-direct), axis=0)
    scale = np.max(np.abs(radial), axis=0)
    limit = float(absolute_tolerance)+float(relative_tolerance)*scale
    if np.any(error > limit):
        raise ValueError(f"atomic Pauli Fourier interpolation exceeds tolerance: {error.max():.3e}")
    return dict(momentum=momentum, radial=radial, ell=l, kappa=k,
                maximum_absolute_error=float(error.max()),
                maximum_scaled_error=float(np.max(error/np.maximum(scale, np.finfo(float).tiny))),
                validation_points=len(query))


def evaluate_pauli_fourier_cache(cache, K_cart, *, center_cart=(0, 0, 0)):
    """Evaluate continuous cached Pauli transforms, (function,2,K).

    No momentum extrapolation is permitted, including for translated atoms.
    Angular functions are shared by every OPF with the same kappa and mj.
    """
    from scipy.interpolate import CubicSpline

    vectors = _vectors(K_cart, "Fourier momentum")
    l, k = _labels(cache['ell'], cache['kappa'])
    grid = np.asarray(cache['momentum'], dtype=np.float64)
    values = np.asarray(cache['radial'])
    if (grid.ndim != 1 or len(grid) < 8 or grid[0] != 0
            or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0)
            or values.shape != (len(grid), len(l)) or not np.all(np.isfinite(values))):
        raise ValueError("invalid atomic Pauli Fourier cache arrays")
    magnitude = np.linalg.norm(vectors, axis=1)
    if np.any(magnitude > grid[-1]):
        raise ValueError("atomic Fourier momentum exceeds the cached range")
    radial = CubicSpline(grid, values, axis=0, extrapolate=False)(magnitude)
    return _pauli_fourier_amplitudes(radial, l, k, vectors, center_cart)


def normalized_delta_fourier(delta_R, radius, weights_dr, ell, kappa,
                             K_cart, *, center_cart=(0, 0, 0)):
    r"""Continuous Fourier transform of U delta_phi, (function,4,K).

    Translation commutes with U.  This is the raw atomic angular transform
    followed by the existing normalized carrier; no new decoupling model.
    """
    vectors = _vectors(K_cart, "Fourier momentum")
    pauli = atomic_pauli_fourier(delta_R, radius, weights_dr, ell, kappa,
                                vectors, center_cart=center_cart)
    return _lift_cartesian(pauli, vectors)


def build_normalized_radial_cache(delta_R, radius, weights_dr, ell, kappa,
                                  momentum, weights_dK, evaluation_radius):
    r"""Hankel-transform R delta_phi and X R delta_phi into a radial cache.

    All radial blocks have shape (n_evaluation_radius,n_OPF).  The input
    momenta and dK weights are supplied explicitly; no accuracy is inferred
    from a heuristic cutoff.  The lower angular channel has kappa -> -kappa
    and ell_small=2|kappa|-1-ell.  Radial derivatives are analytic derivatives
    of spherical Bessel functions, never differences of reconstructed data.
    """
    from scipy.special import spherical_jn

    l, k = _labels(ell, kappa)
    K, wk = _quadrature(momentum, weights_dK, "momentum quadrature")
    x = np.asarray(evaluation_radius, dtype=np.float64)
    if (x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x))
            or np.any(x < 0) or np.any(np.diff(x) <= 0)):
        raise ValueError("cache radii must be finite, increasing and nonnegative")
    radial = radial_fourier_bessel(delta_R, radius, weights_dr, l, K)
    # A spin-up radial amplitude on +z obtains r(K) A and h K r(K) A
    # from the existing lift without duplicating its normalization formula.
    pauli = np.zeros((len(l), 2, len(K)), dtype=np.complex128)
    pauli[:, 0] = radial.T
    lifted = _lift_cartesian(pauli, np.column_stack((0*K, 0*K, K)))
    large_spectrum = lifted[:, 0].T
    small_spectrum = lifted[:, 2].T
    upper = np.empty((len(x), len(l)), dtype=np.complex128)
    dupper = np.empty_like(upper)
    lower = np.empty_like(upper)
    dlower = np.empty_like(upper)
    kr = x[:, None]*K
    spectral_weights = ((2/np.pi)*wk*K*K)[:, None]
    for ki in np.unique(k):
        columns = np.flatnonzero(k == ki)
        li = int(l[columns[0]])
        lb = 2*abs(ki)-1-li
        a = spectral_weights * large_spectrum[:, columns]
        b = -1j**(lb-li) * spectral_weights * small_spectrum[:, columns]
        upper[:, columns] = spherical_jn(li, kr) @ a
        dupper[:, columns] = spherical_jn(li, kr, derivative=True) @ (K[:, None]*a)
        lower[:, columns] = spherical_jn(lb, kr) @ b
        dlower[:, columns] = spherical_jn(lb, kr, derivative=True) @ (K[:, None]*b)
    return dict(radius=x, ell=l, kappa=k, large_R=upper, dlarge_R_dr=dupper,
                small_R=lower, dsmall_R_dr=dlower)


def evaluate_normalized_delta(cache: dict, relative_cart):
    """Evaluate a cached U delta_phi at atom-relative points, (function,4,point).

    Cubic Hermite interpolation uses the exported spectral derivatives.
    There is no implicit atomic cutoff and no assumed norm conservation.
    """
    from scipy.interpolate import CubicHermiteSpline

    vectors = _vectors(relative_cart, "atomic sample coordinates")
    r = np.linalg.norm(vectors, axis=1)
    if np.any(r < cache['radius'][0]) or np.any(r > cache['radius'][-1]):
        raise ValueError("atomic samples lie outside the normalized radial cache; extend and converge its tail")
    large = CubicHermiteSpline(cache['radius'], cache['large_R'],
                              cache['dlarge_R_dr'], axis=0)(r)
    small = CubicHermiteSpline(cache['radius'], cache['small_R'],
                              cache['dsmall_R_dr'], axis=0)(r)
    labels = spinor_function_labels(cache['ell'], cache['kappa'])
    result = np.empty((len(labels), 4, len(vectors)), dtype=np.complex128)
    for row, (i, m2) in enumerate(labels):
        result[row, :2] = spinor_spherical_harmonic(cache['kappa'][i], m2, vectors).T * large[:, i]
        result[row, 2:] = spinor_spherical_harmonic(-cache['kappa'][i], m2, vectors).T * small[:, i]
    return result


def reconstruction_overlap(smooth_samples, corrected_samples, volume_weights):
    """Return local correction to the band Gram; never orthonormalize it.

    Samples have shape (...,band,spin,point); weights integrate d^3r.
    Sum atom-local blocks and add the known smooth band Gram to obtain S.
    Cross terms with the smooth residual are included explicitly.
    """
    smooth = np.asarray(smooth_samples)
    corrected = np.asarray(corrected_samples)
    weights = np.asarray(volume_weights, dtype=np.float64)
    if (smooth.ndim < 3 or corrected.shape != smooth.shape
            or weights.shape != smooth.shape[-1:] or np.any(weights <= 0)
            or not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(smooth)) or not np.all(np.isfinite(corrected))):
        raise ValueError("overlap diagnostic requires paired finite samples and positive volume weights")
    # Expand in the correction to avoid subtracting two large, nearly equal
    # Gram matrices when the reconstruction is weak.
    delta = corrected - smooth
    return (np.einsum('...bsp,...csp,p->...bc', smooth.conj(), delta, weights)
            + np.einsum('...bsp,...csp,p->...bc', delta.conj(), smooth, weights)
            + np.einsum('...bsp,...csp,p->...bc', delta.conj(), delta, weights))
