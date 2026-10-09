"""Atom-local charge fitting and compensated Coulomb integration.

The orbital samples defining C are common to the smooth and local fits.
Local densities are expanded in complex, orthonormal spherical harmonics.
All radial arrays use Hartree atomic units.  See
``docs/theory/augmented-isdf.md`` for the mixed-representation identity and
the smooth--neutral-residual term that must survive compensation.

Atomic radial tables are small, reusable host data.  The Coulomb contractions
below operate on rank-local tiles inside the caller's shard_map; they neither
gather a wavefunction nor construct a band-pair tensor.
"""

from __future__ import annotations

import numpy as np


def _radial_grid(radius, weights):
    """Authenticate a positive, ordered radial quadrature (weights integrate dr)."""
    r = np.asarray(radius, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (r.ndim != 1 or r.size < 2 or w.shape != r.shape
            or not np.all(np.isfinite(r)) or not np.all(np.isfinite(w))
            or np.any(r <= 0) or np.any(np.diff(r) <= 0) or np.any(w <= 0)):
        raise ValueError(
            "atomic radial quadrature requires finite, strictly increasing "
            "positive radii and positive dr weights with the same shape")
    return r, w


def radial_coulomb_potential(density, radius, weights, ell):
    r"""Apply the spherical-harmonic Poisson kernel in O(N_r).

    Parameters
    ----------
    density : (..., N_r) complex array
        Coefficient rho_lm(r) of an orthonormal complex Y_lm.
    radius, weights : (N_r,) float arrays
        Radius in bohr and quadrature weights integrating dr, not r^2 dr.
        The points need not be uniform.  Converge this quadrature separately
        from the density fit.
    ell : int
        Nonnegative spherical-harmonic degree.

    Returns
    -------
    potential : (..., N_r) complex array
        4 pi/(2l+1) [r^(-l-1) int_0^r rho(t)t^(l+2)dt
        + r^l int_r^infinity rho(t)t^(1-l)dt], in Hartree/electron.
        Each diagonal quadrature shell is counted exactly once.  Consequently
        the discrete bilinear form is Hermitian, without symmetrization.
    """
    r, w = _radial_grid(radius, weights)
    rho = np.asarray(density)
    l = int(ell)
    if l != ell or l < 0 or rho.shape[-1:] != r.shape or not np.all(np.isfinite(rho)):
        raise ValueError("radial Coulomb density/degree does not match its grid")
    inside = rho * (w * r ** (l + 2))
    outside = rho * (w * r ** (1 - l))
    a = np.cumsum(inside, axis=-1) - 0.5 * inside
    b = np.cumsum(outside[..., ::-1], axis=-1)[..., ::-1] - 0.5 * outside
    return (4.0 * np.pi / (2 * l + 1)) * (
        a / r ** (l + 1) + b * r ** l)


def radial_coulomb_matrix(density_basis, radius, weights, lm_ell):
    r"""Return K_ij = (rho_i | 1/r | rho_j) for one atom's local basis.

    ``density_basis`` is (N_basis, N_lm, N_r), ``lm_ell`` is (N_lm,).
    Harmonics are orthonormal and consistently ordered by the caller.  The
    returned (N_basis, N_basis) complex128 matrix has Hartree units when the
    basis densities have bohr^-3 units.  This is an atomic precomputation:
    O(N_basis^2 N_lm N_r) work, O(N_basis N_lm N_r) storage, no N_r^2 kernel.
    """
    r, w = _radial_grid(radius, weights)
    rho = np.asarray(density_basis, dtype=np.complex128)
    degrees = np.asarray(lm_ell)
    if (rho.ndim != 3 or rho.shape[1:] != (degrees.size, r.size)
            or degrees.ndim != 1 or not np.all(np.isfinite(rho))):
        raise ValueError("local density basis must have shape (basis,lm,radius)")
    result = np.zeros((rho.shape[0], rho.shape[0]), dtype=np.complex128)
    for j, l in enumerate(degrees):
        potential = radial_coulomb_potential(rho[:, j], r, w, l)
        result += (rho[:, j].conj() * (w * r * r)) @ potential.T
    return result


def compensation_basis(density_basis, radius, weights, lm_ell,
                       *, support_radius, power=6):
    r"""Match every retained external charge multipole with smooth compact g.

    Returns ``(g, moments)`` with the same density-basis shape and
    ``moments (N_basis,N_lm)``.  Each g_lm is proportional to
    r^l [1-(r/R)^2]^power for r<R, normalized to int r^(l+2)g_lm dr.
    ``density_basis-g`` has zero external Poisson field at the represented
    degrees.  Unrepresented degrees and a truncated Fourier expansion of g
    each require independent convergence checks.  A charge compensation is
    not a transverse-current compensation.
    """
    r, w = _radial_grid(radius, weights)
    rho = np.asarray(density_basis, dtype=np.complex128)
    degrees = np.asarray(lm_ell)
    R = float(support_radius)
    p = int(power)
    if (rho.ndim != 3 or rho.shape[1:] != (degrees.size, r.size)
            or degrees.ndim != 1 or np.any(degrees < 0)
            or np.any(degrees != degrees.astype(int))
            or not np.all(np.isfinite(rho)) or not np.isfinite(R)
            or R < r[-1] or p != power or p < 2):
        raise ValueError("invalid local compensation basis, radius or degree")
    g = np.zeros_like(rho)
    moments = np.empty(rho.shape[:2], dtype=np.complex128)
    envelope = np.maximum(1.0 - (r / R) ** 2, 0.0) ** p
    for j, l in enumerate(degrees.astype(int)):
        wr = w * r ** (l + 2)
        moment = rho[:, j] @ wr
        shape = r ** l * envelope
        norm = shape @ wr
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("compensation radial shape has no resolved moment")
        g[:, j] = moment[:, None] * (shape / norm)[None, :]
        moments[:, j] = moment
    return g, moments


def local_basis_fourier(density_basis, radius, weights, lm,
                       wavevectors_cart, *, center_cart):
    r"""Fourier--Bessel integral of an atom-local charge basis.

    ``lm (N_lm,2)`` lists (l,m); wavevectors (N_G,3) are q+G in bohr^-1;
    center (3,) is Cartesian bohr.  Returns (N_basis,N_G) with convention
    int rho(r) exp(-i K.r) d^3r.  No FFT, cell-volume, or Coulomb prefactor
    is inserted here.  The caller converts this integral to its smooth FFT
    convention exactly once.  At K=0 only l=0 survives.
    """
    from scipy.special import spherical_jn, sph_harm_y

    r, w = _radial_grid(radius, weights)
    rho = np.asarray(density_basis, dtype=np.complex128)
    harmonics = np.asarray(lm)
    k = np.asarray(wavevectors_cart, dtype=np.float64)
    center = np.asarray(center_cart, dtype=np.float64)
    if (harmonics.ndim != 2 or harmonics.shape[1] != 2
            or rho.ndim != 3 or rho.shape[1:] != (len(harmonics), len(r))
            or k.ndim != 2 or k.shape[1] != 3 or center.shape != (3,)
            or not np.all(np.isfinite(k)) or not np.all(np.isfinite(center))
            or np.any(harmonics != harmonics.astype(int))):
        raise ValueError("invalid local Fourier basis or Cartesian geometry")
    norm = np.linalg.norm(k, axis=1)
    unit_z = np.divide(k[:, 2], norm, out=np.ones_like(norm), where=norm > 0)
    theta = np.arccos(np.clip(unit_z, -1.0, 1.0))
    phi = np.arctan2(k[:, 1], k[:, 0])
    result = np.zeros((rho.shape[0], len(k)), dtype=np.complex128)
    for j, (l, m) in enumerate(harmonics.astype(int)):
        if l < 0 or abs(m) > l:
            raise ValueError("spherical-harmonic index requires l>=0 and |m|<=l")
        bessel = spherical_jn(l, norm[:, None] * r[None, :])
        radial = (rho[:, j] * (w * r * r)) @ bessel.T
        result += 4.0 * np.pi * (-1j) ** l * radial * sph_harm_y(l, m, theta, phi)
    return result * np.exp(-1j * (k @ center))[None, :]


def mixed_coulomb_tile(smooth, delta, compensation, coulomb):
    r"""One rank-local Fourier tile of the complete mixed charge metric.

    Inputs smooth/delta/compensation are (q,mu,G_tile), already fitted with
    the same C factor and expressed in the same Fourier convention;
    coulomb is (q,G_tile), including the consuming code's units and head.
    Returns (q,mu,nu).  The formula is s*vs+s*vDelta+Delta*vs+g*vg.
    Add the on-site (Delta|v|Delta)-(g|v|g) matrices afterwards.

    This equals PW(s+g) plus PW(s|v|Delta-g)+h.c.; retaining the latter
    is essential even when Delta-g has exactly zero external multipoles.
    No band-pair tensor or C^+ M C^+ intermediate is formed.
    """
    import jax.numpy as jnp

    s, d, g, v = map(jnp.asarray, (smooth, delta, compensation, coulomb))
    if s.ndim != 3 or d.shape != s.shape or g.shape != s.shape or v.shape != (s.shape[0], s.shape[2]):
        raise ValueError("mixed Coulomb inputs must share (q,mu,G_tile)")
    product = lambda a, b: jnp.einsum('qmg,qg,qng->qmn', jnp.conj(a), v, b)
    return product(s, s) + product(s, d) + product(d, s) + product(g, g)


def onsite_coulomb_tile(coefficients, delta_metric, compensation_metric):
    r"""One atom's rank-local (Delta|v|Delta)-(g|v|g) correction.

    coefficients is (q,mu,N_basis), the atom's local zeta coefficients;
    delta_metric/compensation_metric are (N_basis,N_basis), in the same
    units as the Fourier metric.  Returns (q,mu,nu).  Stream atoms through
    a scan in the caller to bound storage.  This difference need not be
    positive semidefinite; positivity belongs to the complete Coulomb tensor.
    """
    import jax.numpy as jnp

    a, kd, kg = map(jnp.asarray, (coefficients, delta_metric, compensation_metric))
    if a.ndim != 3 or kd.shape != (a.shape[2], a.shape[2]) or kg.shape != kd.shape:
        raise ValueError("on-site Coulomb coefficients/metric shape mismatch")
    return jnp.einsum('qmi,ij,qnj->qmn', jnp.conj(a), kd - kg, a)


def radial_coulomb_metric_interpolated(
        radius, lm_ell, *, support_radius, interpolation_degree=3,
        quadrature_order=None):
    r"""Positive Coulomb metric of a piecewise-polynomial physical density.

    This experimental species precomputation leaves the incumbent radial
    collocation metric unchanged.  Samples are values of ``rho_lm(r)``, not
    quadrature delta shells.  Between samples (and to the explicit support
    boundary) a local cardinal polynomial of the requested degree represents
    the density.  The first interval ``[0, r[0]]`` uses the regular extension
    ``rho_lm(r) = rho_lm(r[0]) (r/r[0])**l``.  Its convergence remains an
    independent obligation for an all-electron density with a weak cusp.

    The inner multipole ``M_l(r)=int_0^r t**(l+2) rho_lm(t) dt`` is integrated
    analytically on every polynomial interval and is continuous across its
    boundaries.  The field-energy identity is

    ``K_l = 4*pi int_0^R M_i*(r) M_j(r) r**(-2*l-2) dr``
    ``      + 4*pi/(2*l+1) M_i*(R) M_j(R) R**(-2*l-1)``.

    Positive Gauss weights make this field Gram and its exterior-boundary
    term symmetric positive semidefinite by construction.  Gauss order,
    interpolation degree, sample grid and support are separate convergence
    controls; no physical accuracy is inferred from their default values.

    Parameters
    ----------
    radius : (N_r,) float64
        Strictly increasing positive density-sample radii, in bohr.
    lm_ell : (N_lm,) integer
        Nonnegative degrees. The returned degree axis is their sorted unique
        set, as in the incumbent atomic radial metric.
    support_radius : float
        Compact-density boundary R >= radius[-1], in bohr. The interpolant
        can be nonzero at R; the exterior multipole field is retained exactly.
    interpolation_degree : int
        Local polynomial degree, >= 1 and < N_r. Defaults to cubic.
    quadrature_order : int or None
        Positive Gauss nodes per interval. None resolves from the known
        degree bounds to max(16, max(lm_ell)+interpolation_degree+4).

    Returns
    -------
    dict
        ``metric (N_degree,N_r,N_r)`` is the ordinary 1/r bilinear form in
        Hartree atomic units; no FFT/cell or Rydberg scale is inserted.
        ``moments (N_degree,N_r)`` are exact linear multipole rows for this
        same interpolant. ``quadrature_radius`` and ``quadrature_weights_dr``
        are positive physical integration nodes/weights, shape (N_quad,).
        ``interpolation_map (N_quad,N_r)`` evaluates all nonorigin intervals
        and the l=0 origin interval. For degree-row d, replace its first
        ``origin_row_count`` entries in column zero by
        ``origin_factors[d]``; all other map entries remain unchanged.
        This compact first-column update keeps Fourier integration and
        compensation on precisely the same density without replicating the
        large interpolation map for each degree. Metadata also records the
        resolved polynomial and Gauss orders.

    Notes
    -----
    Atomic host precomputation only: map/multipole storage is O(N_quad*N_r),
    metrics O(N_degree*N_r**2), Gram work O(N_degree*N_quad*N_r**2). It is
    intended to reduce the fitted sample count; large grids still require an
    explicit capacity price. Fourier transforms must use the returned map,
    origin factors and quadrature rather than the old shell weights.
    """
    from math import comb

    r = np.asarray(radius, dtype=np.float64)
    ell = np.asarray(lm_ell)
    R = float(support_radius)
    degree = int(interpolation_degree)
    if (r.ndim != 1 or len(r) < 2 or not np.all(np.isfinite(r))
            or np.any(r <= 0) or np.any(np.diff(r) <= 0)
            or ell.ndim != 1 or len(ell) == 0 or not np.all(np.isfinite(ell))
            or np.any(ell < 0) or np.any(ell != np.round(ell))
            or not np.isfinite(R) or R < r[-1]
            or degree != interpolation_degree or degree < 1 or degree >= len(r)):
        raise ValueError("invalid physical radial interpolant grid, support or degree")
    degrees = np.unique(ell.astype(np.int64))
    order = (max(16, int(degrees[-1])+degree+4) if quadrature_order is None
             else int(quadrature_order))
    if (order < 2 or (quadrature_order is not None and order != quadrature_order)):
        raise ValueError("radial field quadrature order must be an integer >= 2")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u, wu = (nodes+1)/2, weights/2
    edges = np.concatenate(([0.], r, ([R] if R > r[-1] else [])))
    nquad, nr = (len(edges)-1)*order, len(r)
    q_radius = np.empty(nquad)
    q_weights = np.empty(nquad)
    interpolation = np.zeros((nquad, nr))
    intervals = []
    for panel, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        h = b-a
        first, last = panel*order, (panel+1)*order
        q_radius[first:last] = a+h*u
        q_weights[first:last] = h*wu
        if panel == 0:
            interpolation[first:last, 0] = 1.
            continue
        # A fixed-width local stencil centered on this interval.  End
        # intervals extrapolate only to the explicitly supplied 0/R domain.
        left = int(np.searchsorted(r, (a+b)/2)) - (degree+1)//2
        left = max(0, min(left, nr-degree-1))
        columns = np.arange(left, left+degree+1)
        z = (r[columns]-a)/h
        basis = np.empty((degree+1, degree+1))
        for i in range(degree+1):
            other = np.delete(z, i)
            basis[:, i] = np.polynomial.polynomial.polyfromroots(other)/np.prod(z[i]-other)
        interpolation[first:last, columns] = np.polynomial.polynomial.polyval(u, basis).T
        intervals.append((first, last, a, h, columns, basis))

    metrics, moments, origin_factors = [], [], []
    for l in degrees:
        l = int(l)
        values = np.zeros((nquad, nr))
        # Integrate the regular first interval analytically.  Its moment
        # is finite without extrapolating high-l sample noise as rho/r**l.
        values[:order, 0] = q_radius[:order]**(2*l+3)/(r[0]**l*(2*l+3))
        prefix = np.zeros(nr)
        prefix[0] = r[0]**(l+3)/(2*l+3)
        for first, last, a, h, columns, basis in intervals:
            radial_power = np.asarray([
                comb(l+2, k)*a**(l+2-k)*h**k for k in range(l+3)])
            antiderivative = np.zeros((l+degree+4, degree+1))
            for i in range(degree+1):
                polynomial = np.polynomial.polynomial.polymul(radial_power, basis[:, i])
                antiderivative[1:len(polynomial)+1, i] = h*polynomial/np.arange(1,len(polynomial)+1)
            inside = np.polynomial.polynomial.polyval(u, antiderivative).T
            values[first:last] = prefix
            values[first:last, columns] += inside
            prefix[columns] += np.polynomial.polynomial.polyval(1., antiderivative)
        flux = values * (np.sqrt(4*np.pi*q_weights)/q_radius**(l+1))[:, None]
        metric = flux.T @ flux
        metric += (4*np.pi/(2*l+1))*np.outer(prefix,prefix)/R**(2*l+1)
        metrics.append(metric)
        moments.append(prefix.copy())
        origin_factors.append((q_radius[:order]/r[0])**l)
    if any(not np.all(np.isfinite(values)) for values in (
            metrics, moments, origin_factors, interpolation, q_radius, q_weights)):
        raise ValueError("radial field metric is unresolved in float64; revise the grid or angular degree")
    return dict(degrees=degrees, metric=np.asarray(metrics), moments=np.asarray(moments),
                quadrature_radius=q_radius, quadrature_weights_dr=q_weights,
                interpolation_map=interpolation, origin_row_count=order,
                origin_factors=np.asarray(origin_factors),
                interpolation_degree=degree, quadrature_order=order)
