"""
solvers/quadrature.py — Contour integration quadrature for spectral methods.

Provides quadrature nodes and weights for FEAST-style contour eigensolvers.

Usage
-----
    from solvers.quadrature import feast_ellipse_quadrature

    z_nodes, w_weights = feast_ellipse_quadrature(center, half_width, n_quad, gamma=0.2)
"""
from __future__ import annotations

import numpy as np


def feast_ellipse_quadrature(
    center: float,
    half_width: float,
    n_quad: int,
    gamma: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Ellipse trapezoid-rule quadrature for FEAST contour integration.

    Generates nodes on the upper half of an ellipse in the complex plane
    centered on [center - half_width, center + half_width].  For Hermitian
    problems the lower half is obtained by conjugate symmetry, so only
    upper-half nodes are returned.

    The contour integral approximates the spectral projector:
        P = (1 / 2pi i) oint (z I - H)^{-1} dz

    Parameters
    ----------
    center : float
        Center of the target eigenvalue interval.
    half_width : float
        Half-width of the target interval (r_x).
    n_quad : int
        Number of quadrature nodes on the upper half-ellipse.
    gamma : float
        Aspect ratio r_y / r_x of the ellipse.  Smaller gamma gives a
        thinner ellipse (fewer GMRES iterations but worse filter).

    Returns
    -------
    z_nodes : ndarray, shape (n_quad,), complex128
        Quadrature nodes in the upper half-plane.
    w_weights : ndarray, shape (n_quad,), complex128
        Quadrature weights (include the 1/(2pi i) factor and dtheta).
    """
    r_x = half_width
    r_y = gamma * r_x

    # Midpoint quadrature: theta_j = pi(2j-1)/(2N), j = 1..N
    j = np.arange(1, n_quad + 1)
    thetas = np.pi * (2 * j - 1) / (2 * n_quad)

    z = center + r_x * np.cos(thetas) + 1j * r_y * np.sin(thetas)
    w = (1.0 / (2.0 * n_quad)) * (r_y * np.cos(thetas) + 1j * r_x * np.sin(thetas))
    return z.astype(np.complex128), w.astype(np.complex128)
