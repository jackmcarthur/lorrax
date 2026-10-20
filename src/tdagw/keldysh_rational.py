"""Dense reference algebra for snapshot Keldysh screening transfer functions.

This module is a host-side validation owner, not a production screening path.
It represents a matrix rational function as the plain array tuple
``(A, B, C, D)`` with

``H(z) = D + C @ solve(z I - A, B)``.

The representation is deliberately general: poles need not be real, residues
need not be positive, and a lesser/noise realization may contain poles absent
from the retarded response.  Those properties are required by the proposed
full-matrix nonequilibrium closure in ``docs/theory/td_agw_dynamical.md`` and
are outside the contracts of :mod:`gw.shared_pole_constructor`.

The routines below provide the small-matrix oracle for three operations that
must be certified before a distributed implementation is admitted:

* ``sharp(H)(z) = H(conj(z)).H``;
* exact series composition of matrix transfer functions; and
* causal projection by ordered Schur decomposition plus a Sylvester solve.

The latter handles repeated poles without a residue derivative formula.  It
requires every pole to lie strictly off the real axis.  A lossless pole must be
given a declared adiabatic regulator before calling this reference; silently
letting roundoff choose its causal side is refused.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def _arrays(system):
    """Validate and return one ``(A, B, C, D)`` transfer realization."""
    if not isinstance(system, tuple) or len(system) != 4:
        raise TypeError("a rational realization must be the tuple (A, B, C, D)")
    A, B, C, D = (np.asarray(x, dtype=np.complex128) for x in system)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    order = A.shape[0]
    if (B.ndim != 2 or C.ndim != 2 or D.ndim != 2
            or B.shape[0] != order or C.shape[1] != order
            or D.shape != (C.shape[0], B.shape[1])):
        raise ValueError("incompatible state-space shapes")
    if not all(np.isfinite(x).all() for x in (A, B, C, D)):
        raise ValueError("rational realization contains a non-finite value")
    return A, B, C, D


def pole_residue_realization(poles, residues, constant=None):
    """Return a dense realization of a simple-pole matrix expansion.

    Parameters
    ----------
    poles : array_like, shape (n_pole,)
        Complex pole locations.
    residues : array_like, shape (n_pole, n_out, n_in)
        Matrix residue at each pole.  The realization is intentionally not
        rank-compressed: each pole receives ``n_in`` states, which keeps this
        routine a transparent small-matrix reference.
    constant : array_like, shape (n_out, n_in), optional
        Direct term.  The default is zero.

    Returns
    -------
    tuple of ndarray
        ``(A, B, C, D)`` such that
        ``H(z) = D + sum_j residues[j] / (z - poles[j])``.
    """
    poles = np.asarray(poles, dtype=np.complex128)
    residues = np.asarray(residues, dtype=np.complex128)
    if poles.ndim != 1 or residues.ndim != 3 or residues.shape[0] != poles.size:
        raise ValueError("poles/residues must have shapes (p,) and (p,n_out,n_in)")
    if not np.isfinite(poles).all() or not np.isfinite(residues).all():
        raise ValueError("poles and residues must be finite")
    n_out, n_in = residues.shape[1:]
    D = (np.zeros((n_out, n_in), dtype=np.complex128) if constant is None
         else np.asarray(constant, dtype=np.complex128))
    if D.shape != (n_out, n_in) or not np.isfinite(D).all():
        raise ValueError("constant has the wrong shape or a non-finite value")
    if poles.size == 0:
        return (np.zeros((0, 0), dtype=np.complex128),
                np.zeros((0, n_in), dtype=np.complex128),
                np.zeros((n_out, 0), dtype=np.complex128), D.copy())
    A = np.kron(np.diag(poles), np.eye(n_in, dtype=np.complex128))
    B = np.tile(np.eye(n_in, dtype=np.complex128), (poles.size, 1))
    C = np.concatenate(tuple(residues), axis=1)
    return A, B, C, D.copy()


def evaluate(system, z):
    """Evaluate a realization at scalar complex frequency ``z``."""
    A, B, C, D = _arrays(system)
    if A.shape[0] == 0:
        return D.copy()
    return D + C @ np.linalg.solve(complex(z) * np.eye(A.shape[0]) - A, B)


def sharp(system):
    """Return the realization ``H^sharp(z) = H(conj(z)).H``."""
    A, B, C, D = _arrays(system)
    return A.conj().T, C.conj().T, B.conj().T, D.conj().T


def scale(system, factor):
    """Multiply a transfer function by a scalar on its output."""
    A, B, C, D = _arrays(system)
    factor = complex(factor)
    return A.copy(), B.copy(), factor * C, factor * D


def add(left, right):
    """Return the parallel sum of two equally shaped transfer functions."""
    A1, B1, C1, D1 = _arrays(left)
    A2, B2, C2, D2 = _arrays(right)
    if D1.shape != D2.shape:
        raise ValueError("parallel transfer functions must have equal input/output shapes")
    A = sla.block_diag(A1, A2).astype(np.complex128, copy=False)
    B = np.concatenate((B1, B2), axis=0)
    C = np.concatenate((C1, C2), axis=1)
    return A, B, C, D1 + D2


def multiply(left, right):
    """Return the ordered matrix product ``left(z) @ right(z)``.

    The block realization preserves coincident poles as Jordan structure, so
    repeated poles produced by the product are not discarded or split.
    """
    A1, B1, C1, D1 = _arrays(left)
    A2, B2, C2, D2 = _arrays(right)
    if D1.shape[1] != D2.shape[0]:
        raise ValueError("series transfer functions have incompatible inner dimensions")
    n1, n2 = A1.shape[0], A2.shape[0]
    A = np.zeros((n1 + n2, n1 + n2), dtype=np.complex128)
    A[:n1, :n1] = A1
    A[n1:, n1:] = A2
    A[:n1, n1:] = B1 @ C2
    B = np.concatenate((B1 @ D2, B2), axis=0)
    C = np.concatenate((C1, D1 @ C2), axis=1)
    return A, B, C, D1 @ D2


def causal_part(system, *, pole_tolerance=1e-12, direct_tolerance=1e-12):
    """Project a strictly proper rational function onto lower-half-plane poles.

    An ordered complex Schur decomposition puts the lower-half-plane invariant
    subspace first.  Its coupling to the upper-half-plane block is removed by
    a Sylvester solve before the causal block is returned.  This is an exact
    rational projection up to dense linear-algebra roundoff and works when a
    product has repeated poles.

    Parameters
    ----------
    system : tuple of ndarray
        Dense realization ``(A, B, C, D)``.
    pole_tolerance : float
        Poles with ``abs(Im pole) <= pole_tolerance`` are refused.  The caller
        must resolve the statistical/adiabatic limit explicitly.
    direct_tolerance : float
        Maximum accepted direct term.  A nonzero constant has no unique
        strictly causal/anticausal allocation in this convention.

    Returns
    -------
    tuple of ndarray
        A strictly proper realization containing only lower-half-plane poles.
    """
    A, B, C, D = _arrays(system)
    pole_tolerance = float(pole_tolerance)
    direct_tolerance = float(direct_tolerance)
    if pole_tolerance < 0.0 or direct_tolerance < 0.0:
        raise ValueError("projection tolerances must be nonnegative")
    if np.max(np.abs(D), initial=0.0) > direct_tolerance:
        raise ValueError("causal projection requires a strictly proper transfer function")
    if A.shape[0] == 0:
        return (A.copy(), B.copy(), C.copy(), np.zeros_like(D))
    poles = np.linalg.eigvals(A)
    if np.any(np.abs(np.imag(poles)) <= pole_tolerance):
        raise ValueError("causal projection requires a declared regulator for real-axis poles")
    T, Q, n_causal = sla.schur(
        A, output="complex", sort=lambda pole: np.imag(pole) < -pole_tolerance)
    n_causal = int(n_causal)
    Bt, Ct = Q.conj().T @ B, C @ Q
    if n_causal == 0:
        return (np.zeros((0, 0), dtype=np.complex128),
                np.zeros((0, B.shape[1]), dtype=np.complex128),
                np.zeros((C.shape[0], 0), dtype=np.complex128),
                np.zeros_like(D))
    Ac = T[:n_causal, :n_causal]
    Bc = Bt[:n_causal]
    Cc = Ct[:, :n_causal]
    if n_causal < A.shape[0]:
        Au = T[n_causal:, n_causal:]
        coupling = T[:n_causal, n_causal:]
        # Ac X - X Au = -coupling block-diagonalizes the Schur form.
        X = sla.solve_sylvester(Ac, -Au, -coupling)
        Bc = Bc - X @ Bt[n_causal:]
    return Ac, Bc, Cc, np.zeros_like(D)


def screen_retarded(retarded_p, bare):
    """Return ``W^R = (I - v P^R)^-1 v`` as a feedback realization.

    ``retarded_p`` must be strictly proper and ``bare`` is the dense,
    frequency-independent interaction ``v``.  The feedback formula introduces
    the collective poles through ``A + B v C`` without a frequency grid.
    """
    A, B, C, D = _arrays(retarded_p)
    bare = np.asarray(bare, dtype=np.complex128)
    if D.shape[0] != D.shape[1] or bare.shape != D.shape:
        raise ValueError("retarded polarization and bare interaction must be square and equal")
    if np.max(np.abs(D), initial=0.0) > 1e-12:
        raise ValueError("the retarded polarization must be strictly proper")
    return A + B @ bare @ C, B @ bare, bare @ C, bare.copy()


def screened_lesser(retarded_w, lesser_p, *, pole_tolerance=1e-12):
    """Return the one-sided lesser screened interaction reference.

    This computes

    ``C_-[ W^R (P^< - P^<sharp) W^Rsharp ]``

    by exact dense state-space composition and causal projection.  ``W^R`` may
    include a direct bare interaction and ``P^<`` must be strictly proper.
    The returned realization naturally retains lesser/noise poles that are not
    poles of the correlation part of ``W^R``.
    """
    _, _, _, Dp = _arrays(lesser_p)
    if np.max(np.abs(Dp), initial=0.0) > 1e-12:
        raise ValueError("the one-sided lesser polarization must be strictly proper")
    full_p = add(lesser_p, scale(sharp(lesser_p), -1.0))
    full_w = multiply(multiply(retarded_w, full_p), sharp(retarded_w))
    return causal_part(full_w, pole_tolerance=pole_tolerance)
