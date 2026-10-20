"""Dense reference algebra for snapshot Keldysh screening transfer functions.

This module owns host-side transfer algebra for small references and the
explicitly bounded reduced snapshot screening path.
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
    poles = np.diag(A)
    if np.count_nonzero(A) == np.count_nonzero(poles):
        return D + C @ (B / (complex(z) - poles)[:, None])
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


def lossless_residues(system, *, tolerance=1e-8, metric=None, transition_factors=False):
    """Resolve real collective poles and Hermitian residues without pole shifts.

    The small host realization has a square output. Degenerate poles are
    grouped before testing Hermiticity; growing or complex collective modes
    and defective feedback are refused. Energies retain the caller's units.
    """
    A, B, C, D = _arrays(system)
    if not A.size:
        empty = (np.empty(0), np.empty((0, D.shape[0], D.shape[1]), complex))
        return (*empty, []) if transition_factors else empty
    poles, vectors = sla.eig(A)
    span = max(float(np.max(abs(poles))), 1.0)
    if np.max(abs(poles.imag)) > tolerance * span or np.linalg.cond(vectors) > 1e10:
        raise ValueError("GATE snapshot_lossless_stability: nonreal or defective collective poles")
    left, right = C @ vectors, sla.solve(vectors, B)
    if metric is not None:
        metric = np.asarray(metric)
        norms = np.einsum("aj,ab,bj->j", vectors.conj(), metric, vectors).real
        if np.any(abs(norms) < 1e-10):
            raise ValueError("GATE snapshot_lossless_stability: singular collective metric")
        # For A.H S = S A and C = B.H S the residue is explicitly
        # Hermitian: (C x)(C x).H / (x.H S x). No sign is constrained.
        right = left.conj().T / norms[:, None]
    order = np.argsort(poles.real)
    groups, residues, indices = [], [], []
    for j in order:
        residue = np.outer(left[:, j], right[j])
        if groups and abs(poles[j].real - groups[-1]) < tolerance * span:
            residues[-1] += residue
            indices[-1].append(j)
        else:
            groups.append(float(poles[j].real))
            residues.append(residue)
            indices.append([j])
    factors = []
    if metric is not None:
        for i, group in enumerate(indices):
            x = vectors[:, group]
            gram = x.conj().T @ metric @ x
            factor = x @ sla.solve(gram, x.conj().T @ metric @ B)
            factors.append(factor)
            residues[i] = C @ factor
    residues = np.asarray(residues)
    residue_scale = max(float(np.linalg.norm(residues, axis=(1, 2)).max()), 1e-30)
    for residue in residues:
        if np.linalg.norm(residue - residue.conj().T) > tolerance * residue_scale:
            raise ValueError("GATE snapshot_lossless_residue: collective residue is not Hermitian")
    if metric is not None:
        for z in span * np.asarray([-0.4 + 0.7j, 0.3 + 0.6j, 1.1 + 0.9j]):
            reconstructed = D + np.sum(residues / (z - np.asarray(groups))[:, None, None], axis=0)
            expected = evaluate(system, z)
            if np.linalg.norm(reconstructed - expected) > tolerance * max(np.linalg.norm(expected), 1e-30):
                raise ValueError("GATE snapshot_lossless_residue: metric residue reconstruction failed")
    # Symmetrization removes only certified roundoff, never a negative eigenvalue.
    result = (np.asarray(groups), 0.5 * (residues + residues.conj().transpose(0, 2, 1)))
    return (*result, factors) if transition_factors else result


def retarded_reference_branches(system, *, condition_limit=1e10, tolerance=1e-7,
                                evaluation_height=0.0):
    """Assign equilibrium-style T=0 branches to a causal retarded response.

    This is a declared reference prescription for Sigma, not independent
    nonequilibrium fluctuation statistics. For each retarded pole p with
    residue R, the lesser reference has residue -R for Re(p)<0 and zero
    for Re(p)>0. A roundoff-zero real frequency receives -R/2. The greater
    reference is their exact sum with the retarded response. Signed and
    complex residues are retained without a noise-positivity assumption.

    Parameters
    ----------
    system : tuple of ndarray
        Strictly proper square correlation response (A,B,C,D=0), in any
        consistent energy units. Poles must lie below evaluation_height.
    evaluation_height : float
        Positive-lag damping in the subsequent Sigma transform. The
        default zero requires decaying response poles. A positive height
        also admits a growing causal theta(t) response when this damping
        dominates every growth rate; no pole is moved or clipped.

    Returns
    -------
    dict
        Canonical shared-modal ``correlation``, ``lesser``, ``greater`` and
        pole/reconstruction receipts. No poles or amplitudes are shifted.
    """
    A, B, C, D = _arrays(system)
    evaluation_height = float(evaluation_height)
    if not np.isfinite(evaluation_height) or evaluation_height < 0:
        raise ValueError("GATE snapshot_reference_height: finite nonnegative evaluation height required")
    if D.shape[0] != D.shape[1] or np.max(abs(D), initial=0.0) > 1e-12:
        raise ValueError("GATE snapshot_reference_direct: strictly proper square response required")
    if not A.size:
        return dict(correlation=system, lesser=system, greater=system,
                    poles=np.empty(0, complex), zero_frequency_count=0,
                    condition=1.0, reconstruction_error=0.0,
                    signed_gain_pole_count=0, complex_residue_pole_count=0,
                    maximum_pole_growth=0.0, evaluation_height=evaluation_height,
                    laplace_margin=evaluation_height)
    poles, vectors = sla.eig(A)
    condition = float(np.linalg.cond(vectors))
    if not np.isfinite(condition) or condition > condition_limit:
        raise ValueError("GATE snapshot_reference_poles: defective retarded modal realization")
    maximum_growth = float(np.max(poles.imag))
    margin = evaluation_height - maximum_growth
    if margin <= 1e-12:
        raise ValueError(f"GATE snapshot_reference_causality: Sigma Laplace damping "
                         f"{evaluation_height:.9g} must exceed pole growth {maximum_growth:.9g}; "
                         f"margin={margin:.3e}")
    left, right = C @ vectors, sla.solve(vectors, B)
    modal = (np.diag(poles), right, left, D.copy())
    scale = max(float(np.max(abs(poles))), 1.0)
    zero_tolerance = 64 * np.finfo(float).eps * scale
    zero = abs(poles.real) <= zero_tolerance
    weights = np.where(poles.real < -zero_tolerance, -1.0, 0.0)
    weights[zero] = -0.5
    lesser = (modal[0], right, left * weights[None], D.copy())
    greater = (modal[0], right, left * (1 + weights)[None], D.copy())
    gain_count, complex_count = 0, 0
    for pole, l, r in zip(poles, left.T, right):
        residue = np.outer(l, r)
        norm = max(float(np.linalg.norm(residue)), 1e-30)
        hermitian = 0.5 * (residue + residue.conj().T)
        complex_count += int(np.linalg.norm(residue - residue.conj().T) > tolerance * norm)
        if abs(pole.real) > zero_tolerance:
            gain_count += int(sla.eigvalsh(np.sign(pole.real) * hermitian)[0] < -tolerance * norm)
    error = 0.0
    probes = scale * np.asarray([-.37 + .51j, .23 + .87j, 1.13 + .43j])
    probes += 1j * max(maximum_growth, 0.0)
    for z in probes:
        reference = evaluate(system, z)
        error = max(error, float(np.linalg.norm(evaluate(modal, z) - reference)
            / max(np.linalg.norm(reference), 1e-30)))
    if not np.isfinite(error) or error > tolerance:
        raise ValueError("GATE snapshot_reference_reconstruction: retarded modal reconstruction failed")
    return dict(correlation=modal, lesser=lesser, greater=greater, poles=poles,
        zero_frequency_count=int(np.count_nonzero(zero)), condition=condition,
        reconstruction_error=error, zero_frequency_tolerance=zero_tolerance,
        signed_gain_pole_count=gain_count, complex_residue_pole_count=complex_count,
        maximum_pole_growth=maximum_growth, evaluation_height=evaluation_height,
        laplace_margin=margin)


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
