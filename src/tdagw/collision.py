"""tdagw/collision.py — one-sided full-matrix Markov GW operators.

The complex branch tables retain row-shell Sigma_ij(E_i) and column-shell
Sigma_ij(E_j) samples. The particle branch is retarded, the hole advanced.
Their retarded columns are R_p = cond[1] and R_h = val[0]^H. With
M = R_p + R_h + static, Gamma = i(M - M^H), D = i R_h + (i R_h)^H,
and M_H = Herm(M), the full one-sided expression separates into

    -i[M_H, rho] - 1/2 {Gamma, rho} + D.

The Hermitian static term contributes only to M_H. The model carries the
chosen coherent correction separately, exactly once. The optional
subtracted form imposes the equilibrium remainder as a counterterm;
it is not an identity of the raw Markov integral.

Between refreshes the remainder on x = rho - rho_eq is affine:
x\' = -1/2 {Gamma, x} + S, S = C[rho_eq] [- C_eq[rho_eq]]. The existing
Hermitian Gamma eigenbasis gives its exact flow for either sign of tau.
Negative rates are reported, never clipped.

Arrays broadcast over leading trial and k axes, act on trailing band axes,
and use supplied energy units (Hartree in propagation). The propagator
invokes these array functions k-locally inside shard_map.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

__all__ = ["FORMS", "HERMITIAN_MODES", "Dissipator", "RefreshPolicy",
           "dagger", "width", "hermitian_part", "branch_operators", "collision",
           "source", "dissipator", "phi1", "flow", "rate", "hermitian_update", "trace_rates",
           "boundary_diagnostics", "refresh_due"]

#: ``subtracted`` imposes the equilibrium counterterm; ``raw`` does not.
FORMS = ("subtracted", "raw")
#: The held Hermitian part (:func:`hermitian_update`): ``dynamic`` (default) the
#: on-shell change minus its static-W limit, which K δρ carries already;
#: ``full`` the whole change M_H[ρ] - M_H[ρ_eq]; ``none``.
HERMITIAN_MODES = ("dynamic", "full", "none")
#: |z| below which φ1 is its Taylor series (relative error < 1e-16 there).
_PHI1_SERIES = 1e-5


def dagger(a):
    """a^H over the trailing two axes."""
    return jnp.conj(jnp.swapaxes(a, -1, -2))


def width(s):
    """i (S - S^H): -2 Im on the diagonal (positive for a retarded S)."""
    return 1j * (s - dagger(s))


def hermitian_part(s):
    return 0.5 * (s + dagger(s))


def branch_operators(pair_cond, pair_val, static=0.0):
    """One-sided Markov operators ``(gamma, injection, m_h)`` from branch tables.

    Parameters
    ----------
    pair_cond, pair_val : (2, ..., n, n) complex arrays
        Row-shell and column-shell samples in one instantaneous basis, in energy
        units. The particle branch is retarded; the hole branch is time ordered.
    static : (..., n, n) complex array or scalar
        Frequency-independent Hermitian self-energy; enters ``m_h`` only.

    Returns
    -------
    gamma, injection, m_h : (..., n, n) complex arrays
        Hermitian total width, injection D = A_in + A_in^H, and Hermitian
        retarded self-energy. No arithmetic shell averaging is performed.

    The retarded hole column is the adjoint of the advanced *row* table:
    R_p = cond[1], R_h = val[0]^H, A_in = i R_h. Thus gamma =
    i(R_p + R_h - R_p^H - R_h^H), and m_h = Herm(R_p + R_h + static).
    """
    rp, rh = pair_cond[1], dagger(pair_val[0])
    ain = 1j * rh
    return width(rp + rh), ain + dagger(ain), hermitian_part(rp + rh + static)


def _anti(a, b):
    return a @ b + b @ a


def collision(gamma, injection, rho):
    """Split Markov remainder: -1/2 {gamma, rho} + injection.

    The coherent term -i[m_h, rho] is propagated separately, exactly once.
    """
    return -0.5 * _anti(gamma, rho) + injection


def source(gamma, injection, rho_eq, c_eq=None):
    """Affine source on delta rho; optional ``c_eq`` is an equilibrium counterterm.

    Subtracting this counterterm imposes a stationary reference. It is a model
    choice, not an identity of the unsubtracted Markov collision integral.
    """
    s = collision(gamma, injection, rho_eq)
    return s if c_eq is None else s - c_eq


class Dissipator(NamedTuple):
    """The held affine flow x' = -1/2 {Gamma, x} + S, per leading index.

    vecs : (..., n, n) complex, eigenvectors V of Hermitian Gamma (columns).
    vals : (..., n) float, its eigenvalues g (rates). The full-matrix Markov
        approximation can give negative values, which a refresh reports.
    source : (..., n, n) complex, S' = V^H S V.
    The propagator holds it k-sharded like ρ (``ops.collision``); a pad k is
    zero rates and zero source, the identity flow.
    """
    vecs: object
    vals: object
    source: object


def dissipator(gamma, src, *, eigh=None) -> Dissipator:
    """The :class:`Dissipator` of total rate ``gamma`` and source ``src`` (eigh per leading index)."""
    eigh = jnp.linalg.eigh if eigh is None else eigh
    vals, vecs = eigh(hermitian_part(gamma))
    return Dissipator(vecs=vecs, vals=vals, source=dagger(vecs) @ src @ vecs)


def phi1(z):
    """(1 - e^{-z})/z, elementwise, real or complex z (the Taylor series near 0)."""
    small = jnp.abs(z) < _PHI1_SERIES
    zs = jnp.where(small, jnp.ones_like(z), z)
    return jnp.where(small, 1.0 - 0.5 * z + z * z / 6.0, -jnp.expm1(-zs) / zs)


def _pair_rate(vals):
    return 0.5 * (vals[..., :, None] + vals[..., None, :])


def flow(d: Dissipator, x, tau):
    """x(τ) from x(0) = ``x``: exact for held Hermitian Gamma and D."""
    xe = dagger(d.vecs) @ x @ d.vecs
    z = tau * _pair_rate(jnp.real(d.vals))                    # real: the rates are eigenvalues
    ye = jnp.exp(-z).astype(xe.dtype) * xe + (tau * phi1(z)).astype(xe.dtype) * d.source
    return d.vecs @ ye @ dagger(d.vecs)


def rate(d: Dissipator, x):
    """The generator of :func:`flow`: -1/2 {Gamma, x} + D."""
    xe = dagger(d.vecs) @ x @ d.vecs
    return d.vecs @ (-_pair_rate(d.vals).astype(xe.dtype) * xe + d.source) @ dagger(d.vecs)


def hermitian_update(s_t, s_eq, static_t=None, static_eq=None):
    """The held Hermitian term ΔΣ^h added to K δρ (§1.5, §7.1).

    ``s_t``, ``s_eq``: M_H = Herm(R_p + R_h + static) at the refresh state
    and at rho_eq. ``static_t``, ``static_eq``: its static-W limit's
    population-dependent part at the same states (``window_sigma_c(static_limit=True)``:
    sx + diag(head)), or None. Returns the change in M_H minus the change of the
    static limit when given: K δρ already carries the static screened exchange of the
    populations, -W^c(0) δf, which is the static limit of the GN-PPM change, so the
    difference is the dynamical correction and vanishes as the pole energies go to
    infinity (§2: K plus a correction that vanishes in the static limit).
    """
    d = hermitian_part(s_t - s_eq)
    if static_t is None:
        return d
    return d - hermitian_part(static_t - static_eq)


def trace_rates(gamma, injection, rho, gamma_eq=None, injection_eq=None, rho_eq=None):
    """(loss, injection): trace of the two split remainder terms, per leading index.

    With the equilibrium triple, subtract the imposed equilibrium counterterm.
    Real parts; the caller weights the leading k axis. Neither term is assumed
    positive for a general matrix Markov approximation.
    """
    tr = lambda a, b: jnp.real(jnp.einsum("...ab,...ba->...", a, b))
    out = -tr(gamma, rho)
    inn = jnp.real(jnp.trace(injection, axis1=-2, axis2=-1))
    if gamma_eq is not None:
        out = out + tr(gamma_eq, rho_eq)
        inn = inn - jnp.real(jnp.trace(injection_eq, axis1=-2, axis2=-1))
    return out, inn


def boundary_diagnostics(gamma, injection, rho, *, occupation_tolerance):
    """Measure Eq. (8)'s global and current-state Pauli boundary matrices.

    Parameters
    ----------
    gamma, injection, rho : ndarray (..., n, n)
        Already-host-resident, matching small window matrices. ``gamma`` and
        effective injection ``D`` use the same energy/rate unit; ``rho`` is
        dimensionless. For a subtracted model pass the effective injection
        including its counterterm. No microscopic state is gathered here.
    occupation_tolerance : float
        Explicit dimensionless tolerance in [0, 1/2). Eigenvalues within this
        distance of zero or one select the complete empty/filled subspace.

    Returns
    -------
    dict of ndarray (...,)
        ``min_injection`` and ``min_blocking`` are global eigenvalue minima
        of D and Gamma-D. ``empty_min_injection`` and ``filled_min_blocking``
        are minima after compression to the selected density subspaces;
        their ``empty_count``/``filled_count`` are dimensions, with NaN for
        a missing subspace. ``rho_min``/``rho_max`` are occupations and
        ``trace_rate`` is Re Tr(D-Gamma rho), per block, without k weights.
        ``gamma_hermiticity_error``, ``injection_hermiticity_error`` and
        ``rho_hermiticity_error`` report max |A-A^H| on the original inputs;
        quadratic forms use their Hermitian parts.
        Rate minima and trace rate retain the supplied energy/rate unit
        (hbar=1); occupations and counts are dimensionless.

    Notes
    -----
    At an exact boundary the compressed D and Gamma-D give the inward
    first derivatives: both the Hamiltonian commutator and the appropriate
    anticommutator vanish there. At nonzero occupation tolerance these are
    neighborhood probes, not exact derivative tests. Entire subspaces are
    compressed, including degeneracies; diagonal expectations alone miss
    outward directions. Passing at one state does not establish finite-time
    or nonlinear Pauli invariance. No matrix is repaired or clipped.
    """
    tolerance = float(occupation_tolerance)
    if not np.isfinite(tolerance) or not 0.0 <= tolerance < 0.5:
        raise ValueError("occupation_tolerance must be finite and in [0, 0.5)")
    gamma, injection, rho = (np.asarray(a, dtype=np.complex128)
                             for a in (gamma, injection, rho))
    if (rho.ndim < 2 or rho.shape[-1] != rho.shape[-2] or rho.shape[-1] == 0
            or gamma.shape != rho.shape or injection.shape != rho.shape):
        raise ValueError("boundary diagnostics require matching (..., n, n) window matrices")
    if not all(np.isfinite(a).all() for a in (gamma, injection, rho)):
        raise ValueError("boundary diagnostics require finite window matrices")
    hermiticity_errors = {
        f"{name}_hermiticity_error": np.max(
            np.abs(a - np.conj(np.swapaxes(a, -1, -2))), axis=(-2, -1))
        for name, a in (("gamma", gamma), ("injection", injection), ("rho", rho))}
    # Hermitian parts define the reported quadratic forms; inputs are not
    # modified, and their original Hermiticity defects are retained.
    herm = lambda a: 0.5 * (a + np.conj(np.swapaxes(a, -1, -2)))
    gamma, injection, rho = (herm(a) for a in (gamma, injection, rho))
    blocking = gamma - injection
    occupations, vectors = np.linalg.eigh(rho)
    empty = np.abs(occupations) <= tolerance
    filled = np.abs(occupations - 1.0) <= tolerance
    shape = rho.shape[:-2]
    result = dict(min_injection=np.linalg.eigvalsh(injection)[..., 0],
                  min_blocking=np.linalg.eigvalsh(blocking)[..., 0],
                  empty_count=np.count_nonzero(empty, axis=-1),
                  filled_count=np.count_nonzero(filled, axis=-1),
                  empty_min_injection=np.full(shape, np.nan),
                  filled_min_blocking=np.full(shape, np.nan),
                  rho_min=occupations[..., 0], rho_max=occupations[..., -1],
                  trace_rate=np.real(np.trace(injection - gamma @ rho, axis1=-2, axis2=-1)))
    result.update(hermiticity_errors)
    for index in np.ndindex(shape):
        for mask, matrix, key in ((empty, injection, "empty_min_injection"),
                                  (filled, blocking, "filled_min_blocking")):
            basis = vectors[index][:, mask[index]]
            if basis.shape[-1]:
                compressed = basis.conj().T @ matrix[index] @ basis
                result[key][index] = np.linalg.eigvalsh(herm(compressed))[0]
    return result


class RefreshPolicy(NamedTuple):
    """When the held rates are rebuilt (§7.1): host decision at a chunk boundary.

    every : time between refreshes (a.u.); 0 = no cadence.
    df : max |f - f_last| over trials, k and window bands since the last
        refresh; 0 = no tolerance.  With both 0 the rates stay those of t0.
    """
    every: float = 0.0
    df: float = 0.0


def refresh_due(policy: RefreshPolicy, t, t_last, f, f_last):
    """(due, max|f - f_last|, why) at a chunk boundary; ``f`` host (..., n) populations."""
    df = float(np.max(np.abs(np.asarray(f) - np.asarray(f_last)), initial=0.0))
    if policy.df > 0.0 and df >= policy.df:
        return True, df, "df"
    if policy.every > 0.0 and float(t) - float(t_last) >= policy.every * (1.0 - 1e-9):
        return True, df, "cadence"
    return False, df, ""
