"""tdagw/collision.py — the on-shell GW collision term of TD-aGW, on plain arrays.

docs/theory/td_agw_dynamical.md §1.5, Eq. (8).  The causal branches of the Σ τ
sweep split Σ^c into a particle part (weight 1 - f, the ``cond`` branch) and a
hole part (weight f, the ``val`` branch).  Their on-shell averages

    S~_ij = 1/2 [T1_ij + T2_ij],   T1_ij = Σ_ij(e_i),   T2_ij = Σ_ij(e_j)

(the pair tables of ``gw.ppm_sigma.window_sigma_c(branches=True)``) give the
out- and in-scattering rates

    G~> = i (S~_p - S~_p^H),      G~< = s_h i (S~_h - S~_h^H),                 (1)

positive semidefinite on the diagonal; ``s_h`` = :data:`HOLE_BRANCH_SIGN`
because the sweep's hole branch is time ordered (E - e + Ω - iη), the adjoint
of the retarded one, whose width has the opposite sign.  The dissipative part
of Eq. (8) is

    C[ρ] = -1/2 {G~, ρ} + G~<  =  -1/2 {G~>, ρ} + 1/2 {G~<, 1 - ρ},   G~ = G~> + G~<.   (2)

The propagator carries C[ρ] - C[ρ_eq] (``subtracted``: ρ_eq is a fixed point
exactly and x = ρ - ρ_eq is damped by the full G~) or C[ρ] (``raw``), with G~
and the source

    S = -1/2 {G~, ρ_eq} + G~<  [- C[ρ_eq]]                                     (3)

held between refreshes (§7.1).  The flow x' = -1/2 {G~, x} + S is then affine
and exact in the eigenbasis of G~ = V diag(g) V^H,

    x'_ab(τ) = e^{-z_ab} x'_ab(0) + τ φ1(z_ab) S'_ab,   z_ab = (g_a + g_b) τ / 2,   (4)

x' = V^H x V, S' = V^H S V, φ1(z) = (1 - e^{-z})/z.  :class:`Dissipator` holds
(V, g, S'); :func:`flow` is (4) and :func:`rate` its generator (the right-hand
side term of the explicit methods).  The held Hermitian part of the on-shell
operator, :func:`hermitian_update`, joins the K δρ model as a held term.

Every function acts on the trailing (n, n) band axes and broadcasts over any
leading axes (the propagator's (trial, k)): nothing here reduces over k except
:func:`trace_rates`, a diagnostic.  Units are those of the tables (the
propagator: Hartree).  jit-able; the propagator runs them k-locally inside its
``shard_map`` (``tdagw.propagate``).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

__all__ = ["HOLE_BRANCH_SIGN", "FORMS", "HERMITIAN_MODES", "Dissipator", "RefreshPolicy",
           "dagger", "on_shell_average", "width", "hermitian_part", "rates", "collision",
           "source", "dissipator", "phi1", "flow", "rate", "hermitian_update", "trace_rates",
           "refresh_due"]

#: The Σ sweep's hole (``val``) branch is time ordered, E - e + Ω - iη
#: (``tests/test_tdagw_dynsig_sigma.py``'s dense reference, gate G7 pins the
#: sign on the sweep): G~< = -i (S~_h - S~_h^H) = i (S~_h^R - S~_h^R^H).
HOLE_BRANCH_SIGN = -1.0
#: ``subtracted``: C[ρ] - C[ρ_eq] (Eq. 8's equilibrium subtraction); ``raw``: C[ρ].
FORMS = ("subtracted", "raw")
#: The held Hermitian part (:func:`hermitian_update`): ``dynamic`` (default) the
#: on-shell change minus its static-W limit, which K δρ carries already;
#: ``full`` the whole change Herm(S~[ρ]) - Herm(S~[ρ_eq]); ``none``.
HERMITIAN_MODES = ("dynamic", "full", "none")
#: |z| below which φ1 is its Taylor series (relative error < 1e-16 there).
_PHI1_SERIES = 1e-5


def dagger(a):
    """a^H over the trailing two axes."""
    return jnp.conj(jnp.swapaxes(a, -1, -2))


def on_shell_average(pair):
    """S~ = 1/2 (T1 + T2) of a pair table (2, ..., n, n)."""
    return 0.5 * (pair[0] + pair[1])


def width(s):
    """i (S - S^H): -2 Im on the diagonal (positive for a retarded S)."""
    return 1j * (s - dagger(s))


def hermitian_part(s):
    return 0.5 * (s + dagger(s))


def rates(pair_cond, pair_val):
    """(G~>, G~<) of Eq. (1) from each branch's pair tables (2, ..., n, n)."""
    return (width(on_shell_average(pair_cond)),
            HOLE_BRANCH_SIGN * width(on_shell_average(pair_val)))


def _anti(a, b):
    return a @ b + b @ a


def collision(gamma_out, gamma_in, rho):
    """C[ρ] of Eq. (2): -1/2 {G~> + G~<, ρ} + G~<."""
    return -0.5 * _anti(gamma_out + gamma_in, rho) + gamma_in


def source(gamma_out, gamma_in, rho_eq, c_eq=None):
    """S of Eq. (3); ``c_eq`` = C[ρ_eq] at the equilibrium rates (None: the raw form)."""
    s = collision(gamma_out, gamma_in, rho_eq)
    return s if c_eq is None else s - c_eq


class Dissipator(NamedTuple):
    """The held affine flow x' = -1/2 {G~, x} + S of Eq. (4), per leading index.

    vecs : (..., n, n) complex, eigenvectors V of the Hermitized G~ (columns).
    vals : (..., n) float, its eigenvalues g (rates; >= 0 up to the sweep's
        regularization and the PPM's residue asymmetry, which a refresh reports).
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
    """x(τ) of Eq. (4) from x(0) = ``x``: exact for the held G~ and S, any real τ."""
    xe = dagger(d.vecs) @ x @ d.vecs
    z = tau * _pair_rate(jnp.real(d.vals))                    # real: the rates are eigenvalues
    ye = jnp.exp(-z).astype(xe.dtype) * xe + (tau * phi1(z)).astype(xe.dtype) * d.source
    return d.vecs @ ye @ dagger(d.vecs)


def rate(d: Dissipator, x):
    """The generator of :func:`flow`: -1/2 {G~, x} + S."""
    xe = dagger(d.vecs) @ x @ d.vecs
    return d.vecs @ (-_pair_rate(d.vals).astype(xe.dtype) * xe + d.source) @ dagger(d.vecs)


def hermitian_update(s_t, s_eq, static_t=None, static_eq=None):
    """The held Hermitian term ΔΣ^h added to K δρ (§1.5, §7.1).

    ``s_t``, ``s_eq``: the on-shell operator S~ (``window_sigma_c``'s ``onshell``) at the
    refresh state and at ρ_eq.  ``static_t``, ``static_eq``: its static-W limit's
    population-dependent part at the same states (``window_sigma_c(static_limit=True)``:
    sx + diag(head)), or None.  Returns Herm(S~[ρ] - S~[ρ_eq]) minus the change of the
    static limit when given: K δρ already carries the static screened exchange of the
    populations, -W^c(0) δf, which is the static limit of the GN-PPM change, so the
    difference is the dynamical correction and vanishes as the pole energies go to
    infinity (§2: K plus a correction that vanishes in the static limit).
    """
    d = hermitian_part(s_t - s_eq)
    if static_t is None:
        return d
    return d - hermitian_part(static_t - static_eq)


def trace_rates(gamma_out, gamma_in, rho, gamma_out_eq=None, gamma_in_eq=None, rho_eq=None):
    """(out, in): the out- and in-scattering parts of d Tr ρ/dt per leading index.

    Tr(-1/2 {G~>, ρ}) = -Tr(G~> ρ) and Tr(1/2 {G~<, 1 - ρ}) = Tr(G~<) - Tr(G~< ρ); with the
    equilibrium triple given, the same at equilibrium is subtracted (the subtracted form).
    Real parts; the caller weights the leading k axis.
    """
    tr = lambda a, b: jnp.real(jnp.einsum("...ab,...ba->...", a, b))
    out = -tr(gamma_out, rho)
    inn = jnp.real(jnp.trace(gamma_in, axis1=-2, axis2=-1)) - tr(gamma_in, rho)
    if gamma_out_eq is not None:
        out = out + tr(gamma_out_eq, rho_eq)
        inn = inn - (jnp.real(jnp.trace(gamma_in_eq, axis1=-2, axis2=-1)) - tr(gamma_in_eq, rho_eq))
    return out, inn


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
