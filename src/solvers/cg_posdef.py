"""solvers/cg_posdef.py — Preconditioned conjugate gradient for Hermitian PD systems,
batched over a leading band axis.

Written specifically for the DFPT / Sternheimer operator after QE's level-shift
trick (``PHonon/PH/cch_psi_all.f90`` + ``LR_Modules/cgsolve_all.f90``):

    A_v = H_{k-q} − ε_{v,k} + α_pv · P_val^{k-q}

With ``α_pv ≈ 2·(E_max − E_min)`` of the occupied + relevant conduction
eigenvalues, A_v becomes positive-definite on the **entire** Hilbert space:

    * on range(P_val):  eigenvalues ≈ α_pv − (ε_{v'} − ε_{v,k}) > 0.
    * on range(Q):      eigenvalues = ε_c^{k-q} − ε_{v,k} > 0 for an insulator.

This removes the need to project inside iterations — the algorithm is just
unprojected CG, so there is no projector-drift, no pseudo-convergence NaN
windows, and no noise-amplification issue when ‖b‖ is tiny.  If ``b`` is
already Q-projected (as in the density-response case), the solution is
automatically orthogonal to occupied to machine precision: any P_val
component of the solution would have to come from a P_val component of b,
which is zero.

This is **the** algorithm used by QE's ``ph.x`` for static phonon DFPT
and any similar Sternheimer solve.  Reference:

    Dal Corso & Mauri, "cgsolve_all revision 6 Apr 1997"; the theory in
    Baroni, de Gironcoli, Dal Corso, Giannozzi, Rev. Mod. Phys. 73 (2001)
    515 §II.C.

JIT-compatible: fixed ``max_iter`` ⇒ ``lax.fori_loop`` with per-band
convergence masking (once a band's residual drops below ``tol·‖b‖``,
subsequent updates are frozen for that band, mirroring the ``conv(ibnd)``
logic of the QE original).
"""
from __future__ import annotations

import functools
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class CGInfo(NamedTuple):
    """Per-band convergence info from :func:`cg_posdef`."""
    res_norms: jax.Array            # (batch,) float — final ‖b - A x‖
    iters_used: int
    converged: jax.Array            # (batch,) bool


def _batched_dot(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.einsum('vsG,vsG->v', jnp.conj(a), b, optimize=True)


def _batched_real_norm(a: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.real(_batched_dot(a, a)))


def _identity(x: jax.Array) -> jax.Array:
    return x


def cg_posdef(
    apply_A: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    precond: Callable[[jax.Array], jax.Array] | None = None,
    x0: jax.Array | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> tuple[jax.Array, CGInfo]:
    """Preconditioned CG for Hermitian positive-definite ``A``, batched over a
    leading ``v`` axis.

    Algorithm (Fletcher–Reeves CG; mirrors QE's cgsolve_all):

        r_0 = b - A x_0
        z_0 = M⁻¹ r_0        (preconditioned residual)
        p_0 = z_0
        for k = 0, 1, ...:
            α_k = <r_k, z_k> / <p_k, A p_k>
            x_{k+1} = x_k + α_k p_k
            r_{k+1} = r_k - α_k A p_k
            if ‖r_{k+1}‖ < tol · ‖b‖: freeze this band
            z_{k+1} = M⁻¹ r_{k+1}
            β_k = <r_{k+1}, z_{k+1}> / <r_k, z_k>
            p_{k+1} = z_{k+1} + β_k p_k

    Parameters
    ----------
    apply_A : (batch, nspinor, nG) → same
        Hermitian positive-definite matvec.  For Sternheimer this is
        ``x ↦ H_{k-q} x − ε_{v,k} x + α_pv · P_val^{k-q} x``.
    b : (batch, nspinor, nG) complex
        RHS.
    precond : callable, optional
        Symmetric positive-definite preconditioner ``M⁻¹``.  ``None`` →
        identity.  For Sternheimer, TPA; see
        :func:`solvers.sternheimer_precond.make_tpa_preconditioner`.
    x0 : (batch, nspinor, nG), optional — initial guess (default: zero).
    tol : float — relative residual tolerance.
    max_iter : int — static iteration cap.

    Returns
    -------
    x : (batch, nspinor, nG)
    info : CGInfo
    """
    if precond is None:
        precond = _identity
    if x0 is None:
        x0 = jnp.zeros_like(b)

    b_norm = _batched_real_norm(b)
    b_norm_safe = jnp.where(b_norm > 0, b_norm, 1.0)

    x_final = _cg_posdef_core(apply_A, precond, b, x0, b_norm_safe, tol, max_iter)

    residual = b - apply_A(x_final)
    res_norms = _batched_real_norm(residual)
    converged = res_norms < tol * b_norm_safe
    return x_final, CGInfo(
        res_norms=res_norms, iters_used=int(max_iter), converged=converged)


@functools.partial(
    jax.jit, static_argnames=('apply_A', 'precond', 'max_iter'))
def _cg_posdef_core(
    apply_A: Callable,
    precond: Callable,
    b: jax.Array,
    x0: jax.Array,
    b_norm_safe: jax.Array,          # (batch,) float, already clipped > 0
    tol: float,
    max_iter: int,
) -> jax.Array:
    """Inner fori_loop.  Per-band freeze mask when residual drops below tolerance."""
    dtype_c = b.dtype
    dtype_r = jnp.float64 if dtype_c == jnp.complex128 else jnp.float32

    # ── Initialisation ──
    r = b - apply_A(x0)
    z = precond(r)
    # rho_0 = <r, z>  — real and ≥ 0 for HPD M.
    rho = jnp.real(_batched_dot(r, z))
    p = z
    x = x0
    # Bands with zero RHS are already solved — never update.
    alive_mask = (b_norm_safe > 0) & (rho > 0)

    def body(_i, st):
        x, r, z, p, rho, alive_mask = st

        Ap = apply_A(p)
        # α = rho / <p, Ap>  — real for Hermitian A.
        pAp = jnp.real(_batched_dot(p, Ap))
        # Guard pAp against non-positive values (should not happen for PD A,
        # but converged bands can reach pAp = 0 after freeze).
        pAp_safe = jnp.where(pAp > jnp.finfo(dtype_r).eps, pAp, 1.0)
        alpha = rho / pAp_safe                                        # (batch,)
        # Zero out α for dead / converged bands so x stays put.
        alpha = jnp.where(alive_mask, alpha, 0.0)

        alpha_c = alpha[:, None, None].astype(dtype_c)
        x_new = x + alpha_c * p
        r_new = r - alpha_c * Ap

        # Per-band convergence check: once ‖r‖ < tol · ‖b‖, mark dead and
        # freeze this band for all remaining iterations.  Don't zero r or x —
        # we want to preserve the converged answer exactly and avoid perturbing
        # the CG recurrence via where-branches mid-iteration.
        r_new_norm = _batched_real_norm(r_new)
        still_alive = alive_mask & (r_new_norm > tol * b_norm_safe)

        z_new = precond(r_new)
        rho_new = jnp.real(_batched_dot(r_new, z_new))
        # Fletcher–Reeves β.
        rho_safe = jnp.where(rho > jnp.finfo(dtype_r).eps, rho, 1.0)
        beta = jnp.where(alive_mask, rho_new / rho_safe, 0.0)
        beta_c = beta[:, None, None].astype(dtype_c)
        p_new = z_new + beta_c * p

        return (x_new, r_new, z_new, p_new, rho_new, still_alive)

    state = (x, r, z, p, rho, alive_mask)
    x_final, *_ = lax.fori_loop(0, max_iter, body, state)
    return x_final


__all__ = ["cg_posdef", "CGInfo"]
