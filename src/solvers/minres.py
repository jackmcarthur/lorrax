"""solvers/minres.py — Preconditioned projected MINRES for batched Hermitian systems.

Solves, for each v in the leading batch,

    A_eff x_v = b_v_eff,
    A_eff = Π ∘ apply_A ∘ Π,
    b_v_eff = Π b_v,

where ``apply_A`` is a user-supplied Hermitian matvec of shape
``(batch, nspinor, nG) → (batch, nspinor, nG)`` and ``Π`` is an optional
orthogonal projector (e.g. ``Q_{k-q}`` from ``solvers.projectors``).

Algorithm: classical Paige–Saunders MINRES (Paige & Saunders, SINUM **12**
(1975) 617) — symmetric Lanczos tridiagonalisation with rolling Givens
QR of the tridiagonal.  Standard choice for symmetric-indefinite or
symmetric-semidefinite systems.  The Sternheimer operator at ω=0 is
positive-semidefinite inside the conduction subspace ``range(Q_{k-q})``
and identically zero on the occupied subspace; projection keeps the
Krylov sequence clean inside ``range(Π)`` throughout.

JIT-compatible: fixed ``max_iter`` ⇒ ``lax.fori_loop`` with rolling 2-step
Lanczos + 2-step Givens history.  Per-v residual tracking via
:class:`MinresInfo`; no in-loop early exit (cost-free for O(10) bands;
variable iteration counts would force recompilation).

Notation follows Paige–Saunders 1975 §5 directly (``α, β, γ, δ, ε`` for the
tridiagonal entries, ``c, s`` for Givens rotations).  The Hermitian case
keeps ``α, β, c, s`` real, so almost every scalar in the recurrence is
real-valued — we only promote to complex when scaling a state vector.

Reference: SciPy's ``scipy.sparse.linalg.minres`` was used as a
line-by-line correctness check when porting, modulo the projected-operator
and batched-over-v extensions added here.
"""
from __future__ import annotations

import functools
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


# ═══════════════════════════════════════════════════════════════════════
#  Return type
# ═══════════════════════════════════════════════════════════════════════

class MinresInfo(NamedTuple):
    """Per-band convergence info from :func:`minres`."""
    res_norms: jax.Array          # (batch,) float — final ‖b - A x‖
    iters_used: int
    converged: jax.Array          # (batch,) bool — res_norms < tol·‖b‖


# ═══════════════════════════════════════════════════════════════════════
#  Batched dot / norm
# ═══════════════════════════════════════════════════════════════════════

def _batched_dot(a: jax.Array, b: jax.Array) -> jax.Array:
    """Return ``<a_v, b_v> = Σ_{s,G} conj(a[v,s,G]) · b[v,s,G]`` per batch element."""
    return jnp.einsum('vsG,vsG->v', jnp.conj(a), b, optimize=True)


def _batched_real_norm(a: jax.Array) -> jax.Array:
    """``(batch,)`` real L2 norm."""
    return jnp.sqrt(jnp.real(_batched_dot(a, a)))


def _identity(x: jax.Array) -> jax.Array:
    return x


# ═══════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════

def minres(
    apply_A: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    precond: Callable[[jax.Array], jax.Array] | None = None,
    project: Callable[[jax.Array], jax.Array] | None = None,
    x0: jax.Array | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> tuple[jax.Array, MinresInfo]:
    """Preconditioned projected MINRES, batched over a leading ``v`` axis.

    Parameters
    ----------
    apply_A : (batch, nspinor, nG) → same
        Hermitian matvec.  For Sternheimer:
        ``x ↦ H_{k-q} x − ε_{v,k} x`` assembled from
        :func:`psp.h_dft.make_apply_H` plus a per-v eigenvalue shift.
    b : (batch, nspinor, nG)
        Right-hand side (will be internally projected if ``project`` is
        supplied).
    precond : callable, optional
        Symmetric positive-definite preconditioner ``M⁻¹``, same signature
        as ``apply_A``.  For Sternheimer this is
        :func:`solvers.sternheimer_precond.make_tpa_preconditioner`.
        ``None`` → identity.
    project : callable, optional
        Orthogonal projector ``Π`` (e.g. ``Q_{k-q}`` from
        :mod:`solvers.projectors`) applied inside every matvec and to ``b``.
        Enforces the Sternheimer "inside the conduction subspace"
        constraint throughout the solve.  ``None`` → iterate in full space.
    x0 : (batch, nspinor, nG), optional
        Initial guess.  ``None`` → zero (which lies in ``range(Π)``
        trivially).
    tol : float
        Convergence tolerance on ``‖b − A x‖ / ‖b‖``.  Informational only
        (no early exit).
    max_iter : int
        Iteration count.  Static (part of the JIT signature).

    Returns
    -------
    x : (batch, nspinor, nG)
    info : MinresInfo
    """
    if precond is None:
        precond = _identity
    if project is None:
        project = _identity

    b_proj = project(b)
    b_norm = _batched_real_norm(b_proj)
    b_norm_safe = jnp.where(b_norm > 0, b_norm, 1.0)

    if x0 is None:
        x0 = jnp.zeros_like(b)

    x_final = _minres_core(apply_A, precond, project, b_proj, x0, max_iter)

    residual = b_proj - project(apply_A(x_final))
    res_norms = _batched_real_norm(residual)
    converged = res_norms < tol * b_norm_safe
    return x_final, MinresInfo(
        res_norms=res_norms, iters_used=int(max_iter), converged=converged)


# ═══════════════════════════════════════════════════════════════════════
#  JIT core — Paige–Saunders preconditioned MINRES
# ═══════════════════════════════════════════════════════════════════════

@functools.partial(
    jax.jit, static_argnames=('apply_A', 'precond', 'project', 'max_iter'))
def _minres_core(
    apply_A: Callable,
    precond: Callable,
    project: Callable,
    b: jax.Array,
    x0: jax.Array,
    max_iter: int,
) -> jax.Array:
    """Paige–Saunders MINRES, preconditioned + projected, batched over v.

    State carried across iterations (all shape ``(batch, nspinor, nG)``
    for vectors, ``(batch,)`` for scalars):

        v_prev, v_curr    Lanczos basis, last two (M-orthonormal).
        z_prev, z_curr    Preconditioned Lanczos basis (M z = v).
        w_pp,   w_p       Update directions, last two.
        x                 Current iterate.
        beta              β_k, sub-diagonal carried in from previous step.
        c_p,  c_pp        Givens cosines from previous two rotations.
        s_p,  s_pp        Givens sines from previous two rotations.
        eta               Rotated-RHS head entering this step.

    At step k the column of the tridiagonal being processed is
    ``[0, ε_k, δ_k, γ̄_k, β_{k+1}]`` after the two prior rotations; the new
    rotation ``G_k`` zeros ``β_{k+1}`` → the main-diagonal entry becomes
    ``γ_k`` and the rotated RHS head advances.
    """
    dtype_c = b.dtype
    dtype_r = jnp.float64 if dtype_c == jnp.complex128 else jnp.float32
    batch = b.shape[0]
    zero_r = jnp.zeros((batch,), dtype=dtype_r)
    one_r  = jnp.ones((batch,),  dtype=dtype_r)

    # ── Initialisation (k = 0 in the loop corresponds to MINRES step k=1) ──
    # r_0 = b − A x_0   (projected)
    r0 = b - project(apply_A(x0))
    z0 = precond(r0)
    # β_1 = √<r_0, z_0>   (= ‖r_0‖_{M⁻¹}); real because M⁻¹ is HPD.
    beta0 = jnp.sqrt(jnp.maximum(jnp.real(_batched_dot(r0, z0)), 0.0))
    beta0_safe = jnp.where(beta0 > 0, beta0, 1.0)
    v_curr = r0 / beta0_safe[:, None, None].astype(dtype_c)
    z_curr = z0 / beta0_safe[:, None, None].astype(dtype_c)

    v_prev = jnp.zeros_like(b)
    z_prev = jnp.zeros_like(b)
    w_p    = jnp.zeros_like(b)       # w_0
    w_pp   = jnp.zeros_like(b)       # w_{-1}

    # c_0 = c_{-1} = 1,  s_0 = s_{-1} = 0
    c_p, c_pp = one_r, one_r
    s_p, s_pp = zero_r, zero_r
    # Rotated-RHS head at step 1 is β_1 (real).
    eta = beta0

    state = (x0, v_prev, v_curr, z_prev, z_curr,
             w_pp, w_p,
             beta0, c_p, c_pp, s_p, s_pp, eta)

    def body(_i: int, st):
        (x, v_prev, v_curr, z_prev, z_curr,
         w_pp, w_p,
         beta_prev, c_p, c_pp, s_p, s_pp, eta) = st

        # ── 1. Preconditioned Lanczos step ──
        # p = A z_k  (projected)
        p = project(apply_A(z_curr))
        # α_k = <z_k, A z_k>_{M} = <z_k, p>  — real for Hermitian A.
        alpha = jnp.real(_batched_dot(z_curr, p))
        # r_{k+1} = p − α_k v_k − β_k v_{k-1}
        r_new = (p
                 - alpha[:, None, None].astype(dtype_c) * v_curr
                 - beta_prev[:, None, None].astype(dtype_c) * v_prev)
        z_new_raw = precond(r_new)
        # β_{k+1}² = <r_{k+1}, M⁻¹ r_{k+1}> — real and ≥ 0 in exact arith.
        beta_new = jnp.sqrt(
            jnp.maximum(jnp.real(_batched_dot(r_new, z_new_raw)), 0.0))
        beta_new_safe = jnp.where(beta_new > 0, beta_new, 1.0)
        v_new = r_new     / beta_new_safe[:, None, None].astype(dtype_c)
        z_new = z_new_raw / beta_new_safe[:, None, None].astype(dtype_c)

        # ── 2. Apply G_{k-2} to column k ──
        #   ε_k = s_{k-2} β_k,     δ̄_k = c_{k-2} β_k
        # (Hermitian ⇒ s_{k-2} is real; no conjugate needed.)
        eps_k     = s_pp * beta_prev
        delta_bar = c_pp * beta_prev

        # ── 3. Apply G_{k-1} to column k, rows (k-1, k) ──
        #   δ_k   =  c_{k-1} δ̄_k + s_{k-1} α_k
        #   γ̄_k  = -s_{k-1} δ̄_k + c_{k-1} α_k
        delta_k   =  c_p * delta_bar + s_p * alpha
        gamma_bar = -s_p * delta_bar + c_p * alpha

        # ── 4. New Givens G_k to zero β_{k+1} ──
        #   γ_k = √(γ̄_k² + β_{k+1}²);  c_k = γ̄_k / γ_k;  s_k = β_{k+1} / γ_k
        gamma = jnp.sqrt(gamma_bar * gamma_bar + beta_new * beta_new)
        gamma_safe = jnp.where(gamma > 0, gamma, 1.0)
        c_new = gamma_bar / gamma_safe
        s_new = beta_new  / gamma_safe

        # ── 5. Rotated RHS update ──
        #   used_k = c_k η_k   (contributes to x),     η_{k+1} = -s_k η_k
        eta_used = c_new * eta
        eta_next = -s_new * eta

        # ── 6. Update direction w_k = (z_k − δ_k w_{k-1} − ε_k w_{k-2}) / γ_k ──
        w_new = (z_curr
                 - delta_k[:, None, None].astype(dtype_c) * w_p
                 - eps_k[:,   None, None].astype(dtype_c) * w_pp
                ) / gamma_safe[:, None, None].astype(dtype_c)

        # ── 7. Solution update x_k = x_{k-1} + (c_k η_k) w_k ──
        # Mask the update where γ_k = 0 (system solved for that v).
        gamma_mask = (gamma > 0).astype(dtype_c)[:, None, None]
        x_new = x + gamma_mask * eta_used[:, None, None].astype(dtype_c) * w_new

        # ── 8. Advance rolling state ──
        return (x_new, v_curr, v_new, z_curr, z_new,
                w_p, w_new,
                beta_new, c_new, c_p, s_new, s_p, eta_next)

    final = lax.fori_loop(0, max_iter, body, state)
    return final[0]


__all__ = ["minres", "MinresInfo"]
