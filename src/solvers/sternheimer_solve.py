"""solvers/sternheimer_solve.py — The Sternheimer primitive: a single JIT-stable
level-shifted CG solve that reuses ONE compiled kernel across all (k, q) on the
grid, and is ``jax.custom_jvp``-wrappable so that q-derivatives of the induced
density (and hence χ_{G'0}) are obtained via a second application of the same
primitive with a differentiated RHS.

Why this exists
---------------
The generic :func:`solvers.cg_posdef.cg_posdef` takes ``apply_A, precond`` as
static callables.  Each (k, q) in the driver builds fresh closures, which JAX
hashes by identity → cache misses → ≈ 9× JIT retrace on a 9-k-point MoS2 run.
Here we inline the operator directly from its array data, so the whole
Sternheimer solve is one big ``@jax.jit`` whose signature is constant across
(k, q).

The primitive is also the natural shape for the Stage-3 custom JVP.  From
the guide (and Cancès et al. 2023):

    A(θ) x(θ) = b(θ)       (primal)
    A ẋ       = ḃ − Ȧ x   (implicit-derivative)

Both solves use the SAME operator A, differing only in RHS.  The custom JVP
rule here runs the primitive once for the primal, then once more with the
differentiated RHS for the tangent — no autodiff-through-iterations.  This
follows the "short-circuit the primal, not the implicit derivative" advice
the other-agent gave in the Stage-3 writeup.

Public API
----------
* :func:`sternheimer_solve` — the primitive.  Takes a pytree of operator data
  (H arrays + U_val + ε_v + α_pv + precond_weights) and an RHS; returns δu.
* :func:`SternheimerOp` — tiny dataclass bundling the operator pytree.
"""
from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from psp.dft_operators import apply_H_k


# ═══════════════════════════════════════════════════════════════════════
#  Operator pytree
# ═══════════════════════════════════════════════════════════════════════

@jax.tree_util.register_pytree_node_class
class SternheimerOp:
    """Bundle of operator data for the level-shifted Sternheimer system.

    The "level-shifted" operator is

        A_v(x) = H_{k-q} · x  −  ε_{v,k} · x  +  α_pv · P_val^{k-q}(x)

    with α_pv = 2 · (E_max − E_min) of the loaded occupied spectrum.  This is
    positive-definite on the entire Hilbert space (see QE `cgsolve_all.f90`).

    Attributes are JAX arrays — the whole object is a registered pytree, so
    it flows through ``jax.jit`` / ``jax.jvp`` without static-arg retraces.

    Shapes
    ------
    T_diag, mask   : (nG,)                  — kinetic diag + cutoff mask
    V_scf          : (nx, ny, nz)           — real-space SCF potential
    Gx, Gy, Gz     : (nG,) int32            — G-sphere in FFT-box coords
    vnl_Z, vnl_E   : see psp.dft_operators  — nonlocal KB projectors
    U_val          : (nv, nspinor, nG)      — occupied at k-q, orthonormal
    eps_v          : (nv,)                  — eigenvalues at source k
    alpha_pv       : ()  scalar             — level-shift
    precond_diag   : (nv, 1, nG)            — TPA preconditioner weights
    """
    __slots__ = ("T_diag", "V_scf", "Gx", "Gy", "Gz",
                 "vnl_Z", "vnl_E", "mask",
                 "U_val", "eps_v", "alpha_pv", "precond_diag",
                 "fft_grid")

    def __init__(
        self, T_diag, V_scf, Gx, Gy, Gz, vnl_Z, vnl_E, mask,
        U_val, eps_v, alpha_pv, precond_diag, fft_grid,
    ):
        self.T_diag = T_diag
        self.V_scf = V_scf
        self.Gx = Gx
        self.Gy = Gy
        self.Gz = Gz
        self.vnl_Z = vnl_Z
        self.vnl_E = vnl_E
        self.mask = mask
        self.U_val = U_val
        self.eps_v = eps_v
        self.alpha_pv = alpha_pv
        self.precond_diag = precond_diag
        self.fft_grid = fft_grid          # static (tuple of ints)

    def tree_flatten(self):
        children = (self.T_diag, self.V_scf, self.Gx, self.Gy, self.Gz,
                    self.vnl_Z, self.vnl_E, self.mask,
                    self.U_val, self.eps_v, self.alpha_pv, self.precond_diag)
        aux = (self.fft_grid,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (fft_grid,) = aux
        return cls(*children, fft_grid=fft_grid)


# ═══════════════════════════════════════════════════════════════════════
#  Apply A — inline inside the jitted core below
# ═══════════════════════════════════════════════════════════════════════

def _apply_A_inline(op: SternheimerOp, x: jax.Array) -> jax.Array:
    """Sternheimer operator matvec, using ``SternheimerOp``'s arrays.

    A_v(x) = H_{k-q} · x − ε_{v,k} · x + α_pv · P_val^{k-q}(x)

    Written inline (not as its own jit'd fn) so the enclosing
    :func:`_sternheimer_core` has a single fused trace.
    """
    # H·x  — use the shared _apply_H kernel.  apply_H_k lives in dft_operators,
    # which in turn uses static fft_grid dims.  Since op.fft_grid is static
    # (aux pytree data), this re-jits if the grid changes but NOT per-k.
    nx, ny, nz = op.fft_grid
    mask_f = op.mask[None, None, :].astype(x.dtype)
    psi_box = jnp.zeros((*x.shape[:2], nx, ny, nz), dtype=x.dtype)
    psi_box = psi_box.at[:, :, op.Gx, op.Gy, op.Gz].add(x * mask_f)
    Hx = apply_H_k(psi_box, op.T_diag, op.V_scf,
                   op.Gx, op.Gy, op.Gz,
                   op.vnl_Z, op.vnl_E, op.mask)
    # H·x − ε_v·x
    shifted = Hx - op.eps_v[:, None, None].astype(x.dtype) * x
    # α_pv · P_val(x)
    coefs = jnp.einsum('msG,bsG->bm', jnp.conj(op.U_val), x, optimize=True)
    Pv_x = jnp.einsum('bm,msG->bsG', coefs, op.U_val, optimize=True)
    return shifted + op.alpha_pv * Pv_x


def _precond_inline(op: SternheimerOp, r: jax.Array) -> jax.Array:
    """TPA diagonal preconditioner inline."""
    return r * op.precond_diag


# ═══════════════════════════════════════════════════════════════════════
#  CG core — fused JIT, reused across all (k, q) with matching shapes
# ═══════════════════════════════════════════════════════════════════════

def _batched_dot(a, b):
    return jnp.einsum('vsG,vsG->v', jnp.conj(a), b, optimize=True)


def _batched_real_norm(a):
    return jnp.sqrt(jnp.real(_batched_dot(a, a)))


@functools.partial(jax.jit, static_argnames=('max_iter',))
def _sternheimer_core(op: SternheimerOp, b: jax.Array, tol: float, max_iter: int) -> jax.Array:
    """Level-shifted preconditioned CG for  A_v · δu = −b.

    Returns δu of the same shape as ``b``.  The solution lies in range(Q_{k-q})
    automatically when ``b`` does (A is block-diagonal on the
    range(Q) ⊕ range(P_val) decomposition).

    Per-band convergence freeze (mirrors QE's ``conv(ibnd)``): once a band's
    residual drops below tol·‖b‖, subsequent iterations don't update it.
    Dead-band mask: if ‖b‖ ≈ 0 for some v, that row stays at 0 throughout
    (avoids the MINRES-style noise amplification).

    RHS sign convention: we solve A·δu = −b where b is the CONSTRUCTED source
    Q_{k-q}·V_pert·u_{v,k} (its natural sign).  The caller does not negate b.
    """
    dtype_c = b.dtype
    dtype_r = jnp.float64 if dtype_c == jnp.complex128 else jnp.float32
    batch = b.shape[0]

    rhs = -b                                            # actual RHS for CG
    x0 = jnp.zeros_like(b)
    r0 = rhs - _apply_A_inline(op, x0)                  # = rhs
    z0 = _precond_inline(op, r0)
    rho0 = jnp.real(_batched_dot(r0, z0))               # ≥ 0 for HPD M
    b_norm = _batched_real_norm(rhs)
    b_norm_safe = jnp.where(b_norm > 0, b_norm, 1.0)

    # Dead-band: if ‖b‖ ≈ 0 the solution is x=0 and running CG on noise
    # would blow up.  Mask out those rows from the start.
    _DEAD = jnp.asarray(1e-14, dtype=dtype_r)
    alive0 = (b_norm > _DEAD)

    def body(_i, state):
        x, r, z, p, rho, alive = state

        Ap = _apply_A_inline(op, p)
        pAp = jnp.real(_batched_dot(p, Ap))
        eps_r = jnp.asarray(jnp.finfo(dtype_r).eps, dtype=dtype_r)
        pAp_safe = jnp.where(pAp > eps_r, pAp, 1.0)
        alpha = jnp.where(alive, rho / pAp_safe, 0.0)

        alpha_c = alpha[:, None, None].astype(dtype_c)
        x_new = x + alpha_c * p
        r_new = r - alpha_c * Ap

        # Per-band freeze: drop from the "alive" set once ‖r‖ < tol·‖b‖.
        r_new_norm = _batched_real_norm(r_new)
        still_alive = alive & (r_new_norm > tol * b_norm_safe)

        z_new = _precond_inline(op, r_new)
        rho_new = jnp.real(_batched_dot(r_new, z_new))
        rho_safe = jnp.where(rho > eps_r, rho, 1.0)
        beta = jnp.where(alive, rho_new / rho_safe, 0.0)
        beta_c = beta[:, None, None].astype(dtype_c)
        p_new = z_new + beta_c * p

        return (x_new, r_new, z_new, p_new, rho_new, still_alive)

    state0 = (x0, r0, z0, z0, rho0, alive0)
    x_final, *_ = lax.fori_loop(0, max_iter, body, state0)
    return x_final


# ═══════════════════════════════════════════════════════════════════════
#  Public primitive with custom JVP
# ═══════════════════════════════════════════════════════════════════════

@functools.partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def sternheimer_solve(op: SternheimerOp, b: jax.Array,
                      tol: float = 1e-6, max_iter: int = 100) -> jax.Array:
    """Solve  A_v · δu = −b  via level-shifted CG.  Single-compile.

    Primal + custom JVP, so ``jax.jvp`` / ``jax.jacfwd`` work without
    autodiffing through the CG iterations.  The derivative of δu with respect
    to any element of ``op`` or ``b`` is obtained by differentiating the
    converged linear equation implicitly:

        A ẋ = ḃ − Ȧ x

    which is the same operator A — one extra CG solve with the right RHS.

    Parameters
    ----------
    op : SternheimerOp
        Operator pytree (H arrays, U_val, ε_v, α_pv, precond_diag).
    b : (nv, nspinor, nG) complex
        Source (will be negated internally to form the CG RHS).
    tol : float — convergence tolerance.
    max_iter : int — static iteration cap.
    """
    return _sternheimer_core(op, b, tol, max_iter)


@sternheimer_solve.defjvp
def _sternheimer_solve_jvp(tol, max_iter, primals, tangents):
    """JVP rule via implicit differentiation.

    Let  A x = b  with  x = sternheimer_solve(op, b).  Then

        A · ẋ = ḃ − Ȧ · x
                      └──── linearise apply_A(op, x) wrt op, at fixed x ────┘

    so ``ẋ`` is obtained by calling the primitive again with the tangent RHS.
    """
    op, b = primals
    op_dot, b_dot = tangents

    # Primal solve.
    x = _sternheimer_core(op, b, tol, max_iter)

    # Compute Ȧ · x  via jax.jvp of _apply_A_inline w.r.t. op at fixed x.
    _, A_dot_x = jax.jvp(
        lambda o: _apply_A_inline(o, x),
        (op,),
        (op_dot,),
    )

    # Effective RHS for the tangent solve:  rhs_tangent = ḃ − Ȧ · x
    # The primitive negates its input, so we pass  +(ḃ − Ȧ·x)  and it
    # internally solves A·ẋ = −(ḃ − Ȧ·x); but the WANTED equation is
    # A·ẋ = ḃ − Ȧ·x, so we instead pass the NEGATED quantity.
    rhs_tangent = -(b_dot - A_dot_x)                    # fed to _sternheimer_core which negates → + (ḃ − Ȧx)
    x_dot = _sternheimer_core(op, rhs_tangent, tol, max_iter)

    return x, x_dot


__all__ = [
    "SternheimerOp",
    "sternheimer_solve",
]
