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

from psp.dft_operators import apply_H_k_from_G
from solvers.projectors import expand_subspace, subspace_coefficients


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

    Schur initial-guess fields (all None if unused — plain CG starts from zero):
    U_extra        : (M, nspinor, nG) or None — low-energy Ritz / conduction
                     vectors at k-q, orthonormal and orthogonal to U_val
    eps_extra      : (M,) or None             — eigenvalues of H_{k-q} on U_extra
    """
    __slots__ = ("T_diag", "V_scf", "Gx", "Gy", "Gz",
                 "vnl_Z", "vnl_E", "mask",
                 "U_val", "eps_v", "alpha_pv", "precond_diag",
                 "U_extra", "eps_extra",
                 "fft_grid")

    def __init__(
        self, T_diag, V_scf, Gx, Gy, Gz, vnl_Z, vnl_E, mask,
        U_val, eps_v, alpha_pv, precond_diag, fft_grid,
        U_extra=None, eps_extra=None,
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
        self.U_extra = U_extra              # may be None
        self.eps_extra = eps_extra          # may be None
        self.fft_grid = fft_grid            # static (tuple of ints)

    def tree_flatten(self):
        # None-valued Schur fields become empty leaves for pytree compat.
        children = (self.T_diag, self.V_scf, self.Gx, self.Gy, self.Gz,
                    self.vnl_Z, self.vnl_E, self.mask,
                    self.U_val, self.eps_v, self.alpha_pv, self.precond_diag,
                    self.U_extra, self.eps_extra)
        aux = (self.fft_grid, self.U_extra is None)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (fft_grid, schur_off) = aux
        # Preserve the None when rebuilding after a pytree transform.
        (*core, U_ext, eps_ext) = children
        if schur_off:
            U_ext = None; eps_ext = None
        return cls(*core, fft_grid=fft_grid, U_extra=U_ext, eps_extra=eps_ext)


# ═══════════════════════════════════════════════════════════════════════
#  Apply A — inline inside the jitted core below
# ═══════════════════════════════════════════════════════════════════════

def _apply_A_inline(op: SternheimerOp, x: jax.Array) -> jax.Array:
    """Sternheimer operator matvec, using ``SternheimerOp``'s arrays.

    A_v(x) = H_{k-q} · x − ε_{v,k} · x + α_pv · P_val^{k-q}(x)

    Written inline (not as its own jit'd fn) so the enclosing
    :func:`_sternheimer_core` has a single fused trace.
    """
    # H·x via the G-sphere entry point — one scatter (for FFT path) instead
    # of the scatter→gather round-trip the box-input ``apply_H_k`` forces.
    Hx = apply_H_k_from_G(x, op.T_diag, op.V_scf,
                           op.Gx, op.Gy, op.Gz,
                           op.vnl_Z, op.vnl_E, op.mask)
    # H·x − ε_v·x
    shifted = Hx - op.eps_v[:, None, None].astype(x.dtype) * x
    # α_pv · P_val(x)
    coefs = subspace_coefficients(op.U_val, x)
    Pv_x = expand_subspace(op.U_val, coefs)
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


def cond_subspace_sos_solve(op: SternheimerOp, b: jax.Array) -> jax.Array:
    """Explicit T-block solution = sum-over-states formula restricted to the
    cond-Ritz subspace ``op.U_extra``.

    Equivalent to using the projector  P_cond_N = Σ_{m<N} |U_m⟩⟨U_m|  in
    place of  Q_{k-q} = 1 − P_val  in the Sternheimer source projection.
    Returns the closed-form δu inside that subspace; no CG iteration.

    Used in two contexts:
      1. As a CG warm-start (see :func:`_sternheimer_core` ``use_schur``).
      2. As the *only* solve when comparing how fast each component of
         χ_{G'0}(q) converges with the truncated-cond-N basis size — set
         ``sos_only=True`` on the chi pipeline.
    """
    return _schur_initial_guess(op, b)


def _schur_initial_guess(op: SternheimerOp, b: jax.Array) -> jax.Array:
    """Explicit T-block solution (= standard sum-over-states formula over the
    loaded extra-Ritz subspace) used as the CG initial guess.

    Since U_extra are Ritz vectors of H_{k-q} with eigenvalues eps_extra, and
    they're orthogonal to U_val (so α_pv·P_val·U_extra = 0), the T-block of
    the level-shifted operator A = H_{k-q} − ε_{v,k} + α_pv·P_val is diagonal:

        A_TT[m, m'] = (eps_extra[m] − eps_v) · δ_{m, m'}

    and the T-block of the solution   A · x = −b   reads

        x^T_v(r) = −Σ_m  ⟨U_m | b_v⟩_cell / (eps_extra[m] − eps_v[v]) · U_m(r)

    That's the k·p / SoS formula applied to the first M ``extra`` bands.
    CG from this initial guess then only has to correct the residual beyond
    those M bands — the hard (low-energy / small-denominator) part is
    handled explicitly.
    """
    # coefs[v, m] = ⟨U_m | b_v⟩_cell = Σ_{s,G} conj(U_extra[m,s,G]) · b[v,s,G]
    coefs = subspace_coefficients(op.U_extra, b)
    # Energy denominator  (eps_extra[m] − eps_v[v])   (M, ) − (v,) → (v, M)
    denom = op.eps_extra[None, :] - op.eps_v[:, None]                 # (batch, M)
    # Guard against near-zero denominator (crossings between extra-Ritz
    # and occupied energies).  For a proper insulator this never happens,
    # but include a clamp so a degenerate band doesn't blow up.
    denom_safe = jnp.where(jnp.abs(denom) > 1e-8, denom, 1.0)
    y = jnp.where(jnp.abs(denom) > 1e-8, -coefs / denom_safe, 0.0)     # (batch, M)
    # x0[v,s,G] = Σ_m y[v,m] · U_m[s,G]
    return expand_subspace(op.U_extra, y)


@functools.partial(jax.jit, static_argnames=('max_iter', 'use_schur'))
def _sternheimer_core(op: SternheimerOp, b: jax.Array,
                      tol: float, max_iter: int,
                      use_schur: bool = False) -> jax.Array:
    """Level-shifted preconditioned CG for  A_v · δu = −b.

    Returns δu of the same shape as ``b``.  The solution lies in range(Q_{k-q})
    automatically when ``b`` does (A is block-diagonal on the
    range(Q) ⊕ range(P_val) decomposition).

    When ``use_schur`` is True the CG is warm-started from the explicit
    T-block solution over op.U_extra (see :func:`_schur_initial_guess`).
    Equivalent to the Cancès-et-al. Schur split in initial-guess form
    (guide §3 final paragraph).  Reduces CG iteration count for systems
    where small-denominator low-energy conduction modes dominate the
    spectrum of the iterate.

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
    if use_schur:
        x0 = _schur_initial_guess(op, b)
    else:
        x0 = jnp.zeros_like(b)
    r0 = rhs - _apply_A_inline(op, x0)                  # = rhs for x0=0
    z0 = _precond_inline(op, r0)
    rho0 = jnp.real(_batched_dot(r0, z0))               # ≥ 0 for HPD M
    b_norm = _batched_real_norm(rhs)
    b_norm_safe = jnp.where(b_norm > 0, b_norm, 1.0)

    # Dead-band: if ‖b‖ ≈ 0 the solution is x=0 and running CG on noise
    # would blow up.  Mask out those rows from the start.
    _DEAD = jnp.asarray(1e-14, dtype=dtype_r)
    alive0 = (b_norm > _DEAD)

    def body(state):
        x, r, z, p, rho, alive, i = state

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

        return (x_new, r_new, z_new, p_new, rho_new, still_alive, i + 1)

    def cond(state):
        # Continue as long as ANY band is still un-converged AND we haven't
        # hit max_iter.  Switching from a fixed-iteration ``fori_loop`` to
        # this ``while_loop`` eliminates the ~50% of iterations that ran
        # purely on already-converged bands at QE-DFPT-style tolerances.
        _, _, _, _, _, alive, i = state
        return jnp.logical_and(jnp.any(alive), i < max_iter)

    state0 = (x0, r0, z0, z0, rho0, alive0, jnp.asarray(0, dtype=jnp.int32))
    final_state = lax.while_loop(cond, body, state0)
    return final_state[0]


# ═══════════════════════════════════════════════════════════════════════
#  Public primitive with custom JVP
# ═══════════════════════════════════════════════════════════════════════

@functools.partial(jax.custom_jvp, nondiff_argnums=(2, 3, 4))
def sternheimer_solve(op: SternheimerOp, b: jax.Array,
                      tol: float = 1e-6, max_iter: int = 100,
                      use_schur: bool = False) -> jax.Array:
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
    return _sternheimer_core(op, b, tol, max_iter, use_schur)


@sternheimer_solve.defjvp
def _sternheimer_solve_jvp(tol, max_iter, use_schur, primals, tangents):
    """JVP rule via implicit differentiation.

    Let  A x = b  with  x = sternheimer_solve(op, b).  Then

        A · ẋ = ḃ − Ȧ · x
                      └──── linearise apply_A(op, x) wrt op, at fixed x ────┘

    so ``ẋ`` is obtained by calling the primitive again with the tangent RHS.
    The tangent solve also uses Schur warm-start when ``use_schur`` is True —
    its T-block initial guess is the standard k·p formula applied to the
    differentiated RHS (same Ritz vectors, same denominators).
    """
    op, b = primals
    op_dot, b_dot = tangents

    # Primal solve.
    x = _sternheimer_core(op, b, tol, max_iter, use_schur)

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
    x_dot = _sternheimer_core(op, rhs_tangent, tol, max_iter, use_schur)

    return x, x_dot


__all__ = [
    "SternheimerOp",
    "sternheimer_solve",
]
