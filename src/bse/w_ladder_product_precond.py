"""Product-form (Casida-style) preconditioner for the ladder screening solve.

WHY THIS EXISTS.  The ladder resolvent solves ``(z - H) x = b`` at ``z = 0``,
an INTERIOR point of the symplectic +/- spectrum (scalar-Si nband=20: gap
radius ~5e-2 Ry against a spectral radius ~2.6 Ry).  The exact-diagonal
preconditioner leaves the two hard irreducible q blocks at 242 (band-12 fit)
to 436-440 (band-14 fit) GMRES iterations — a 440-vector Krylov basis in an
8192-dim pair space, which is both the wall-time and the memory problem.
RPA never sees this because ``A - B = D`` makes the Casida ``z^2`` reduction
exact and definite; the ladder rung broke that identity, and the production
solver fell back to unstructured GMRES on the indefinite two-sided problem.

THE DERIVATION (record, not sketch).  The ladder operator has the hybrid
anti-resonant row (see ``bse_ring_comm``'s row derivation and claim 0242):

    H = [[ A,  B ], [ -Bt, -At ]],   A  = D + V - W_d,      B  = V - W_d^B,
                                     At = D + V - conj(W_d), Bt = V - conj(W_d^B)

with ``V`` the Hermitian ring kernel, ``W_d`` real-symmetric to O(Im W_R) and
``W_d^B`` complex-symmetric with the same tiny imaginary scale (measured
``max|Im diag| = 2.8e-6 Ry`` on the nband=20 deck — the solver-hygiene
report).  In the half-sum coordinates ``u = x + y, v = x - y``:

    H  ~=  [[ i e_plus,  P ],       P = D - Re(W_d) + Re(W_d^B)
            [ Q, i e_minus ]],      Q = D + 2 V - Re(W_d) - Re(W_d^B)

    e_plus/minus = -(Im W_d +/- Im W_d^B)  -- O(1e-6 Ry), DROPPED by M.

Both ``P`` and ``Q`` are Hermitian (D real diagonal, V Hermitian, the real
parts of the rung kernels real-symmetric).  Dropping the epsilon blocks, at
``z = 0``:

    H^{-1}  ~=  T^{-1} [[0, Q^{-1}], [P^{-1}, 0]] T,     T: (x,y) -> (u,v)

so ONE preconditioner application is: form (r_u, r_v) = (r_x + r_y,
r_x - r_y); approximately solve ``Q u = r_v`` and ``P v = r_u`` (Hermitian
inner solves); return x = (u+v)/2, y = (u-v)/2.  The preconditioned outer
operator differs from the identity by two named error terms only:

  * the dropped epsilon blocks, relative size ``O(|Im W| / gap)`` ~ 1e-4;
  * the inner-solve inexactness, which a FLEXIBLE outer absorbs by
    construction (``w_ladder_precond.fgmres_solve_core`` stores Z).

MATRIX-FREE P AND Q AT ONE MATVEC EACH.  ``matvec([un ; s*un])`` returns
``[(A + sB)un ; -(At + sBt)un]``: the two half-sum factors are one full
production matvec each, using BOTH output blocks —

    Q un = (top - bottom)/2   at s = +1
    P un = (top + bottom)/2   at s = -1

(this is ``bse_nontda.make_ab_appliers`` extended to keep the second block;
the hybrid row makes the second block carry the tilde combinations, which is
exactly what the Re-projection needs).

INNER SOLVES.  Fixed-trip Jacobi-preconditioned CG (fori_loop, jit-stable,
``unroll=1``): ``P`` is diagonally dominant (D plus a small rung difference)
and converges in a few steps; ``Q`` carries the ring and rung spectrum and
sets the cost.  Fixed trip counts are hyperparameters of the PRECONDITIONER,
not accuracy knobs: the outer's true-residual acceptance is unchanged.
Definiteness of P and Q is PROBED (Lanczos bound) before trusting CG; an
indefinite finding is a refusal with the measured bound, not a silent
fallback.

MEMORY.  The point of the exercise: outer FGMRES basis ~<= 2 * n_outer
vectors plus CG's four working vectors, against the diagonal-preconditioned
baseline's ``max_iter + 1`` (441 at the current cap) — an order of magnitude
on the solve's high-water at equal pair-space width.

SCOPE.  Prototype quality, single-process probes; the production integration
(operand plumbing, P>1 legs, cache keys) is the follow-on and goes through
the same ``fgmres_solve_core`` seam unchanged.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def make_half_sum_appliers(matvec, operands, sh):
    """(apply_P, apply_Q): one full matvec each, both output blocks used."""

    def _apply(un, sign):
        stacked = jax.lax.with_sharding_constraint(
            jnp.stack([un, sign * un], axis=0).astype(jnp.complex128),
            sh.X_full)
        out = matvec(stacked, *operands)
        top, bottom = out[0], out[1]
        if sign > 0:                          # (A+B)u ; -(At+Bt)u
            return 0.5 * (top - bottom)
        return 0.5 * (top + bottom)           # (A-B)u ; +(At-Bt)u

    return (lambda u: _apply(u, -1.0)), (lambda u: _apply(u, 1.0))


def _jacobi_cg(apply_op, diag_op, rhs, n_steps):
    """Fixed-trip Jacobi-preconditioned CG for a Hermitian operator.

    Fixed trips keep the body jit-stable inside the outer while_loop; the
    flexible outer absorbs the fixed-accuracy result.  ``diag_op`` is the
    operator's diagonal (real), clamped away from zero so a tiny entry
    cannot poison the scaling.
    """
    d = jnp.maximum(jnp.abs(diag_op.real), 1e-8)

    def prec(r):
        return r / d

    nrm = jnp.linalg.norm(rhs)
    safe = nrm > 0.0

    def _sdiv(a, b):
        return a / jnp.where(jnp.abs(b) > 0.0, b, 1.0)

    x = jnp.zeros_like(rhs)
    r = rhs
    zv = prec(r)
    p = zv
    rz = jnp.vdot(r, zv)

    def body(_, carry):
        x, r, p, rz = carry
        Ap = apply_op(p)
        alpha = _sdiv(rz, jnp.vdot(p, Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        zn = prec(r)
        rz_new = jnp.vdot(r, zn)
        beta = _sdiv(rz_new, rz)
        p = zn + beta * p
        return x, r, p, rz_new

    x, r, p, rz = jax.lax.fori_loop(0, n_steps, body, (x, r, p, rz),
                                    unroll=1)
    return jnp.where(safe, x, jnp.zeros_like(x))


def _fixed_gmres(apply_op, diag_op, rhs, m):
    """Fixed-trip inner GMRES(m), Jacobi-scaled — no Hermitian assumption.

    The finite-q half-sum factors are measurably non-normal (the hybrid row
    couples q with -q rather than elementwise-conjugating), so the inner
    engine must not assume CG's symmetry.  Fixed m, small dense lstsq at the
    end; zero-rhs safe.
    """
    d = jnp.maximum(jnp.abs(diag_op.real), 1e-8)
    b = rhs / d
    nrm = jnp.linalg.norm(b)
    safe = nrm > 0.0
    v0 = jnp.where(safe, b / jnp.where(safe, nrm, 1.0), b)
    V = jnp.zeros((m + 1,) + rhs.shape, dtype=rhs.dtype).at[0].set(v0)
    H = jnp.zeros((m + 1, m), dtype=rhs.dtype)

    def body(k, carry):
        V, H = carry
        w = apply_op(V[k]) / d

        def arnoldi(i, c):
            w_l, H_l = c
            h = jnp.vdot(V[i], w_l)
            return w_l - h * V[i], H_l.at[i, k].set(h)

        w, H = jax.lax.fori_loop(0, k + 1, arnoldi, (w, H), unroll=1)
        hn = jnp.linalg.norm(w)
        H = H.at[k + 1, k].set(hn)
        V = V.at[k + 1].set(jnp.where(hn > 0.0, w / jnp.where(hn > 0.0, hn, 1.0), w))
        return V, H

    V, H = jax.lax.fori_loop(0, m, body, (V, H), unroll=1)
    g = jnp.zeros((m + 1,), dtype=rhs.dtype).at[0].set(nrm)
    y = jnp.linalg.lstsq(H, g, rcond=None)[0]
    x = jnp.tensordot(y, V[:m], axes=(0, 0))
    return jnp.where(safe, x, jnp.zeros_like(x))


def make_product_preconditioner(matvec, sh, *, n_cg_p: int = 6,
                                n_cg_q: int = 24, inner: str = "gmres"):
    """The ``precond(v, z, precond_args)`` callable for ``fgmres_solve_core``.

    ``precond_args = (operands, dref)`` — the matvec's runtime operand tuple
    and the REAL resonant-block diagonal used for both inner Jacobi scalings
    (the diagonals of P and Q differ from ``Re(diag_h[0])`` only by rung
    diagonal terms already inside it; a preconditioner-grade approximation,
    stated rather than hidden).  Runtime arrays ride the args pytree, never a
    closure — the seam's P>1 contract.
    """

    def precond(vec, z, precond_args):
        del z
        operands, dref = precond_args
        apply_P, apply_Q = make_half_sum_appliers(matvec, operands, sh)
        r_x, r_y = vec[0], vec[1]
        r_u = r_x + r_y
        r_v = r_x - r_y
        solver = _fixed_gmres if inner == "gmres" else _jacobi_cg
        u = solver(apply_Q, dref, r_v, n_cg_q)
        v = solver(apply_P, dref, r_u, n_cg_p)
        x = 0.5 * (u + v)
        y = 0.5 * (u - v)
        return jnp.stack([x, y], axis=0)

    return precond


def lanczos_extremal_bound(apply_op, probe, n_steps: int = 24):
    """(min, max) Ritz bounds of a Hermitian operator from plain Lanczos.

    Definiteness probe only — full reorthogonalization is deliberately
    omitted at this depth; bounds are used as a refusal gate, not physics.
    """
    v = probe / jnp.linalg.norm(probe)
    alphas, betas = [], []
    v_prev = jnp.zeros_like(v)
    beta = jnp.asarray(0.0, dtype=jnp.float64)
    for _ in range(n_steps):
        w = apply_op(v) - beta * v_prev
        alpha = jnp.vdot(v, w).real
        w = w - alpha * v
        alphas.append(alpha)
        beta_new = jnp.linalg.norm(w)
        betas.append(beta_new)
        v_prev, v, beta = v, w / jnp.maximum(beta_new, 1e-30), beta_new
    import numpy as np
    T = np.diag(np.asarray(alphas))
    off = np.asarray(betas[:-1])
    T += np.diag(off, 1) + np.diag(off, -1)
    ev = np.linalg.eigvalsh(T)
    return float(ev[0]), float(ev[-1])


def make_single_block_appliers(matvec, operands, sh):
    """(apply_A, apply_At, apply_B): the individual blocks, one matvec each.

    ``matvec([u;0]) = [A u ; -Bt u]`` and ``matvec([0;u]) = [B u ; -At u]`` —
    the TDA-Schur preconditioner needs A, At and the coupling B separately.
    """

    def _lift(un, row):
        z = jnp.zeros_like(un)
        pair = (un, z) if row == 0 else (z, un)
        stacked = jax.lax.with_sharding_constraint(
            jnp.stack(pair, axis=0).astype(jnp.complex128), sh.X_full)
        return matvec(stacked, *operands)

    def apply_A(u):
        return _lift(u, 0)[0]

    def apply_At(u):
        return -_lift(u, 1)[1]

    def apply_B(u):
        return _lift(u, 1)[0]

    return apply_A, apply_At, apply_B


def make_tda_schur_preconditioner(matvec, sh, *, n_in_a: int = 12,
                                  n_in_at: int = 12, inner: str = "gmres"):
    """Block-triangular TDA-Schur M^{-1} for the symplectic interior shift.

    The +/- indefiniteness of ``z - H`` at interior ``z`` lives entirely in
    the block structure: ``A`` (and ``At``) are gap-bounded DEFINITE
    operators.  Preconditioning with the block triangle

        M = [[ A - z,  B ], [ 0,  -At - z ]]

    costs two definite TDA-class inner solves plus one coupling multiply per
    application, and moves the outer convergence onto the coupling strength
    ``||B||/gap`` instead of the two-sided spread — the classic saddle cure.
    Inner solves are fixed-trip and loose; the flexible outer absorbs it.

    Apply: solve ``(-At - z) y = r_y``; then ``(A - z) x = r_x - B y``.
    """

    def precond(vec, z, precond_args):
        operands, dref = precond_args
        apply_A, apply_At, apply_B = make_single_block_appliers(
            matvec, operands, sh)
        solver = _fixed_gmres if inner == "gmres" else _jacobi_cg
        r_x, r_y = vec[0], vec[1]

        def op_y(u):
            return -apply_At(u) - z * u

        y = solver(op_y, dref, r_y, n_in_at)

        def op_x(u):
            return apply_A(u) - z * u

        x = solver(op_x, dref, r_x - apply_B(y), n_in_a)
        return jnp.stack([x, y], axis=0)

    return precond
