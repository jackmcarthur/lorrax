"""Experimental sample-only tangential rational Krylov, outside the GW driver.

Implements iterative_rk_jack.md §§3–9 for F(s)=B†(sI-T)^-1 B.  T is
Hermitian positive semidefinite and s is in Ry².  The physical screened
interaction is W-v=2F; exported physical factors are sqrt(2) times Ritz
readouts.  Complex supports still give a Hermitian Ritz model.

Pair arrays have a leading RHS axis; callers retain their named sharding.
Only the shifted solver sees pair space.  Reduced data are plain arrays,
preallocated by the caller; no model-order truncation or pole repair occurs.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl


def screening_operands(data):
    """Pack the existing RPA operands and positive transition square root.

    Delta is in Ry. Zero padded transitions retain zero vertices; negative
    physical transitions must be refused by the deck gate before this call.
    """
    delta = data['eps_c'].T[None, :, None, :] - data['eps_v'].T[None, None, :, :]
    half = jnp.sqrt(jnp.maximum(delta.real, 0)).astype(jnp.complex128)
    args = tuple(data[k] for k in ('psi_c_X', 'psi_c_Y', 'psi_v_X', 'psi_v_Y',
                                   'eps_c', 'eps_v', 'W_R', 'V_q0', 'M_X', 'M_Y'))
    return half, args


def build_screening_actions(matvec, gen, snapshot, sh, *, nk, nspinor):
    """Reuse the production ring for T, B, B†, with balanced spin vertices.

    T is exactly w_omega_chain's D_half H_RPA[D_half x;D_half x]_X.
    Production snapshot includes spin degeneracy g while gen does not:
    B=sqrt(g)D_half gen and B†=snapshot D_half/sqrt(g). Therefore W-v=2F.
    q=(Nr,r), x=(r,c,v,k); q rows must use the loader's centroid order.
    Runtime operands prevent freezing per-q wavefunctions in compiled code.
    """
    from bse.bse_ring_comm import ring_spin_degeneracy
    root_g = ring_spin_degeneracy(nspinor) ** .5

    @jax.jit
    def apply_t(x, operands):
        half, args = operands
        u = jax.lax.with_sharding_constraint(half * x, sh.X)
        doubled = jax.lax.with_sharding_constraint(jnp.stack((u, u)), sh.X_full)
        return jax.lax.with_sharding_constraint(half * matvec(doubled, *args)[0], sh.X)

    @jax.jit
    def apply_b(q, operands):
        half, args = operands
        drive = jax.lax.with_sharding_constraint(jnp.broadcast_to(q.T[:, :, None],
                                                   (q.shape[1], q.shape[0], nk)), sh.S)
        return jax.lax.with_sharding_constraint(root_g * half * gen(drive, args[0], args[2], args[7]), sh.X)

    @jax.jit
    def apply_bh(x, operands):
        half, args = operands
        return snapshot(half * x / root_g, args[1], args[3], args[7])

    return apply_t, apply_b, apply_bh


def column_cg(apply_t, rhs, shift, *, maxiter, tol):
    """Solve (T-shift I)x=rhs by batched independent Hermitian CG.

    Parameters
    ----------
    apply_t : callable
        Matrix-free T action, preserving rhs shape and named sharding.
    rhs : jax.Array
        Complex128 (r, *pair_shape), pair dimensions distributed on x/y.
    shift : scalar
        Real negative squared frequency in Ry²; caller validates this.
    maxiter : int
        Compile-time bound. Converged columns have zero search directions.
    tol : float
        Relative Euclidean residual tolerance per nonzero RHS.

    Returns
    -------
    x, receipt : tuple
        Solution and small arrays counting iterations, useful/issued matvec
        columns, final true residuals, and breakdowns. The final residual
        matvec is counted. This is column-CG, not a block-Gram CG recurrence.
    """
    axes = tuple(range(1, rhs.ndim))
    broad = (rhs.shape[0],) + (1,) * len(axes)

    def dot(a, b):
        return jnp.sum(jnp.conj(a) * b, axis=axes).real

    b2 = dot(rhs, rhs)
    threshold = tol * tol * b2
    live = b2 > 0
    initial = (jnp.zeros_like(rhs), rhs, rhs, b2, live,
               jnp.zeros(rhs.shape[0], jnp.int32), jnp.int32(0),
               jnp.int32(0), jnp.zeros_like(live))

    def step(_, state):
        def advance(state):
            x, r, p, rr, live, iterations, useful, issued, broken = state
            ap = apply_t(p) - shift * p
            pap = dot(p, ap)
            bad = live & ((pap <= 0) | ~jnp.isfinite(pap))
            moving = live & ~bad
            alpha = jnp.where(moving, rr / jnp.where(moving, pap, 1), 0)
            x = x + alpha.reshape(broad) * p
            r = r - alpha.reshape(broad) * ap
            rr_new = dot(r, r)
            next_live = moving & (rr_new > threshold)
            beta = jnp.where(next_live, rr_new / jnp.where(rr > 0, rr, 1), 0)
            p = jnp.where(next_live.reshape(broad), r + beta.reshape(broad) * p, 0)
            return (x, r, p, rr_new, next_live, iterations + live,
                    useful + jnp.sum(live, dtype=jnp.int32),
                    issued + rhs.shape[0], broken | bad)

        return jax.lax.cond(jnp.any(state[4]), advance, lambda x: x, state)

    x, _, _, _, _, iterations, useful, issued, broken = jax.lax.fori_loop(
        0, maxiter, step, initial)
    residual = rhs - (apply_t(x) - shift * x)
    relative = jnp.sqrt(dot(residual, residual) / jnp.where(b2 > 0, b2, 1))
    return x, dict(iterations=iterations, relative=relative, breakdown=broken,
                   matvec_columns=issued + rhs.shape[0],
                   useful_matvec_columns=useful + jnp.sum(b2 > 0),
                   rhs_solves=jnp.sum(b2 > 0))


def empty_samples(n_ports, k_max):
    """Allocate §2's reduced buffers (complex128) with zero inactive columns."""
    return dict(xi=jnp.zeros(k_max, jnp.complex128),
                Q=jnp.zeros((n_ports, k_max), jnp.complex128),
                Y=jnp.zeros((n_ports, k_max), jnp.complex128),
                S=jnp.zeros((k_max, k_max), jnp.complex128),
                H=jnp.zeros((k_max, k_max), jnp.complex128),
                active=jnp.zeros(k_max, bool), m=jnp.int32(0))


def append_samples(state, shift, q, y, gram):
    """Insert §4 cross terms and the directly contracted confluent Gram block.

    q/y are (Nr,r), gram is Xnew†Xnew (r,r); state uses maximum shapes.
    Every accepted block must use a support distinct from all older supports.
    Reused-support detection is returned as a refusal flag, never regularized.
    """
    m, active = state['m'], state['active']
    zero = jnp.int32(0)
    den = shift - state['xi'].conj()
    reused = jnp.any(active & (den == 0))
    a = state['Y'].conj().T @ q
    b = state['Q'].conj().T @ y
    cross_s = jnp.where(active[:, None], (a - b) / jnp.where(den != 0, den, 1)[:, None], 0)
    cross_h = shift * cross_s - a
    update = jax.lax.dynamic_update_slice
    s = update(state['S'], cross_s, (zero, m))
    s = update(s, cross_s.conj().T, (m, zero))
    s = update(s, gram, (m, m))
    h = update(state['H'], cross_h, (zero, m))
    h = update(h, cross_h.conj().T, (m, zero))
    hnew = shift * gram - y.conj().T @ q
    h = update(h, (hnew + hnew.conj().T) * .5, (m, m))
    out = dict(xi=update(state['xi'], jnp.full(q.shape[1], shift), (m,)),
               Q=update(state['Q'], q, (zero, m)), Y=update(state['Y'], y, (zero, m)),
               S=s, H=h,
               active=update(active, jnp.ones(q.shape[1], bool), (m,)),
               m=m + q.shape[1])
    return out, reused


def masked_factor(state, s):
    """Factor §13's P(sS-H)P+(I-P), with diagonal Gram equilibration."""
    active = state['active']
    diag = jnp.real(jnp.diag(state['S']))
    scale = jnp.where(active, 1 / jnp.sqrt(jnp.where(diag > 0, diag, 1)), 1)
    mask = active[:, None] & active[None, :]
    pencil = jnp.where(mask, (s * state['S'] - state['H']) * scale[:, None] * scale[None, :], 0)
    pencil = pencil + jnp.diag((~active).astype(pencil.dtype))
    return jsl.lu_factor(pencil), scale


def residual_action(state, g0, s, v, factor):
    """Apply exact sample-only R†R to port block v, per note §8.

    Reduced adjoint solves reuse the same LU factorization. No pair array is
    formed. Arrays Q/Y=(Nr,K), S/H=(K,K), G0=(Nr,Nr), v=(Nr,r).
    """
    lu, scale = factor

    def solve(rhs, adjoint=False):
        return scale[:, None] * jsl.lu_solve(lu, scale[:, None] * rhs,
                                             trans=2 if adjoint else 0)

    q, y, gram, xi = state['Q'], state['Y'], state['S'], state['xi']
    c = solve(y.conj().T @ v)
    a = v - q @ c
    d = (s - xi)[:, None] * c
    t = g0 @ a - y @ d
    u = y.conj().T @ a - gram @ d
    return t - y @ solve(q.conj().T @ t + (s.conjugate() - xi.conj())[:, None] * u,
                          adjoint=True)


def residual_tangents(state, g0, s, whiten, seed, *, n_power):
    """§9 fixed-count subspace iteration for R†R q=lambda G0 q.

    whiten=(Nr,Nr) is precomputed G0^-1/2, or identity for the absolute arm.
    seed=(Nr,r) fixes the trial block. Returns Euclidean-orthonormal tangents
    and the largest residual Ritz value; the eigensystem is only r by r.
    """
    factor = masked_factor(state, s)

    def action(v):
        return whiten.conj().T @ residual_action(state, g0, s, whiten @ v, factor)

    def power(_, v):
        return jnp.linalg.qr(action(v), mode='reduced')[0]

    v = jax.lax.fori_loop(0, n_power, power, jnp.linalg.qr(seed, mode='reduced')[0])
    small = v.conj().T @ action(v)
    values, vectors = jnp.linalg.eigh((small + small.conj().T) * .5)
    q = whiten @ v @ vectors[:, ::-1]
    return jnp.linalg.qr(q, mode='reduced')[0], values[-1]


def log_support_indicator(grid, xi, theta, active):
    """§6 scalar log chi(s), excluding inactive slots, one vmap over supports."""
    def one(s):
        return jnp.sum(jnp.where(active, jnp.log(jnp.abs(s-xi)) - jnp.log(jnp.abs(s-theta)), 0))
    return jax.vmap(one)(grid)


def ritz_model(state):
    """Whiten the full active Gram without selecting or repairing any state.

    Returns padded Ritz values, physical factors sqrt(2)Y U, and the active
    equilibrated Gram eigenvalues. Nonpositive Gram values propagate NaNs:
    the host gate must refuse. Inactive eigenvalues are placed above the
    active spectrum, so the first m entries correspond to actual states.
    """
    active = state['active']
    mask = active[:, None] & active[None, :]
    diag = jnp.real(jnp.diag(state['S']))
    scale = jnp.where(active, 1 / jnp.sqrt(jnp.where(diag > 0, diag, 1)), 1)
    gram = jnp.where(mask, state['S'] * scale[:, None] * scale[None, :], 0)
    gram = gram + jnp.diag((~active) * (state['S'].shape[0] + 1.))
    g, u = jnp.linalg.eigh(gram)
    w = scale[:, None] * u / jnp.sqrt(g)[None, :]
    h = w.conj().T @ state['H'] @ w
    live = jnp.arange(g.size) < state['m']
    bound = jnp.linalg.norm(h) + 1.
    h = jnp.where(live[:, None] & live[None, :], h, 0) + jnp.diag((~live) * bound)
    theta, v = jnp.linalg.eigh((h + h.conj().T) * .5)
    factor = jnp.sqrt(2.) * state['Y'] @ w @ v
    return theta, factor * live[None, :], g
