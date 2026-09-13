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

# Numerical oscillator-strength support for §9 whitening. This fixed support
# guard never selects the delivered rational-model order.
METRIC_SUPPORT_RTOL = 1e-3


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


def empty_samples(n_ports, k_max):
    """Allocate §2's reduced buffers (complex128) with zero inactive columns."""
    return dict(xi=jnp.zeros(k_max, jnp.complex128),
                Q=jnp.zeros((n_ports, k_max), jnp.complex128),
                Y=jnp.zeros((n_ports, k_max), jnp.complex128),
                S=jnp.zeros((k_max, k_max), jnp.complex128),
                H=jnp.zeros((k_max, k_max), jnp.complex128),
                active=jnp.zeros(k_max, bool), m=jnp.int32(0))


def make_block_qr(sharding):
    """Device-only TSQR; pair arrays stay tiled, only small R factors gather.

    Uses ARKC's measured TSQR construction (w_omega_chain_cost, 5d748a6b),
    with its small-R SVD on device to support the fixed CG loop. Numerical
    null search vectors are zeroed at eps*r, independently of model order.
    """
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    axes, count = tuple(sharding.mesh.axis_names), sharding.mesh.size

    def local(block):
        r = block.shape[0]
        matrix = block.reshape(r,-1).T
        if matrix.shape[0] < r:
            raise ValueError('block QR requires at least r local pair rows')
        qlocal, rlocal = jnp.linalg.qr(matrix,mode='reduced')
        small = jax.lax.all_gather(rlocal,axes,axis=0,tiled=False).reshape(count*r,r)
        qsmall, rr = jnp.linalg.qr(small,mode='reduced')
        section = jax.lax.dynamic_slice_in_dim(qsmall,jax.lax.axis_index(axes)*r,r,axis=0)
        u, values, _ = jnp.linalg.svd(rr,full_matrices=False)
        keep = values > jnp.finfo(values.dtype).eps*r*values[0]
        q = (qlocal @ section @ (u*keep[None,:])).T.reshape(block.shape)
        return q,keep

    return jax.jit(shard_map(local,mesh=sharding.mesh,in_specs=sharding.spec,
                             out_specs=(sharding.spec,P()),check_vma=False))


def block_cg(apply_t, rhs, shift, *, maxiter, tol, orthonormalize):
    """QR-stabilized coupled CG for (T-shift I)X=B with fixed storage.

    The block Galerkin step solves (P†AP) alpha=P†R. The next search
    block is Rnew-P(P†AP)^-1 AP†Rnew, then TSQR normalizes it. This
    equivalent CG recurrence avoids inversion of the nearly singular R†R.
    Converged RHS columns become no-ops; search-space numerical nulls are
    counted separately and never remove a requested model tangent.
    Arrays are complex128 (r,c,v,k), in Ry², tiled by their existing named mesh.
    """
    axes = tuple(range(1,rhs.ndim))
    broad = (rhs.shape[0],)+(1,)*len(axes)

    def inner(a,b):
        return jnp.einsum('acvk,bcvk->ab',a.conj(),b)

    def combine(p,c):
        return jnp.einsum('ab,acvk->bcvk',c,p)

    bnorm = jnp.sqrt(jnp.sum(jnp.abs(rhs)**2,axis=axes))
    b = rhs/jnp.where(bnorm>0,bnorm,1).reshape(broad)
    live = bnorm>0
    p,search = orthonormalize(b)
    initial = (jnp.zeros_like(b),b,p,search,live,
               jnp.zeros(rhs.shape[0],jnp.int32),jnp.int32(0),jnp.int32(0),
               jnp.zeros_like(live),jnp.int32(rhs.shape[0]))

    def step(_,state):
        def advance(state):
            x,r,p,search,live,iterations,useful,issued,broken,minimum = state
            ap = apply_t(p)-shift*p
            mask = search[:,None] & search[None,:]
            pap = inner(p,ap)
            pap = jnp.where(mask,(pap+pap.conj().T)*.5,0)+jnp.diag(~search)
            bad = (jnp.linalg.eigvalsh(pap)[0]<=0) | ~jnp.any(search)
            alpha = jnp.linalg.solve(pap,inner(p,r))
            xn = x+combine(p,alpha)
            rn = r-combine(ap,alpha)
            rr = jnp.sum(jnp.abs(rn)**2,axis=axes)
            next_live = live & (rr>tol*tol) & ~bad
            rn = jnp.where(next_live.reshape(broad),rn,0)
            beta = jnp.linalg.solve(pap,inner(ap,rn))
            pn,search_new = orthonormalize(rn-combine(p,beta))
            finite = jnp.all(jnp.isfinite(xn)) & jnp.all(jnp.isfinite(pn))
            bad = bad | ~finite
            return (xn,rn,pn,search_new,next_live & ~bad,iterations+live,
                    useful+jnp.sum(search,dtype=jnp.int32),issued+rhs.shape[0],
                    broken | (live & bad),jnp.minimum(minimum,jnp.sum(search,dtype=jnp.int32)))
        return jax.lax.cond(jnp.any(state[4]),advance,lambda x:x,state)

    x,_,_,_,_,iterations,useful,issued,broken,minimum = jax.lax.fori_loop(0,maxiter,step,initial)
    x = x*bnorm.reshape(broad)
    residual = rhs-(apply_t(x)-shift*x)
    relative = jnp.sqrt(jnp.sum(jnp.abs(residual)**2,axis=axes))/jnp.where(bnorm>0,bnorm,1)
    return x,dict(iterations=iterations,relative=relative,breakdown=broken,
                  matvec_columns=issued+rhs.shape[0],search_rank_min=minimum,
                  useful_matvec_columns=useful+jnp.sum(bnorm>0),rhs_solves=jnp.sum(bnorm>0))


def preconditioned_block_cg(apply_t, rhs, shift, diagonal, *, maxiter, tol,
                            orthonormalize):
    """Symmetric free-diagonal preconditioning of the negative-axis solve.

    diagonal is the positive transition-square operator Delta² in Ry²,
    with broadcast-compatible pair sharding. P=Delta²-shift is positive.
    The norm-equivalence factor converts the requested original-system tol
    into an internal scaled tolerance; no empirical threshold is introduced.
    The separately checked original-system residual costs one extra T block.
    """
    positive = diagonal-shift
    root = jax.lax.rsqrt(positive)
    inner_tol = tol*jnp.sqrt(jnp.min(positive)/jnp.max(positive))

    def scaled(v):
        return root*(apply_t(root*v)-shift*root*v)

    y,receipt = block_cg(scaled,root*rhs,jnp.float64(0),maxiter=maxiter,
                         tol=inner_tol,orthonormalize=orthonormalize)
    x = root*y
    residual = rhs-(apply_t(x)-shift*x)
    axes = tuple(range(1,rhs.ndim))
    b2 = jnp.sum(jnp.abs(rhs)**2,axis=axes)
    relative = jnp.sqrt(jnp.sum(jnp.abs(residual)**2,axis=axes)/jnp.where(b2>0,b2,1))
    return x,dict(receipt,relative=relative,internal_tolerance=inner_tol,
                  matvec_columns=receipt['matvec_columns']+rhs.shape[0],
                  useful_matvec_columns=receipt['useful_matvec_columns']+jnp.sum(b2>0))


def complex_shifted_cg(apply_t, rhs, shift, *, maxiter, tol, orthonormalize,
                       diagonal=None):
    """Solve (shift I-T)x=rhs using a Hermitian positive normal operator.

    For shift=c+id, D=(T-cI)^2+d²I and x=(conj(shift)I-T)D^-1 rhs.
    This Extension B action retains CG's fixed storage and convergence masks;
    it squares the shifted condition number, which the measured iteration
    count exposes. Nonzero d is required. The final receipt reports the true
    original-system residual, not the normal-equation stopping residual.
    Each D application costs two T columns per RHS; both final T actions are
    included. An optional free transition-square diagonal preconditions D by
    (Delta²-Re shift)²+(Im shift)². The original-system tolerance is unchanged.
    """
    def normal(v):
        av = apply_t(v) - shift.real * v
        return apply_t(av) - shift.real * av + shift.imag**2 * v

    if diagonal is None:
        y, receipt = block_cg(normal, rhs, jnp.float64(0), maxiter=maxiter, tol=tol,
                              orthonormalize=orthonormalize)
    else:
        normal_diagonal = (diagonal-shift.real)**2+shift.imag**2
        y, receipt = preconditioned_block_cg(normal, rhs, jnp.float64(0),
            normal_diagonal, maxiter=maxiter, tol=tol, orthonormalize=orthonormalize)
    x = shift.conjugate() * y - apply_t(y)
    residual = rhs - (shift * x - apply_t(x))
    axes = tuple(range(1, rhs.ndim))
    b2 = jnp.sum(jnp.abs(rhs)**2, axis=axes)
    relative = jnp.sqrt(jnp.sum(jnp.abs(residual)**2, axis=axes) /
                         jnp.where(b2 > 0, b2, 1))
    receipt = dict(receipt, normal_relative=receipt['relative'], relative=relative,
                   matvec_columns=2*receipt['matvec_columns'] + 2*rhs.shape[0],
                   useful_matvec_columns=2*receipt['useful_matvec_columns'] +
                                           2*jnp.sum(b2 > 0))
    return x, receipt


def append_samples(state, shift, q, y, gram, confluent_gram=None):
    """Insert §4 cross terms and the directly contracted confluent Gram block.

    q/y are (Nr,r), gram is Xnew†Xnew (r,r); state uses maximum shapes.
    A support coinciding with an old conjugate support needs confluent_gram
    (Kmax,r), the old/new overlaps, obtained by a further resolvent action or
    a derivative oracle in a reproduction gate. Without it the returned flag
    refuses the block. No difference quotient denominator is regularized.
    """
    m, active = state['m'], state['active']
    zero = jnp.int32(0)
    den = shift - state['xi'].conj()
    reused = jnp.any(active & (den == 0))
    a = state['Y'].conj().T @ q
    b = state['Q'].conj().T @ y
    cross_s = jnp.where(active[:, None], (a - b) / jnp.where(den != 0, den, 1)[:, None], 0)
    if confluent_gram is not None:
        cross_s = jnp.where((active & (den == 0))[:, None],
                            confluent_gram, cross_s)
        reused = jnp.bool_(False)
    # Hermitian form of H_ij=xi_j S_ij-Y_i†Q_j. The average agrees
    # algebraically with that expression and treats noisy confluent samples
    # on both sides equally, matching a global Hermitian pencil assembly.
    cross_h = .5 * ((shift + state['xi'].conj())[:, None] * cross_s - a - b)
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


def append_infinity(state, qinf, g0, g1):
    """Append states B qinf using G0=B†B and G1=B†TB (M1/M3).

    For finite X, X†TBq = conj(xi)Y†q - Q†G0q.  This is the
    coordinator's fixed-budget A2: these columns were reserved inside K.
    No truncation follows; residual selection must finish before this append.
    """
    m, zero = state['m'], jnp.int32(0)
    update = jax.lax.dynamic_update_slice
    gi = state['Y'].conj().T @ qinf
    hi = state['xi'].conj()[:, None] * gi - state['Q'].conj().T @ g0 @ qinf
    gram = update(state['S'], gi, (zero, m))
    gram = update(gram, gi.conj().T, (m, zero))
    gram = update(gram, qinf.conj().T @ g0 @ qinf, (m, m))
    h = update(state['H'], hi, (zero, m))
    h = update(h, hi.conj().T, (m, zero))
    h = update(h, qinf.conj().T @ g1 @ qinf, (m, m))
    return dict(xi=update(state['xi'], jnp.full(qinf.shape[1], jnp.inf+0j), (m,)),
                Q=update(state['Q'], qinf, (zero, m)),
                Y=update(state['Y'], g0 @ qinf, (zero, m)), S=gram, H=h,
                active=update(state['active'], jnp.ones(qinf.shape[1], bool), (m,)),
                m=m+qinf.shape[1])


def build_adaptive_loop(apply_t, apply_b, apply_bh, sh, *, k_max, r_add,
                        n_grid, shortlist, n_power, cg_maxiter, cg_tol,
                        pair_shape, store_truth, block_callback=None, pair_diagonal=None,
                        absolute_residual=False, metric_support_rtol=METRIC_SUPPORT_RTOL,
                        hybrid_candidates=False):
    """Compile §§11–13's fixed-shape outer scan and per-column CG masks.

    The returned callable accepts runtime (operands, G0, grid, spectral_ends,
    K_requested). Every block records order, interpolation and Gram guards;
    a failure sets done and skips every subsequent pair-space solve.  The
    caller must refuse any nonzero failure flag before exporting a model.
    Spectral ends and grid are in Ry²; no Sigma information enters selection.
    Extension B optionally adds complex candidates. store_truth retains Si's sharded X and
    direct Gram/H diagnostics; the sample-only algorithm never reads X.
    pair_diagonal supplies the free positive transition-square diagonal for
    the production preconditioner; planted generic operators may omit it.
    An optional host diagnostic callback receives only the small receipt once
    per attempted block. It never controls selection or changes the model.
    absolute_residual selects the preregistered R†R comparison; the numerical
    metric support is fixed at trace time and never selects the model order.
    """
    if n_grid < shortlist:
        raise ValueError('candidate grid must cover the shortlist')
    n_outer = (k_max + r_add - 1) // r_add
    zero = jnp.int32(0)
    orthonormalize = make_block_qr(sh.X)

    @jax.jit
    def run(operands, g0, grid, spectral_ends, requested):
        nr = g0.shape[0]
        state = empty_samples(nr, k_max)
        evals, evecs = jnp.linalg.eigh((g0+g0.conj().T)*.5)
        metric_live = evals > evals[-1]*metric_support_rtol
        invroot = jnp.where(metric_live, 1/jnp.sqrt(jnp.where(metric_live, evals, 1)), 0)
        whiten = evecs * invroot[None, :]
        if absolute_residual:
            whiten = jnp.eye(nr, dtype=g0.dtype)
        seed = evecs[:, -r_add:]
        trial = jnp.sin(jnp.arange(nr)[:, None] * (jnp.arange(r_add)[None, :]+1) + .7).astype(jnp.complex128)
        truth_size = k_max if store_truth else 0
        basis = jax.lax.with_sharding_constraint(jnp.zeros((truth_size,)+pair_shape, jnp.complex128), sh.X)
        exact_s = jnp.zeros((truth_size, truth_size), jnp.complex128)
        exact_h = jnp.zeros_like(exact_s)
        invalid = (requested % r_add != 0) | (requested > k_max) | (requested <= 0)
        initial = (state, jnp.zeros(k_max), basis, exact_s, exact_h, invalid)

        def step(carry, index):
            def accept(carry):
                state, theta, basis, exact_s, exact_h, done = carry
                a, b = spectral_ends

                def choose(_):
                    indicators = log_support_indicator(grid, state['xi'], theta, state['active'])
                    if hybrid_candidates:
                        # Equal fixed shortlist shares make both contours enter
                        # Extension B's max merit; no pair solve in this scan.
                        on_axis = grid.imag == 0
                        _, ia = jax.lax.top_k(jnp.where(on_axis, indicators, -jnp.inf), shortlist//2)
                        _, ib = jax.lax.top_k(jnp.where(~on_axis, indicators, -jnp.inf), shortlist//2)
                        ids = jnp.concatenate((ia, ib))
                    else:
                        _, ids = jax.lax.top_k(indicators, shortlist)

                    def candidate(i):
                        s = grid[i]
                        q, value = residual_tangents(state, g0, s, whiten, trial, n_power=n_power)
                        score = value * (b-s.real)/(a-s.real)
                        if hybrid_candidates:
                            distance = jnp.abs(s-jnp.clip(s.real,a,b))
                            score = jnp.where(s.imag==0,score,jnp.sqrt(jnp.maximum(value,0))/distance)
                        return q, score

                    qs, scores = jax.lax.map(candidate, ids)
                    best = jnp.argmax(scores)
                    return grid[ids[best]], qs[best], scores[best].real

                shift, q, score = jax.lax.cond(state['m']==0,
                    lambda _: (-jnp.sqrt(a*b)+0j, seed, jnp.float64(1)), choose, None)
                rhs = -apply_b(q, operands)
                def real_solve(_):
                    if pair_diagonal is None:
                        x,cg = block_cg(lambda v:apply_t(v,operands),rhs,shift.real,
                                        maxiter=cg_maxiter,tol=cg_tol,orthonormalize=orthonormalize)
                    else:
                        x,cg = preconditioned_block_cg(lambda v:apply_t(v,operands),rhs,
                            shift.real,pair_diagonal(operands),maxiter=cg_maxiter,tol=cg_tol,
                            orthonormalize=orthonormalize)
                    return x,{key:cg[key] for key in ('iterations','relative','breakdown','matvec_columns','rhs_solves')}

                def complex_solve(_):
                    x,cg = complex_shifted_cg(lambda v:apply_t(v,operands),-rhs,shift,
                        maxiter=cg_maxiter,tol=cg_tol,orthonormalize=orthonormalize,
                        diagonal=None if pair_diagonal is None else pair_diagonal(operands))
                    return x,{key:cg[key] for key in ('iterations','relative','breakdown','matvec_columns','rhs_solves')}

                if hybrid_candidates:
                    x,cg = jax.lax.cond(shift.imag==0,real_solve,complex_solve,None)
                else:
                    x,cg = real_solve(None)
                y = apply_bh(x, operands)
                gram = jnp.einsum('acvk,bcvk->ab', x.conj(), x)
                new, reused = append_samples(state, shift, q, y, gram)
                theta, factor, g = ritz_model(new)
                live = new['active']
                predicted = factor @ ((factor.conj().T @ new['Q']) /
                    jnp.where(live[None, :], new['xi'][None, :]-theta[:, None], 1))
                error = jnp.linalg.norm(predicted-2*new['Y'],axis=0) / jnp.maximum(jnp.linalg.norm(2*new['Y'],axis=0),1e-300)
                interpolation = jnp.max(jnp.where(live,error,0))
                gmax = jnp.max(jnp.where(live,g,0))
                ratio = g[0]/gmax
                failure = (jnp.int32(reused)*1 + jnp.int32(~jnp.all(jnp.isfinite(g)) | (ratio <= 1e-12))*2
                           + jnp.int32(~jnp.isfinite(interpolation) | (interpolation>1e-10))*4
                           + jnp.int32(jnp.any(cg['breakdown']) | (jnp.max(cg['relative'])>cg_tol))*8
                           + jnp.int32(theta[0]<=0)*16)
                gs, gh = jnp.float64(0), jnp.float64(0)
                matvecs = cg['matvec_columns']
                if store_truth:
                    basis = jax.lax.with_sharding_constraint(jax.lax.dynamic_update_slice(
                        basis,x,(state['m'],zero,zero,zero)),sh.X)
                    tx = apply_t(x,operands)
                    cs = jnp.einsum('acvk,bcvk->ab',basis.conj(),x)
                    ch = jnp.einsum('acvk,bcvk->ab',basis.conj(),tx)
                    exact_s = jax.lax.dynamic_update_slice(exact_s,cs,(zero,state['m']))
                    exact_s = jax.lax.dynamic_update_slice(exact_s,cs.conj().T,(state['m'],zero))
                    exact_h = jax.lax.dynamic_update_slice(exact_h,ch,(zero,state['m']))
                    exact_h = jax.lax.dynamic_update_slice(exact_h,ch.conj().T,(state['m'],zero))
                    gs = jnp.linalg.norm(exact_s-new['S'])/jnp.linalg.norm(exact_s)
                    gh = jnp.linalg.norm(exact_h-new['H'])/jnp.linalg.norm(exact_h)
                    matvecs = matvecs+r_add
                receipt = dict(m=new['m'],shift=shift,score=score,interpolation=interpolation,
                               gram_ratio=ratio,gram_relative=gs,h_relative=gh,failure=failure,
                               rhs_solves=cg['rhs_solves'].astype(jnp.int32),
                               matvec_columns=matvecs.astype(jnp.int32),cg_iterations=cg['iterations'],
                               cg_relative=cg['relative'])
                if block_callback is not None:
                    jax.debug.callback(block_callback, receipt)
                return (new,theta,basis,exact_s,exact_h,(failure!=0)|(new['m']>=requested)),receipt

            def skip(carry):
                receipt=dict(m=jnp.int32(0),shift=jnp.complex128(0),score=jnp.float64(0),
                             interpolation=jnp.float64(0),gram_ratio=jnp.float64(0),
                             gram_relative=jnp.float64(0),h_relative=jnp.float64(0),failure=jnp.int32(0),
                             rhs_solves=jnp.int32(0),matvec_columns=jnp.int32(0),
                             cg_iterations=jnp.zeros(r_add,jnp.int32),cg_relative=jnp.zeros(r_add))
                return carry,receipt

            return jax.lax.cond(carry[-1],skip,accept,carry)

        return jax.lax.scan(step,initial,jnp.arange(n_outer,dtype=jnp.int32))

    return run
