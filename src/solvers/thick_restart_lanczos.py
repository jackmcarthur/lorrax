"""
solvers/thick_restart_lanczos.py — thick-restart Lanczos, fixed shape.

Finds the lowest ``n_eig`` eigenvalues of a Hermitian operator given only a
callable matvec.  No physics knowledge — works for any Hermitian eigenproblem.

WHY THIS EXISTS, AND WHAT IT IS FOR
-----------------------------------
Unrestarted Lanczos (``solvers.lanczos.lanczos_eig_jit``) is the cheapest
correct solver on a 1024-dim deck and this module does not beat it there.  What
it fixes is **memory at production dimension**.  Unrestarted Lanczos stores the
whole Krylov basis: at ``bse_dim = 1e6`` a 400-vector basis is 6.4 GB, and at
``1e7`` it is 64 GB and the solve does not fit.  Thick restart caps the basis at
``m_max`` slots — typically ``3·n_eig … 5·n_eig`` — and pays for the cap in
extra matvecs.

Method: Wu & Simon, *Thick-restart Lanczos method for large symmetric
eigenvalue problems*, SIAM J. Matrix Anal. Appl. 22(2):602 (2000); and Wu,
Canning, Simon & Wang, *Thick-restart Lanczos method for electronic structure
calculations*, J. Comput. Phys. 154:156 (1999), which is this exact application
domain.

SHAPE CONVENTION
----------------
Identical to ``solvers.davidson``: the basis ``Q`` has shape
``(m_max + 1, *trailing)`` where axis 0 is the basis axis and ``trailing`` is
whatever encodes one vector — flat ``(dim,)``, exciton ``(nc, nv, nk)``,
plane-wave ``(n_channels, dim)``.  Every contraction is an ellipsis einsum, so
the trailing axes are never reshaped, never flattened and never gathered, and
the sharding of ``Q`` propagates to every derived quantity.

DISTRIBUTED STRUCTURE
---------------------
**This module opens no ``shard_map`` region of its own.**  Every operation here
is either an ellipsis einsum whose batch axis is replicated and whose trailing
axes carry the caller's sharding, or a small ``(m_max, m_max)`` dense solve on
replicated data.  GSPMD lowers the einsums to exactly the collectives the CGS2
argument below predicts.  The only ``shard_map`` in the whole solve is the one
the caller's matvec already owns, at its own (largest possible) scope; this
module calls it and adds nothing around it.

FIXED SHAPE — THE POINT OF THE FILE
-----------------------------------
``m_max``, ``n_keep``, ``n_restarts`` and ``n_eig`` are Python ints, so every
slice below is static and every loop trip count is a compile-time constant.
The basis, the projected matrix, and the coefficient arrays are allocated once
at their final size and updated in place.  Nothing in the iteration is traced
at more than one shape:

* one program for the first (cold) cycle, whose inner loop runs ``0 … m_max``;
* one program for the restart cycle, whose inner loop runs ``n_keep … m_max``,
  reused for every subsequent cycle by an outer ``lax.fori_loop``;
* one program for the final Ritz reconstruction.

Contrast ``solvers.davidson``, whose subspace grows ``n_eig, 2·n_eig, …`` and
which therefore traces ``_ritz_and_residuals`` and ``_ortho_expand`` once per
subspace width.

THE ARROWHEAD, WHICH IS THE WHOLE METHOD
----------------------------------------
After a thick restart the retained basis vectors are *Ritz* vectors, not
Lanczos vectors, so the recurrence is not three-term across the restart.  The
projected matrix takes the **arrowhead** form: a diagonal block of ``n_keep``
Ritz values, a full row and column coupling them to the first new Lanczos
vector with weights ``s_i = β_last · conj(Y[m_max-1, i])``, and the ordinary
tridiagonal tail below that.  Getting this right is the entire difference
between thick-restart Lanczos and merely restarting Lanczos, which loses the
retained information.

ORTHOGONALISATION
-----------------
Full reorthogonalisation against every slot ``i ≤ j``, by two classical
Gram-Schmidt passes (CGS2) written as GEMMs — the machinery from
``perf/reorth-batch-2026-08-08``, transplanted rather than re-derived, with the
window mask set to ``arange(m_max+1) <= j``.  This is **two collectives per
Lanczos step** carrying an ``(m_max+1,)`` payload, against the shipped MGS
sweep's ``j+1`` collectives each carrying a 16-byte scalar.  Thick restart does
not merely benefit from this, it *requires* full reorthogonalisation: the
retained Ritz block is not part of any three-term recurrence, so nothing else
keeps the new vectors orthogonal to it.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


# ═══════════════════════════════════════════════════════════════════════
#  CGS2 — transplanted from perf/reorth-batch-2026-08-08 (lanczos.py)
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  Retrace instrument
# ═══════════════════════════════════════════════════════════════════════
#  Incremented INSIDE traced bodies, so one increment is one TRACE of that
#  body.  The fixed-shape claim is exactly the statement that these counts
#  do not depend on ``n_restarts``: run the solver with 2 restart cycles and
#  with 20 and the numbers are identical.  ``compile_cache_stats()`` cannot
#  make that statement -- with the persistent cache warm it reads 0 compiles
#  no matter how many distinct programs the loop dispatches.
TRACE_COUNTS: dict = {}


def _tally(name) -> None:
    TRACE_COUNTS[name] = TRACE_COUNTS.get(name, 0) + 1


def reset_trace_counts() -> None:
    TRACE_COUNTS.clear()



def _cgs2(Q, z, sel):
    """Two classical Gram-Schmidt passes of ``z`` against the basis ``Q``.

    ``Q``   ``(M, *trailing)`` — slots past the current iteration are exactly
            zero, so they contribute nothing even before the mask.
    ``z``   ``(*trailing,)``.
    ``sel`` ``(M,)`` bool — the window mask.

    TWO collectives total, one per pass, each an ``(M,)`` all-reduce.  The
    second contraction sums over the replicated basis axis and emits none.

    The mask is not redundant with the zero fill: slot ``j`` holds the current
    basis vector and is non-zero, so it must be masked explicitly.
    """
    def _pass(zz):
        # Contracts every SHARDED trailing axis -> exactly one psum, shape (M,).
        h = jnp.einsum('m...,...->m', jnp.conj(Q), zz, optimize=True)
        h = jnp.where(sel, h, jnp.zeros((), dtype=h.dtype))
        return zz - jnp.einsum('m,m...->...', h, Q, optimize=True)
    return _pass(_pass(z))


def _window(j, n_slots: int):
    """Boolean ``(n_slots,)`` selector for ``{i : i <= j}``.

    ``j`` is traced; ``n_slots`` is static.  Full reorthogonalisation, which
    thick restart requires — see the module docstring.
    """
    return jnp.arange(int(n_slots)) <= j


# ═══════════════════════════════════════════════════════════════════════
#  Solver
# ═══════════════════════════════════════════════════════════════════════

def thick_restart_lanczos_eig(
    matvec: Callable[[jax.Array], jax.Array],
    trailing_shape: tuple[int, ...],
    *,
    n_eig: int = 20,
    m_max: int = 80,
    n_keep: int | None = None,
    n_restarts: int = 8,
    seed: int = 42,
    dtype=jnp.complex128,
    X0: jax.Array | None = None,
    sharding=None,
    _drop_arrowhead: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Lowest ``n_eig`` eigenpairs by thick-restart Lanczos.

    Parameters
    ----------
    matvec : (b, *trailing) -> (b, *trailing)
        Hermitian matvec on a BATCH of vectors.  This module only ever calls
        it with ``b = 1``; the batch axis is kept so the callable is the same
        object ``solvers.davidson`` takes, and so the caller's sharding
        constraint applies unchanged.  Sharding of the trailing axes is the
        caller's business — nothing here reshapes them.
    trailing_shape : tuple[int, ...]
        Shape of ONE vector (everything after the basis axis).
    n_eig : int
        Number of lowest eigenvalues wanted.
    m_max : int
        Basis slots.  The memory cap.  ``3·n_eig … 5·n_eig`` is the useful
        range; below ``n_eig + 2`` the method cannot make progress.
    n_keep : int, optional
        Ritz vectors retained across a restart.  Default ``n_eig + 10``.
        Must satisfy ``n_eig <= n_keep < m_max``.
    n_restarts : int
        Restart cycles after the first (cold) cycle.  Total matvec
        applications are ``m_max + n_restarts · (m_max - n_keep)``.
    X0 : (k, *trailing), optional
        Explicit starting vector(s); only ``X0[0]`` is used.  Default is a
        deterministic complex Gaussian from ``seed``.
    sharding : jax.sharding.Sharding, optional
        Applied to the preallocated basis so the compile cache key matches the
        caller's production sharding.

    Returns
    -------
    eigenvalues : (n_eig,) float64
    eigenvectors : (n_eig, *trailing)
    alpha_im_max : () float64
        Largest ``|Im <q, Hq>|`` seen.  For a Hermitian operator this is zero
        in exact arithmetic at every iteration, converged or not, so it is a
        free detector for "the matvec did not return H·q" — the same invariant
        ``solvers.lanczos`` carries, kept because it costs nothing and because
        thick restart rotates the basis through the Ritz vectors every cycle
        and would otherwise smear a recurrence defect across the whole
        retained block.
    """
    if n_keep is None:
        n_keep = n_eig + 10
    m_max = int(m_max)
    n_keep = int(n_keep)
    n_eig = int(n_eig)
    if not (0 < n_eig <= n_keep < m_max):
        raise ValueError(
            f"thick_restart_lanczos_eig: need 0 < n_eig ({n_eig}) <= n_keep "
            f"({n_keep}) < m_max ({m_max})")

    tr = tuple(int(t) for t in trailing_shape)
    M = m_max + 1                      # +1 slot for the residual vector

    # ── starting vector ────────────────────────────────────────────────
    if X0 is not None:
        q0 = jnp.asarray(X0[0], dtype=dtype)
    else:
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        q0 = (jax.random.normal(k1, tr, dtype=jnp.float64)
              + 1j * jax.random.normal(k2, tr, dtype=jnp.float64)).astype(dtype)
    q0 = q0 / jnp.sqrt(jnp.sum(jnp.abs(q0) ** 2))

    Q0 = jnp.zeros((M,) + tr, dtype=dtype).at[0].set(q0)
    if sharding is not None:
        Q0 = lax.with_sharding_constraint(Q0, sharding)

    def _apply(q):
        """H applied to one vector, through the caller's batched matvec."""
        return matvec(q[None, ...])[0]

    # ── one Lanczos step at slot j, full CGS2 reorthogonalisation ──────
    def _step(j, carry):
        _tally('lanczos_step')
        Q, alpha, beta, alpha_im = carry
        qj = Q[j]
        z = _apply(qj)
        # ONE complex dot.  .real drives the recurrence; .imag is the free
        # Hermitian-form residual (see the module docstring / lanczos.py).
        a_c = jnp.sum(jnp.conj(qj) * z)
        alpha = alpha.at[j].set(a_c.real)
        alpha_im = alpha_im.at[j].set(jnp.abs(a_c.imag))
        z = z - a_c.real * qj
        # Full reorthogonalisation subsumes the -beta_{j-1} q_{j-1} term AND
        # the arrowhead couplings to the retained Ritz block, which is why
        # thick restart needs it rather than merely benefiting from it.
        z = _cgs2(Q, z, _window(j, M))
        b = jnp.sqrt(jnp.sum(jnp.abs(z) ** 2)).real
        beta = beta.at[j].set(b)
        Q = Q.at[j + 1].set(z / jnp.maximum(b, 1e-300))
        return (Q, alpha, beta, alpha_im)

    # ── assemble the projected matrix (fixed (m_max, m_max)) ───────────
    def _build_T(alpha, beta, theta_k, s_arrow, first: bool):
        """Arrowhead + tridiagonal tail, all-static slicing.

        ``first`` selects the cold cycle, whose T is plain tridiagonal.
        Both branches are resolved at TRACE time (Python bool), so each
        produces its own fixed-shape program rather than a select.
        """
        _tally('build_T_first' if first else 'build_T_restart')
        T = jnp.zeros((m_max, m_max), dtype=dtype)
        if first:
            T = T + jnp.diag(alpha.astype(dtype))
            off = beta[:m_max - 1].astype(dtype)
            return T + jnp.diag(off, 1) + jnp.diag(off, -1)
        # retained Ritz block: diagonal of Ritz values
        idx = jnp.arange(n_keep)
        T = T.at[idx, idx].set(theta_k.astype(dtype))
        # the arrow: couplings from each retained Ritz vector to slot n_keep
        T = T.at[:n_keep, n_keep].set(s_arrow)
        T = T.at[n_keep, :n_keep].set(jnp.conj(s_arrow))
        # the ordinary tridiagonal tail, slots n_keep .. m_max-1
        tail = jnp.arange(n_keep, m_max)
        T = T.at[tail, tail].set(alpha[n_keep:m_max].astype(dtype))
        tl = jnp.arange(n_keep, m_max - 1)
        off = beta[n_keep:m_max - 1].astype(dtype)
        T = T.at[tl, tl + 1].set(off)
        T = T.at[tl + 1, tl].set(off)
        return T

    def _restart(Q, T, beta_last):
        _tally('restart')
        """Rotate the basis onto the lowest n_keep Ritz vectors.

        Returns the new basis, the retained Ritz values, and the arrow
        couplings for the next cycle.
        """
        T = 0.5 * (T + jnp.conj(T).T)
        theta, Y = jnp.linalg.eigh(T)
        theta_k = theta[:n_keep]
        Y_k = Y[:, :n_keep]                       # (m_max, n_keep)
        # New basis vectors are Ritz vectors of the OLD basis.
        Qk = jnp.einsum('mn,m...->n...', Y_k, Q[:m_max], optimize=True)
        # Slot n_keep is the residual vector the last step produced.
        Q_new = (jnp.zeros_like(Q)
                 .at[:n_keep].set(Qk)
                 .at[n_keep].set(Q[m_max]))
        # s_i = beta_last * conj(Y[m_max-1, i]) — the arrowhead couplings.
        s_arrow = beta_last.astype(dtype) * jnp.conj(Y_k[m_max - 1, :])
        if _drop_arrowhead:
            # RED TWIN ONLY.  Dropping the arrow is precisely the difference
            # between thick-restart Lanczos and naively restarting Lanczos:
            # the retained Ritz vectors stop coupling to the new Krylov
            # directions and the retained information is wasted.
            s_arrow = jnp.zeros_like(s_arrow)
        return Q_new, theta_k, s_arrow

    # ── cold cycle: plain Lanczos filling all m_max slots ──────────────
    alpha = jnp.zeros((m_max,), dtype=jnp.float64)
    beta = jnp.zeros((m_max,), dtype=jnp.float64)
    alpha_im = jnp.zeros((m_max,), dtype=jnp.float64)

    Q, alpha, beta, alpha_im = lax.fori_loop(
        0, m_max, _step, (Q0, alpha, beta, alpha_im))
    T = _build_T(alpha, beta, None, None, first=True)
    Q, theta_k, s_arrow = _restart(Q, T, beta[m_max - 1])

    # ── restart cycles: identical program, run n_restarts times ────────
    def _cycle(_c, carry):
        Q, theta_k, s_arrow, alpha, beta, alpha_im = carry
        Q, alpha, beta, alpha_im = lax.fori_loop(
            n_keep, m_max, _step, (Q, alpha, beta, alpha_im))
        T = _build_T(alpha, beta, theta_k, s_arrow, first=False)
        Q, theta_k, s_arrow = _restart(Q, T, beta[m_max - 1])
        return (Q, theta_k, s_arrow, alpha, beta, alpha_im)

    if n_restarts > 0:
        Q, theta_k, s_arrow, alpha, beta, alpha_im = lax.fori_loop(
            0, n_restarts, _cycle,
            (Q, theta_k, s_arrow, alpha, beta, alpha_im))

    # ── final Rayleigh-Ritz on the retained block ──────────────────────
    # After the last _restart, Q[:n_keep] ARE the Ritz vectors and theta_k
    # their values, already sorted ascending by eigh.
    eigenvalues = theta_k[:n_eig]
    eigenvectors = Q[:n_eig]
    return eigenvalues, eigenvectors, jnp.max(alpha_im)
