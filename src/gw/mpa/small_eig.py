"""Eigenvalues of a SMALL non-symmetric complex matrix, in pure jax.

WHY THIS EXISTS, AND IT IS A MEASUREMENT AND NOT A PREFERENCE.
``jnp.linalg.eigvals`` does lower to the GPU -- the perf fleet's fit lane
established that on 2026-08-10, against an older belief that it fell back
to the host -- but the lowering it reaches, ``cusolver_geev_ffi``, does
not batch.  Measured on this tree at ``n_p = 8``, complex128:

    n_elements   vmap(jnp.linalg.eigvals)
    1024         1012 us/element
    4096         1045 us/element

The cost per element does not move with the batch, which is the signature
of a routine that loops over the batch dimension one matrix at a time.
The same measurement on an A100 in the perf lane's units is 28.1
us/element scaling as ``n_p**1.87``; either way it is the dominant term
of the fit -- 91 % of the jitted per-element cost on this device -- and
it is dominated by per-matrix launch and setup, not by the O(n**3)
arithmetic of an 8x8 problem, which is nanoseconds.

The routine also computes BOTH eigenvector sets and discards them: the
custom call returns a five-tuple of which the fit reads element 1, and
XLA cannot dead-code-eliminate inside a custom call.  ``lax.linalg.eig``
with both ``compute_*_eigenvectors=False`` measures the same, so that
waste is not reachable from the jax side either.

So the pole-finding gets an implementation that stays in XLA.  It is
worth being explicit that this CHANGES THE ANSWER: a different
root-finder returns different roots in the last digits, which is why it
is selectable rather than substituted, and why the equivalence gate for
it is the campaign's W-rebuild-at-samples norm rather than bit-identity.
The perf lane named this exact trade when it declined to make it: *"that
is a re-certification, not a performance change."*  This module is the
re-certification's subject, not a quiet swap.

THE ALGORITHM, AND WHY IT IS JIT-CLEAN BY CONSTRUCTION
-----------------------------------------------------
Textbook Hessenberg-QR, with every count fixed at trace time:

1. **Hessenberg reduction**, ``n - 2`` Householder reflections applied as
   full ``n x n`` rank-one updates.  The count is ``n - 2`` because the
   matrix is ``n x n``, not because of anything in the data.
2. **Shifted QR sweeps with stage-wise deflation.**  The active block
   shrinks on a FIXED schedule -- ``m = n-1, n-2, ..., 1`` -- rather than
   when a subdiagonal entry is judged small.  Each stage gets a fixed
   ``n_sweeps`` Wilkinson-shifted QR steps, and each step is ``n - 1``
   Givens rotations of which the ones at ``k >= m`` are forced to the
   identity by a mask.  A textbook implementation tests
   ``|H[m, m-1]| < tol`` and deflates early; that test is data-dependent
   control flow, which is precisely what cannot appear here.  Paying for
   a fixed number of sweeps whether or not they were needed is the price
   of a static shape, and at ``n = 8`` it is a price of arithmetic that
   was already free.
3. The eigenvalues are the diagonal of the converged quasi-triangular
   matrix.  In COMPLEX arithmetic the Schur form is genuinely triangular
   -- there are no real 2x2 blocks holding a conjugate pair -- so a
   single-shift iteration suffices and no Francis double shift is needed.

The shift is applied to the ACTIVE BLOCK ONLY (``jnp.where(idx <= m)``).
Shifting the whole diagonal would move the already-deflated eigenvalues
in the trailing corner, which is a way of losing answers already found.

This is the ``vmap``/``jit`` contract the rest of ``gw.mpa`` keeps: one
element per call, static shapes throughout, no data-dependent branch, so
the caller's ``jax.vmap`` and ``jax.jit`` apply unchanged.
"""

import jax.numpy as jnp
from jax import lax

#: QR sweeps per deflation stage.  The honest question is not "how many
#: iterations does convergence take" but "how many does the WORST element
#: of a block take", so this is measured, on 512 real Loewner matrices
#: per row, as the worst relative eigenvalue disagreement with LAPACK
#: over the batch and the fraction of elements past 1e-9:
#:
#:     rung            sweeps 8        12         16         32
#:     n_p=4  rank 4   1.2e-01/30%  3.1e-04/1%  1.9e-10/0%  1.9e-10/0%
#:     n_p=8  rank 8   2.2e-02/15%  1.9e-09/2%  1.9e-09/2%  1.9e-09/2%
#:     n_p=10 rank 10  2.4e-03/13%  6.3e-09/7%  6.3e-09/7%  6.3e-09/7%
#:     n_p=12 rank 12  1.3e-03/ 6%  8.0e-09/2%  8.0e-09/2%  8.0e-09/2%
#:     n_p=8  rank 3   1.0e-13/ 0%  1.0e-13/0%  1.0e-13/0%  8.9e-14/0%
#:     n_p=10 rank 3   4.6e-14/ 0%  4.3e-14/0%  4.9e-14/0%  7.7e-14/0%
#:
#: TWO DIFFERENT THINGS ARE VISIBLE IN THAT TABLE AND THEY MUST NOT BE
#: CONFUSED.  Up to sixteen sweeps the numbers FALL, and that is genuine
#: non-convergence: an element left at 1e-1 has an eigenvalue the
#: iteration simply had not reached.  Past sixteen they STOP MOVING --
#: sw16, sw24 and sw32 agree to the digit -- and a plateau that more
#: iterations cannot lower is not an iteration error at all.  It is the
#: conditioning of the eigenvalue problem itself: on full-rank data the
#: Loewner matrix runs cond ~1e5-1e8, so its eigenvalues carry a
#: perturbation sensitivity of about ``eps * cond``, which is the 1e-9
#: the plateau sits at.  LAPACK is not a reference to more digits than
#: that, and the disagreement there is not evidence about this routine.
#:
#: It is also exactly why the campaign states equivalence in the
#: W-rebuild-at-samples norm rather than pole-for-pole: the poles of an
#: ill-conditioned pencil are not the reproducible object, and the
#: screening they rebuild is.
#:
#: Note the last two rows.  The RANK-DEFICIENT case -- more poles fitted
#: than the data supports, which is the production situation the null
#: guard exists for -- is converged at eight sweeps and sits at 1e-13,
#: because a truncated pseudo-inverse hands the iteration a large exact
#: null space and eigenvalues at zero need no sweeps at all.  The hard
#: case is the well-posed one, which is the opposite of the intuition
#: and the reason this ladder was re-measured after a first pass read it
#: off rank-deficient tiles and concluded eight was plenty.
#:
#: SIXTEEN IS SHIPPED: the knee of the falling part, where every rung
#: reaches its conditioning floor.  At ``n_p = 8`` it costs about 11 us
#: per element against the vendor eigensolver's 1155.
DEFAULT_SWEEPS = 16


def _hessenberg(H, n):
    """Upper-Hessenberg by ``n - 2`` Householder reflections.

    Applied as full ``n x n`` rank-one updates rather than to shrinking
    trailing sub-blocks.  That is ``O(n**4)`` where the blocked form is
    ``O(n**3)``, and at ``n = 8`` it is 4096 flops against 512 -- both of
    which are nothing beside a single kernel launch.  What it buys is
    that every operand has the same shape at every ``k``, so the whole
    reduction fuses instead of becoming a staircase of differently-shaped
    slices.
    """

    idx = jnp.arange(n)
    for k in range(n - 2):
        # The column below the diagonal, embedded at full length so the
        # reflector is an n-vector like every other one.
        x = jnp.where(idx > k, H[:, k], 0.0 + 0.0j)
        nx = jnp.linalg.norm(x)
        x0 = H[k + 1, k]
        # alpha = -e^{i arg x0} ||x||: the sign is chosen AWAY from x0 so
        # that v = x - alpha e_{k+1} adds magnitudes instead of
        # subtracting them.  Choosing it towards x0 is the classic
        # cancellation that costs half the digits when x0 dominates.
        ax0 = jnp.abs(x0)
        phase = jnp.where(ax0 > 0, x0 / jnp.where(ax0 > 0, ax0, 1.0),
                          1.0 + 0.0j)
        v = x + jnp.where(idx == k + 1, phase * nx, 0.0 + 0.0j)
        vn = jnp.linalg.norm(v)
        # vn == 0 means the column was already zero below the diagonal;
        # v = 0 then makes both updates below vanish, i.e. P = I.
        v = jnp.where(vn > 0, v / jnp.where(vn > 0, vn, 1.0), 0.0 + 0.0j)
        H = H - 2.0 * jnp.outer(v, v.conj() @ H)
        H = H - 2.0 * jnp.outer(H @ v, v.conj())
    return H


def _wilkinson_shift(H, m):
    """The eigenvalue of the active block's trailing 2x2 nearest ``H[m, m]``.

    ``m`` is traced (the deflation stage index), so this indexes with
    dynamic scalars; the SHAPE of everything it touches is static.
    """

    a = H[m - 1, m - 1]
    b = H[m - 1, m]
    c = H[m, m - 1]
    d = H[m, m]
    tr = a + d
    det = a * d - b * c
    disc = jnp.sqrt(tr * tr / 4.0 - det)
    l1 = tr / 2.0 + disc
    l2 = tr / 2.0 - disc
    s = jnp.where(jnp.abs(l1 - d) <= jnp.abs(l2 - d), l1, l2)
    # A non-finite shift would poison the whole matrix rather than one
    # pole; falling back to an unshifted step keeps the sweep linear
    # instead of divergent, and the fit's range guard drops whatever
    # comes out.
    return jnp.where(jnp.isfinite(jnp.abs(s)), s, 0.0 + 0.0j)


def _qr_sweep(H, m, n):
    """One Wilkinson-shifted QR step on the active block ``0..m``."""

    idx = jnp.arange(n)
    shift = jnp.where(idx <= m, _wilkinson_shift(H, m), 0.0 + 0.0j)
    shift = shift.astype(H.dtype)
    H = H - jnp.diag(shift)

    # --- Q^H H, as n-1 Givens rotations of adjacent rows.  The rotations
    # at k >= m are forced to the identity, which is how the deflated
    # tail is left alone without a shape change.
    def left(k, carry):
        H, cs = carry
        a = H[k, k]
        b = H[k + 1, k]
        r = jnp.sqrt(jnp.abs(a) ** 2 + jnp.abs(b) ** 2)
        live = (r > 0) & (k < m)
        safe_r = jnp.where(r > 0, r, 1.0).astype(H.dtype)
        c = jnp.where(live, a / safe_r, 1.0 + 0.0j)
        s = jnp.where(live, b / safe_r, 0.0 + 0.0j)
        rk = H[k]
        rk1 = H[k + 1]
        H = H.at[k].set(c.conj() * rk + s.conj() * rk1)
        H = H.at[k + 1].set(-s * rk + c * rk1)
        cs = cs.at[0, k].set(c).at[1, k].set(s)
        return H, cs

    cs0 = jnp.zeros((2, max(n - 1, 1)), dtype=H.dtype)
    H, cs = lax.fori_loop(0, max(n - 1, 0), left, (H, cs0))

    # --- R Q, the same rotations from the right in the SAME order.
    def right(k, carry):
        H, cs = carry
        c = cs[0, k]
        s = cs[1, k]
        ck = H[:, k]
        ck1 = H[:, k + 1]
        H = H.at[:, k].set(c * ck + s * ck1)
        H = H.at[:, k + 1].set(-s.conj() * ck + c.conj() * ck1)
        return H, cs

    H, _ = lax.fori_loop(0, max(n - 1, 0), right, (H, cs))
    return H + jnp.diag(shift)


def eigvals(A, *, n_sweeps=DEFAULT_SWEEPS):
    """Eigenvalues of ONE small non-symmetric complex matrix.

    Parameters
    ----------
    A
        ``(n, n)`` complex.  One element's matrix; the caller vmaps.
    n_sweeps
        Shifted-QR sweeps per deflation stage.  Static.

    Returns
    -------
    ``(n,)`` complex128, in no particular order -- the caller sorts.
    ``jnp.linalg.eigvals`` makes no ordering promise either, which is why
    ``pade_fit`` has always sorted the roots itself.
    """

    M = jnp.asarray(A, dtype=jnp.complex128)
    n = M.shape[-1]
    if M.ndim != 2 or M.shape[0] != n:
        raise ValueError(
            f"GATE small_eig_square: A has shape {tuple(M.shape)}. FALSE "
            "case: A is a square 2-D array; this kernel takes ONE matrix "
            "and is vmapped for a batch.")
    if n == 1:
        return M[0:1, 0]

    H = _hessenberg(M, n)

    def stage(i, H):
        # m counts DOWN from n-1: stage i works the leading (m+1) block
        # and freezes H[m, m] as an eigenvalue.  Fixed schedule, so the
        # trip count is a constant and the body is traced once.
        m = n - 1 - i
        return lax.fori_loop(0, n_sweeps, lambda _, h: _qr_sweep(h, m, n), H)

    H = lax.fori_loop(0, n - 1, stage, H)
    return jnp.diagonal(H)
