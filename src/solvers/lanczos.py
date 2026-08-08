"""
solvers/lanczos.py — Lanczos iterative eigensolvers.

Finds the lowest n_eig eigenvalues of a Hermitian operator H given only
a callable matvec.  No physics knowledge — works for any Hermitian
eigenproblem.

Five variants:
  - simple_lanczos_eig:              Python-loop, full reorthogonalization
  - lanczos_eig_jit:                 lax.fori_loop, partial reorth (JIT-able)
  - block_lanczos_eig:               Block Lanczos, Python loop, shaped vectors
  - block_lanczos_eig_jit:           Block Lanczos in lax.fori_loop
  - block_lanczos_eig_jit_converged: as above, Ritz-stability exit

Every one of them carries the **α-Hermiticity invariant** — see the section
below ``import numpy as np``.  ``⟨q, Hq⟩`` is real for a Hermitian H, so the
imaginary part of ``α`` (which the recurrence computes and used to discard) is
a free detector for "the matvec did not return H·q".  It is always on because
it is free; it is the only invariant in this codebase that sits on COLLECTIVE
OUTPUT rather than on a construction tile.

Usage
-----
    from solvers.lanczos import lanczos_eig_jit
    eigenvalues, eigenvectors = lanczos_eig_jit(matvec, n=1000, n_eig=10)
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


# ===========================================================================
# Reorthogonalisation: why the default is FULL and not a window
# ===========================================================================
#
# The shipped defaults used to be ``n_reorth=2`` here and ``10`` in
# ``bse/bse_lanczos.py``.  Both are footguns, and the measurement says the
# window is not a speed/accuracy dial at all — it is a correctness switch with
# one correct setting.
#
# MEASURED, stored dense Si BSE matrix, n=1024, single-vector Lanczos, error on
# the lowest 20 eigenvalues against the dense spectrum:
#
#     window     k=200    k=400    k=600    k=800    k=1000
#     2          4.3 meV  86       94       134      142
#     10         4.3      86       94       134      142
#     30         4.3      86       94       134      142
#     full       4.3      1.2      0.037    8.1e-4   4e-4
#
# Windows of 2, 10 and 30 are INDISTINGUISHABLE from each other: the basis has
# already lost orthogonality by the time the window falls off the end, so
# widening it changes nothing.  All of them plateau at 4.3 meV and then get
# monotonically WORSE with more iterations — at k=1000, ``‖QᴴQ − I‖ = 0.35``
# and there are 91 Ritz values below λ₂₀ where there should be exactly 20.
# That is the ghost-eigenvalue mechanism: a lost direction is re-discovered as
# a duplicate copy of an already-converged root.  At n=4096 the best a windowed
# run achieves is 21.3 meV.
#
# "More iterations makes it worse" is the property that makes a partial window
# unsafe as a DEFAULT.  A caller who asks for more work and gets a worse answer
# has no way to notice from the outside.
#
# Production escaped this because ``bse/bse_jax.py`` passes the -1 sentinel,
# which resolves to full reorth — but every test in the tree ALSO pinned full
# reorth explicitly, so nothing in CI ever exercised the shipped defaults.  The
# defaults are now the sentinel, and ``tests/test_lanczos_reorth_default.py``
# covers them AT THE DEFAULT (no ``n_reorth=`` argument anywhere in it).
#
# The window is kept as an option, not deleted: it is a legitimate memory/time
# trade for a caller who has measured their own spectrum and knows a short
# basis suffices.  It just must not be what you get by not choosing.

#: Sentinel: reorthogonalise against the ENTIRE basis built so far.
#: Same convention as ``bse/bse_jax.py --n-reorth`` and
#: ``bse/exciton_bands.py``, so one value means one thing everywhere.
FULL_REORTH = -1


def resolve_n_reorth(n_reorth: int | None, depth: int) -> int:
    """Resolve the reorth window against the basis depth it will run to.

    ``FULL_REORTH`` (-1) and ``None`` both mean "the whole basis", expressed
    as ``depth`` — the number of iterations (single-vector) or blocks (block
    variants) the loop can reach.  Any other value passes through as a window
    width.  Centralised so the sentinel cannot mean -1 iterations in one
    variant and full reorth in another.
    """
    if n_reorth is None or int(n_reorth) < 0:
        return int(depth)
    return int(n_reorth)


# ===========================================================================
# The Hermitian-form invariant the recurrence already computes
# ===========================================================================
#
# WHY THIS IS HERE
# ----------------
# Every Lanczos variant below forms
#
#     α_j = ⟨q_j, H q_j⟩            (scalar variants)
#     α_j = Q_jᴴ H Q_j              (block variants, (bs, bs))
#
# and then throws away the part of it that is a free integrity check.  The
# scalar variants took ``.real`` and discarded ``Im α``; the block variants
# built T and symmetrised it with ``(T + Tᴴ)/2`` (see ``_build_block_tridiag``),
# which discards ``α − αᴴ``.  For a Hermitian H both discarded quantities are
# ZERO in exact arithmetic — a Hermitian form has a real value, a Hermitian
# Gram block is Hermitian — for ANY q, converged or not, at every iteration.
#
# That makes ``Im α`` a detector for "the matvec did not return H·q".  It is
# blind to nothing that matters and it costs nothing, because the complex dot
# product that produces it is already on the critical path.
#
# The concrete motivation (2026-07-29): ``jax.lax.psum_scatter`` under
# ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo`` silently returns wrong data in
# ~5 % of executions, always in output segment 0, with a plausible magnitude
# and a zero exit code (``wk_REL/UPSTREAM_gloo_psum_scatter_corruption.md``).
# The BSE matvec issues two of them per iteration
# (``bse/bse_stack_matvec.py:126,129``).  A corrupted segment makes the
# returned vector not equal to H·q, which breaks the Hermitian form by the
# size of the error.  The archaeology found that ``check_hermitian`` runs at
# five sites in this codebase and NONE of them is downstream of a
# reduce-scatter — every invariant sat on construction tiles.  This is the one
# that sits on collective OUTPUT, and it is the reason a whole campaign of
# 1913 job logs contains no detection of a bug that was firing all along.
#
# THE QUANTITY, AND WHY IT IS check_hermitian's
# ---------------------------------------------
# The tridiagonal T built from these α's is Hermitian by construction, so
# ``(T − Tᴴ)_jj = α_j − conj(α_j) = 2i·Im α_j``.  Reporting
#
#     rel = max_j |Im α_j| / max_j |α_j|
#
# is therefore literally ``max|A − Aᴴ| / max|A|`` (up to the factor 2)
# restricted to T's diagonal — the SAME residual, scaled against the tile's own
# scale, that ``common.sanity.check_hermitian`` computes.  Both paths report
# through ``sanity.report_hermitian_residual`` so there is one verdict, one
# tolerance and one message, not two.  Normalising by ``max_j |α_j|`` rather
# than by the per-iteration ``|α_j|`` is deliberate and is check_hermitian's own
# convention: a single α passing near zero (a Krylov direction nearly orthogonal
# to Hq) must not manufacture a false positive.
#
# THE TOLERANCE — DERIVED, NOT TUNED
# ----------------------------------
# With ``‖q‖₂ = 1`` and u = 2⁻⁵³ = 1.11e-16 the unit roundoff of float64, the
# computed ``Im α`` has exactly two sources, and both scale with a CONTRACTION
# LENGTH times u:
#
#   (1) the dot product itself.  α = Σᵢ conj(qᵢ) zᵢ over n terms; the standard
#       bound is |fl(Σ) − Σ| ≤ γ_n Σ|qᵢ zᵢ| ≤ γ_n‖q‖‖z‖ = γ_n‖z‖ with
#       γ_n = nu/(1−nu) ≈ n·u (pairwise summation, which XLA/BLAS actually use,
#       reduces this to O(log n · u) — we keep the pessimistic n·u).
#
#   (2) the matvec.  z = fl(Hq) = Hq + δz with ‖δz‖ ≲ c_H·u·‖H‖, where c_H is
#       the effective accumulation depth of the matvec.  ⟨q, δz⟩ has no reason
#       to be real, so it lands in Im α at ≲ c_H·u·‖H‖.  For the BSE matvec the
#       deep chains are the two reduce-scatter contractions over μ and ν
#       (length N_mu each) plus the k-FFT (nk log nk) and the c/v einsums, so
#       c_H ≈ 2·N_mu + nk·log₂nk + n_c + n_v = O(N_mu).
#
# Adding them and dividing by scale = max_j|α_j| = θ‖H‖ (θ = O(1) once the
# Krylov space has sampled the spectrum; θ ≥ 0.1 is generous for BSE, whose H
# is gapped and positive):
#
#     rel ≲ (n + c_H)·u / θ
#
# At the LARGEST production BSE shape on this stack (N_mu = 10015, n = n_c·n_v·nk
# ≈ 4·10³):  (4·10³ + 2·10⁴)·1.11e-16 / 0.1 ≈ 2.7e-11.
#
# ``ALPHA_HERM_RTOL = 1e-9`` therefore sits ~40× above the worst-case round-off
# budget of the largest shape we run (and ~10⁶× above what a small deck
# actually measures), while the corruption it exists to catch is a RELATIVE
# perturbation of order 1e-2…1e-1 of the matvec output — 7 to 8 orders of
# magnitude above the threshold.  There is no tuning freedom in that gap: any
# tolerance in [1e-11, 1e-4] gives the same verdict on every case we have.  It
# is not chosen to make a test pass; it is the round-off bound rounded up.
#
# COST — why this is always-on and not behind LORRAX_SANITY
# ---------------------------------------------------------
# Scalar variants: ``jnp.vdot`` already produces a complex scalar, so ``.imag``
# is the half of it that was being discarded — at worst two extra length-n
# multiply-accumulate passes, against a matvec that is orders of magnitude more
# expensive, and against the reorthogonalisation's own n_reorth dot products
# in the same loop body.  Block variants: α_j is already fully materialised
# (the recurrence subtracts ``Q_j @ α_j``), so the residual is bs² = O(10) flops
# on a tile that is already in registers.  Neither adds a collective, a device
# sync, or a full-tile pass.  Per the owner's rule, a free invariant is
# always-on: it reports through ``report_hermitian_residual(..., always=True)``,
# which bypasses the ``LORRAX_SANITY`` *cost* escape hatch while still honouring
# ``strict``.
#
# The residual leaves the traced region through ONE ``jax.debug.callback`` per
# solve (unordered, three float64 scalars).  It cannot be a return value: these
# solvers run inside ``bse_lanczos._full_run``'s outer jit with fixed
# ``out_shardings``, and you cannot raise from inside a ``fori_loop``.

ALPHA_HERM_RTOL = 1e-9

_ALPHA_FORMS = {
    "vec": ("<q,Hq>",
            "alpha_j = <q_j, H q_j> is REAL for any Hermitian H and any q_j, "
            "so a nonzero imaginary part"),
    "block": ("Q^H H Q",
              "alpha_j = Q_j^H H Q_j is HERMITIAN for any Hermitian H and any "
              "Q_j, so a nonzero antihermitian part"),
}

_ALPHA_CAUSE = (
    " means the matvec did not return exactly H*q.  On the BSE decks this "
    "stack runs, the KNOWN and MEASURED source is the OPERATOR TILE, not the "
    "solver and not a collective: the asymmetry ORIGINATES IN THE DIRECT "
    "PER-Q V_q ASSEMBLY (the unfold path is four orders cleaner -- 1.27e-7 "
    "against 6.03e-3 direct -- and the split is keyed on a discrete "
    "zone-boundary predicate) and PROPAGATES INTO W; the matvec then carries "
    "it faithfully and the gate reports it.  CALIBRATION (Si 8v8c, "
    "2026-08-07): the gate fires at ~1.3e-06 relative; the same "
    "max|H - H^H| = 8.0776e-06 is reproduced BIT-IDENTICALLY on a "
    "single-process login-node DENSE probe with no reduce-scatter anywhere, "
    "and is identical between the n=1024 and n=4096 matrices -- a "
    "deterministic, size-independent number that random collective "
    "corruption cannot produce.  Its worth in the EIGENVALUES is 0.0003 ueV "
    "on the lowest 12, i.e. FAR below any physics tolerance.  So at this "
    "magnitude: the eigenvalues are usable; the asymmetry is a real defect "
    "in the V_q assembly worth fixing at its source, not a reason to "
    "discard the solve.  "
    "Escalate only if the relative residual is ORDERS above ~1e-06 (>=1e-04 "
    "is the corruption regime), or if it VARIES between otherwise identical "
    "runs -- non-determinism is the signature that separates a collective "
    "fault from a deterministic tile defect.  In that case the other "
    "candidates are: a silent reduce-scatter corruption "
    "(jax.lax.psum_scatter under JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo "
    "returns wrong data in ~5% of executions, always output segment 0 -- "
    "see wk_REL/UPSTREAM_gloo_psum_scatter_corruption.md -- so re-run under "
    "JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi, clean in 584/584), or a "
    "mis-transposed shard."
)

#: Below this relative residual the gate is a WARNING about W's construction,
#: not a verdict on the eigenvalues.  See ``_ALPHA_CAUSE`` for the
#: calibration.  Above it, the corruption story is back on the table.
ALPHA_HERM_KNOWN_W_RTOL = 1e-4


def _report_alpha_herm(name: str, form: str, dev, scale, worst) -> bool:
    """Host-side half of the α-Hermiticity gate (called from a debug callback).

    ``dev = max_j|Im α_j|`` (or ``max_j max|α_j − α_jᴴ|`` for the block
    variants), ``scale = max_j|α_j|``, ``worst`` = the iteration index where
    ``dev`` was attained, which is what a human needs to know next.

    THREE bands, because the residual has three meanings and one message
    cannot carry all of them:

    ``rel <= ALPHA_HERM_RTOL`` (1e-9)
        Round-off.  One OK line.

    ``ALPHA_HERM_RTOL < rel <= ALPHA_HERM_KNOWN_W_RTOL`` (1e-4)
        The band every BSE run on this stack currently lands in (~1.3e-6).
        This is a TRUE POSITIVE and is NOT silenced — but it is a defect in
        the direct per-q V_q assembly worth 0.0003 ueV on the lowest 12
        eigenvalues, so it is reported as a WARNING about how the operator
        was built rather than as a verdict on the solve.  It deliberately does NOT carry the
        ``LORRAX SANITY FAILURE`` token and does NOT raise under
        ``LORRAX_SANITY=strict``: that token means "this run's outputs are
        suspect", and at 1e-6 they demonstrably are not.  Firing it at
        FAILURE severity on every single BSE run is why it was being
        ignored, which is the worst state for a gate to be in.

    ``rel > ALPHA_HERM_KNOWN_W_RTOL``
        Out of the calibrated regime.  Full failure path, ``strict`` raises.

    WHEN THIS BANDING IS WRONG: if the V_q assembly is fixed at its source
    and the residual drops to round-off, the middle band should never fire
    again — and if it still does, the calibration in ``_ALPHA_CAUSE`` no
    longer describes reality and the threshold must come back down.
    Equally, a middle-band residual that VARIES between otherwise identical
    runs is not the tile (the tile is deterministic); that is
    the collective-corruption signature and it deserves the failure path
    even at 1e-6.  Neither condition can be detected from a single scalar,
    which is why both are named in the message rather than coded here.
    """
    from common import sanity

    dev = float(np.asarray(dev))
    scale = float(np.asarray(scale))
    worst = int(np.asarray(worst))
    rel = dev / scale if scale > 0.0 else dev
    label, why = _ALPHA_FORMS[form]
    finite = np.isfinite(dev) and np.isfinite(scale)

    if finite and ALPHA_HERM_RTOL < rel <= ALPHA_HERM_KNOWN_W_RTOL:
        try:
            first = jax.process_index() == 0
        except Exception:
            first = True
        if first:
            print(
                f"  *** LORRAX WARNING: {name} alpha (Hermitian form "
                f"{label}) is not Hermitian to round-off: "
                f"max|A-Aᴴ|/max|A| = {rel:.3e} > {ALPHA_HERM_RTOL:.0e} "
                f"(abs {dev:.3e} on scale {scale:.3e}, worst j={worst}).  "
                f"{why}{_ALPHA_CAUSE}  This residual is INSIDE the "
                f"calibrated band (<= {ALPHA_HERM_KNOWN_W_RTOL:.0e}), so "
                f"the eigenvalues from this solve are usable; the defect is "
                f"in the operator's construction and should be fixed there. "
                f"***",
                flush=True)
        return False

    ok = sanity.report_hermitian_residual(
        f"{name} alpha (Hermitian form {label})", dev, scale,
        rtol=ALPHA_HERM_KNOWN_W_RTOL, always=True,
        cause=(f"This is {rel / ALPHA_HERM_KNOWN_W_RTOL:.0f}x ABOVE the "
               f"calibrated W-asymmetry band, so the known benign cause does "
               f"NOT explain it."),
        detail=f"Worst iteration: j={worst}.  {why}{_ALPHA_CAUSE}",
    )
    if ok:
        try:
            first = jax.process_index() == 0
        except Exception:
            first = True
        if first:
            print(f"  lanczos[{name}]: alpha non-Hermitian part / "
                  f"max|alpha| = {rel:.3e} "
                  f"(tol {ALPHA_HERM_RTOL:.0e}, worst j={worst})  OK",
                  flush=True)
    return ok


def _emit_alpha_herm(name: str, alpha_im, alpha_re,
                     form: str = "vec") -> None:
    """Reduce the per-iteration α residual to 3 scalars and ship them out.

    ``alpha_im`` : (n_iter,) real — |Im α_j| (scalar variants) or
    ``max|α_j − α_jᴴ|`` (block variants).  ``alpha_re`` : (n_iter,) real —
    |α_j| (or ``max|α_j|``).  Traced-safe: pure reductions plus one unordered
    ``jax.debug.callback``, no collectives, no device sync.
    """
    dev = jnp.max(alpha_im)
    scale = jnp.max(alpha_re)
    worst = jnp.argmax(alpha_im)
    jax.debug.callback(
        lambda d, s, w: _report_alpha_herm(name, form, d, s, w),
        dev, scale, worst)


def _block_alpha_stats(alpha_all):
    """Per-block ``(max|α − αᴴ|, max|α|)`` for a stack of (bs, bs) α blocks.

    Exactly ``check_hermitian``'s two numbers, evaluated per iteration on a
    tile that is already resident.  O(n_iter · bs²) — free.
    """
    herm = jnp.conj(jnp.swapaxes(alpha_all, -1, -2))
    dev = jnp.max(jnp.abs(alpha_all - herm), axis=(-2, -1))
    scale = jnp.max(jnp.abs(alpha_all), axis=(-2, -1))
    return dev, scale


def block_lanczos_eig(
    matvec: Callable[[jax.Array], jax.Array],
    shape: tuple[int, ...],
    n_eig: int = 20,
    block_size: int = 4,
    max_iter: int = 50,
    tol: float = 1e-8,
    seed: int = 42,
) -> tuple[jax.Array, jax.Array]:
    """Block Lanczos for lowest eigenvalues of a Hermitian operator.

    Parameters
    ----------
    matvec : (block_size, *shape) -> (block_size, *shape)
        Hermitian matvec operating on a block of vectors.
    shape : tuple
        Shape of a single state vector.
    n_eig : int
        Number of lowest eigenvalues to compute.
    block_size : int
        Number of vectors per Lanczos block.
    max_iter : int
        Maximum number of block iterations.
    tol : float
        Convergence tolerance on beta norm.
    seed : int
        Random seed for initial vectors.

    Returns
    -------
    eigenvalues : (n_eig,)
    eigenvectors : (n_eig, *shape)
    """
    n_flat = int(np.prod(shape))
    key = jax.random.PRNGKey(seed)

    k1, k2 = jax.random.split(key)
    Q0 = jax.random.normal(k1, (block_size, *shape), dtype=jnp.float64)
    Q0 = Q0 + 1j * jax.random.normal(k2, (block_size, *shape), dtype=jnp.float64)

    Q0_flat = Q0.reshape(block_size, n_flat)
    Q0_flat, _ = jnp.linalg.qr(Q0_flat.T)
    Q0_flat = Q0_flat.T

    Q_blocks = [Q0_flat]
    alpha_blocks = []
    beta_blocks = []

    Q_current = Q0_flat.reshape(block_size, *shape)

    for j in range(max_iter):
        Z = matvec(Q_current)
        Z_flat = Z.reshape(block_size, n_flat)
        Q_current_flat = Q_current.reshape(block_size, n_flat)

        alpha_j = Q_current_flat.conj() @ Z_flat.T
        alpha_blocks.append(alpha_j)

        Z_flat = Z_flat - alpha_j.T @ Q_current_flat
        if j > 0:
            Q_prev_flat = Q_blocks[-2]
            Z_flat = Z_flat - beta_blocks[-1].T @ Q_prev_flat

        for Q_old in Q_blocks:
            proj = Z_flat @ Q_old.conj().T
            Z_flat = Z_flat - proj @ Q_old

        Z_flat_T, R = jnp.linalg.qr(Z_flat.T)
        # P1: beta_j = R (NOT R.T).  With Z_col = Z_flat.T = Q R, the block
        # recurrence residual is Z = Q_{j+1} R, so the sub-diagonal block of T is
        # exactly R and the super-diagonal is R^H.  The old ``R.T`` transposed
        # both off-diagonal blocks (they came out conj(R)/R.T instead of R/R^H),
        # which the final (T+T^H)/2 masked for eigenVALUES but corrupted the
        # T->Q_all mapping used for eigenVECTORS (solver_program P1).
        beta_j = R
        beta_blocks.append(beta_j)

        beta_norm = jnp.linalg.norm(beta_j)
        if beta_norm < tol * block_size:
            print(f"Block Lanczos converged at iteration {j+1}")
            break

        Q_next_flat = Z_flat_T.T
        Q_blocks.append(Q_next_flat)
        Q_current = Q_next_flat.reshape(block_size, *shape)

    # α-Hermiticity gate.  ``alpha_j = Q_jᴴ H Q_j`` is a Hermitian Gram block
    # for any Q_j; the (T + Tᴴ)/2 below would silently absorb any violation.
    # This path is eager, so ``check_hermitian`` applies verbatim to the tiny
    # (n_blocks, bs, bs) stack — no callback needed.
    from common import sanity
    sanity.check_hermitian(
        "block_lanczos_eig alpha (Hermitian form Q^H H Q)",
        jnp.stack(alpha_blocks), rtol=ALPHA_HERM_RTOL)

    n_blocks = len(alpha_blocks)
    T_size = n_blocks * block_size
    T = jnp.zeros((T_size, T_size), dtype=jnp.complex128)

    for i, alpha in enumerate(alpha_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        T = T.at[start:end, start:end].set(alpha)

        if i < len(beta_blocks) - 1:
            beta = beta_blocks[i]
            T = T.at[end:end + block_size, start:end].set(beta)
            T = T.at[start:end, end:end + block_size].set(beta.conj().T)

    T = (T + T.conj().T) / 2

    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T.real)[:n_eig]
    eigenvalues = evals_T[idx].real

    Q_all = jnp.concatenate(Q_blocks[:n_blocks], axis=0)
    eigenvectors_flat = vecs_T[:, idx].T @ Q_all
    eigenvectors = eigenvectors_flat.reshape(n_eig, *shape)

    norms = jnp.linalg.norm(eigenvectors.reshape(n_eig, -1), axis=1, keepdims=True)
    eigenvectors = eigenvectors / norms.reshape(n_eig, *([1] * len(shape)))

    return eigenvalues, eigenvectors


def simple_lanczos_eig(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    max_iter: int = 100,
    seed: int = 42,
) -> tuple[jax.Array, jax.Array]:
    """Simple Lanczos with full reorthogonalization (Python loop).

    Parameters
    ----------
    matvec : (n,) -> (n,)
        Hermitian matvec on flat vectors.
    n : int
        Vector dimension.
    n_eig : int
        Number of lowest eigenvalues to compute.
    max_iter : int
        Maximum Lanczos iterations.  Clamped to ``n``.
    seed : int
        Random seed for initial vector.

    Returns
    -------
    eigenvalues : (n_eig,)
    eigenvectors : (n_eig, n)

    Krylov-exhaustion clamp: ``max_iter`` is clamped to ``n`` for the same
    reason as the block variants (196c30b) and ``lanczos_eig_jit``.  This
    loop has a ``beta < 1e-12`` break, which catches EXACT exhaustion — but
    the break is on a traced value read in Python, so it only works because
    this variant is not jitted, and it does not protect against beta merely
    becoming tiny rather than zero.  The clamp is the structural bound; the
    break is the opportunistic early exit.
    """
    max_iter = max(1, min(int(max_iter), int(n)))
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    q = jax.random.normal(k1, (n,), dtype=jnp.float64)
    q = q + 1j * jax.random.normal(k2, (n,), dtype=jnp.float64)
    q = q / jnp.linalg.norm(q)

    Q = jnp.zeros((n, max_iter + 1), dtype=jnp.complex128)
    Q = Q.at[:, 0].set(q)
    alpha = jnp.zeros((max_iter,), dtype=jnp.float64)
    beta = jnp.zeros((max_iter,), dtype=jnp.float64)
    # |Im α_j| — the half of the Hermitian form this loop used to discard.
    alpha_im = jnp.zeros((max_iter,), dtype=jnp.float64)

    for j in range(max_iter):
        z = matvec(q)
        alpha_c = jnp.vdot(q, z)           # ONE dot; both halves are used.
        alpha = alpha.at[j].set(alpha_c.real)
        alpha_im = alpha_im.at[j].set(jnp.abs(alpha_c.imag))

        if j > 0:
            z = z - beta[j - 1] * Q[:, j - 1]
        z = z - alpha[j] * q

        for i in range(j + 1):
            proj = jnp.vdot(Q[:, i], z)
            z = z - proj * Q[:, i]

        beta = beta.at[j].set(jnp.linalg.norm(z))
        if beta[j] < 1e-12:
            max_iter = j + 1
            break

        q = z / beta[j]
        Q = Q.at[:, j + 1].set(q)

    _emit_alpha_herm("simple_lanczos_eig",
                     alpha_im[:max_iter], jnp.abs(alpha[:max_iter]))

    T = jnp.diag(alpha[:max_iter])
    if max_iter > 1:
        off = beta[:max_iter - 1]
        T = T + jnp.diag(off, 1) + jnp.diag(off, -1)

    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]

    eigenvalues = evals_T[idx]
    eigenvectors = (Q[:, :max_iter] @ vecs_T[:, idx]).T

    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / norms

    return eigenvalues, eigenvectors


def lanczos_eig_jit(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    max_iter: int = 100,
    seed: int = 42,
    n_reorth: int = FULL_REORTH,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled Lanczos using lax.fori_loop.

    Parameters
    ----------
    matvec : (n,) -> (n,)
        Hermitian matvec on flat vectors.
    n : int
        Vector dimension.
    n_eig : int
        Number of lowest eigenvalues to compute.
    max_iter : int
        Maximum Lanczos iterations (fixed for JIT).  Clamped to ``n`` — see
        the Krylov-exhaustion note below.
    seed : int
        Random seed for initial vector.
    n_reorth : int
        Window size for partial reorthogonalization.  Default
        ``FULL_REORTH`` (-1) = reorthogonalise against the whole basis; see
        the module-level note on why a finite window is not a tunable here.

    Returns
    -------
    eigenvalues : (n_eig,)
    eigenvectors : (n_eig, n)

    Krylov-exhaustion clamp: ``max_iter`` is clamped to ``n``.  Identical
    rationale to the block variants (added there in 196c30b and absent here
    until now): past ``n`` steps the Krylov space is the whole space, the
    residual collapses, ``z / max(beta, 1e-15)`` manufactures a junk
    direction from the 1e-15 floor, and the resulting α/β put Ritz values
    anywhere — including BELOW the true spectrum.  Latent on the decks we
    run today (2400 iterations against n=4096) but one deck edit from
    firing, and this is the path production takes at ``block_size=1,
    rtol=0`` — the default, and the pinned test.
    """
    max_iter = max(1, min(int(max_iter), int(n)))
    n_reorth = resolve_n_reorth(n_reorth, max_iter)
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    q0 = jax.random.normal(k1, (n,), dtype=jnp.float64)
    q0 = q0 + 1j * jax.random.normal(k2, (n,), dtype=jnp.float64)
    q0 = q0 / jnp.linalg.norm(q0)

    # +1 column so the last iteration does not overwrite Q[:, max_iter-1] (P1).
    Q = jnp.zeros((n, max_iter + 1), dtype=jnp.complex128)
    Q = Q.at[:, 0].set(q0)
    alpha = jnp.zeros((max_iter,), dtype=jnp.float64)
    beta = jnp.zeros((max_iter,), dtype=jnp.float64)
    # |Im α_j| — carried alongside α, checked once after the loop.  Two extra
    # float64 slots of carry; see the module header for the cost argument.
    alpha_im = jnp.zeros((max_iter,), dtype=jnp.float64)

    def lanczos_step(j, carry):
        Q, alpha, beta, alpha_im, q_prev = carry
        z = matvec(q_prev)

        # ONE complex dot product.  ``.real`` drives the recurrence; ``.imag``
        # is the Hermitian-form residual that used to be discarded here.
        alpha_c = jnp.vdot(q_prev, z)
        alpha_j = alpha_c.real
        alpha = alpha.at[j].set(alpha_j)
        alpha_im = alpha_im.at[j].set(jnp.abs(alpha_c.imag))

        z = z - alpha_j * q_prev
        q_prev_prev = Q[:, jnp.maximum(j - 1, 0)]
        beta_prev = jnp.where(j > 0, beta[j - 1], 0.0)
        z = z - beta_prev * q_prev_prev

        def reorth_body(i, z_acc):
            valid = i < j
            q_i = Q[:, i]
            proj = jnp.where(valid, jnp.vdot(q_i, z_acc), 0.0 + 0j)
            return z_acc - proj * q_i

        start_idx = jnp.maximum(0, j - n_reorth)
        z = lax.fori_loop(start_idx, j + 1, reorth_body, z)

        beta_j = jnp.linalg.norm(z)
        beta = beta.at[j].set(beta_j)

        q_next = z / jnp.maximum(beta_j, 1e-15)
        Q = Q.at[:, j + 1].set(q_next)

        return (Q, alpha, beta, alpha_im, q_next)

    init_carry = (Q, alpha, beta, alpha_im, q0)
    Q, alpha, beta, alpha_im, _ = lax.fori_loop(
        0, max_iter, lanczos_step, init_carry)

    _emit_alpha_herm("lanczos_eig_jit", alpha_im, jnp.abs(alpha))

    T = jnp.diag(alpha)
    off_diag = beta[:-1]
    T = T + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)

    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]
    eigenvalues = evals_T[idx]

    eigenvectors = (Q[:, :max_iter] @ vecs_T[:, idx]).T
    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / jnp.maximum(norms, 1e-15)

    return eigenvalues, eigenvectors


def _build_block_tridiag(alpha_all, beta_all, max_iter: int, bs: int):
    """Build the block-tridiagonal T from per-iter (bs,bs) blocks.

    Done inside the jit by ``lax.fori_loop`` so the trace-time HLO stays
    O(1) instead of unrolling ``max_iter`` slot updates. Used by both
    the fixed-iter and convergence-driven block Lanczos paths.
    """
    T_size = bs * max_iter
    T = jnp.zeros((T_size, T_size), dtype=jnp.complex128)

    def body(j, T):
        s = j * bs
        T = lax.dynamic_update_slice(T, alpha_all[j], (s, s))
        # Off-diagonal beta only when j+1 < max_iter (zero alpha/beta past
        # the end keeps the slot a no-op even when j is at the boundary).
        T = lax.dynamic_update_slice(T, beta_all[j], (s + bs, s))
        T = lax.dynamic_update_slice(
            T, jnp.conj(beta_all[j]).T, (s, s + bs))
        return T

    T = lax.fori_loop(0, max_iter - 1, body, T)
    # Final diagonal block (no off-diagonal past the end).
    s_last = (max_iter - 1) * bs
    T = lax.dynamic_update_slice(T, alpha_all[max_iter - 1], (s_last, s_last))
    return (T + jnp.conj(T).T) * 0.5


def block_lanczos_eig_jit(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    block_size: int = 4,
    max_iter: int = 50,
    seed: int = 42,
    n_reorth: int = FULL_REORTH,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled block Lanczos using ``lax.fori_loop``.

    Same algorithm as :func:`block_lanczos_eig`, but all state lives in
    pre-allocated arrays so the body fits in ``lax.fori_loop`` and the
    caller's outer jit can fuse this with the matvec.  The matvec
    operates on a *block* of trial vectors

        matvec : (block_size, n) -> (block_size, n)

    so the BSE-style ring matvec processes ``block_size`` vectors per
    call.  That makes the per-call GEMMs ``block_size`` times larger
    (better arithmetic intensity / GPU occupancy) and reduces the host
    dispatch count by ``block_size`` for the same total Krylov
    dimension.

    The total Krylov dimension is ``block_size * max_iter``; pick
    ``max_iter`` so this is comparable to a single-vector Lanczos's
    ``max_iter``.

    Parameters
    ----------
    matvec : (block_size, n) -> (block_size, n)
        Hermitian matvec on a block of flat vectors.
    n : int
        Single-vector dimension.
    n_eig : int
        Number of lowest eigenvalues to compute.
    block_size : int
        Vectors per Lanczos block.
    max_iter : int
        Block iterations (fixed for JIT). Total Krylov size = block_size·max_iter.
    seed : int
        Random seed for initial block.
    n_reorth : int
        Window size (in *blocks*) for partial reorthogonalisation.  Default
        ``FULL_REORTH`` (-1) = the whole basis; see the module-level
        reorthogonalisation note for why a finite window is not a default.

    Krylov-exhaustion clamp: the Krylov space cannot exceed the vector
    space, so ``max_iter`` is clamped to ``floor(n / block_size)``.
    Running past exhaustion is not benign — the residual block collapses,
    QR of a ~zero block returns junk directions, and the manufactured
    α/β blocks put Ritz values ANYWHERE, including BELOW the true
    spectrum (measured on the 4v4c MoS2 exciton window, n=144 with a
    requested 320-dim Krylov: spurious states 60-100 meV under the dense
    ground state).  At the clamp the Krylov space spans (almost) the
    whole space and the extremal Ritz values are dense-quality.
    """
    bs = int(block_size)
    max_iter = max(1, min(int(max_iter), int(n) // bs))
    n_reorth = resolve_n_reorth(n_reorth, max_iter)
    T_size = bs * int(max_iter)

    # Initial orthonormal block via QR of random complex Gaussian.
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    Q0 = (jax.random.normal(k1, (n, bs), dtype=jnp.float64)
          + 1j * jax.random.normal(k2, (n, bs), dtype=jnp.float64))
    Q0, _ = jnp.linalg.qr(Q0)                          # (n, bs)

    # Ring buffer of all Q-blocks: (max_iter + 1, n, bs) — the +1 slot holds the
    # final Q_next so the last iteration does NOT overwrite Q_{max_iter-1} (the
    # slot-overwrite bug, solver_program P1: it corrupted the last Krylov block
    # in the eigenvector reconstruction).  alpha/beta: (max_iter, bs, bs).
    Q_all = jnp.zeros((int(max_iter) + 1, n, bs), dtype=jnp.complex128)
    Q_all = Q_all.at[0].set(Q0)
    alpha_all = jnp.zeros((int(max_iter), bs, bs), dtype=jnp.complex128)
    beta_all = jnp.zeros((int(max_iter), bs, bs), dtype=jnp.complex128)

    def body(j, carry):
        Q_all, alpha_all, beta_all = carry
        Q_j = Q_all[j]                                 # (n, bs)
        # Block matvec over (bs, n) → (bs, n); transpose to (n, bs).
        Z = matvec(Q_j.T).T                            # (n, bs)

        alpha_j = jnp.conj(Q_j).T @ Z                  # (bs, bs)
        alpha_all = alpha_all.at[j].set(alpha_j)
        Z = Z - Q_j @ alpha_j

        # Subtract Q_{j-1} · β_{j-1}^H (skip on j=0).
        Q_jm1 = Q_all[jnp.maximum(j - 1, 0)]
        beta_prev = beta_all[jnp.maximum(j - 1, 0)]
        Z = jnp.where(j > 0, Z - Q_jm1 @ jnp.conj(beta_prev).T, Z)

        # Partial reorth over the last n_reorth blocks.
        def reorth_body(i, Z_acc):
            valid = i < j
            Q_i = Q_all[i]
            proj = jnp.where(valid, jnp.conj(Q_i).T @ Z_acc, jnp.zeros((bs, bs), dtype=Z_acc.dtype))
            return Z_acc - Q_i @ proj
        start = jnp.maximum(0, j - n_reorth)
        Z = lax.fori_loop(start, j + 1, reorth_body, Z)

        # QR(Z) → next block + β_j.  Write to slot j+1 (always valid with the
        # +1 buffer, 1..max_iter) — no clobber of the current Q_j.
        Q_next, beta_j = jnp.linalg.qr(Z)              # (n, bs), (bs, bs)
        beta_all = beta_all.at[j].set(beta_j)
        Q_all = Q_all.at[j + 1].set(Q_next)
        return (Q_all, alpha_all, beta_all)

    Q_all, alpha_all, beta_all = lax.fori_loop(
        0, int(max_iter), body, (Q_all, alpha_all, beta_all))

    # α-Hermiticity gate, BEFORE _build_block_tridiag's (T + Tᴴ)/2 absorbs it.
    _emit_alpha_herm("block_lanczos_eig_jit", *_block_alpha_stats(alpha_all),
                     form="block")

    # Block-tridiagonal T built inside-jit (no Python loop unroll).
    T = _build_block_tridiag(alpha_all, beta_all, int(max_iter), bs)

    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]
    eigenvalues = evals_T[idx]

    # Q_all is (max_iter + 1, n, bs); the T basis is the first max_iter blocks.
    Q_full = jnp.transpose(Q_all[:int(max_iter)], (1, 0, 2)).reshape(n, T_size)
    eigenvectors = (Q_full @ vecs_T[:, idx]).T          # (n_eig, n)
    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / jnp.maximum(norms, 1e-15)
    return eigenvalues, eigenvectors


def block_lanczos_eig_jit_converged(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    block_size: int = 4,
    max_iter: int = 50,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    check_every: int = 4,
    min_iter: int | None = None,
    seed: int = 42,
    n_reorth: int = FULL_REORTH,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convergence-driven block Lanczos via ``lax.while_loop``.

    Same algorithm as :func:`block_lanczos_eig_jit`, but the iteration
    count is decided by Ritz-eigenvalue stability rather than fixed:

      every ``check_every`` block-iters, build the partial T (with
      future-block α/β set to zero), eigh it, and compare the lowest
      ``n_eig`` Ritz values against the previous check.  Exit when

          max_i |λ_i - λ_i_prev| < rtol·max(|λ_i|, atol).

    The pre-allocated buffers fix the upper bound at ``max_iter`` block
    iterations (so ``max_iter * block_size`` total Krylov dimension);
    the ``while_loop`` carry includes the running iteration count and
    the previous Ritz values for comparison.

    Returns (eigenvalues, eigenvectors, n_iter_done) — the third value
    is the actual block iteration count where the loop exited (≤
    ``max_iter``).
    """
    bs = int(block_size)
    # Krylov-exhaustion clamp — same rationale as block_lanczos_eig_jit:
    # past floor(n/bs) blocks the residual collapses and QR manufactures
    # junk directions with arbitrary (even sub-spectrum) Ritz values.
    M = max(1, min(int(max_iter), int(n) // bs))
    n_reorth = resolve_n_reorth(n_reorth, M)
    T_size = bs * M
    if min_iter is None:
        min_iter = max(2 * check_every, max(1, n_eig // bs + 1))
    min_iter = int(min(min_iter, M))

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    Q0 = (jax.random.normal(k1, (n, bs), dtype=jnp.float64)
          + 1j * jax.random.normal(k2, (n, bs), dtype=jnp.float64))
    Q0, _ = jnp.linalg.qr(Q0)

    # +1 Krylov slot so the final block does not overwrite Q_{M-1} (P1).
    Q_all = jnp.zeros((M + 1, n, bs), dtype=jnp.complex128).at[0].set(Q0)
    alpha_all = jnp.zeros((M, bs, bs), dtype=jnp.complex128)
    beta_all = jnp.zeros((M, bs, bs), dtype=jnp.complex128)
    last_evals = jnp.full((n_eig,), jnp.inf, dtype=jnp.float64)
    converged = jnp.bool_(False)

    def step(j, Q_all, alpha_all, beta_all):
        Q_j = Q_all[j]
        Z = matvec(Q_j.T).T
        alpha_j = jnp.conj(Q_j).T @ Z
        alpha_all = alpha_all.at[j].set(alpha_j)
        Z = Z - Q_j @ alpha_j
        Q_jm1 = Q_all[jnp.maximum(j - 1, 0)]
        beta_prev = beta_all[jnp.maximum(j - 1, 0)]
        Z = jnp.where(j > 0, Z - Q_jm1 @ jnp.conj(beta_prev).T, Z)

        def reorth_body(i, Z_acc):
            valid = i < j
            Q_i = Q_all[i]
            proj = jnp.where(
                valid, jnp.conj(Q_i).T @ Z_acc,
                jnp.zeros((bs, bs), dtype=Z_acc.dtype))
            return Z_acc - Q_i @ proj
        start = jnp.maximum(0, j - n_reorth)
        Z = lax.fori_loop(start, j + 1, reorth_body, Z)

        Q_next, beta_j = jnp.linalg.qr(Z)
        beta_all = beta_all.at[j].set(beta_j)
        Q_all = Q_all.at[j + 1].set(Q_next)
        return Q_all, alpha_all, beta_all

    def cond(state):
        j, _, _, _, _, conv = state
        return jnp.logical_and(j < M, jnp.logical_not(conv))

    def body(state):
        j, Q_all, alpha_all, beta_all, last_evals, _ = state
        Q_all, alpha_all, beta_all = step(j, Q_all, alpha_all, beta_all)

        # Convergence check — only every ``check_every`` iters and after
        # ``min_iter`` warmup. ``jax.lax.cond`` keeps both branches
        # constant-cost (no Python-level branching).
        do_check = jnp.logical_and(
            (j + 1) >= min_iter,
            ((j + 1) % check_every) == 0,
        )

        def _check_branch(args):
            alpha_all, beta_all, last_evals, j_done = args
            T = _build_block_tridiag(alpha_all, beta_all, M, bs)
            # Mask inactive (zero) part of T by adding a large constant
            # to its diagonal — pushes inactive eigvals out of the
            # spectrum so jnp.sort()[:n_eig] picks only real Ritz vals.
            LARGE = jnp.asarray(1.0e6, dtype=T.real.dtype)
            pos = jnp.arange(M * bs)
            active = (j_done + 1) * bs                        # completed iters × bs
            mask = (pos >= active).astype(T.real.dtype) * LARGE
            T = T + jnp.diag(mask).astype(T.dtype)
            ev = jnp.linalg.eigvalsh(T)
            ev = jnp.sort(ev)[:n_eig]
            scale = jnp.maximum(jnp.abs(ev), atol)
            delta = jnp.max(jnp.abs(ev - last_evals) / scale)
            new_conv = delta < rtol
            return ev, new_conv

        def _skip_branch(args):
            _, _, last_evals, _ = args
            return last_evals, jnp.bool_(False)

        new_evals, new_conv = lax.cond(
            do_check, _check_branch, _skip_branch,
            (alpha_all, beta_all, last_evals, j),
        )
        return (j + 1, Q_all, alpha_all, beta_all, new_evals, new_conv)

    init = (jnp.int32(0), Q_all, alpha_all, beta_all, last_evals, converged)
    j_final, Q_all, alpha_all, beta_all, _, _ = lax.while_loop(cond, body, init)

    # α-Hermiticity gate, BEFORE _build_block_tridiag's (T + Tᴴ)/2 absorbs it.
    # Blocks past ``j_final`` were never written and are exactly zero, so they
    # contribute 0 to both the deviation and the scale — no mask needed.
    _emit_alpha_herm("block_lanczos_eig_jit_converged",
                     *_block_alpha_stats(alpha_all), form="block")

    # Final eigh — mask inactive blocks the same way as the convergence
    # check, otherwise the zero eigvals from unfilled iters dominate
    # ``sort()[:n_eig]``.
    T = _build_block_tridiag(alpha_all, beta_all, M, bs)
    LARGE_F = jnp.asarray(1.0e6, dtype=T.real.dtype)
    pos_F = jnp.arange(T_size)
    active_F = j_final * bs
    mask_F = (pos_F >= active_F).astype(T.real.dtype) * LARGE_F
    T = T + jnp.diag(mask_F).astype(T.dtype)
    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]
    eigenvalues = evals_T[idx]
    Q_full = jnp.transpose(Q_all[:M], (1, 0, 2)).reshape(n, T_size)
    eigenvectors = (Q_full @ vecs_T[:, idx]).T
    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / jnp.maximum(norms, 1e-15)
    return eigenvalues, eigenvectors, j_final
