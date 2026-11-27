"""
solvers/lanczos.py — the simple Lanczos eigensolver and its Hermiticity gate.

``simple_lanczos_eig`` finds the lowest n_eig eigenvalues of a Hermitian
operator H given only a callable matvec: a Python loop with full
reorthogonalization, used for spectral bounds (``solvers.dos``).  No physics
knowledge — works for any Hermitian eigenproblem.  (The jitted block Lanczos
and its BSE route were retired 2026-11-27; the Davidson is the BSE
eigensolver.)

It carries the **α-Hermiticity invariant** — see the section below
``import numpy as np``.  ``⟨q, Hq⟩`` is real for a Hermitian H, so the
imaginary part of ``α`` (which the recurrence computes and used to discard) is
a free detector for "the matvec did not return H·q".  It is always on because
it is free.

``_supported_start_block`` is the random start block restricted to a
physical-transition support, used by the Davidson's start subspace
(``bse.bse_davidson_helpers``).
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from distrib_la import eigh_local

from common.log import get_logger

log = get_logger(__name__)


# ===========================================================================
# The Hermitian-form invariant the recurrence already computes
# ===========================================================================
#
# WHY THIS IS HERE
# ----------------
# The Lanczos recurrence forms
#
#     α_j = ⟨q_j, H q_j⟩
#
# and would throw away the part of it that is a free integrity check:
# ``.real`` discards ``Im α``.  For a Hermitian H that discarded quantity is
# ZERO in exact arithmetic — a Hermitian form has a real value — for ANY q,
# converged or not, at every iteration.
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
# magnitude above the threshold.  There is no tuning freedom in that gap.  It
# is not chosen to make a test pass; it is the round-off bound rounded up.
#
# CORRECTION (2026-08-08).  This paragraph used to end "any tolerance in
# [1e-11, 1e-4] gives the same verdict on every case we have".  A case now
# exists that the window does NOT classify uniformly: si_bse_debug, on the
# Miller-(0,0,0) head-slot rule, measures rel ~1.2e-06 -- squarely inside it.
# The claim was too strong and is retired rather than re-asserted.  The
# CONSTANT is untouched and stays untouched: with the head-slot defect fixed
# (see ``_ALPHA_CAUSE``) the same deck measures 3.165e-14 against the same
# 1e-9, i.e. 4.5 orders of headroom, so the tolerance has now been validated
# end to end rather than merely derived.
#
# COST — why this is always-on and not behind LORRAX_SANITY
# ---------------------------------------------------------
# ``jnp.vdot`` already produces a complex scalar, so ``.imag``
# is the half of it that was being discarded — at worst two extra length-n
# multiply-accumulate passes, against a matvec that is orders of magnitude more
# expensive, and against the reorthogonalisation's own dot products
# in the same loop body.  It adds no collective, device
# sync, or full-tile pass.  Per the owner's rule, a free invariant is
# always-on: it reports through ``report_hermitian_residual(..., always=True)``,
# which bypasses the ``LORRAX_SANITY`` *cost* escape hatch while still honouring
# ``strict``.
#
ALPHA_HERM_RTOL = 1e-9

_ALPHA_FORMS = {
    "vec": ("<q,Hq>",
            "alpha_j = <q_j, H q_j> is REAL for any Hermitian H and any q_j, "
            "so a nonzero imaginary part"),
}

# WHY THIS TEXT WAS REWRITTEN (2026-08-08, alongside the head-slot fix on
# this branch).  It used to name the gloo ``psum_scatter`` corruption as
# "the known cause on this stack" and instruct the reader to re-run under
# ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` before believing any
# eigenvalue.  That text predates the diagnosis and is falsified three ways
# on the only deck that has ever fired this gate: the residual is
# bit-identical across fourteen P=4 runs (the corruption is
# non-deterministic, ~5% of executions), it is present at P=1 where there is
# no collective at all (9.365e-07), and it reproduced bit-identically on a
# single-process login-node dense probe with no reduce-scatter anywhere.
# Sending every reader to the collective is a large part of why this line
# went unread on a gate that was telling the truth.
#
# The full evidence -- the stage-by-stage reciprocity walk, the dense
# control that pins the whole non-Hermiticity on the W term, the tolerance
# derivation above, and the one-commit A/B that takes 1.155e-06 to
# 3.165e-14 -- is written up in (agent workspace, not in-tree):
#     ~/lorrax_bse_perf_2026-08-08/HERMITICITY_INVESTIGATION.md
# with the A/B logs at
#     /pscratch/sd/j/jackm/vcoul_head_0808/_reports/bse4p_{control,fixed}_400.log
# and the two frozen references the fix moves catalogued in
#     ~/lorrax_service_phase/NOTE_vcoul_head_refreeze.md
_ALPHA_CAUSE = (
    " means the matvec did not return H*q -- the operator, not the "
    "algorithm, is wrong.  WHAT WAS MEASURED: dev = max_j|Im alpha_j| "
    "over the Krylov vectors "
    "this solve actually visited, divided by scale = max_j|alpha_j|; both "
    "are printed above, so the ratio is already normalised by the "
    "operator's own magnitude and cannot be explained away by the dynamic "
    "range of the tiles.  alpha_j is real (Hermitian) for ANY Hermitian H "
    "and ANY q_j, so the recurrence, the reorthogonalisation and the "
    "iteration count cannot manufacture this number.  KNOWN CAUSE on this "
    "stack, and the first thing to check: the mini-BZ Coulomb head used to "
    "be injected at the slot labelled Miller (0,0,0), and that label is "
    "NOT equivariant under q -> -q -- the BGW wrap sends a zone-boundary "
    "component to +1/2 rather than -1/2, so slot (0,0,0) at +q pairs with "
    "a slot carrying the BARE value at -q.  V_q therefore carried a "
    "q <-> -q reciprocity break, W inherited it, and the matvec transports "
    "it faithfully into alpha.  Measured on si_bse_debug: rel 1.155e-06 "
    "under the label rule, 3.165e-14 once the head is injected at "
    "argmin|q+G|^2 over the tied set.  If this tree's vcoul.v_qG_table "
    "still takes a per-q ``v_head_miniBZ`` table rather than "
    "``v_head_fn`` + ``head_tie_rtol``, it predates that fix and this is "
    "what you are looking at.  Other candidates, in order: any other "
    "non-Hermitian W/V tile fed to the matvec; a mis-transposed shard; "
    "and -- ONLY if the residual does not reproduce bit-for-bit at fixed "
    "process count -- the gloo reduce-scatter corruption "
    "(jax.lax.psum_scatter under JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo "
    "returns wrong data in ~5% of executions, always output segment 0; "
    "see wk_REL/UPSTREAM_gloo_psum_scatter_corruption.md), which re-runs "
    "clean under ...=mpi.  DETERMINISM IS THE DISCRIMINATOR: an operator "
    "defect is bit-stable at fixed configuration and survives P=1; the "
    "corruption is neither."
)


def _report_alpha_herm(name: str, form: str, dev, scale, worst) -> bool:
    """Host-side half of the α-Hermiticity gate.

    Reached from the ``jax.debug.callback`` of :func:`_emit_alpha_herm`.

    ``dev = max_j|Im alpha_j|``, ``scale = max_j|alpha_j|``, ``worst`` = the iteration index where
    ``dev`` was attained, which is what a human needs to know next.
    """
    from common import sanity

    dev = float(np.asarray(dev))
    scale = float(np.asarray(scale))
    worst = int(np.asarray(worst))
    rel = dev / scale if scale > 0.0 else dev
    label, why = _ALPHA_FORMS[form]
    ok = sanity.report_hermitian_residual(
        f"{name} alpha (Hermitian form {label})", dev, scale,
        rtol=ALPHA_HERM_RTOL, always=True,
        detail=f"Worst iteration: j={worst}.  {why}{_ALPHA_CAUSE}",
    )
    if ok:
        log.debug(f"  lanczos[{name}]: alpha non-Hermitian part / "
                  f"max|alpha| = {rel:.3e} "
                  f"(tol {ALPHA_HERM_RTOL:.0e}, worst j={worst})  OK")
    return ok


def _emit_alpha_herm(name: str, alpha_im, alpha_re,
                     form: str = "vec") -> None:
    """Reduce the per-iteration α residual to 3 scalars and ship them out.

    ``alpha_im`` : (n_iter,) real — |Im α_j|.  ``alpha_re`` : (n_iter,) real —
    |α_j|.  Traced-safe: pure reductions plus one unordered
    ``jax.debug.callback``, no collectives, no device sync.
    """
    dev = jnp.max(alpha_im)
    scale = jnp.max(alpha_re)
    worst = jnp.argmax(alpha_im)
    jax.debug.callback(
        lambda d, s, w: _report_alpha_herm(name, form, d, s, w),
        dev, scale, worst)


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
        Maximum Lanczos iterations.
    seed : int
        Random seed for initial vector.

    Returns
    -------
    eigenvalues : (n_eig,)
    eigenvectors : (n_eig, n)

    Krylov-exhaustion clamp: the Krylov space cannot exceed the vector space,
    so ``max_iter`` is clamped to ``n``.  Running past exhaustion is not
    benign — the residual collapses, the normalisation divides by ~0, and the
    manufactured alpha/beta put Ritz values ANYWHERE, including BELOW the true
    spectrum.
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

    evals_T, vecs_T = eigh_local(T)
    idx = jnp.argsort(evals_T)[:n_eig]

    eigenvalues = evals_T[idx]
    eigenvectors = (Q[:, :max_iter] @ vecs_T[:, idx]).T

    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / norms

    return eigenvalues, eigenvectors


def _supported_start_block(n, bs, seed, support, *, vector_shape=None,
                           support_shape=None, constrain=None):
    """The random complex ``(n, bs)`` start block and the supported count.

    ``support`` is a boolean mask over the ``n`` entries, ``support_shape`` the
    extent of the supported corner of ``vector_shape`` (not both): the Gaussian
    draw covers the supported entries only, in row-major order, and is
    scattered into zeros on the rest.
    """
    if support_shape is not None:
        if support is not None:
            raise ValueError('Lanczos accepts support or support_shape, not both')
        vector_shape = (int(n),) if vector_shape is None else tuple(vector_shape)
        support_shape = tuple(support_shape)
        if (len(support_shape) != len(vector_shape) or
                any(not isinstance(s, (int, np.integer)) or isinstance(s, bool)
                    or not 0 < s <= p for s, p in zip(support_shape, vector_shape))):
            raise ValueError('Lanczos support_shape must be positive integer extents within vector_shape')
        n_active = int(np.prod(support_shape))
        draw_shape = (*support_shape, bs)
        active = None
    else:
        active = (None if support is None
                  else np.flatnonzero(np.asarray(support).reshape(-1)))
        n_active = int(n) if active is None else int(active.size)
        draw_shape = (n_active, bs)
    if active is not None and int(np.size(support)) != int(n):
        raise ValueError('Lanczos support must have n entries')
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    Q0 = (jax.random.normal(k1, draw_shape, dtype=jnp.float64)
          + 1j * jax.random.normal(k2, draw_shape, dtype=jnp.float64))
    if support_shape is not None:
        # The row-major draw is identical to (n_active, bs). Keeping its
        # physical axes visible lets GSPMD generate only local RNG tiles.
        rows = jnp.moveaxis(Q0, -1, 0)
        if constrain is not None:
            rows = constrain(rows)
        rows = jnp.pad(rows, ((0, 0), *tuple(
            (0, p-s) for s, p in zip(support_shape, vector_shape))))
        Q0 = rows.reshape(bs, n).T
    if active is not None:
        Q0 = jnp.zeros((int(n), bs), dtype=Q0.dtype).at[active].set(Q0)
    return Q0, n_active
