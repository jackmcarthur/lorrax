"""Pivoted-Cholesky pruning of over-sampled ISDF candidate points.

Implements the q=0 candidate-pruning stage described in
``pivoted_cholesky.md`` (sandbox root). The idea: k-means gives a set of M
candidate points ``{r̃_a}`` (M > N_μ); the pair-product rows
``z_{a,(vck)} = φ*_{v,k}(r̃_a) ψ_{c,k}(r̃_a)`` define a Hermitian PSD Gram
matrix ``G^{(0)} ∈ ℂ^{M×M}``. Greedy pivoted Cholesky picks the N_μ pivots
with the largest residual Schur-complement diagonal, and the corresponding
``r̃_a`` become the final ISDF points. This is strictly better than picking
on amplitude alone because it targets the coherence structure of the
valence-conduction pair-product space the ISDF fit will actually use.

Architectural map to ``gw/isdf_fitting.py``:

    pair_density                      ←→  per-k open-spin P^{(v/c)}_{αβ}(a,b)
                                          (rank-5; same einsum at candidates
                                          r̃_a not chosen r_μ)
    gram_q0_from_pair                 ←→  q=0 cross-product (no k→q FFT)
    (nothing)                         ←→  pivoted_cholesky_select  (new)

The Gram is built row-sharded over the ``('x','y')`` mesh and the select
step runs on it in place: per iteration it costs one ``pmax`` for the
pivot value, one for the tie-break, and one ``psum`` of the (k_keep,)
pivot row — O(k_keep) comm, everything else local.  Column ``p`` needs no
collective at all, which is why the Gram is ROW-sharded.

Shapes (following the md):

    phi_val_cand   (nk, nv_eff, M)   complex  φ_{v,k}(r̃_a)
    psi_cond_cand  (nk, nc_eff, M)   complex  ψ_{c,k}(r̃_a)
    G              (M, M)            complex  Hermitian PSD
    L              (M, k_keep)       complex  Cholesky columns (padded)
    piv            (k_keep,)         int32    pivot indices (−1 past rank)
    d_final        (M,)              real     Schur-complement residuals

``nv_eff`` / ``nc_eff`` fold the spinor axis into the band axis
(nv_eff = nv_bands · nspinor), matching the md's "assume spin has already
been folded" convention.
"""

from __future__ import annotations

import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from common.shard_map import shard_map
from jax.experimental import multihost_utils as _mh
from functools import partial
from typing import TYPE_CHECKING

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

# SMALL_ISSUES row 38.  ``WfnLoader`` is used at MODULE SCOPE for nothing but
# two parameter annotations, and ``from __future__ import annotations`` above
# means those are strings that are never evaluated — so this import bought a
# hard dependency on ``wfn_loader`` (and, through it, h5py) for every consumer
# of this module, including the ones that only want the select kernel.  That
# edge is what let a clobbered PYTHONPATH on Perlmutter kill an h5py-less
# import of a module that does not need h5py.  Under TYPE_CHECKING the
# annotations still resolve for a type checker and cost nothing at run time.
if TYPE_CHECKING:                                                   # pragma: no cover
    from wfn_loader import WfnLoader
from common import timing
from common.collectives import device_put_process_local

from . import distribution as dist
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

import symmetry_maps                                            # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# Reference — single-device greedy pivoted Cholesky
# ═══════════════════════════════════════════════════════════════════════
#
# The production path is ``make_sharded_pivoted_cholesky_select`` below;
# this is the same greedy recurrence written without any collective, and
# it is what ``tests/test_centroid_distribution.py`` gates the sharded
# kernel against.  Keep the two in step.


# THE STOPPING RULE, AND WHY IT IS A CONTRACT AND NOT A DIAGNOSTIC
# ─────────────────────────────────────────────────────────────────────────
# Both kernels below compute ``floor = sqrt(eps) · max(diag G)`` — relative
# to the largest initial diagonal, which is the scale-invariant choice, and
# in the sharded kernel it is a ``pmax`` so it agrees across shard counts.
# Until 2026-08-07 that number was computed and then used for NOTHING but
# reporting ``rank``: the recurrence ran all ``k_keep`` iterations whatever
# the residual did.  MEASURED consequence, on this box, reproducing
# ``CENTROID_GEN_ASSESSMENT.md`` §4.1-§4.3 (probe ``pc_repro.py``):
#
#   * Gram of true rank 10, k_keep=40.  ``pivot_val`` clamps to ``eps``, so
#     ``denom = sqrt(eps) ≈ 1.5e-8`` and each column is the previous one
#     SQUARED over 1.5e-8 — a geometric blow-up.  Column norms past the
#     cliff went 3.2e-07 … 5.2e+04, 2.2e+16, 1.1e+39, 1.4e+85, inf, nan;
#     the first non-finite column was j = 22, twelve iterations past a true
#     rank of 10.  ``argmax`` over NaN returns the FIRST NaN index, so the
#     pivots past the rank came back ``0 1 3 4 5 7 8 9 10 11 13 …`` — the
#     first unpicked indices, chosen by array position and not by any
#     residual criterion.  Those are real candidate indices, so the pad
#     guard below did not fire and the caller got what looked like a normal
#     pivot list.  ``d_taken`` is masked to clean zeros past ``rank``, so
#     the three numbers the wrapper prints never showed it; only
#     ``trR_over_trG`` came back NaN.
#   * Orbit mode, M=96 in 12 orbits of 8, k_keep=20.  Once every orbit is
#     inactive ``masked_d`` is uniformly −inf, ``pivot_val`` clamps to
#     ``eps`` rather than NaN and ``argmax`` over a uniform array returns 0:
#     pivots came back ``[54 40 30 86 72 6 32 62 22 88 8 64 0 0 0 0 0 0 0 0]``
#     — twelve genuine pivots, then index 0 repeated eight times, finite
#     arithmetic and a nonsense answer.
#   * Indefinite Gram (λ_min = −4.95e-01 against λ_max = 4.95e+02).  Reported
#     rank 23 of 24, all 24 pivots distinct, ``L`` entirely finite,
#     ``min(d_final)`` exactly 0.0 — NO signal anywhere in the return tuple
#     that the input was not positive semidefinite, because the
#     ``jnp.maximum(…, 0.0)`` on the Schur update destroys the classic PSD
#     detector before it can be observed.  LAPACK's ``pstrf`` returns
#     ``INFO > 0`` for exactly this case.
#
# The fix is three lines of arithmetic and no new machinery:
#
#   1. ``denom`` clamps at ``floor`` instead of at ``eps``.  On a healthy
#      input ``pivot_val > floor`` so ``max(pivot_val, floor) == pivot_val``
#      and every number is BIT-IDENTICAL to before; past the cliff the
#      divisor can no longer be 1.5e-8 and the blow-up has no fuel.
#   2. ``take = pivot_val > floor`` gates the writes.  Past the stop ``L``
#      takes exact zeros, ``piv`` keeps the −1 sentinel it was initialised
#      with, ``d`` and ``trR_over_trG`` freeze.  ``rank`` is then exactly
#      the number of pivots taken — a HARD CONTRACT, not a diagnostic —
#      and the caller's sentinel guard fires naturally.  This is deliberately
#      NOT a ``lax.cond``: the predicate is a ``pmax`` result and therefore
#      identical on every shard, but putting the sharded kernel's
#      collectives inside a conditional region is a hazard for no gain,
#      since the halted branch costs the same as a taken one either way.
#   3. The Schur update keeps a clamp for the recurrence's own safety but
#      the PSD detector now reads the value BEFORE it — ``d_min_raw``, the
#      most negative pre-clamp residual diagonal seen over active rows.
#      The clamp no longer destroys the detector; it just no longer feeds
#      it.  The wrapper refuses when ``d_min_raw < −floor``, which is the
#      ``pstrf`` INFO in the one form a jitted kernel can return it.
#
# WHAT DID NOT CHANGE, DELIBERATELY.  The PIVOT RULE.  Greedy
# max-residual-diagonal IS LAPACK ``?pstrf``'s rule, and this work adopts
# ``pstrf``'s SEMANTICS around it — terminate on a scale-relative
# tolerance, report the achieved rank, signal indefiniteness INFO-style —
# without touching the selection itself.  Every healthy-input result is
# bit-identical to the pre-2026-08-07 kernel; that is checked, not asserted
# (``pc_ab.py``, five cases including orbit mode, ``piv``/``L``/``rank``/
# ``d_final``/``d_taken``/``trR`` all byte-equal).
#
# THE TOLERANCE POLICY, AND WHY IT IS A KNOB.  The floor is
# ``tol_rel · max(diag G)`` with ``tol_rel`` defaulting to ``sqrt(eps)``
# (~1.49e-08 in float64).  Relative to the largest INITIAL diagonal is the
# scale-invariant choice and is what makes the answer independent of how G
# is normalised; the sharded kernel takes that maximum through a ``pmax``
# so the floor is identical at every shard count.  LAPACK's ``?pstrf``
# defaults to ``n·eps·max(diag)`` instead, which at the production
# M = 2580 is 5.7e-13 — some four orders LOOSER than the default here.
# ``sqrt(eps)`` is kept as the default because it is the number this kernel
# has always computed for its ``rank`` report, so adopting it as the
# stopping rule leaves every existing deck's reported rank unmoved; a
# caller that wants LAPACK's own policy passes ``tol_rel=n*eps``.  The
# override is a parameter on all three entry points and, at the driver
# seam, the ``LORRAX_CENTROID_PC_TOL`` environment variable.
#
# LAPACK PANEL BLOCKING DOES NOT TRANSFER HERE, AND IT WAS MEASURED.
# ``?pstrf`` blocks because it factors the FULL matrix: k = n = M, so the
# trailing update's O(M·b·M) per panel and the left-looking O(M·k) per step
# are the same cost.  This kernel SELECTS ``k_keep`` columns out of M
# candidates with k_keep << M — 42 of 2580 at the Si-class point, 900 of
# 13872 at the MoS₂-class one.  The trailing update refreshes ALL M columns
# of a working Gram; the algorithm only ever READS column p, at k of the M
# columns, chosen adaptively.  So blocking pays M/k times the arithmetic:
#
#   shape                left-looking   panel-blocked   ratio
#   D3 (M=2580, k=42)      4.55e+06       2.87e+08       63x
#   MoS₂ (M=13872, k=900)  1.12e+10       1.74e+11       15x
#
# A prototype confirmed it (``blocked_proto.py``): the panel form reproduces
# the pivot sequence EXACTLY at every shape tested — the blocking is correct
# — and runs 3.5x to 16.7x SLOWER, worst at the D3 shape where M/k is worst.
#
# And it does not buy the round trips it was reached for.  Pivot selection
# is inherently sequential: each pivot needs its own ``pmax`` on the value
# and its own ``pmax`` on the tie-break index, whatever the panel width.
# Blocking shrinks the ``L[p, :]`` psum from k_keep to b and adds one
# all-gather per panel; the per-iteration COUNT is unchanged.
#
# The reduction that IS available needed no blocking at all, and is what
# this kernel does.  MEASURED in the lowered HLO, per iteration of the
# while body, on a real 2×2:
#
#   point mode, k_keep=20
#     before  f64[], s32[], c128[20], f64[]          = 4
#     after   f64[], s32[], c128[20]                 = 3
#   orbit mode, k_keep=12
#     before  f64[], s32[], c128[12], s32[], f64[]   = 5
#     after   f64[], s32[], c128[13]                 = 3
#
# Two changes, neither touching the arithmetic: the trace-ratio psum is a
# pure diagnostic and moves OUT of the loop (local partials, one psum at the
# end), and the orbit-id broadcast RIDES the ``L[p, :]`` psum instead of
# taking its own trip — visible above as c128[12]+s32[] becoming c128[13].
# 1.33x fewer round trips in point mode, 1.67x in orbit mode, zero extra
# flops, results bit-identical.  ``tests/test_centroid_distribution.py``
# gates the count in the HLO, because NCCL latency is not measurable on an
# emulated-device box and the count is the honest proxy.
#
# REGISTERED, NOT BUILT: block-greedy selection (take the top-b entries of
# one snapshot of d per round) WOULD batch the pivot pmax and get the round
# trips down by ~b.  It also CHANGES WHICH PIVOTS ARE CHOSEN on
# well-separated Grams, not just on ties, so it needs its own owner ruling
# and a regeneration story for every existing centroid file.  Not a
# refactor; a different algorithm.
#
# The refusal DISCIPLINE — a named condition, the measured evidence, and
# what to do instead — is borrowed from ``distrib_la``.  Its DISPATCH is
# not: this kernel stays local (assessment R7 — it is a selection and not a
# factorisation, the orbit-kill rule is crystallography rather than linear
# algebra, no backend has a masked-active-set ``pstrf`` to dispatch to, and
# at 0.165 s on a production Gram it is 3-4 % of the driver's wall).


@partial(jax.jit, static_argnames=('k_keep', 'tol_rel'))
def pivoted_cholesky_select(
    G: jnp.ndarray,
    k_keep: int,
    orbit_id: jnp.ndarray | None = None,
    *,
    tol_rel: float | None = None,
):
    """Greedy pivoted Cholesky on an Hermitian PSD ``G``. STOPS at the
    numerical-rank floor. Returns ``(piv, L, rank, d_final, d_taken,
    trR_over_trG, psd_info)``.

    ``rank`` is the number of pivots actually taken and is a CONTRACT:
    ``piv[rank:] == -1``, ``L[:, rank:] == 0``, ``d_taken[rank:] == 0``.
    ``rank < k_keep`` means the kernel could not certify what was asked for
    and the returned pivot list is deliberately incomplete rather than
    padded with noise — see the block comment above for what the old
    always-run-k_keep behaviour did instead.

    ``psd_info`` is ``(d_min_raw, at_row, at_step)`` — the most negative
    pre-clamp residual diagonal observed, the candidate row that attained
    it, and the iteration it happened on.  ``d_min_raw < -tol_rel·max(diag
    G)`` says the input was not positive semidefinite, and the two indices
    are what lets the refusal NAME the pivot rather than merely assert the
    condition, which is the ``pstrf`` ``INFO`` contract.

    ``tol_rel`` overrides the stopping tolerance, relative to the largest
    initial diagonal; ``None`` means ``sqrt(eps)``.  See the block comment
    above for why that, and not LAPACK's ``n·eps``, is the default.

    When ``orbit_id`` is given (shape ``(M,)`` int), each pivot iteration
    marks the **whole orbit** of the picked point as inactive — i.e. one
    pivot per orbit. With a sym-invariant Gram (e.g. ρ-symmetric ISDF
    candidate Gram), all orbit members of the picked pivot have the same
    residual diagonal and the column update on any one of them is, by
    symmetry, the optimal full-orbit removal. The caller unfolds picked
    pivots through their orbits at output time to recover the full
    centroid set.  Asking for more orbits than exist now STOPS at the last
    real orbit instead of repeating index 0.
    """
    M = G.shape[0]
    real_dtype = G.real.dtype
    eps = jnp.finfo(real_dtype).eps
    minus_inf = jnp.array(-jnp.inf, dtype=real_dtype)
    if orbit_id is None:
        orbit_id = jnp.arange(M, dtype=jnp.int32)         # each point its own orbit

    diag_raw = jnp.real(jnp.diag(G))
    # The floor moves ABOVE the loop: it is the stopping rule now, not a
    # post-hoc label on a number the loop already ruined.
    tol = jnp.sqrt(eps) if tol_rel is None else jnp.asarray(tol_rel,
                                                            real_dtype)
    floor = tol * jnp.max(diag_raw)
    diag0 = jnp.maximum(diag_raw, 0.0)
    trG = jnp.sum(diag0)

    # The initial diagonal is itself a PSD statement: a negative entry on
    # the diagonal of a PSD matrix is impossible, and step -1 names it.
    at0 = jnp.argmin(diag_raw).astype(jnp.int32)
    neg0 = diag_raw[at0] < 0.0
    init = (
        diag0,                                                       # d
        jnp.zeros((M, k_keep), dtype=G.dtype),                       # L
        -jnp.ones((k_keep,), dtype=jnp.int32),                       # piv
        jnp.ones((M,), dtype=bool),                                  # active
        jnp.zeros((k_keep,), dtype=real_dtype),                      # d_taken
        jnp.zeros((k_keep + 1,), dtype=real_dtype).at[0].set(1.0),   # trR/trG
        jnp.minimum(diag_raw[at0], jnp.zeros((), dtype=real_dtype)),  # d_min
        jnp.where(neg0, at0, jnp.int32(-1)),                         # at_row
        jnp.where(neg0, jnp.int32(-1), jnp.int32(-1)),               # at_step
    )
    col_ids = jnp.arange(k_keep)

    def body(j, carry):
        (d, L, piv, active, d_taken, trR_over_trG,
         d_min_raw, d_min_at, d_min_j) = carry

        masked_d = jnp.where(active, d, minus_inf)
        p = jnp.argmax(masked_d)
        # THE STOP.  ``pivot_val`` clamps at ``floor`` (not ``eps``), so on a
        # healthy input this is bit-for-bit the old arithmetic and past the
        # rank the divisor can no longer manufacture a blow-up.
        take = masked_d[p] > floor
        pivot_val = jnp.maximum(masked_d[p], floor)

        # L[:, j] = (G[:, p] - Σ_{i<j} L[:, i] · conj(L[p, i])) / sqrt(d[p])
        prev_mask = (col_ids < j).astype(G.dtype)
        corr = L @ (jnp.conj(L[p, :]) * prev_mask)
        denom = jnp.sqrt(pivot_val)
        newcol = (G[:, p] - corr) / denom
        # Pivot entry exactly sqrt(d[p]) — kills rounding drift.
        newcol = newcol.at[p].set(denom.astype(G.dtype))
        newcol = jnp.where(take, newcol, jnp.zeros_like(newcol))

        L = L.at[:, j].set(newcol)
        piv = piv.at[j].set(jnp.where(take, p, -1).astype(jnp.int32))
        d_taken = d_taken.at[j].set(
            jnp.where(take, pivot_val, 0.0).astype(real_dtype))

        # Schur-complement update; d_new[p] ≈ 0 by the cleanup above.  The
        # PSD detector reads d_raw BEFORE the clamp, over ACTIVE rows only
        # (inactive rows carry −inf and would swamp the minimum).
        d_raw = d - jnp.abs(newcol) ** 2
        masked_raw = jnp.where(active, d_raw, jnp.inf)
        step_at = jnp.argmin(masked_raw).astype(jnp.int32)
        step_min = masked_raw[step_at]
        # Keep the row and the step alongside the value, so the refusal can
        # NAME the pivot the way pstrf's INFO does instead of only asserting
        # that indefiniteness happened somewhere.
        beats = take & (step_min < d_min_raw)
        d_min_at = jnp.where(beats, step_at, d_min_at)
        d_min_j = jnp.where(beats, j.astype(jnp.int32), d_min_j)
        d_min_raw = jnp.where(beats, step_min, d_min_raw)
        # ``d_new`` past the stop is just the frozen residual with the −inf
        # kill markers clipped away, so the trace ratio holds its last real
        # value instead of going −inf/NaN.
        d_new = jnp.maximum(jnp.where(take, d_raw, d), 0.0)
        trR_over_trG = trR_over_trG.at[j + 1].set(jnp.sum(d_new) / trG)
        # Mark p (or its whole orbit, if orbit_id was provided) inactive.
        kill_mask = (orbit_id == orbit_id[p]) & take
        d = jnp.where(kill_mask, minus_inf, jnp.where(take, d_new, d))
        active = active & ~kill_mask

        return (d, L, piv, active, d_taken, trR_over_trG,
                d_min_raw, d_min_at, d_min_j)

    (d, L, piv, _, d_taken, trR_over_trG,
     d_min_raw, d_min_at, d_min_j) = lax.fori_loop(0, k_keep, body, init)
    d_final = jnp.where(jnp.isfinite(d), d, 0.0)
    # Effective rank = #pivots taken.  With the stopping rule above this is
    # exactly the loop's trip count: ``d_taken[j] > floor`` for every taken
    # pivot by construction and 0 for every untaken one, so the count is a
    # contract rather than the §4.4 assumption that ``d_taken`` happens to
    # be monotone past the numerical rank (it is not, when it is noise).
    rank = jnp.sum(d_taken > floor).astype(jnp.int32)
    return (piv, L, rank, d_final, d_taken, trR_over_trG,
            (d_min_raw, d_min_at, d_min_j))




# ═══════════════════════════════════════════════════════════════════════
# The kernel's INFO, read on the host
# ═══════════════════════════════════════════════════════════════════════


def refuse_unless_select_certified(
    piv,
    rank: int,
    psd_info,
    *,
    n_keep: int,
    M: int,
    M_pad: int | None = None,
    orbit_id=None,
    d0max: float,
    tol_rel: float | None = None,
) -> None:
    """Raise unless the select delivered ``n_keep`` certified pivots.

    A jitted kernel cannot raise, so it REPORTS and this REFUSES — the same
    division of labour LAPACK's ``pstrf`` makes with its ``INFO`` code, and
    the reason assessment R7 says to borrow ``distrib_la``'s refusal
    discipline rather than its dispatch.  Three conditions, each of which
    used to be a silent wrong answer that passed every downstream shape
    check; the measured before-behaviour of all three is in the block
    comment above :func:`pivoted_cholesky_select`.

    It is a free function, and public, for one reason: every one of these
    refusals needs a constructible-FALSE twin, and building one through
    :func:`prune_candidates_by_pivoted_cholesky` would mean standing up a
    WFN and a symmetry table to test an arithmetic contract.  The tests
    hand it real kernel output.

    Parameters
    ----------
    piv, rank, psd_info
        Straight off the kernel — the pivot list, its hard-contract rank
        and the ``(d_min_raw, at_row, at_step)`` triple.  A bare float is
        also accepted for ``psd_info``, in which case the refusal cannot
        name the pivot.
    n_keep
        What was asked for.  Orbits in orbit mode, points otherwise.
    M, M_pad
        Logical candidate count and the zero-padded extent the sharded
        select actually ran on.  ``M_pad=None`` means "unpadded".
    orbit_id
        ``(M,)`` int or ``None``.  Read only to count available orbits and
        to name the unit in the message.
    d0max
        ``max(diag G)`` — the scale the noise floor is relative to.
    tol_rel
        The stopping tolerance the kernel ran with; ``None`` means
        ``sqrt(eps)``.  Must match, or the floor quoted in the refusal
        would not be the one the kernel stopped on.
    """
    piv_np = np.asarray(piv)
    M_pad = int(M if M_pad is None else M_pad)
    tol = (float(np.sqrt(np.finfo(np.float64).eps)) if tol_rel is None
           else float(tol_rel))
    floor = tol * float(d0max)
    rank_i = int(rank)
    if isinstance(psd_info, (tuple, list)):
        d_min = float(psd_info[0])
        at_row, at_step = int(psd_info[1]), int(psd_info[2])
    else:
        d_min, at_row, at_step = float(psd_info), -1, -1

    # (1) NOT POSITIVE SEMIDEFINITE.  The Schur update keeps a clamp for the
    # recurrence's own safety, but the detector now reads the value BEFORE
    # it, so the clamp no longer destroys the signal.  ``pstrf`` reports
    # this as INFO = the order of the leading minor that failed; the
    # equivalent here is the STEP it failed on and the candidate ROW that
    # carried it, and both are named rather than left to be re-derived.
    if d_min < -floor:
        where = (f"candidate row {at_row} at step "
                 f"{'the initial diagonal' if at_step < 0 else at_step}"
                 if at_row >= 0 else "an unrecorded row")
        raise RuntimeError(
            f"pivoted-Cholesky REFUSES: the Gram is not positive "
            f"semidefinite.  The residual Schur diagonal reached "
            f"{d_min:.6e} on {where} — past the noise floor -{floor:.6e} "
            f"(= -{tol:.3e}·max diag G = -{tol:.3e}·{float(d0max):.6e}), "
            f"i.e. by {abs(d_min) / max(floor, 1e-300):.3g}x the floor.  A "
            f"PSD matrix cannot do that in any arithmetic.  Until "
            f"2026-08-07 a ``jnp.maximum(..., 0.0)`` on the update absorbed "
            f"exactly this signal and the kernel reported a clean rank with "
            f"every pivot distinct, L entirely finite and min(d_final) "
            f"exactly 0.0 — measured on a Gram indefinite by one part in a "
            f"thousand.  The q=0 Gram is a sum of outer products and should "
            f"be PSD by construction, so this says the ASSEMBLY is wrong (a "
            f"conjugation, a k-weight, a band window), not that the "
            f"selection needs loosening.")

    # (2) THE RECURRENCE STOPPED.  ``rank`` is a contract now: short means
    # ``piv`` carries -1 sentinels and there is no set of the requested size
    # to deliver.  The two causes get different text because the fixes are
    # different — one is "the pool is too small", the other is "the pool is
    # numerically flat", and the second is the one that has cost eV.
    if rank_i < int(n_keep):
        unit = "orbits" if orbit_id is not None else "points"
        n_avail = (int(np.unique(np.asarray(orbit_id)).size)
                   if orbit_id is not None else int(M))
        if n_avail <= rank_i:
            cause = (f"the candidate pool CONTAINS only {n_avail} {unit}, "
                     f"so there was nothing left to pick.  Raise "
                     f"--oversample for a richer pool, or lower N.")
        else:
            cause = (f"{n_avail} {unit} were available, but the residual "
                     f"Schur diagonal fell to the noise floor {floor:.3e} "
                     f"after {rank_i} of them — the pool is numerically "
                     f"RANK-DEFICIENT, not short.  Widen the prune window "
                     f"(--prune-n-cond, or --prune-window vc_x_vc) so the "
                     f"Gram sees the pair densities Sigma actually "
                     f"consumes, or lower N.")
        raise RuntimeError(
            f"pivoted-Cholesky REFUSES: asked for {int(n_keep)} {unit}, "
            f"certified {rank_i}.\n"
            f"  cause : {cause}\n"
            f"  effect: piv[{rank_i}:] is the -1 sentinel, so there is no "
            f"set of {int(n_keep)} {unit} to return.\n"
            f"Before 2026-08-07 this ran to k_keep regardless.  Past the "
            f"numerical rank the divisor clamped to sqrt(eps) and the "
            f"factor blew up geometrically to Inf and then NaN (MEASURED: "
            f"first non-finite column twelve iterations past a true rank of "
            f"10), after which argmax over NaN handed back the first "
            f"unpicked indices IN ARRAY ORDER — real candidate indices, so "
            f"the pad guard did not fire and nothing downstream noticed.  "
            f"In orbit mode the same exhaustion stayed finite and returned "
            f"index 0 repeated once per missing orbit.")

    # (3) PAD ROWS AND SENTINELS.  HARD GUARD, not a comment: an
    # out-of-range index would silently index past the candidate list (or
    # wrap from the end) and the centroid set would be quietly wrong — the
    # rc=0-with-garbage class.  UNCONDITIONAL since 2026-08-07; it used to
    # run only under ``if n_pad``, which is exactly the branch that cannot
    # see a -1 sentinel on an unpadded problem.
    bad = piv_np[(piv_np >= int(M)) | (piv_np < 0)]
    if bad.size:
        raise RuntimeError(
            f"pivoted-Cholesky REFUSES: {bad.size} pivot(s) {bad[:8]} lie "
            f"outside the candidate range [0,{int(M)}) (M_pad={M_pad}).  "
            f"Either the active mask that makes pad rows unpickable did not "
            f"hold, or a stopping sentinel survived the rank contract "
            f"above.  Refusing to emit a centroid set built from either.")


# ═══════════════════════════════════════════════════════════════════════
# Orbit-BLOCK select — the repair for the stale-residual defect
# ═══════════════════════════════════════════════════════════════════════
#
# THE DEFECT THIS REPLACES.  ``pivoted_cholesky_select`` in orbit mode does
# ONE rank-1 Schur update per admitted orbit and then marks every member of
# that orbit inactive (``kill_mask = (orbit_id == orbit_id[p])``, :352).  So
# admitting a 48-point orbit deflates the residual by ONE direction while
# spending 48 points.  Its own docstring justifies this with
#
#     "all orbit members of the picked pivot have the same residual diagonal
#      and the column update on any one of them is, by symmetry, the optimal
#      full-orbit removal"
#
# and the first clause is true (the diagonal of a sym-invariant Gram is a
# class function) while the second does not follow: equal DIAGONALS do not
# make the members linearly DEPENDENT.  An orbit spans up to ``n_sym``
# directions; a rank-1 update removes one.  MEASURED contradiction on the Si
# 4x4x4 anchor: an orbit-mode 768-point set is reported by the select as 18
# certified directions, and the zeta back-solve on that very set keeps
# 768/768 modes per q.  If the docstring's claim held, zeta would keep 18.
#
# CONSEQUENCE.  After the first admitted orbit the residual ``d`` that ranks
# every later orbit is stale by (orbit_size - 1) directions, and it goes on
# being stale, so the greedy ranks candidates on a residual that still counts
# directions already paid for.  The delivered set is full rank but spans the
# WRONG directions -- which is exactly the failure shape the rank gate cannot
# see, because in orbit mode it is stated on the orbit count.
#
# THE REPAIR.  Block-pivoted Cholesky with the orbits as blocks:
#   * score each ACTIVE orbit by its residual TRACE (= sum of member
#     diagonals).  For a sym-invariant Gram that is orbit_size x d_rep, so
#     multiplicity weighting is automatic -- an orbit that costs 48 points is
#     ranked on the 48 points' worth of trace it removes, not on one point's.
#   * admit the argmax orbit and deflate by EVERY member of it, one rank-1
#     update each, largest residual first.
# The number of rank-1 deflations then EQUALS the number of points delivered,
# which is the property the orbit path violated.
#
# COST.  One rank-1 update per delivered point -- identical to point mode,
# and independent of the FFT grid size.  The k-means candidate pool is
# untouched: this changes only which pool members are kept.

#: Guard for the host-side implementation below.  The dense host path holds
#: one (M, M) complex128 Gram; at the Si/MoS2-fixture scale (M ~ 1e3) that is
#: tens of MB.  Above this the sharded-kernel port is required -- the host
#: path REFUSES rather than silently gathering a multi-GB Gram to one rank.
ORBIT_BLOCK_HOST_MAX_M_DEFAULT = 4096


def orbit_block_select_host(G, orbit_id, *, n_orbit_keep, n_point_budget=None,
                            tol_rel=None, score="trace"):
    """Block-pivoted Cholesky with orbits as blocks.  Host/numpy.

    Parameters
    ----------
    G : (M, M) complex/real ndarray -- the candidate Gram (logical part only,
        no zero pad).
    orbit_id : (M,) int ndarray.
    n_orbit_keep : int -- maximum number of orbits to admit.
    n_point_budget : int or None -- stop before the delivered point count
        would exceed this.  Whole orbits only; never a partial orbit.
    tol_rel : float or None -- stopping tolerance relative to max initial
        diagonal.  ``None`` -> sqrt(eps), matching the jitted kernel.
    score : {"trace", "maxdiag"} -- how an orbit is ranked.  ``"trace"`` is
        the block generalisation and carries the multiplicity weighting;
        ``"maxdiag"`` reproduces the OLD ranking rule while keeping the NEW
        (correct) deflation, so the two effects can be attributed separately.

    Returns
    -------
    order : (n_orbits_taken,) int ndarray -- admitted orbit LABELS, in the
        order they were admitted.
    piv_points : (n_points_taken,) int ndarray -- every pivoted point index,
        in pivot order.  ``len(piv_points)`` IS the point-granularity rank:
        one certified direction per delivered point.
    trR_over_trG : float -- residual trace fraction after the last block.
    """
    G = np.asarray(G)
    M = G.shape[0]
    oid = np.asarray(orbit_id, dtype=np.int64)
    if oid.shape[0] != M:
        raise ValueError(f"orbit_id has {oid.shape[0]} entries for M={M}")
    real_dtype = np.real(G).dtype
    eps = np.finfo(real_dtype).eps
    tol = np.sqrt(eps) if tol_rel is None else float(tol_rel)

    d = np.real(np.diag(G)).astype(np.float64).copy()
    trG = float(np.maximum(d, 0.0).sum())
    floor = tol * float(d.max())
    d = np.maximum(d, 0.0)

    labels, inv = np.unique(oid, return_inverse=True)
    n_orb = labels.shape[0]
    members = [np.where(inv == a)[0] for a in range(n_orb)]
    orb_alive = np.ones(n_orb, dtype=bool)

    L = np.zeros((M, M), dtype=G.dtype)     # columns filled as we pivot
    piv_points: list[int] = []
    order: list[int] = []
    n_pts = 0
    j = 0

    for _ in range(int(n_orbit_keep)):
        if not orb_alive.any():
            break
        if score == "trace":
            sc = np.array([d[members[a]].sum() if orb_alive[a] else -np.inf
                           for a in range(n_orb)])
        elif score == "maxdiag":
            sc = np.array([d[members[a]].max() if orb_alive[a] else -np.inf
                           for a in range(n_orb)])
        else:
            raise ValueError(f"score must be 'trace' or 'maxdiag'; got {score!r}")
        a = int(np.argmax(sc))
        if not np.isfinite(sc[a]):
            break
        mem = members[a]
        # Stop BEFORE overrunning the budget; whole orbits only.
        if n_point_budget is not None and n_pts + mem.shape[0] > int(n_point_budget):
            orb_alive[a] = False
            continue
        if float(d[mem].max()) <= floor:
            break

        # --- deflate by the WHOLE orbit: one rank-1 update per member ------
        remaining = list(mem)
        while remaining:
            loc = int(np.argmax(d[remaining]))
            p = int(remaining.pop(loc))
            dp = float(d[p])
            if dp <= floor:
                # This member adds no certified direction.  Keep it in the
                # delivered set (the orbit is admitted whole) but do not
                # count it as a certified direction.
                continue
            corr = L[:, :j] @ np.conj(L[p, :j])
            newcol = (G[:, p] - corr) / np.sqrt(dp)
            newcol[p] = np.sqrt(dp)
            L[:, j] = newcol
            d = np.maximum(d - np.abs(newcol) ** 2, 0.0)
            d[p] = 0.0
            piv_points.append(p)
            j += 1
        orb_alive[a] = False
        d[mem] = 0.0                       # admitted; never re-rank them
        order.append(int(labels[a]))
        n_pts += mem.shape[0]
        if n_point_budget is not None and n_pts >= int(n_point_budget):
            break

    trR = float(d.sum()) / trG if trG > 0 else 0.0
    return (np.asarray(order, dtype=np.int64),
            np.asarray(piv_points, dtype=np.int64), trR)


# ═══════════════════════════════════════════════════════════════════════
# R2 — rank at POINT granularity
# ═══════════════════════════════════════════════════════════════════════

#: Above this many delivered points the point-granularity rank is reported
#: as "not measured" rather than measured: the check is a dense
#: ``eigvalsh`` on the kept sub-Gram, O(n³), and at the MoS₂-class point
#: (n ≈ 13 000) that is minutes and gigabytes.  Raise it deliberately via
#: ``LORRAX_CENTROID_POINT_RANK_CAP`` when the answer is worth the wait.
#: 4096 covers every deck in the project's fixtures and the whole frontier
#: ladder (whose largest set is 1908).
POINT_RANK_CAP_DEFAULT = 4096


def point_granularity_rank(G, keep_mask, *, tol_rel=None, cap=None):
    """Independent directions in the DELIVERED POINT SET, not in the orbits.

    THE CONFUSION THIS REMOVES.  In orbit mode the greedy select deflates
    the Schur complement by ONE direction per orbit while removing all
    ``n_sym`` members from contention, so the rank it reports counts
    ORBITS.  D3's gate passed at "42 of 42 directions certified" and the
    file it blessed contained 1908 POINTS; the ζ back-solve then truncated
    to 1440-1455 modes per q (23.7-24.5 %, logged eight times per leg and
    read by nobody) because the 60-band pair-product space on a 24³ grid
    saturates around 1450 numerical directions.  Asking for 1908 centroids
    does not raise that ceiling — it hands the pseudo-inverse a
    rank-deficient Gram.

    Nothing in the centroid pipeline ever checked the delivered set at
    point granularity, and in orbit mode it structurally cannot: the number
    the rank gate reads is not comparable to the point count.  This is that
    number, measured directly, so the log can say "42 orbits, 1908 points,
    N independent directions" and an operator can see the ζ truncation
    coming before spending a 7 GiB restart file on it.

    Returns ``(rank, n_points, reason)``.  ``rank`` is ``None`` when the
    measurement was skipped and ``reason`` says why — never a silent
    absence, because "no number" and "the number is fine" must not look
    alike in a log.
    """
    keep_mask = np.asarray(keep_mask, dtype=bool)
    n_pts = int(keep_mask.sum())
    if cap is None:
        env = os.environ.get("LORRAX_CENTROID_POINT_RANK_CAP")
        cap = int(env) if env else POINT_RANK_CAP_DEFAULT
    if n_pts == 0:
        return None, 0, "empty point set"
    if n_pts > int(cap):
        return (None, n_pts,
                f"skipped: {n_pts} points exceeds the O(n^3) cap {int(cap)} "
                f"(raise LORRAX_CENTROID_POINT_RANK_CAP to measure anyway)")
    try:
        sub = np.asarray(jax.device_get(G))[np.ix_(keep_mask, keep_mask)]
    except Exception as exc:                                  # noqa: BLE001
        return None, n_pts, f"skipped: could not gather the Gram ({exc})"
    d0max = float(np.real(np.diag(sub)).max())
    tol = (float(np.sqrt(np.finfo(np.float64).eps)) if tol_rel is None
           else float(tol_rel))
    # The SAME relative floor the select stops on, so the two numbers in
    # the log line are measured against one policy and can be compared.
    ev = np.linalg.eigvalsh(0.5 * (sub + sub.conj().T))
    return int((ev > tol * d0max).sum()), n_pts, ""


def point_rank_closure_note(G, keep_mask, rank, *, tol_rel=None):
    """Whether that rank lands inside a degenerate block.  REPORT ONLY.

    :func:`point_granularity_rank` selects nothing — it hands an operator a
    number to size ``n_keep`` by — so this neither snaps nor refuses.  But
    the number is a rank cut like any other, and when it lands inside a
    degenerate block every ``n_keep`` in that block buys a symmetry-arbitrary
    slice of an eigenspace: the pivoted-Cholesky select is then choosing
    between directions the spectrum cannot distinguish, and no amount of care
    in the pivot order can make that choice covariant.  Saying so here is the
    difference between an operator picking 1908 points and understanding why
    the ζ back-solve truncated to 1450.

    Returns ``""`` when the cut falls in a gap.  Kept OUT of
    :func:`point_granularity_rank`'s ``reason`` field on purpose: that field
    is contracted to mean "the measurement was skipped and ``rank`` is
    ``None``", and its caller only prints it in that case, so a closure note
    smuggled into it would be silently dropped.
    """
    from common import spectral_closure
    sub = np.asarray(jax.device_get(G))[np.ix_(np.asarray(keep_mask, dtype=bool),
                                               np.asarray(keep_mask, dtype=bool))]
    ev = np.linalg.eigvalsh(0.5 * (sub + sub.conj().T))
    info = spectral_closure.cluster_at_cut(ev, int(rank))
    if not info["fired"]:
        return ""
    lo, hi = info["n_keep_dropped"], info["n_keep_kept"]
    return (f"the point rank {int(rank)} lands INSIDE a degenerate block of "
            f"{len(info['members'])} eigenvalues (relative gap at the cut "
            f"{info['gap_rel']:.2e}, rtol {info['rtol']:.1e}); the block runs "
            f"from {lo} to {hi}.  Any n_keep strictly between those selects "
            f"PART of an eigenspace and is not point-group invariant; the two "
            f"legal ranks are {lo} and {hi}, and the ruling of 2026-08-10 "
            f"takes the LOWER — see common/spectral_closure.")


# ═══════════════════════════════════════════════════════════════════════
# Step 4 — end-to-end wrapper
# ═══════════════════════════════════════════════════════════════════════


def prune_candidates_by_pivoted_cholesky(
    wfn: "WfnLoader",
    sym: symmetry_maps.SymMaps,
    cand_idx: np.ndarray,
    n_keep: int,
    mesh: Mesh,
    *,
    n_val: int | None = None,
    n_cond: int | None = None,
    band_range_left: tuple[int, int] | None = None,
    band_range_right: tuple[int, int] | None = None,
    band_norms: np.ndarray | None = None,
    k_weights: np.ndarray | None = None,
    verbose: bool = True,
    bispinor: bool = False,
    orbit_id: np.ndarray | None = None,
    use_phdf5: bool = False,
    tol_rel: float | None = None,
    n_point_budget: int | None = None,
):
    """End-to-end pruning: gather wfns → Gram → pivoted Cholesky → keep.

    Requires a 2-D mesh ``('x', 'y')`` (single-device callers pass a 1×1
    mesh — same shape gw_jax uses). Wavefunction loading goes through
    ``load_centroids_band_chunked`` so the prune path is agnostic to which
    G-space backend (WFNReader / phdf5 / future jax-multihost) is in use.

    When ``orbit_id`` is provided (one int per candidate, equal for sym-
    equivalent candidates), PC picks one pivot per orbit and the returned
    ``keep_idx`` is the union of orbits of the picked pivots — guaranteed
    orbit-closed under the sym group used to assign ``orbit_id``. In that
    mode ``n_keep`` counts ORBITS (final unfolded centroid count is
    ``Σ orbit_size`` for picked orbits).

    Returns ``(keep_idx, rank, G, d_final, d_taken, trR_over_trG,
    psd_info)``.

    ``n_point_budget`` is THE FLOOR (owner ruling, 2026-08-10: "everything
    the user has input on they should be specifying in units of points, and
    we should be choosing the quantity of orbits that comes closest to that
    number of points without exceeding it").  Orbit mode's ``n_keep`` is a
    count of ORBITS, so the POINT total it lands on is whatever the picked
    orbits happen to sum to and can overrun the number the user typed.  Give
    this the user's point count and the delivered set is truncated to the
    longest prefix of the pivot order whose point total does not exceed it —
    a prefix and not a knapsack, because the pivot order is a quality ranking
    produced by deflating in that order.  ``None`` (the default) is the
    historical behaviour: ``n_keep`` orbits, whatever that costs in points.
    Ignored outside orbit mode, where ``n_keep`` already IS the point count.

    ``tol_rel`` overrides the select's stopping tolerance (relative to the
    largest initial Gram diagonal).  ``None`` reads
    ``LORRAX_CENTROID_PC_TOL`` from the environment and falls back to
    ``sqrt(eps)`` — the number this kernel has always computed for its rank
    report, kept as the default so no existing deck's reported rank moves.
    LAPACK ``?pstrf``'s own policy is ``n·eps``; pass it explicitly to get
    it.

    REFUSES, rather than returning a plausible-looking set, when the kernel
    could not do what was asked: a non-PSD Gram, a pool that runs out of
    orbits, a pool that is numerically rank-deficient, or a pivot outside
    the candidate range.  ``rank == n_keep`` on every path that returns.
    Those refusals are the assessment's R1 and they are stated in full
    beside the kernel; the short version is that each one used to be a
    silent wrong answer that passed every downstream shape check.
    """
    M = int(cand_idx.shape[0])
    n_tot = int(wfn.nbands)
    asymmetric = (band_range_left is not None and band_range_right is not None)
    if tol_rel is None:
        _env_tol = os.environ.get("LORRAX_CENTROID_PC_TOL")
        tol_rel = float(_env_tol) if _env_tol else None

    if not asymmetric:
        # Legacy (n_val, n_cond) path — left = (0, n_val), right = (n_val, n_val + n_cond).
        if n_val is None:
            n_val = int(wfn.nelec)
        if n_cond is None:
            n_cond = min(n_val, n_tot - n_val)
        if n_val + n_cond > n_tot:
            raise ValueError(
                f"wfn.nbands={n_tot} < n_val + n_cond = {n_val} + {n_cond}"
            )
        max_band = int(n_val) + int(n_cond)
    else:
        if band_range_left[1] > n_tot or band_range_right[1] > n_tot:
            raise ValueError(
                f"wfn.nbands={n_tot} < max(left={band_range_left[1]}, "
                f"right={band_range_right[1]})"
            )
        max_band = max(int(band_range_left[1]), int(band_range_right[1]))

    # Plane-wave-basis sanity check. For centroid pruning to be
    # meaningful, the pair-product space must be significantly smaller
    # than the full plane-wave basis; once we include > 50 % of the
    # available PW degrees of freedom, the candidate-vs-grid distinction
    # blurs and the user should be pruning the real-space grid directly.
    ngk_max = int(np.max(wfn.ngk)) if hasattr(wfn, 'ngk') else None
    nspinor = int(wfn.nspinor)
    if ngk_max is not None:
        npw_basis = ngk_max * nspinor  # size of the plane-wave basis per k
        if max_band > 0.5 * npw_basis:
            raise ValueError(
                f"Requested band window touches band {max_band}, which "
                f"exceeds 50 % of the plane-wave basis size "
                f"({0.5 * npw_basis:.0f} = 0.5 · ngk_max · nspinor = "
                f"0.5 · {ngk_max} · {nspinor}). Centroid pruning is "
                f"ill-posed in this regime — prune on the full real-space "
                f"grid directly instead."
            )

    select_axis = dist.require_axes(mesh, dist.MESH_AXES,
                                    "prune_candidates_by_pivoted_cholesky")

    # The sharded select kernel requires M to be divisible by the product
    # of the mesh axis sizes (each shard owns M/n_dev rows). Orbit-unfold
    # counts can land on awkward M (special-position orbits don't all have
    # size n_sym), so check up-front and give a hint instead of letting
    # ``make_sharded_pivoted_cholesky_select`` fail with a cryptic message.
    # ── M need not divide the mesh: ZERO-PAD + ACTIVE-MASK ──────────────────
    # The select kernel is a shard_map and needs M % n_dev == 0.  M is an
    # ORBIT-UNFOLD count (special-position orbits are not all of size n_sym),
    # so it is not controllable from the CLI: at the rung-5 point
    # M = 13872 = 2^4*3*17^2, which divides 1,2,4,8,16 but NOT 32 or 64 — the
    # old refusal here is exactly what blocked P=64 centroid generation
    # (measured, job 7879533).
    # Instead of refusing, pad G to M_pad with ZERO rows and columns and start
    # those rows INACTIVE.  A zero row/col contributes nothing to trG, nothing
    # to d0max, and nothing to the Schur update, and an inactive row can never
    # be picked, so every reported quantity (trG, trR/trG, d_taken, rank, piv)
    # is identical to the unpadded problem.  This is the same zero-pad + mask
    # contract ``runtime.padding.padded_mu_extent`` applies to mu everywhere
    # else in LORRAX.
    n_dev = dist.n_shards(mesh, select_axis)
    # NOTE the pad already exists: ``build_gram_q0_via_loadwfns`` builds through
    # ``Meta.from_system(n_rmu=M)``, whose ``n_rmu_padded =
    # padded_mu_extent(M, world_size) = round_up(M, n_dev)`` (common/meta.py
    # :117), and ψ pad rows are zero, so G comes back (M_pad, M_pad) with the
    # trailing rows/cols EXACTLY ZERO.  The select never knew that, so it
    # refused instead.  M_pad is read off G below rather than recomputed here —
    # recomputing it was a real bug (job 7879553: I padded a second time on top
    # of the builder's pad and produced 13904, which 64 does not divide).

    if verbose:
        window_tag = (f"left={band_range_left}, right={band_range_right}, "
                      f"norms={'on' if band_norms is not None else 'off'}"
                      if asymmetric else f"n_val={n_val}, n_cond={n_cond}")
        print(f"[pivoted_cholesky] M={M}, n_keep={n_keep}, {window_tag} "
              f"(load_wfns 2-D, mesh axes {mesh.axis_names})")

    with timing.section("prune.gram"):
        G = build_gram_q0_via_loadwfns(
            wfn, sym, jnp.asarray(cand_idx),
            n_val=n_val, n_cond=n_cond,
            mesh_xy=mesh, bispinor=bispinor, verbose=verbose,
            band_range_left=band_range_left,
            band_range_right=band_range_right,
            band_norms=band_norms,
            use_phdf5=use_phdf5,
        )
        # Reshard ('x','y') → row-sharded for the column-major pivot scan.
        G = jax.lax.with_sharding_constraint(
            G, NamedSharding(mesh, PartitionSpec(select_axis, None)),
        )
        G.block_until_ready()
    # M_pad is whatever the builder produced (round_up(M, n_dev)); the trailing
    # n_pad rows/cols are zero-padding, NOT candidates.
    M_pad = int(G.shape[0])
    n_pad = M_pad - M
    if M_pad % n_dev != 0:
        raise ValueError(
            f"Gram came back with M_pad={M_pad}, which the mesh product "
            f"{n_dev} does not divide (logical M={M}). The sharded select "
            f"needs an even row split; expected round_up(M, {n_dev})="
            f"{-(-M // n_dev) * n_dev}.")

    if verbose:
        # Report the LOGICAL diagonal (pads are zero and would drag the min to
        # 0.0, changing a diagnostic the gates compare across P).
        diag = jnp.real(jnp.diag(G))[:M]
        print(f"[pivoted_cholesky] G built, shape=({M}, {M}), "
              f"diag range [{float(diag.min()):.3e}, {float(diag.max()):.3e}]")
        if n_pad:
            print(f"[pivoted_cholesky] zero-pad for the sharded select: "
                  f"M {M} -> {M_pad} (+{n_pad} inactive rows) so M_pad % "
                  f"{n_dev} == 0; pads carry d=0 and start INACTIVE")

    # ── ORBIT-BLOCK SELECT — THE DEFAULT FOR ORBIT MODE ────────────────────
    # The old orbit branch of the point-mode kernel deflated ONE direction
    # per admitted orbit while spending up to n_sym points; it is wrong, not
    # merely slower, so it is not kept behind a flag.  Set
    # LORRAX_ORBIT_BLOCK_SELECT=host to run the numpy oracle instead (it is
    # the reference the sharded kernel is gated against); =legacy re-enables
    # the old behaviour for bisection ONLY and prints that it is doing so.
    _ob = os.environ.get("LORRAX_ORBIT_BLOCK_SELECT", "").strip().lower()
    if orbit_id is not None and _ob != "legacy":
        _oid_np = np.asarray(orbit_id)
        _labels, _dense = np.unique(_oid_np, return_inverse=True)
        _n_orb = int(_labels.shape[0])
        _budget = int(n_point_budget) if n_point_budget is not None else int(M)
        _kpts = min(int(M), _budget)
        if _ob == "host":
            _cap = int(os.environ.get("LORRAX_ORBIT_BLOCK_HOST_MAX_M",
                                      ORBIT_BLOCK_HOST_MAX_M_DEFAULT))
            if M > _cap:
                raise NotImplementedError(
                    f"LORRAX_ORBIT_BLOCK_SELECT=host is the numpy ORACLE and "
                    f"M={M} exceeds LORRAX_ORBIT_BLOCK_HOST_MAX_M={_cap}.  "
                    f"Drop the override to use the sharded kernel.")
            with timing.section("prune.select"):
                _Gh = np.asarray(_mh.process_allgather(G, tiled=True))[:M, :M]
                _order, _piv_pts, _trR = orbit_block_select_host(
                    _Gh, _dense, n_orbit_keep=_n_orb,
                    n_point_budget=_budget, tol_rel=tol_rel, score="trace")
            _piv_np = np.asarray(_piv_pts)
        else:
            with timing.section("prune.select"):
                _dense_pad = _dense.astype(np.int32)
                if n_pad:
                    _dense_pad = np.concatenate(
                        [_dense_pad, np.full((n_pad,), _n_orb, dtype=np.int32)])
                _oid_jax = device_put_process_local(
                    _dense_pad, NamedSharding(mesh, PartitionSpec(select_axis)))
                _active_init = None
                if n_pad:
                    _act = np.ones((M_pad,), dtype=bool)
                    _act[M:] = False
                    _active_init = device_put_process_local(
                        _act, NamedSharding(mesh, PartitionSpec(select_axis)))
                _score = os.environ.get(
                    "LORRAX_ORBIT_BLOCK_SCORE", "ratio").strip().lower()
                _sel_step = make_sharded_orbit_block_select(
                    mesh, M_pad, _kpts, _n_orb,
                    mesh_axis=select_axis, tol_rel=tol_rel, score=_score)
                (_piv, _L, _rank_j, _d_final, _d_taken, _trR_vec,
                 _psd) = _sel_step(G, _oid_jax, _active_init)
                _piv.block_until_ready()
            del _L
            _piv_np = np.asarray(_piv)
            _piv_np = _piv_np[_piv_np >= 0]
            _trR = float(np.asarray(_trR_vec)[int(_rank_j)])
            _order = _dense[_piv_np][
                np.sort(np.unique(_dense[_piv_np], return_index=True)[1])]

        # Whole-orbit point budget, on the orbit order the pivots induce.
        _sizes = np.bincount(_dense, minlength=_n_orb)
        _cum = np.cumsum(_sizes[_order])
        _k = int(np.searchsorted(_cum, _budget, side="right"))
        if _k < 1:
            raise RuntimeError(
                f"pivoted-Cholesky REFUSES: the point budget {_budget} is "
                f"smaller than the first orbit the block order ranked "
                f"({int(_sizes[_order[0]])} points).  Raise N.")
        _order = _order[:_k]
        _sel = np.isin(_dense, _order)
        keep_idx = np.asarray(cand_idx)[_sel]
        _n_del = int(_sel.sum())
        _rank = int(np.isin(_dense[_piv_np], _order).sum())
        if verbose:
            _path = ("host oracle" if _ob == "host"
                     else f"sharded, score={os.environ.get('LORRAX_ORBIT_BLOCK_SCORE', 'ratio').strip().lower()}")
            print(f"[pivoted_cholesky] ORBIT-BLOCK select ({_path}): "
                  f"{_order.shape[0]} orbits -> {_n_del} points, "
                  f"{_rank} rank-1 deflations "
                  f"({_rank}/{_n_del} = {_rank / max(_n_del, 1):.4f} certified "
                  f"directions per delivered point), tr(R)/tr(G)={_trR:.3e}")
            print(f"[pivoted_cholesky] orbit-aware: {_order.shape[0]} orbits "
                  f"→ {_n_del} unfolded centroids (orbit-closed)")
        # THE GATE THIS PATH CAN ACTUALLY STATE.  The old orbit gate compared
        # a rank counted in ORBITS against an orbit target, which is why it
        # printed "18/18 certified -- PASS" over a set that spanned the wrong
        # directions.  Here the deflation count IS at point granularity.
        _tol = float(os.environ.get("LORRAX_CENTROID_RANK_TOL", "0.01"))
        if _rank < int(np.ceil((1.0 - _tol) * _n_del)):
            raise RuntimeError(
                f"pivoted-Cholesky REFUSES (orbit-block): delivered "
                f"{_n_del} points but only {_rank} of them added an "
                f"independent direction at tol*max(diag G).  The admitted "
                f"orbits are numerically rank-deficient as POINT sets; the "
                f"file would claim {_n_del} centroids and span {_rank}.  "
                f"Lower N, or widen the prune window so the Gram sees the "
                f"pair densities the orbits actually carry.")
        return keep_idx, _rank, _n_del, None, None
    if orbit_id is not None and _ob == "legacy":
        print("[pivoted_cholesky] WARNING: LORRAX_ORBIT_BLOCK_SELECT=legacy — "
              "running the OLD orbit branch, which deflates one direction per "
              "orbit and is known to cost 56x of BerkeleyGW agreement on the "
              "Si anchor.  For bisection only.")

    # Run select on the row-sharded Gram. Orbit-aware mode passes orbit_id
    # row-sharded the same way as G; the body marks the whole orbit
    # inactive after each pivot pick (orbit_id of the pivot is broadcast
    # via psum-with-mask, same idiom as the L[p, :] broadcast).
    with timing.section("prune.select"):
        select_step = make_sharded_pivoted_cholesky_select(
            mesh, M_pad, n_keep, mesh_axis=select_axis, tol_rel=tol_rel,
        )
        # Pad mask: real candidates active, pads inactive.  None when n_pad==0
        # so the P=1 / already-divisible paths take the byte-identical old
        # code path (no extra operand, no extra shard_map input).
        active_init = None
        if n_pad:
            _act = np.ones((M_pad,), dtype=bool)
            _act[M:] = False
            active_init = device_put_process_local(
                _act, NamedSharding(mesh, PartitionSpec(select_axis)),
            )
        if orbit_id is None:
            (piv, L, rank, d_final, d_taken, trR_over_trG,
             psd_info) = select_step(G, None, active_init)
        else:
            # Process-local placement, NOT plain ``jax.device_put``: the
            # latter fires JAX's hidden ``assert_equal`` all-gather
            # (P × M × 4 bytes) on a multi-process mesh (scorecard AA.1).
            # ``orbit_id`` is a pure function of the candidate list +
            # symmetry ops, identical on every rank by construction;
            # ``LORRAX_CHECK_REPLICA=1`` restores the assertion.
            _oid = np.asarray(orbit_id, dtype=np.int32)
            if n_pad:
                # Pads get an orbit id no real candidate can hold, so the
                # orbit-kill mask (orbit_id == orbit_id_of_pivot) can never
                # reach them — belt and braces on top of the active mask.
                _oid = np.concatenate(
                    [_oid, np.full((n_pad,), -1, dtype=np.int32)])
            orbit_id_jax = device_put_process_local(
                _oid, NamedSharding(mesh, PartitionSpec(select_axis)),
            )
            (piv, L, rank, d_final, d_taken, trR_over_trG,
             psd_info) = select_step(G, orbit_id_jax, active_init)
        piv.block_until_ready()
    del L

    if verbose:
        print(f"[pivoted_cholesky] picked-pivot residuals: "
              f"first={float(d_taken[0]):.3e}, "
              f"mid={float(d_taken[n_keep // 2]):.3e}, "
              f"last={float(d_taken[-1]):.3e}")
        print(f"[pivoted_cholesky] tr(R_k)/tr(G): "
              f"first={float(trR_over_trG[1]):.3e}, "
              f"mid={float(trR_over_trG[n_keep // 2 + 1]):.3e}, "
              f"last={float(trR_over_trG[n_keep]):.3e}")

    piv_np = np.asarray(piv)
    diag_host = np.real(np.asarray(jnp.diag(G)))
    psd_host = (float(np.asarray(psd_info[0])), int(np.asarray(psd_info[1])),
                int(np.asarray(psd_info[2])))
    rank_i = int(rank)
    refuse_unless_select_certified(
        piv_np, rank_i, psd_host, n_keep=n_keep, M=M, M_pad=M_pad,
        orbit_id=orbit_id, d0max=float(diag_host.max()), tol_rel=tol_rel)

    if orbit_id is None:
        keep_idx = np.asarray(cand_idx)[piv_np]
    else:
        # Unfold: kept = union of orbits of picked pivots.
        orbit_id_np = np.asarray(orbit_id)
        piv_used = piv_np
        if n_point_budget is not None:
            # THE FLOOR.  Truncate the pivot order to the longest prefix that
            # fits the user's POINT budget.  Without this the delivered count
            # is Σ orbit_size over whatever orbits the greedy picked, which
            # overruns the number the user typed whenever the orbits it
            # ranked highest are the large ones.
            _labels, _inv = np.unique(orbit_id_np, return_inverse=True)
            _sizes = np.bincount(_inv)
            _sz_in_order = _sizes[np.searchsorted(_labels,
                                                  orbit_id_np[piv_np])]
            _cum = np.cumsum(_sz_in_order)
            _k = int(np.searchsorted(_cum, int(n_point_budget), side="right"))
            if _k < 1:
                raise RuntimeError(
                    f"pivoted-Cholesky REFUSES: the point budget "
                    f"{int(n_point_budget)} is smaller than the first orbit "
                    f"the pivot order ranked ({int(_sz_in_order[0])} "
                    f"points), so the largest union of whole orbits that fits "
                    f"is empty.  Orbit sizes present: "
                    f"{np.array2string(np.unique(_sizes))}.  Raise N.")
            piv_used = piv_np[:_k]
            if verbose and _k < len(piv_np):
                print(f"[pivoted_cholesky] ORBIT-FLOORED: "
                      f"{int(n_point_budget)} points requested -> REALIZED "
                      f"{int(_cum[_k - 1])} ({_k} of {len(piv_np)} picked "
                      f"orbits kept; the next holds "
                      f"{int(_sz_in_order[_k])} and would overrun).  The "
                      f"floor SPENDS LESS and never rounds up.")
        picked_orbits = orbit_id_np[piv_used]
        in_kept = np.isin(orbit_id_np, picked_orbits)
        keep_idx = np.asarray(cand_idx)[in_kept]
        if verbose:
            print(f"[pivoted_cholesky] orbit-aware: {len(piv_used)} orbits picked "
                  f"→ {len(keep_idx)} unfolded centroids (orbit-closed)")
        # R2 — certify at POINT granularity, which is the granularity of the
        # FILE being written.  Orbit mode only: in point mode ``rank`` is
        # already the point count and there is nothing to reconcile.
        pt_rank, n_pts, why = point_granularity_rank(
            G, in_kept, tol_rel=tol_rel)
        if verbose:
            if pt_rank is None:
                print(f"  [point rank] {rank_i} orbits, {n_pts} points, "
                      f"independent directions NOT MEASURED — {why}")
            else:
                print(f"  [point rank] {rank_i} orbits, {n_pts} points, "
                      f"{pt_rank} independent directions "
                      f"({100.0 * pt_rank / max(1, n_pts):.1f}% of the "
                      f"points)")
                if pt_rank < n_pts:
                    print(f"  [point rank] NOTE: {n_pts - pt_rank} of the "
                          f"{n_pts} delivered points add no independent "
                          f"direction at tol*max(diag G).  The zeta "
                          f"back-solve will truncate about that many modes "
                          f"per q; D3 shipped a 7 GiB restart file to learn "
                          f"the same thing downstream.")
                _note = point_rank_closure_note(G, in_kept, pt_rank,
                                                tol_rel=tol_rel)
                print(f"  [point rank] closure: " + (
                    _note if _note else
                    "the rank cut falls in a gap — no degenerate block is "
                    "sliced at this tolerance."))
    d_final_np = np.asarray(_mh.process_allgather(d_final, tiled=True))[:M]
    if n_pad:
        G = G[:M, :M]        # hand back the LOGICAL Gram, not the padded one
    return (keep_idx, rank_i, G, d_final_np, np.asarray(d_taken),
            np.asarray(trR_over_trG), psd_host)


# ═══════════════════════════════════════════════════════════════════════
# Multi-device — sharded pivoted-Cholesky select
# ═══════════════════════════════════════════════════════════════════════
# Consumes the row-sharded G ∈ ℂ^(M×M) that ``build_gram_q0_via_loadwfns``
# produces and runs the same greedy select as ``pivoted_cholesky_select``.
# Sharded along M: each device owns (M_slab, M) of G and (M_slab, k_keep)
# of L.
#
# Collectives per iteration (one per Lloyd-like step):
#
#   pmax(local_pv, 'x')       — 1 scalar: finds the global pivot value
#   pmax(-winner_p, 'x')      — 1 int32:  breaks ties by lowest device idx
#   psum(local_Lp, 'x')       — (k_keep,) array: broadcasts L[p, :] from
#                                its owning shard to every device
#
# Total comm per iter: O(k_keep). Total over k_keep iters: O(k_keep²).
# Matmul/Schur update are local — each device does L_slab @ (scalar)
# + elementwise ops on (M_slab,)- and (M_slab, k_keep)-shaped arrays.
#
# Column access `G[:, global_p]`: because G is ROW-sharded, each device's
# local slab already contains its portion of column p — no collective.
# This is why row-sharding is preferred over column-sharding for this
# algorithm.


def make_sharded_pivoted_cholesky_select(
    mesh: Mesh,
    M: int,
    k_keep: int,
    *,
    mesh_axis: str | tuple[str, ...] = 'x',
    tol_rel: float | None = None,
):
    """Sharded pivoted-Cholesky select on a row-sharded Gram.  STOPS at the
    numerical-rank floor, exactly as ``pivoted_cholesky_select`` does, and
    returns the same 7-tuple: ``(piv, L, rank, d_final, d_taken,
    trR_over_trG, psd_info)`` with shardings (replicated, row-sharded,
    replicated, row-sharded-1d, replicated, replicated, replicated).

    The stopping predicate is ``global_pv > floor``, and BOTH sides of it
    are ``pmax`` results — so every shard computes the same bool and the
    two kernels agree on where to stop at any shard count.  That is the
    property ``tests/test_centroid_distribution.py`` gates at >1 shard on an
    emulated mesh; before 2026-08-07 that gate ran both sides at 1×1 and
    every collective in here was satisfied vacuously."""
    dist.require_axes(mesh, mesh_axis, "make_sharded_pivoted_cholesky_select")
    n_dev = dist.n_shards(mesh, mesh_axis)
    if M % n_dev != 0:
        raise ValueError(f"M={M} must be divisible by product of mesh axes "
                         f"{mesh_axis} (= {n_dev})")
    M_slab = M // n_dev

    row_shard = PartitionSpec(mesh_axis, None)
    row_shard_1d = PartitionSpec(mesh_axis)
    rep = PartitionSpec()

    # Input layouts: G alone, or with ``orbit_id`` and/or ``active_init``, each
    # row-sharded the same way as G's row dim.  ``active_init`` is the
    # zero-pad mask (see ``prune_candidates_by_pivoted_cholesky``): rows marked
    # False are never eligible to be picked, which is what lets the caller pad
    # M up to a multiple of the mesh size instead of being refused.
    in_specs_no_orbit = (row_shard,)
    in_specs_orbit    = (row_shard, row_shard_1d)
    out_specs = (rep, row_shard, rep, row_shard_1d, rep, rep,
                 (rep, rep, rep))

    @jax.jit
    def step(G, orbit_id=None, active_init=None):
        def body_local(G_slab, orbit_id_slab=None, active_slab=None):
            real_dtype = G_slab.real.dtype
            eps = jnp.finfo(real_dtype).eps
            minus_inf = jnp.array(-jnp.inf, dtype=real_dtype)
            my_idx = lax.axis_index(mesh_axis)

            # Local diagonal of G: each device owns rows [my_idx*M_slab,
            # (my_idx+1)*M_slab); the diag entry sits at col == row.
            col_ids_local = my_idx * M_slab + jnp.arange(M_slab)
            local_diag_raw = jnp.real(
                G_slab[jnp.arange(M_slab), col_ids_local])
            local_diag = jnp.maximum(local_diag_raw, 0.0)
            trG = lax.psum(jnp.sum(local_diag), axis_name=mesh_axis)
            col_ids_k = jnp.arange(k_keep)
            # The floor moves ABOVE the loop, same as the reference kernel:
            # it is the stopping rule, and ``pmax`` makes it identical on
            # every shard so the two kernels stop at the same iteration.
            d0max_global = lax.pmax(jnp.max(local_diag_raw),
                                    axis_name=mesh_axis)
            tol = (jnp.sqrt(eps) if tol_rel is None
                   else jnp.asarray(tol_rel, real_dtype))
            floor = tol * d0max_global

            # A negative entry on the INITIAL diagonal is already a PSD
            # statement; step -1 names it.  Row indices are kept GLOBAL so
            # the refusal names a candidate, not a slab offset.
            at0_loc = jnp.argmin(local_diag_raw).astype(jnp.int32)
            at0_glob = (my_idx * M_slab + at0_loc).astype(jnp.int32)
            neg0 = local_diag_raw[at0_loc] < 0.0

            # Pad rows enter with d = 0 (their G row/col is exactly zero), so
            # they contribute nothing to trG, to d0max, or to the Schur update.
            # Starting them INACTIVE is what makes them unpickable: relying on
            # the tie-break (pads sit at the highest global indices and the
            # pivot rule takes the LOWEST index among ties) would work today but
            # is an accident, not a contract.
            active0 = (jnp.ones((M_slab,), dtype=bool) if active_slab is None
                       else active_slab.astype(bool))

            init = (
                local_diag,                                              # d_slab
                jnp.zeros((M_slab, k_keep), dtype=G_slab.dtype),         # L_slab
                -jnp.ones((k_keep,), dtype=jnp.int32),                   # piv
                active0,                                                 # active
                jnp.zeros((k_keep,), dtype=real_dtype),                  # d_taken
                # trR partials, LOCAL: slot 0 holds this shard's share of
                # trG so the post-loop psum makes trR_over_trG[0] exactly 1.
                jnp.zeros((k_keep + 1,), dtype=real_dtype).at[0].set(
                    jnp.sum(local_diag)),
                # d_min_raw / at_row / at_step — the pstrf INFO triple.
                jnp.minimum(local_diag_raw[at0_loc],
                            jnp.zeros((), dtype=real_dtype)),
                jnp.where(neg0, at0_glob, jnp.int32(-1)),
                jnp.int32(-1),
            )

            def body(j, carry):
                (d, L, piv, active, d_taken, trR_over_trG,
                 d_min_raw, d_min_at, d_min_j) = carry

                # Pick global pivot: per-device argmax then pmax + tie-break
                # to lowest global index.
                masked_d = jnp.where(active, d, minus_inf)
                local_p_idx = jnp.argmax(masked_d)
                local_pv = masked_d[local_p_idx]
                global_pv = lax.pmax(local_pv, mesh_axis)
                local_global_p = (my_idx * M_slab + local_p_idx).astype(jnp.int32)
                winner_p = jnp.where(
                    local_pv >= global_pv, local_global_p, jnp.int32(2**30),
                )
                global_p = -lax.pmax(-winner_p, mesh_axis)
                # THE STOP.  Both operands are pmax results, so this bool is
                # identical on every shard — no shard can run an iteration
                # another one skipped, and no collective goes unmatched.
                take = global_pv > floor
                pivot_val = jnp.maximum(global_pv, floor)

                # Column p of G (no collective: G is row-sharded).
                gcol_slab = G_slab[:, global_p]

                # Row p of L: broadcast from owning shard via masked psum.
                my_has_p = (global_p // M_slab == my_idx)
                local_p_rel = global_p - my_idx * M_slab
                safe_idx = jnp.clip(local_p_rel, 0, M_slab - 1)
                local_Lp = jnp.where(
                    my_has_p, L[safe_idx, :], jnp.zeros_like(L[safe_idx, :]),
                )
                if orbit_id_slab is None:
                    L_p = lax.psum(local_Lp, mesh_axis)
                else:
                    # ONE psum, not two.  The orbit id of the picked pivot
                    # rides the SAME masked broadcast as L[p, :] — it is the
                    # same idiom, from the same owner, at the same point in
                    # the iteration, and it was a second round trip purely
                    # because it was written a few lines further down.  Orbit
                    # ids are small integers and complex128 carries them
                    # exactly, so packing costs nothing in precision.
                    _oid_term = jnp.where(
                        my_has_p, orbit_id_slab[safe_idx], jnp.int32(0),
                    ).astype(G_slab.dtype)
                    _fused = lax.psum(
                        jnp.concatenate([local_Lp, _oid_term[None]]),
                        mesh_axis)
                    L_p = _fused[:k_keep]
                    orbit_id_p = jnp.round(
                        jnp.real(_fused[k_keep])).astype(jnp.int32)

                # New column.
                prev_mask = (col_ids_k < j).astype(G_slab.dtype)
                corr = L @ (jnp.conj(L_p) * prev_mask)
                denom = jnp.sqrt(pivot_val)
                newcol = (gcol_slab - corr) / denom
                # Pivot-row entry exactly sqrt(d[p]), only on the owner.
                fix_row_mask = my_has_p & (jnp.arange(M_slab) == local_p_rel)
                newcol = jnp.where(fix_row_mask, denom.astype(G_slab.dtype), newcol)
                newcol = jnp.where(take, newcol, jnp.zeros_like(newcol))

                L = L.at[:, j].set(newcol)
                piv = piv.at[j].set(jnp.where(take, global_p, jnp.int32(-1)))
                d_taken = d_taken.at[j].set(jnp.where(take, pivot_val, 0.0))

                # Schur update; the PSD detector reads the residual BEFORE
                # the clamp, over this shard's ACTIVE rows only.
                d_raw = d - jnp.abs(newcol) ** 2
                masked_raw = jnp.where(active, d_raw, jnp.inf)
                step_at = jnp.argmin(masked_raw).astype(jnp.int32)
                step_min = masked_raw[step_at]
                beats = take & (step_min < d_min_raw)
                d_min_at = jnp.where(
                    beats, (my_idx * M_slab + step_at).astype(jnp.int32),
                    d_min_at)
                d_min_j = jnp.where(beats, j.astype(jnp.int32), d_min_j)
                d_min_raw = jnp.where(beats, step_min, d_min_raw)
                d_new = jnp.maximum(jnp.where(take, d_raw, d), 0.0)
                # LOCAL partial only.  The psum that turns these into the
                # global trace ratio runs ONCE, after the loop, on the whole
                # (k_keep+1,) vector — it is a pure DIAGNOSTIC and paying a
                # collective round trip per iteration for a number nobody
                # reads until the end was the cheapest 25% on the hot path.
                trR_over_trG = trR_over_trG.at[j + 1].set(jnp.sum(d_new))
                if orbit_id_slab is None:
                    kill_mask = my_has_p & (jnp.arange(M_slab) == local_p_rel)
                else:
                    # ``orbit_id_p`` came back on the FUSED psum above.
                    kill_mask = orbit_id_slab == orbit_id_p
                kill_mask = kill_mask & take
                active = active & ~kill_mask
                d = jnp.where(kill_mask, minus_inf,
                              jnp.where(take, d_new, d))

                return (d, L, piv, active, d_taken, trR_over_trG,
                        d_min_raw, d_min_at, d_min_j)

            (d_final, L_out, piv_out, _, d_taken, trR_over_trG,
             d_min_raw, d_min_at, d_min_j) = lax.fori_loop(
                0, k_keep, body, init)
            d_final = jnp.where(jnp.isfinite(d_final), d_final, 0.0)
            # The one psum the per-iteration diagnostic was costing.
            trR_over_trG = lax.psum(trR_over_trG, axis_name=mesh_axis) / trG
            rank = jnp.sum(d_taken > floor).astype(jnp.int32)
            # THREE reductions, ONCE, after the loop — not per iteration:
            # the global minimum, then the row and step that attained it,
            # tie-broken to the lowest global row exactly as the pivot rule
            # is.  Putting these on the hot path would be a third collective
            # per iteration for a number only the refusal reads.
            g_min = -lax.pmax(-d_min_raw, axis_name=mesh_axis)
            mine = d_min_raw == g_min
            far = jnp.int32(2 ** 30)
            g_at = -lax.pmax(-jnp.where(mine, d_min_at, far),
                             axis_name=mesh_axis)
            g_j = -lax.pmax(-jnp.where(mine, d_min_j, far),
                            axis_name=mesh_axis)
            return (piv_out, L_out, rank, d_final, d_taken, trR_over_trG,
                    (g_min, g_at, g_j))

        specs = [row_shard]
        args = [G]
        if orbit_id is not None:
            specs.append(row_shard_1d); args.append(orbit_id)
        if active_init is not None:
            specs.append(row_shard_1d); args.append(active_init)
        has_orbit = orbit_id is not None
        has_active = active_init is not None

        def _entry(*a):
            g = a[0]
            i = 1
            oid = a[i] if has_orbit else None
            if has_orbit:
                i += 1
            act = a[i] if has_active else None
            return body_local(g, oid, act)

        return shard_map(
            _entry, mesh=mesh,
            in_specs=tuple(specs), out_specs=out_specs,
            check_vma=False,
        )(*args)

    return step



def make_sharded_orbit_block_select(
    mesh: Mesh,
    M: int,
    k_points: int,
    n_orb: int,
    *,
    mesh_axis: str | tuple[str, ...] = 'x',
    tol_rel: float | None = None,
    score: str = "ratio",
):
    """Sharded ORBIT-BLOCK pivoted-Cholesky select.

    Same row-sharded Gram, same collectives idioms and same stopping rule as
    ``make_sharded_pivoted_cholesky_select``; two things differ, and they are
    the whole repair:

      1. **The pivot search is restricted to the OPEN orbit.**  When the open
         orbit has no residual left, the next orbit is opened by the best
         score over the orbits not yet admitted (a ``segment_sum`` plus one
         fused ``psum`` on an ``(n_orb+1,)`` pair -- orbit counts are tens to
         hundreds, so this is cheap next to the O(M) work already in the
         body).

         THE COST MODEL, because the choice of score is not a preference.
         Admitting orbit ``o`` costs ``s_o`` POINTS out of the user's point
         budget and removes at most ``s_o * d_rep`` of residual trace.  That
         is a knapsack with cost ``s_o`` and value ``<= s_o * d_rep``, and
         greedy for knapsack ranks by value per unit COST:

             value / cost  =  (s_o * d_rep) / s_o  =  d_rep      <- "ratio"
             value         =   s_o * d_rep                       <- "trace"

         ``trace`` is the right greedy only for an UNBUDGETED pick-k-orbits
         problem -- which is the orbit-counting cost model the OLD code used,
         and exactly the thing this commit series set out to remove.  The
         delivered file is counted in POINTS, so ``ratio`` is the consistent
         score and is the default.  The two orderings COINCIDE when every
         orbit has the same size and diverge only where special positions
         (``|stabiliser| > 1``) enter: on Si 4x4x4 the pools carry sizes
         {8, 12, 24, 48}, so a size-8 orbit at ``d_rep = 5`` (5 units/point)
         is ranked above a size-48 orbit at ``d_rep = 1`` (1 unit/point),
         where ``trace`` ranks them 40 against 48 and takes the worse buy.
         ``score='trace'`` is retained so the pair stays measurable.

      2. **The Schur update kills the PIVOT, not the orbit.**  The old body
         did one rank-1 update and then dropped all ``n_sym`` members
         (``kill_mask = orbit_id == orbit_id_p``), so admitting a 48-point
         orbit deflated ONE direction while spending 48 points and every
         later orbit was ranked on a residual stale by 47 of them.

    ``k_points`` is therefore a POINT budget and the trip count is one
    rank-1 update per delivered point -- the same loop shape and the same
    cost as point mode, and independent of the FFT grid size.  **The k-means
    candidate pool is untouched:** this changes only which pool members are
    kept, so the owner's constraint that pruning must not run over the full
    grid is preserved by construction.

    ``orbit_id`` must be DENSE in ``[0, n_orb)``; zero-pad rows must carry
    ``n_orb`` as a sentinel (they land in a bucket nothing can open).

    Returns ``(piv, L, rank, d_final, d_taken, trR_over_trG, psd_info)`` with
    the same shardings as the point-mode factory.  ``piv`` is in pivot order
    and groups by orbit, so the host can read the admitted-orbit order off it
    and apply the whole-orbit point budget exactly as before.
    """
    dist.require_axes(mesh, mesh_axis, "make_sharded_orbit_block_select")
    n_dev = dist.n_shards(mesh, mesh_axis)
    if M % n_dev != 0:
        raise ValueError(f"M={M} must be divisible by product of mesh axes "
                         f"{mesh_axis} (= {n_dev})")
    M_slab = M // n_dev

    row_shard = PartitionSpec(mesh_axis, None)
    row_shard_1d = PartitionSpec(mesh_axis)
    rep = PartitionSpec()
    out_specs = (rep, row_shard, rep, row_shard_1d, rep, rep,
                 (rep, rep, rep))

    @jax.jit
    def step(G, orbit_id, active_init=None):
        def body_local(G_slab, orbit_id_slab, active_slab=None):
            real_dtype = G_slab.real.dtype
            eps = jnp.finfo(real_dtype).eps
            minus_inf = jnp.array(-jnp.inf, dtype=real_dtype)
            my_idx = lax.axis_index(mesh_axis)

            col_ids_local = my_idx * M_slab + jnp.arange(M_slab)
            local_diag_raw = jnp.real(G_slab[jnp.arange(M_slab), col_ids_local])
            local_diag = jnp.maximum(local_diag_raw, 0.0)
            trG = lax.psum(jnp.sum(local_diag), axis_name=mesh_axis)
            col_ids_k = jnp.arange(k_points)
            d0max_global = lax.pmax(jnp.max(local_diag_raw), axis_name=mesh_axis)
            tol = (jnp.sqrt(eps) if tol_rel is None
                   else jnp.asarray(tol_rel, real_dtype))
            floor = tol * d0max_global

            at0_loc = jnp.argmin(local_diag_raw).astype(jnp.int32)
            at0_glob = (my_idx * M_slab + at0_loc).astype(jnp.int32)
            neg0 = local_diag_raw[at0_loc] < 0.0

            avail0 = (jnp.ones((M_slab,), dtype=bool) if active_slab is None
                      else active_slab.astype(bool))
            oid = orbit_id_slab.astype(jnp.int32)

            init = (
                local_diag,                                          # d
                jnp.zeros((M_slab, k_points), dtype=G_slab.dtype),   # L
                -jnp.ones((k_points,), dtype=jnp.int32),             # piv
                avail0,                                              # unadmitted
                avail0,                                              # unpivoted
                jnp.int32(-1),                                       # cur orbit
                jnp.zeros((k_points,), dtype=real_dtype),            # d_taken
                jnp.zeros((k_points + 1,), dtype=real_dtype).at[0].set(
                    jnp.sum(local_diag)),
                jnp.minimum(local_diag_raw[at0_loc],
                            jnp.zeros((), dtype=real_dtype)),
                jnp.where(neg0, at0_glob, jnp.int32(-1)),
                jnp.int32(-1),
            )

            def body(j, carry):
                (d, L, piv, unadm, unpiv, cur, d_taken, trR,
                 d_min_raw, d_min_at, d_min_j) = carry

                # ---- is the OPEN orbit still worth pivoting? --------------
                in_cur = (oid == cur) & unpiv
                pv_cur = lax.pmax(jnp.max(jnp.where(in_cur, d, minus_inf)),
                                  axis_name=mesh_axis)
                need_new = pv_cur <= floor

                # ---- open the next orbit by residual TRACE ----------------
                # Computed unconditionally (jit has no cheap branch here) and
                # selected below; both terms are O(M) like the argmax already
                # in this body.
                _val = jax.ops.segment_sum(
                    jnp.where(unadm, d, 0.0), oid,
                    num_segments=n_orb + 1, indices_are_sorted=False)
                _cnt = jax.ops.segment_sum(
                    jnp.where(unadm, 1.0, 0.0).astype(_val.dtype), oid,
                    num_segments=n_orb + 1, indices_are_sorted=False)
                # ONE psum for both halves -- same idiom as the fused
                # L[p,:]/orbit-id broadcast in the point-mode body.
                _fused = lax.psum(jnp.concatenate([_val, _cnt]),
                                  axis_name=mesh_axis)
                _val, _cnt = _fused[:n_orb + 1], _fused[n_orb + 1:]
                if score == "ratio":
                    seg = _val / jnp.maximum(_cnt, 1.0)
                elif score == "trace":
                    seg = _val
                else:
                    raise ValueError(
                        f"score must be 'ratio' or 'trace'; got {score!r}")
                # A fully-admitted orbit has no remaining members: exclude it
                # explicitly rather than relying on its score being 0.
                seg = jnp.where(_cnt > 0, seg, -jnp.inf)
                seg = seg.at[n_orb].set(-jnp.inf)      # the pad sentinel bucket
                cand = jnp.argmax(seg).astype(jnp.int32)
                cur_new = jnp.where(need_new, cand, cur)

                in_cur = (oid == cur_new) & unpiv
                masked_d = jnp.where(in_cur, d, minus_inf)
                local_p_idx = jnp.argmax(masked_d)
                local_pv = masked_d[local_p_idx]
                global_pv = lax.pmax(local_pv, mesh_axis)
                local_global_p = (my_idx * M_slab + local_p_idx).astype(jnp.int32)
                winner_p = jnp.where(local_pv >= global_pv, local_global_p,
                                     jnp.int32(2 ** 30))
                global_p = -lax.pmax(-winner_p, mesh_axis)
                take = global_pv > floor
                pivot_val = jnp.maximum(global_pv, floor)

                # Mark the newly-opened orbit as admitted (whole).
                opened = need_new & take
                unadm = unadm & ~(opened & (oid == cur_new))

                gcol_slab = G_slab[:, global_p]
                my_has_p = (global_p // M_slab == my_idx)
                local_p_rel = global_p - my_idx * M_slab
                safe_idx = jnp.clip(local_p_rel, 0, M_slab - 1)
                local_Lp = jnp.where(my_has_p, L[safe_idx, :],
                                     jnp.zeros_like(L[safe_idx, :]))
                L_p = lax.psum(local_Lp, mesh_axis)

                prev_mask = (col_ids_k < j).astype(G_slab.dtype)
                corr = L @ (jnp.conj(L_p) * prev_mask)
                denom = jnp.sqrt(pivot_val)
                newcol = (gcol_slab - corr) / denom
                fix_row_mask = my_has_p & (jnp.arange(M_slab) == local_p_rel)
                newcol = jnp.where(fix_row_mask, denom.astype(G_slab.dtype),
                                   newcol)
                newcol = jnp.where(take, newcol, jnp.zeros_like(newcol))

                L = L.at[:, j].set(newcol)
                piv = piv.at[j].set(jnp.where(take, global_p, jnp.int32(-1)))
                d_taken = d_taken.at[j].set(jnp.where(take, pivot_val, 0.0))

                d_raw = d - jnp.abs(newcol) ** 2
                masked_raw = jnp.where(unpiv, d_raw, jnp.inf)
                step_at = jnp.argmin(masked_raw).astype(jnp.int32)
                step_min = masked_raw[step_at]
                beats = take & (step_min < d_min_raw)
                d_min_at = jnp.where(
                    beats, (my_idx * M_slab + step_at).astype(jnp.int32),
                    d_min_at)
                d_min_j = jnp.where(beats, j.astype(jnp.int32), d_min_j)
                d_min_raw = jnp.where(beats, step_min, d_min_raw)
                d_new = jnp.maximum(jnp.where(take, d_raw, d), 0.0)
                trR = trR.at[j + 1].set(jnp.sum(d_new))

                # THE KEY DIFFERENCE: kill the PIVOT only.  Its orbit siblings
                # stay pivotable, so the orbit is deflated by as many
                # directions as it delivers points.
                kill = my_has_p & (jnp.arange(M_slab) == local_p_rel) & take
                unpiv = unpiv & ~kill
                d = jnp.where(kill, 0.0, jnp.where(take, d_new, d))

                return (d, L, piv, unadm, unpiv, cur_new, d_taken, trR,
                        d_min_raw, d_min_at, d_min_j)

            (d_final, L_out, piv_out, _unadm, _unpiv, _cur, d_taken, trR,
             d_min_raw, d_min_at, d_min_j) = lax.fori_loop(
                0, k_points, body, init)
            d_final = jnp.where(jnp.isfinite(d_final), d_final, 0.0)
            trR = lax.psum(trR, axis_name=mesh_axis) / trG
            rank = jnp.sum(d_taken > floor).astype(jnp.int32)
            g_min = -lax.pmax(-d_min_raw, axis_name=mesh_axis)
            mine = d_min_raw == g_min
            far = jnp.int32(2 ** 30)
            g_at = -lax.pmax(-jnp.where(mine, d_min_at, far),
                             axis_name=mesh_axis)
            g_j = -lax.pmax(-jnp.where(mine, d_min_j, far),
                            axis_name=mesh_axis)
            return (piv_out, L_out, rank, d_final, d_taken, trR,
                    (g_min, g_at, g_j))

        specs = [row_shard, row_shard_1d]
        args = [G, orbit_id]
        has_active = active_init is not None
        if has_active:
            specs.append(row_shard_1d); args.append(active_init)

        def _entry(*a):
            return body_local(a[0], a[1], a[2] if has_active else None)

        return shard_map(
            _entry, mesh=mesh,
            in_specs=tuple(specs), out_specs=out_specs,
            check_vma=False,
        )(*args)

    return step


# ═══════════════════════════════════════════════════════════════════════
# Full 2-D Gram pipeline: load_wfns → pair density → q=0 Gram
# ═══════════════════════════════════════════════════════════════════════
#
# Uses the same data-loading path as the gw_jax ISDF fit:
#
#   read_Gvecs_to_devices(...)                       — full-BZ G-space wfns
#      ↓
#   get_sharded_wfns_centroids(...)                  — iFFT + gather at
#                                                      candidate points;
#                                                      returns psi_rmu_Y and
#                                                      psi_rmuT_X (the latter
#                                                      already conjugated)
#      ↓
#   compute_pair_density_spin_traced(psi_rmuT_X, psi_rmu_Y, mesh)
#      ↓    P_k[mu_X, nu_Y] = Σ_{n,s} ψ*(μ) ψ(ν)        (gw_jax convention)
#
# Called once for valence (→ P_v_k) and once for conduction (→ P_c_k). Then
# at q=0:
#
#   G[mu_X, nu_Y] = Σ_k w_k · conj(P_v_k) · P_c_k
#                = gw.isdf_fitting.compute_gram_q0_from_left_right(
#                      P_v_k, P_c_k, k_weights, mesh
#                  )
#
# The conj() on P_v_k flips it from gw_jax's Σ_v φ*(μ)φ(ν) to the
# valence-projector form Σ_v φ(a)φ*(b) the Gram definition needs.
#
# Uses full-BZ unfold with uniform k-weights = 1/nk_tot (read_Gvecs_to_devices
# unfolds symmetry, so IBZ-weighted IBZ data are not the inputs). This is the
# correct convention to match gw_jax's pair-density pipeline exactly.


def build_gram_q0_via_loadwfns(
    wfn: "WfnLoader",
    sym: symmetry_maps.SymMaps,
    cand_idx: jnp.ndarray,
    n_val: int | None = None,
    n_cond: int | None = None,
    mesh_xy: Mesh | None = None,
    *,
    bispinor: bool = False,
    verbose: bool = True,
    band_range_left: tuple[int, int] | None = None,
    band_range_right: tuple[int, int] | None = None,
    band_norms: np.ndarray | None = None,
    band_chunk_size: int = 64,
    use_phdf5: bool = False,
    memory_per_device_gb: float | None = None,
) -> jnp.ndarray:
    """Build the q=0 candidate Gram on a 2-D mesh using gw_jax's data path.

    Two call modes:

    * Simple ``(n_val, n_cond)`` (legacy): left window = ``(0, n_val)``,
      right window = ``(n_val, n_val + n_cond)``. This is the literal
      valence × conduction pair-product Gram used in the original assay.

    * Explicit ``(band_range_left, band_range_right)`` (gw_jax / ISDF
      convention): left = ``(b0, b3)`` = "all val + sigma cond", right =
      ``(b1, b4)`` = "sigma val + all cond". Matches the windowing used
      by ``gw_init.fit_zeta`` → ``isdf_fitting.fit_zeta_chunked_to_h5``.
      Passing ``band_norms`` additionally applies the pseudoband
      normalization ``ψ /= max(norm, 1.0)`` on both left and right
      (same clamp recipe as ``isdf_fitting.py:838-847``).

    Full-BZ unfold: one ``load_wfns.read_Gvecs_to_devices`` per window,
    ``get_sharded_wfns_centroids`` at the candidate indices, sharded
    pair densities via ``compute_pair_density_spin_traced``, combined
    with the q=0 sum via ``compute_gram_q0_from_left_right``. k-weights
    are uniform ``1 / nk_tot`` because we've unfolded.

    Args:
        wfn: open WFNReader.
        sym: matching SymMaps.
        cand_idx: (M, 3) int32 FFT-grid indices of candidate points.
            Must be a ``jnp.ndarray`` (will be ``jnp.asarray``-ed if not).
        n_val: valence-window size for the legacy mode (see above).
            Required when ``band_range_left`` is not given.
        n_cond: conduction-window size for the legacy mode. Required
            when ``band_range_left`` is not given.
        mesh_xy: 2-D device mesh with axes ``'x'`` and ``'y'``. (Other
            axis names work too, as long as both are present; the pair
            density / Gram helpers hard-code the axis names ``'x'`` and
            ``'y'`` at present — we follow that convention.)
        bispinor: if True, upcast the spin structure to 4 components
            (matches gw_jax's bispinor mode). Default False.
        verbose: print progress lines.
        band_range_left: optional explicit left window (start, end).
            When given, takes precedence over (n_val, n_cond).
        band_range_right: optional explicit right window.
        band_norms: optional (nbands,) array of band norms
            (``wfn.band_norms``) for pseudoband reweighting. When given,
            applied to both left and right ψ via
            ``ψ /= max(norm_slice, 1.0)`` before the pair-density
            einsum.

    Returns:
        G: (M, M) complex, sharded ``P('x','y')`` on the mesh — ready to
           be reshard-constrained to a 1-D row-shard for the select
           stage.
    """
    # Lazy imports — these modules pull in the full gw_jax dep chain and
    # we don't want to charge the single-device prune path for it.
    from common.meta import Meta
    from common.wfn_transforms import load_centroids_band_chunked
    from isdf import (
        pair_density,
        gram_q0_from_pair,
    )

    # Resolve windows.
    if band_range_left is None or band_range_right is None:
        if n_val is None or n_cond is None:
            raise ValueError(
                "Must supply either (n_val, n_cond) or "
                "(band_range_left, band_range_right)"
            )
        # v×(v+c) default: left = (0, n_val), right = (0, n_val+n_cond).
        # The centroids that prune-Cholesky picks then span the val×val
        # diagonals that V_H and any G_RI band-diagonal projection
        # consume, on top of the val×cond pair densities χ₀/W/Σ_xc need.
        # On MoS2 4×4 this cut V_H |err| at the CBM ~3× vs the legacy
        # v×c window (right=(n_val, n_val+n_cond)) at the same centroid
        # count.  Σ_xc is unaffected since the (v+c) right pool is a
        # superset of the legacy cond range; conditioning of the Gram
        # only improves (more PSD contributions).  Callers needing the
        # strict legacy v×c Gram should pass ``band_range_left=(0,nval)``
        # and ``band_range_right=(nval, nval+ncond)`` explicitly.
        left_range = (0, int(n_val))
        right_range = (0, int(n_val) + int(n_cond))
    else:
        left_range = (int(band_range_left[0]), int(band_range_left[1]))
        right_range = (int(band_range_right[0]), int(band_range_right[1]))

    nb_left = left_range[1] - left_range[0]
    nb_right = right_range[1] - right_range[0]
    if nb_left <= 0 or nb_right <= 0:
        raise ValueError(
            f"Empty band window: left={left_range} right={right_range}"
        )

    # Meta's nband must cover whichever of left/right reaches higher.
    max_band = max(left_range[1], right_range[1])
    # Keep Meta.b0..b4 consistent with the *legacy* nval/ncond semantics
    # when the caller passed those; otherwise use (max_band, max_band) so
    # the metadata bounds don't constrain anything downstream.
    meta_nval = int(n_val) if n_val is not None else nb_left
    meta_ncond = int(n_cond) if n_cond is not None else max(1, max_band - meta_nval)

    M = int(cand_idx.shape[0])
    cand_idx = jnp.asarray(cand_idx, dtype=jnp.int64)

    meta = Meta.from_system(
        wfn, sym,
        nval=meta_nval, ncond=meta_ncond,
        nband=max_band,
        n_rmu=M,
        bispinor=bispinor,
    )

    kw = jnp.ones((sym.nk_tot,), dtype=jnp.float64) / float(sym.nk_tot)

    # Memory budget for the band-+k-chunker. ``load_centroids_band_chunked``
    # reads ``meta.memory_per_device_gb`` to size the FFT-box per chunk; if
    # the caller didn't pin a budget, auto-detect device HBM the same way
    # gw_config does so the prune path tracks whatever the rest of LORRAX
    # is using.
    if memory_per_device_gb is None or memory_per_device_gb <= 0:
        try:
            from common.gpu_utils import get_device_memory_gb
            memory_per_device_gb = float(get_device_memory_gb())
        except Exception:
            memory_per_device_gb = 0.0  # falls back to the 36 GB default
    setattr(meta, "memory_per_device_gb", float(memory_per_device_gb))

    # Optional pseudoband norms — same clamp recipe as isdf_fitting.
    if band_norms is not None:
        band_norms_np = np.asarray(band_norms, dtype=np.float64)
        if band_norms_np.shape[0] < max_band:
            raise ValueError(
                f"band_norms has {band_norms_np.shape[0]} entries but "
                f"the left/right windows touch band {max_band}"
            )
        norms_l = np.maximum(
            band_norms_np[left_range[0]:left_range[1]], 1.0,
        )
        norms_r = np.maximum(
            band_norms_np[right_range[0]:right_range[1]], 1.0,
        )
        norms_l_j = jnp.asarray(norms_l, dtype=jnp.float64)
        norms_r_j = jnp.asarray(norms_r, dtype=jnp.float64)
    else:
        norms_l_j = None
        norms_r_j = None

    if verbose:
        print(f"[pivoted_cholesky] 2-D Gram build via load_wfns: "
              f"nk_tot={sym.nk_tot}, left={left_range} (nb={nb_left}), "
              f"right={right_range} (nb={nb_right}), M={M}, "
              f"norms={'on' if band_norms is not None else 'off'}, "
              f"backend={'phdf5' if use_phdf5 else 'WFNReader'}, "
              f"budget={meta.memory_per_device_gb:g} GB/device, "
              f"band_chunk_size={band_chunk_size}")

    # ---- Left window ----
    with timing.section("left.load"):
        psi_l_rmu_Y, psi_l_rmuT_X = load_centroids_band_chunked(
            wfn, sym, meta, cand_idx, bispinor, mesh_xy, left_range,
            band_chunk_size=band_chunk_size, use_phdf5=use_phdf5,
        )
        if norms_l_j is not None:
            # Y shape (nk, nb, ns, n_rmu); X shape (nk, n_rmu, nb, ns)
            psi_l_rmu_Y = psi_l_rmu_Y / norms_l_j[None, :, None, None]
            psi_l_rmuT_X = psi_l_rmuT_X / norms_l_j[None, None, :, None]
        psi_l_rmu_Y.block_until_ready()

    # ---- Single-device column-blocked path (size-ladder wall fix) ----
    # The full open-spin pair tensors are (nk, ns, ns, M, M): 98 GB EACH at
    # M~9.8k (c7000 kmeans killed a 192 GB node — 2026-07-28 job 7878309).
    # Per-element contraction order is unchanged by blocking the OUTPUT
    # columns, so G is numerically the same map; only materialization moves.
    # Multi-device meshes keep the original path untouched (the 'y'-sharded
    # column axis must not be sliced locally).
    n_dev_total = mesh_xy.devices.size
    col_block = 0
    # ── MULTI-DEVICE BLOCKING (2026-08-15) ──────────────────────────────────
    # This used to be `if n_dev_total == 1`, i.e. the blocking that exists to
    # stop the pair density from being materialised whole was disabled on
    # exactly the meshes that hit the wall.  MEASURED: the orbit-mode
    # candidate pool at oversample 6 on Si 4x4x4 (M = 3724, nk = 8, nspinor
    # = 2) asks for a single 13.44 GiB allocation ON EACH DEVICE of a 2x2
    # mesh and dies -- 2*nk*ns^2*M*16 B/column x M columns = 14.2 GB, which
    # is the request to the byte, so the (nk,ns,ns,M,M) intermediate is NOT
    # being spread by the mesh.  That ceiling was then mistaken for a
    # property of the algorithm and used to bound a design decision.
    #
    # Blocking the OUTPUT columns does not change the contraction order, so G
    # is the same map either way; only materialisation moves.  The one thing
    # a sharded mesh needs on top of the single-device path is that each
    # block's column extent divide the 'y' shard count, so the slice reshards
    # cleanly instead of splitting a shard.  Single-device behaviour is
    # untouched and byte-identical: same branch, same auto-budget, same
    # env override.
    _n_y = 1
    try:
        if 'y' in mesh_xy.axis_names:
            _n_y = int(mesh_xy.shape['y'])
    except Exception:
        _n_y = 1
    if True:
        # LORRAX_GRAM_COL_BLOCK: explicit column-block width; a falsy token
        # means "no override", i.e. the auto budget below.  This USED to be a bare
        # presence test — ``=0`` and ``=off`` are the two spellings a user
        # reaches for to DISABLE a knob, and they did the opposite or
        # crashed: "0" is a non-empty string, so it took the override
        # branch and ``max(256, 0)`` turned "off" into the SMALLEST legal
        # block (maximum blocking), while "off" died in ``int()`` mid-run
        # after the left window had already been loaded.  Same falsy
        # vocabulary as ``runtime._env_falsy`` and every other LORRAX knob.
        env_cb = os.environ.get("LORRAX_GRAM_COL_BLOCK", "").strip()
        if env_cb.lower() in ("", "0", "false", "no", "off"):
            env_cb = ""
        if env_cb:
            try:
                col_block = max(256, int(env_cb))
            except ValueError:
                raise ValueError(
                    f"LORRAX_GRAM_COL_BLOCK={env_cb!r} is neither a positive "
                    f"integer column width nor a falsy token "
                    f"('', 0, false, no, off)."
                ) from None
        else:
            nk_, _, ns_, M_ = psi_l_rmu_Y.shape
            bytes_per_col = 2 * nk_ * ns_ * ns_ * M_ * 16
            budget_bytes = float(meta.memory_per_device_gb) * 1e9 * 0.25
            col_block = max(256, int(budget_bytes // max(bytes_per_col, 1)))
        if n_dev_total > 1 and col_block:
            # Round UP to a whole number of 'y' shards: a block that splits a
            # shard would force a partial-shard slice on every block.
            col_block = ((col_block + _n_y - 1) // _n_y) * _n_y
        if col_block >= psi_l_rmu_Y.shape[3]:
            col_block = 0  # one full block == the original computation

    if col_block:
        M_cols = psi_l_rmu_Y.shape[3]
        if verbose:
            print(f"[pivoted_cholesky] column-blocked Gram: M={M_cols}, "
                  f"col_block={col_block} "
                  f"({-(-M_cols // col_block)} blocks; "
                  f"{'single-device' if n_dev_total == 1 else f'{n_dev_total}-device'} path)")
        with timing.section("right.load"):
            psi_r_rmu_Y, psi_r_rmuT_X = load_centroids_band_chunked(
                wfn, sym, meta, cand_idx, bispinor, mesh_xy, right_range,
                band_chunk_size=band_chunk_size, use_phdf5=use_phdf5,
            )
            if norms_r_j is not None:
                psi_r_rmu_Y = psi_r_rmu_Y / norms_r_j[None, :, None, None]
                psi_r_rmuT_X = psi_r_rmuT_X / norms_r_j[None, None, :, None]
            psi_r_rmu_Y.block_until_ready()
        with timing.section("q0_sum"):
            g_blocks = []
            for c0 in range(0, M_cols, col_block):
                c1 = min(c0 + col_block, M_cols)
                # Slicing a 'y'-sharded axis yields an array whose sharding
                # pjit will not accept against pair_density's declared
                # in_shardings, so each block is re-placed explicitly.  The
                # block width is a whole number of 'y' shards (rounded above),
                # so this is a local relabel, not a reshuffle.
                _yspec = NamedSharding(mesh_xy, PartitionSpec(None, None, None, 'y'))
                _l_blk = jax.device_put(psi_l_rmu_Y[..., c0:c1], _yspec)
                _r_blk = jax.device_put(psi_r_rmu_Y[..., c0:c1], _yspec)
                P_l_b = pair_density(psi_l_rmuT_X, _l_blk, mesh_xy)
                P_r_b = pair_density(psi_r_rmuT_X, _r_blk, mesh_xy)
                G_b = gram_q0_from_pair(P_l_b, P_r_b, kw, mesh_xy=mesh_xy,
                                        symmetrize=False)
                G_b.block_until_ready()
                g_blocks.append(G_b)
                del P_l_b, P_r_b, _l_blk, _r_blk
            G = jnp.concatenate(g_blocks, axis=1)
            # Same Hermitian symmetrization the unblocked kernel applies,
            # once, on the assembled square matrix.
            G = 0.5 * (G + jnp.conj(G.T))
            G.block_until_ready()
        del psi_l_rmu_Y, psi_l_rmuT_X, psi_r_rmu_Y, psi_r_rmuT_X
        return G

    with timing.section("left.pair"):
        P_l_k = pair_density(psi_l_rmuT_X, psi_l_rmu_Y, mesh_xy)
        P_l_k.block_until_ready()
    del psi_l_rmu_Y, psi_l_rmuT_X

    # ---- Right window ----
    with timing.section("right.load"):
        psi_r_rmu_Y, psi_r_rmuT_X = load_centroids_band_chunked(
            wfn, sym, meta, cand_idx, bispinor, mesh_xy, right_range,
            band_chunk_size=band_chunk_size, use_phdf5=use_phdf5,
        )
        if norms_r_j is not None:
            psi_r_rmu_Y = psi_r_rmu_Y / norms_r_j[None, :, None, None]
            psi_r_rmuT_X = psi_r_rmuT_X / norms_r_j[None, None, :, None]
        psi_r_rmu_Y.block_until_ready()
    with timing.section("right.pair"):
        P_r_k = pair_density(psi_r_rmuT_X, psi_r_rmu_Y, mesh_xy)
        P_r_k.block_until_ready()
    del psi_r_rmu_Y, psi_r_rmuT_X

    # ---- q=0 Gram: sum_k w_k · Σ_{αβ} conj(P_l_k,αβ) · P_r_k,αβ ----
    # γ̃ identity (charge channel) — open-spin Frobenius reduction.
    with timing.section("q0_sum"):
        G = gram_q0_from_pair(P_l_k, P_r_k, kw, mesh_xy=mesh_xy)
        G.block_until_ready()
    return G
