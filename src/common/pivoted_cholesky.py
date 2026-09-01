"""Pure pivoted-Cholesky row selection and numerical certificates.

This L2 module owns the greedy recurrence shared by centroid construction and
GW downfolding.  It knows only about a Hermitian PSD matrix, optional integer
group labels, an optional active-row mask, and a JAX mesh.  ``orbit_id`` is
retained as the compatibility spelling for arbitrary group labels; the
kernel assigns no physical interpretation to them.

The reference and row-sharded kernels return
``(piv, L, rank, d_final, d_taken, trR_over_trG, psd_info)``.  For an
``(M, M)`` Gram and ``k_keep`` requested rows, their shapes are ``(k_keep,)``,
``(M, k_keep)``, scalar, ``(M,)``, ``(k_keep,)``, ``(k_keep + 1,)``, and
three scalars.  The sharded kernel requires the Gram, optional labels, and
optional mask to be row-sharded across ``mesh_axis``; all outputs except
``L`` and ``d_final`` are replicated.

Selection semantics are deterministic: global ties choose the lowest row
index. The compatibility grouped selector returns one representative per
label. The block selector instead admits a complete group under a point budget
and pivots every member before scoring another group. The reference
implementation is the parity oracle for the distributed path.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec

from common.shard_map import shard_map


__all__ = [
    "pivoted_cholesky_select",
    "make_sharded_pivoted_cholesky_select",
    "group_block_pivoted_cholesky_select",
    "make_sharded_group_block_pivoted_cholesky_select",
    "make_sharded_group_panel_pivoted_cholesky_select",
]


def _mesh_axis_size(
    mesh: Mesh,
    mesh_axis: str | tuple[str, ...],
    who: str,
) -> int:
    """Validate ``mesh_axis`` and return its total shard count."""
    axes = (mesh_axis,) if isinstance(mesh_axis, str) else tuple(mesh_axis)
    if not axes:
        raise ValueError(f"{who}: mesh_axis must name at least one axis")
    missing = tuple(axis for axis in axes if axis not in mesh.axis_names)
    if missing:
        raise ValueError(
            f"{who}: mesh axes {missing} are absent from {mesh.axis_names}"
        )
    return math.prod(int(mesh.shape[axis]) for axis in axes)


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

    ``rank`` is the number of numerically independent pivots and is a
    contract: ``L[:, rank:] == 0`` and ``d_taken[rank:] == 0``.  Selection
    still delivers distinct active rows after the numerical floor is reached;
    ``piv[j] == -1`` means only that no active row remained at step ``j``.

    ``psd_info`` is ``(d_min_raw, at_row, at_step)`` — the most negative
    pre-clamp residual diagonal observed, the candidate row that attained
    it, and the iteration it happened on.  ``d_min_raw < -tol_rel·max(diag
    G)`` says the input was not positive semidefinite, and the two indices
    are what lets the refusal NAME the pivot rather than merely assert the
    condition, which is the ``pstrf`` ``INFO`` contract.

    ``tol_rel`` overrides the stopping tolerance, relative to the largest
    initial diagonal; ``None`` means ``sqrt(eps)``.

    When ``orbit_id`` is given (shape ``(M,)`` int), each pivot iteration
    marks every row with the picked row's label inactive.  Thus at most one
    row per group is returned; asking for more groups than exist produces
    the ``-1`` exhaustion sentinel.
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
        # …AND THE CONTINUATION, which is a different question from the stop.
        # ``take`` says "this pivot adds an independent DIRECTION"; ``avail``
        # says "there is still a row to hand back".  Past the numerical
        # rank the two diverge, and the selection keeps DELIVERING rows
        # (largest frozen residual first, ties to the lowest index — the same
        # deterministic rule as above) while ``rank`` below keeps counting
        # only the certified ones.  Whether a rank-deficient pool is accepted
        # is a caller policy, not part of this recurrence.
        # This is only safe because the clamp above already removed the
        # 2026-08-07 blow-up: with ``take`` false, ``newcol`` is exactly zero,
        # so ``d`` is unchanged and no divisor can run away.
        avail = jnp.any(active)

        # L[:, j] = (G[:, p] - Σ_{i<j} L[:, i] · conj(L[p, i])) / sqrt(d[p])
        prev_mask = (col_ids < j).astype(G.dtype)
        corr = L @ (jnp.conj(L[p, :]) * prev_mask)
        denom = jnp.sqrt(pivot_val)
        newcol = (G[:, p] - corr) / denom
        # Pivot entry exactly sqrt(d[p]) — kills rounding drift.
        newcol = newcol.at[p].set(denom.astype(G.dtype))
        newcol = jnp.where(take, newcol, jnp.zeros_like(newcol))

        L = L.at[:, j].set(newcol)
        # ``avail``, not ``take``: a delivered pivot is a real candidate index
        # even when it certifies no new direction.  ``-1`` now means ONLY
        # "the pool ran out", which is the structural refusal.
        piv = piv.at[j].set(jnp.where(avail, p, -1).astype(jnp.int32))
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
        # Mark p (or its whole label group, if orbit_id was provided) inactive.
        # ``avail``, not ``take``: without this the loop STALLS past the
        # numerical rank — it re-picks the same p every remaining iteration
        # and delivers nothing — which is what made the rank deficiency an
        # unavoidable refusal rather than a reportable fact.
        kill_mask = (orbit_id == orbit_id[p]) & avail
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


@partial(jax.jit, static_argnames=('point_budget', 'n_groups', 'tol_rel'))
def group_block_pivoted_cholesky_select(
    G: jnp.ndarray,
    point_budget: int,
    group_id: jnp.ndarray,
    *,
    n_groups: int,
    tol_rel: float | None = None,
    active_init: jnp.ndarray | None = None,
):
    """Select complete labelled groups under a point budget.

    This is block pivoting in the selection sense: a group is admitted as a
    whole, then every one of its rows is used in the ordinary rank-1 Schur
    recurrence before another group is scored.  Groups are scored by mean
    residual diagonal (residual value per point spent).  A group is eligible
    only when all of its active rows fit in the remaining point budget.

    ``group_id`` must be dense in ``[0, n_groups)``.  An optional inactive
    padding row may use ``n_groups`` as a sentinel.  The return contract is
    the same seven-tuple as :func:`pivoted_cholesky_select`; trailing ``-1``
    pivots mean that no complete remaining group fit the budget.  Numerical
    rank and delivered point count are intentionally separate: after the
    residual reaches ``tol_rel`` the selected rows are still returned, but
    their factor columns and ``d_taken`` entries are zero.
    """
    M = G.shape[0]
    real_dtype = G.real.dtype
    eps = jnp.finfo(real_dtype).eps
    minus_inf = jnp.array(-jnp.inf, dtype=real_dtype)
    active0 = (jnp.ones((M,), dtype=bool) if active_init is None
               else active_init.astype(bool))

    diag_raw = jnp.real(jnp.diag(G))
    diag0 = jnp.maximum(diag_raw, 0.0)
    d0 = jnp.where(active0, diag0, 0.0)
    tol = (jnp.sqrt(eps) if tol_rel is None
           else jnp.asarray(tol_rel, real_dtype))
    floor = tol * jnp.max(jnp.where(active0, diag_raw, minus_inf))
    trG = jnp.sum(d0)
    group_sizes = jax.ops.segment_sum(
        active0.astype(jnp.int32), group_id,
        num_segments=n_groups + 1, indices_are_sorted=False)

    initial_for_min = jnp.where(active0, diag_raw, jnp.inf)
    at0 = jnp.argmin(initial_for_min).astype(jnp.int32)
    neg0 = initial_for_min[at0] < 0.0
    init = (
        d0,
        jnp.zeros((M, point_budget), dtype=G.dtype),
        -jnp.ones((point_budget,), dtype=jnp.int32),
        active0,                         # not yet admitted
        active0,                         # admitted but not yet pivoted, or future
        jnp.int32(-1),                   # open group
        jnp.int32(0),                    # rows remaining in open group
        jnp.int32(0),                    # points committed by whole groups
        jnp.zeros((point_budget,), dtype=real_dtype),
        jnp.zeros((point_budget + 1,), dtype=real_dtype).at[0].set(1.0),
        jnp.minimum(initial_for_min[at0], jnp.zeros((), dtype=real_dtype)),
        jnp.where(neg0, at0, jnp.int32(-1)),
        jnp.int32(-1),
    )
    col_ids = jnp.arange(point_budget)

    def body(j, carry):
        (d, L, piv, unadmitted, available, current, current_left,
         committed, d_taken, trR, d_min_raw, d_min_at, d_min_j) = carry

        need_group = current_left == 0

        def choose_group(_):
            values = jax.ops.segment_sum(
                jnp.where(unadmitted, d, 0.0), group_id,
                num_segments=n_groups + 1, indices_are_sorted=False)
            remaining_budget = jnp.int32(point_budget) - committed
            eligible = ((group_sizes > 0)
                        & (group_sizes <= remaining_budget)
                        & (jnp.arange(n_groups + 1) < n_groups))
            scores = jnp.where(
                eligible,
                values / jnp.maximum(group_sizes.astype(real_dtype), 1.0),
                minus_inf,
            )
            candidate = jnp.argmax(scores).astype(jnp.int32)
            return candidate, jnp.max(scores) > minus_inf

        # Score groups only at a block boundary.  On a generic orbit this
        # removes ``group_size - 1`` redundant segment reductions per block.
        candidate, has_candidate = lax.cond(
            need_group,
            choose_group,
            lambda _: (current, jnp.bool_(False)),
            operand=None,
        )
        opened = need_group & has_candidate
        current = jnp.where(opened, candidate, current)
        safe_group = jnp.clip(candidate, 0, n_groups)
        opened_size = group_sizes[safe_group]
        current_left = jnp.where(opened, opened_size, current_left)
        committed = committed + jnp.where(opened, opened_size, 0)
        unadmitted = unadmitted & ~(opened & (group_id == current))

        in_current = available & (group_id == current)
        masked_d = jnp.where(in_current, d, minus_inf)
        p_raw = jnp.argmax(masked_d).astype(jnp.int32)
        pivot_raw = masked_d[p_raw]
        avail = pivot_raw > minus_inf
        p = jnp.where(avail, p_raw, jnp.int32(0))
        take = avail & (pivot_raw > floor)
        pivot_val = jnp.maximum(pivot_raw, floor)

        prev_mask = (col_ids < j).astype(G.dtype)
        corr = L @ (jnp.conj(L[p, :]) * prev_mask)
        denom = jnp.sqrt(pivot_val)
        newcol = (G[:, p] - corr) / denom
        newcol = newcol.at[p].set(denom.astype(G.dtype))
        newcol = jnp.where(take, newcol, jnp.zeros_like(newcol))
        L = L.at[:, j].set(newcol)
        piv = piv.at[j].set(jnp.where(avail, p, jnp.int32(-1)))
        d_taken = d_taken.at[j].set(jnp.where(take, pivot_val, 0.0))

        d_raw = d - jnp.abs(newcol) ** 2
        masked_raw = jnp.where(available, d_raw, jnp.inf)
        step_at = jnp.argmin(masked_raw).astype(jnp.int32)
        step_min = masked_raw[step_at]
        beats = take & (step_min < d_min_raw)
        d_min_at = jnp.where(beats, step_at, d_min_at)
        d_min_j = jnp.where(beats, j.astype(jnp.int32), d_min_j)
        d_min_raw = jnp.where(beats, step_min, d_min_raw)
        d_new = jnp.maximum(jnp.where(take, d_raw, d), 0.0)
        trR = trR.at[j + 1].set(jnp.sum(d_new) / trG)

        kill = (jnp.arange(M) == p) & avail
        available = available & ~kill
        current_left = current_left - jnp.where(avail, 1, 0)
        d = jnp.where(kill, 0.0, jnp.where(take, d_new, d))
        return (d, L, piv, unadmitted, available, current, current_left,
                committed, d_taken, trR, d_min_raw, d_min_at, d_min_j)

    (d, L, piv, _, available, _, _, _, d_taken, trR,
     d_min_raw, d_min_at, d_min_j) = lax.fori_loop(
        0, point_budget, body, init)
    d_final = jnp.where(available, d, 0.0)
    rank = jnp.sum(d_taken > floor).astype(jnp.int32)
    return (piv, L, rank, d_final, d_taken, trR,
            (d_min_raw, d_min_at, d_min_j))


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
    n_dev = _mesh_axis_size(
        mesh, mesh_axis, "make_sharded_pivoted_cholesky_select"
    )
    if M % n_dev != 0:
        raise ValueError(f"M={M} must be divisible by product of mesh axes "
                         f"{mesh_axis} (= {n_dev})")
    M_slab = M // n_dev

    row_shard = PartitionSpec(mesh_axis, None)
    row_shard_1d = PartitionSpec(mesh_axis)
    rep = PartitionSpec()

    # Input layouts: G alone, or with ``orbit_id`` and/or ``active_init``, each
    # row-sharded the same way as G's row dim. Rows marked False in
    # ``active_init`` are never eligible for selection.
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
                # THE CONTINUATION — see the reference kernel for the whole
                # argument.  ``global_pv`` is ``-inf`` exactly when no shard
                # holds an active candidate, so this bool is a pmax result
                # too and is identical on every shard.
                avail = global_pv > minus_inf

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
                piv = piv.at[j].set(jnp.where(avail, global_p, jnp.int32(-1)))
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
                kill_mask = kill_mask & avail
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


def make_sharded_group_block_pivoted_cholesky_select(
    mesh: Mesh,
    M: int,
    point_budget: int,
    n_groups: int,
    *,
    mesh_axis: str | tuple[str, ...] = 'x',
    tol_rel: float | None = None,
):
    """Row-sharded counterpart of
    :func:`group_block_pivoted_cholesky_select`.

    The Gram and group labels are row-sharded across ``mesh_axis``.  Group
    scores use one all-reduce of ``n_groups + 1`` residual sums when a new
    group is opened; no group-score collective runs while the admitted group
    is being deflated. The ordinary pivot value, global tie-break and
    factor-row broadcast keep the same collective order as the point selector.
    Group sizes are reduced once before the loop.
    """
    n_dev = _mesh_axis_size(
        mesh, mesh_axis, "make_sharded_group_block_pivoted_cholesky_select"
    )
    if M % n_dev != 0:
        raise ValueError(f"M={M} must be divisible by product of mesh axes "
                         f"{mesh_axis} (= {n_dev})")
    if point_budget < 1 or point_budget > M:
        raise ValueError(
            f"point_budget must lie in [1, M={M}]; got {point_budget}")
    if n_groups < 1 or n_groups > M:
        raise ValueError(f"n_groups must lie in [1, M={M}]; got {n_groups}")
    M_slab = M // n_dev

    row_shard = PartitionSpec(mesh_axis, None)
    row_shard_1d = PartitionSpec(mesh_axis)
    rep = PartitionSpec()
    out_specs = (rep, row_shard, rep, row_shard_1d, rep, rep,
                 (rep, rep, rep))

    @jax.jit
    def step(G, group_id, active_init=None):
        def body_local(G_slab, group_id_slab, active_slab=None):
            real_dtype = G_slab.real.dtype
            eps = jnp.finfo(real_dtype).eps
            minus_inf = jnp.array(-jnp.inf, dtype=real_dtype)
            my_idx = lax.axis_index(mesh_axis)
            local_rows = jnp.arange(M_slab)
            global_rows = my_idx * M_slab + local_rows
            local_diag_raw = jnp.real(G_slab[local_rows, global_rows])
            local_diag = jnp.maximum(local_diag_raw, 0.0)
            active0 = (jnp.ones((M_slab,), dtype=bool)
                       if active_slab is None else active_slab.astype(bool))
            local_diag = jnp.where(active0, local_diag, 0.0)
            trG = lax.psum(jnp.sum(local_diag), axis_name=mesh_axis)
            d0max = lax.pmax(
                jnp.max(jnp.where(active0, local_diag_raw, minus_inf)),
                axis_name=mesh_axis)
            tol = (jnp.sqrt(eps) if tol_rel is None
                   else jnp.asarray(tol_rel, real_dtype))
            floor = tol * d0max
            group_id_slab = group_id_slab.astype(jnp.int32)
            local_sizes = jax.ops.segment_sum(
                active0.astype(jnp.int32), group_id_slab,
                num_segments=n_groups + 1, indices_are_sorted=False)
            group_sizes = lax.psum(local_sizes, axis_name=mesh_axis)

            initial_for_min = jnp.where(active0, local_diag_raw, jnp.inf)
            at0_loc = jnp.argmin(initial_for_min).astype(jnp.int32)
            at0_glob = (my_idx * M_slab + at0_loc).astype(jnp.int32)
            neg0 = initial_for_min[at0_loc] < 0.0
            init = (
                local_diag,
                jnp.zeros((M_slab, point_budget), dtype=G_slab.dtype),
                -jnp.ones((point_budget,), dtype=jnp.int32),
                active0,                         # not admitted
                active0,                         # not pivoted
                jnp.int32(-1),                   # open group
                jnp.int32(0),                    # rows left in open group
                jnp.int32(0),                    # points committed
                jnp.zeros((point_budget,), dtype=real_dtype),
                jnp.zeros((point_budget + 1,), dtype=real_dtype).at[0].set(
                    jnp.sum(local_diag)),
                jnp.minimum(initial_for_min[at0_loc],
                            jnp.zeros((), dtype=real_dtype)),
                jnp.where(neg0, at0_glob, jnp.int32(-1)),
                jnp.int32(-1),
            )
            col_ids = jnp.arange(point_budget)

            def body(j, carry):
                (d, L, piv, unadmitted, available, current, current_left,
                 committed, d_taken, trR, d_min_raw,
                 d_min_at, d_min_j) = carry

                need_group = current_left == 0

                def choose_group(_):
                    local_values = jax.ops.segment_sum(
                        jnp.where(unadmitted, d, 0.0), group_id_slab,
                        num_segments=n_groups + 1, indices_are_sorted=False)
                    values = lax.psum(local_values, axis_name=mesh_axis)
                    remaining_budget = jnp.int32(point_budget) - committed
                    eligible = ((group_sizes > 0)
                                & (group_sizes <= remaining_budget)
                                & (jnp.arange(n_groups + 1) < n_groups))
                    scores = jnp.where(
                        eligible,
                        values / jnp.maximum(
                            group_sizes.astype(real_dtype), 1.0),
                        minus_inf,
                    )
                    candidate = jnp.argmax(scores).astype(jnp.int32)
                    return candidate, jnp.max(scores) > minus_inf

                # ``need_group`` and its inputs are replicated, so every
                # shard enters the collective branch together.
                candidate, has_candidate = lax.cond(
                    need_group,
                    choose_group,
                    lambda _: (current, jnp.bool_(False)),
                    operand=None,
                )
                opened = need_group & has_candidate
                current = jnp.where(opened, candidate, current)
                safe_group = jnp.clip(candidate, 0, n_groups)
                opened_size = group_sizes[safe_group]
                current_left = jnp.where(opened, opened_size, current_left)
                committed = committed + jnp.where(opened, opened_size, 0)
                unadmitted = unadmitted & ~(
                    opened & (group_id_slab == current))

                in_current = available & (group_id_slab == current)
                masked_d = jnp.where(in_current, d, minus_inf)
                local_p_idx = jnp.argmax(masked_d).astype(jnp.int32)
                local_pv = masked_d[local_p_idx]
                global_pv = lax.pmax(local_pv, axis_name=mesh_axis)
                local_global_p = (my_idx * M_slab + local_p_idx).astype(
                    jnp.int32)
                winner_p = jnp.where(
                    local_pv >= global_pv, local_global_p, jnp.int32(2**30))
                global_p_raw = -lax.pmax(-winner_p, axis_name=mesh_axis)
                avail = global_pv > minus_inf
                global_p = jnp.where(avail, global_p_raw, jnp.int32(0))
                take = avail & (global_pv > floor)
                pivot_val = jnp.maximum(global_pv, floor)

                gcol_slab = G_slab[:, global_p]
                my_has_p = (global_p // M_slab == my_idx) & avail
                local_p_rel = global_p - my_idx * M_slab
                safe_idx = jnp.clip(local_p_rel, 0, M_slab - 1)
                local_Lp = jnp.where(
                    my_has_p, L[safe_idx, :], jnp.zeros_like(L[safe_idx, :]))
                L_p = lax.psum(local_Lp, axis_name=mesh_axis)

                prev_mask = (col_ids < j).astype(G_slab.dtype)
                corr = L @ (jnp.conj(L_p) * prev_mask)
                denom = jnp.sqrt(pivot_val)
                newcol = (gcol_slab - corr) / denom
                fix_row = my_has_p & (local_rows == local_p_rel)
                newcol = jnp.where(
                    fix_row, denom.astype(G_slab.dtype), newcol)
                newcol = jnp.where(take, newcol, jnp.zeros_like(newcol))
                L = L.at[:, j].set(newcol)
                piv = piv.at[j].set(jnp.where(
                    avail, global_p, jnp.int32(-1)))
                d_taken = d_taken.at[j].set(jnp.where(
                    take, pivot_val, 0.0))

                d_raw = d - jnp.abs(newcol) ** 2
                masked_raw = jnp.where(available, d_raw, jnp.inf)
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

                kill = my_has_p & (local_rows == local_p_rel)
                available = available & ~kill
                current_left = current_left - jnp.where(avail, 1, 0)
                d = jnp.where(kill, 0.0, jnp.where(take, d_new, d))
                return (d, L, piv, unadmitted, available, current,
                        current_left, committed, d_taken, trR,
                        d_min_raw, d_min_at, d_min_j)

            (d, L, piv, _, available, _, _, _, d_taken, trR,
             d_min_raw, d_min_at, d_min_j) = lax.fori_loop(
                0, point_budget, body, init)
            d_final = jnp.where(available, d, 0.0)
            trR = lax.psum(trR, axis_name=mesh_axis) / trG
            rank = jnp.sum(d_taken > floor).astype(jnp.int32)
            g_min = -lax.pmax(-d_min_raw, axis_name=mesh_axis)
            mine = d_min_raw == g_min
            far = jnp.int32(2**30)
            g_at = -lax.pmax(
                -jnp.where(mine, d_min_at, far), axis_name=mesh_axis)
            g_j = -lax.pmax(
                -jnp.where(mine, d_min_j, far), axis_name=mesh_axis)
            return (piv, L, rank, d_final, d_taken, trR,
                    (g_min, g_at, g_j))

        specs = [row_shard, row_shard_1d]
        args = [G, group_id]
        has_active = active_init is not None
        if has_active:
            specs.append(row_shard_1d)
            args.append(active_init)

        def _entry(*a):
            return body_local(a[0], a[1], a[2] if has_active else None)

        return shard_map(
            _entry, mesh=mesh, in_specs=tuple(specs), out_specs=out_specs,
            check_vma=False,
        )(*args)

    return step


def make_sharded_group_panel_pivoted_cholesky_select(
    mesh: Mesh,
    M: int,
    point_budget: int,
    group_start,
    group_size,
    *,
    mesh_axis: str | tuple[str, ...] = 'x',
    tol_rel: float | None = None,
):
    """Panelized whole-group selection on an orbit-local row layout.

    Each group must occupy a contiguous interval wholly owned by one row
    shard.  Opening a group then costs two collectives: its global score and
    one fused broadcast of the owner's previous factor rows, residual
    diagonal, canonical IDs, and small Gram block.  The group's rank-``b``
    update is a local GEMM plus triangular solve; no collective occurs inside
    its member loop.

    ``step(G, group_id, canonical_id, active_init)`` returns the usual
    seven-tuple.  Pivots and PSD row receipts are canonical IDs, while ``L``
    and ``d_final`` remain in the supplied packed row order.
    """
    n_dev = _mesh_axis_size(
        mesh, mesh_axis, "make_sharded_group_panel_pivoted_cholesky_select")
    if M % n_dev:
        raise ValueError(
            f"M={M} must be divisible by product of mesh axes "
            f"{mesh_axis} (= {n_dev})")
    if point_budget < 1 or point_budget > M:
        raise ValueError(
            f"point_budget must lie in [1, M={M}]; got {point_budget}")
    starts_np = jnp.asarray(group_start, dtype=jnp.int32)
    sizes_np = jnp.asarray(group_size, dtype=jnp.int32)
    if starts_np.ndim != 1 or sizes_np.shape != starts_np.shape \
            or int(starts_np.size) < 1:
        raise ValueError("group_start/group_size must be equal nonempty vectors")
    # Validate on the host before tracing; communication kernels may trust it.
    starts_host = list(map(int, group_start))
    sizes_host = list(map(int, group_size))
    M_slab = M // n_dev
    for group, (start, size) in enumerate(zip(starts_host, sizes_host)):
        if size < 1 or start < 0 or start + size > M:
            raise ValueError(
                f"group {group} interval [{start}, {start + size}) is outside "
                f"[0, {M})")
        if start // M_slab != (start + size - 1) // M_slab:
            raise ValueError(
                f"group {group} interval [{start}, {start + size}) crosses "
                f"row-shard boundaries of width {M_slab}")
    n_groups = len(starts_host)
    max_group = max(sizes_host)

    row_shard = PartitionSpec(mesh_axis, None)
    row_shard_1d = PartitionSpec(mesh_axis)
    rep = PartitionSpec()
    out_specs = (rep, row_shard, rep, row_shard_1d, rep, rep,
                 (rep, rep, rep))

    @jax.jit
    def step(G, group_id, canonical_id, active_init):
        def body_local(G_slab, group_slab, canonical_slab, active_slab):
            real_dtype = G_slab.real.dtype
            minus_inf = jnp.array(-jnp.inf, dtype=real_dtype)
            plus_inf = jnp.array(jnp.inf, dtype=real_dtype)
            far = jnp.int32(2 ** 30)
            my_idx = lax.axis_index(mesh_axis)
            local_rows = jnp.arange(M_slab, dtype=jnp.int32)
            global_rows = my_idx * M_slab + local_rows
            group_slab = group_slab.astype(jnp.int32)
            canonical_slab = canonical_slab.astype(jnp.int32)
            active0 = active_slab.astype(bool)
            local_diag_raw = jnp.real(
                G_slab[local_rows, global_rows])
            d0 = jnp.where(active0, jnp.maximum(local_diag_raw, 0.0), 0.0)
            trG = lax.psum(jnp.sum(d0), axis_name=mesh_axis)
            d0max = lax.pmax(
                jnp.max(jnp.where(active0, local_diag_raw, minus_inf)),
                axis_name=mesh_axis)
            eps = jnp.finfo(real_dtype).eps
            tol = (jnp.sqrt(eps) if tol_rel is None
                   else jnp.asarray(tol_rel, real_dtype))
            floor = tol * d0max

            min0 = jnp.min(jnp.where(active0, local_diag_raw, plus_inf))
            at0 = jnp.min(jnp.where(
                active0 & (local_diag_raw == min0), canonical_slab, far))
            neg0 = min0 < 0.0
            storage = point_budget + max_group
            init = (
                jnp.int32(0),                         # groups processed
                jnp.bool_(True),                      # another group may fit
                jnp.int32(0),                         # points committed
                d0,
                jnp.zeros((M_slab, storage), dtype=G_slab.dtype),
                -jnp.ones((storage,), dtype=jnp.int32),
                active0,
                jnp.zeros((n_groups,), dtype=bool),
                jnp.zeros((storage,), dtype=real_dtype),
                jnp.zeros((storage + 1,), dtype=real_dtype).at[0].set(
                    jnp.sum(d0)),
                jnp.minimum(min0, jnp.zeros((), dtype=real_dtype)),
                jnp.where(neg0, at0, jnp.int32(-1)),
                jnp.int32(-1),
            )
            panel_rows = jnp.arange(max_group, dtype=jnp.int32)
            factor_cols = jnp.arange(max_group, dtype=jnp.int32)

            def keep_going(carry):
                iteration, running, committed = carry[:3]
                return ((iteration < n_groups) & running
                        & (committed < point_budget))

            def open_group(carry):
                (iteration, _running, committed, d, L, piv, available,
                 chosen, d_taken, trR, d_min_raw, d_min_at,
                 d_min_j) = carry

                local_values = jax.ops.segment_sum(
                    jnp.where(available, d, 0.0), group_slab,
                    num_segments=n_groups + 1, indices_are_sorted=False)
                values = lax.psum(local_values, axis_name=mesh_axis)
                remaining = jnp.int32(point_budget) - committed
                eligible = ((sizes_np <= remaining) & ~chosen)
                scores = jnp.where(
                    eligible,
                    values[:n_groups] / sizes_np.astype(real_dtype),
                    minus_inf)
                candidate = jnp.argmax(scores).astype(jnp.int32)
                has_candidate = jnp.max(scores) > minus_inf
                block_size = jnp.where(
                    has_candidate, sizes_np[candidate], jnp.int32(0))
                start = jnp.where(
                    has_candidate, starts_np[candidate], jnp.int32(0))
                members = start + panel_rows
                valid_member = panel_rows < block_size
                owner = start // M_slab
                mine = (my_idx == owner) & has_candidate
                local_member = members - my_idx * M_slab
                safe_local = jnp.clip(local_member, 0, M_slab - 1)
                safe_global = jnp.clip(members, 0, M - 1)

                # One owner broadcast per admitted group.  The complex carrier
                # fuses four faces that would otherwise be four round trips.
                local_L = jnp.where(
                    (mine & valid_member)[:, None],
                    L[safe_local, :point_budget], 0.0)
                local_G = G_slab[safe_local[:, None],
                                 safe_global[None, :]]
                local_G = jnp.where(
                    mine & valid_member[:, None] & valid_member[None, :],
                    local_G, 0.0)
                local_d = jnp.where(
                    mine & valid_member, d[safe_local], 0.0)
                local_canonical = jnp.where(
                    mine & valid_member, canonical_slab[safe_local], 0)
                payload = jnp.concatenate([
                    local_L.reshape(-1),
                    local_G.reshape(-1),
                    local_d.astype(G_slab.dtype),
                    local_canonical.astype(G_slab.dtype),
                ])
                payload = lax.psum(payload, axis_name=mesh_axis)
                n_l = max_group * point_budget
                n_g = max_group * max_group
                L_group = payload[:n_l].reshape(max_group, point_budget)
                G_group = payload[n_l:n_l + n_g].reshape(
                    max_group, max_group)
                d_group = jnp.real(payload[n_l + n_g:n_l + n_g + max_group])
                canonical_group = jnp.rint(jnp.real(
                    payload[n_l + n_g + max_group:])).astype(jnp.int32)
                prev = (jnp.arange(point_budget) < committed).astype(
                    G_slab.dtype)
                L_group_prev = L_group * prev[None, :]
                residual_group = (
                    G_group - L_group_prev @ jnp.conj(L_group_prev).T)

                # Replicated small pivoted Cholesky.  It fixes the member
                # order and triangular factor without another collective.
                small_init = (
                    d_group,
                    jnp.zeros((max_group, max_group), dtype=G_slab.dtype),
                    -jnp.ones((max_group,), dtype=jnp.int32),
                    jnp.zeros((max_group,), dtype=real_dtype),
                    valid_member & has_candidate,
                )

                def small_body(column, small):
                    d_s, F, relative_piv, taken, unused = small
                    masked = jnp.where(unused, d_s, minus_inf)
                    pivot_value = jnp.max(masked)
                    tied_id = jnp.min(jnp.where(
                        unused & (masked == pivot_value),
                        canonical_group, far))
                    relative = jnp.argmin(jnp.where(
                        unused & (canonical_group == tied_id),
                        canonical_group, far)).astype(jnp.int32)
                    avail = pivot_value > minus_inf
                    take = avail & (pivot_value > floor)
                    safe_value = jnp.maximum(pivot_value, floor)
                    corr = F @ (jnp.conj(F[relative, :])
                                * (factor_cols < column))
                    new_column = (
                        residual_group[:, relative] - corr) / jnp.sqrt(
                            safe_value)
                    new_column = new_column.at[relative].set(
                        jnp.sqrt(safe_value).astype(G_slab.dtype))
                    new_column = jnp.where(
                        take & valid_member, new_column, 0.0)
                    F = F.at[:, column].set(new_column)
                    relative_piv = relative_piv.at[column].set(
                        jnp.where(avail, relative, -1))
                    taken = taken.at[column].set(
                        jnp.where(take, safe_value, 0.0))
                    d_next = jnp.maximum(jnp.where(
                        take, d_s - jnp.abs(new_column) ** 2, d_s), 0.0)
                    unused = unused & ~(
                        (panel_rows == relative) & avail)
                    return d_next, F, relative_piv, taken, unused

                _, F, relative_piv, panel_taken, _ = lax.fori_loop(
                    0, max_group, small_body, small_init)
                safe_relative = jnp.clip(relative_piv, 0, max_group - 1)
                pivot_members = safe_global[safe_relative]
                pivot_canonical = canonical_group[safe_relative]
                member_valid_in_order = factor_cols < block_size
                L_piv_prev = L_group_prev[safe_relative]
                G_columns = G_slab[:, pivot_members]
                cross = G_columns - (
                    L[:, :point_budget] @ jnp.conj(L_piv_prev).T)
                cross = jnp.where(member_valid_in_order[None, :], cross, 0.0)
                triangular = F[safe_relative, :]
                taken_mask = panel_taken > 0.0
                triangular = triangular.at[
                    factor_cols, factor_cols].set(jnp.where(
                        taken_mask, jnp.diag(triangular),
                        jnp.ones((max_group,), dtype=G_slab.dtype)))
                W = jax.scipy.linalg.solve_triangular(
                    jnp.conj(triangular), cross.T, lower=True).T
                W = jnp.where(taken_mask[None, :], W, 0.0)

                # The selected owner rows are the small factor by definition;
                # pinning them removes solve roundoff from later block scores.
                def fix_owner_row(row, panel):
                    old = panel[safe_local[row]]
                    replacement = jnp.where(
                        mine & valid_member[row], F[row], old)
                    return panel.at[safe_local[row]].set(replacement)

                W = lax.fori_loop(0, max_group, fix_owner_row, W)

                # Diagnostics and exact per-point trace history need a cheap
                # elementwise scan, not another factorization or collective.
                def install(column, state):
                    (d_i, L_i, piv_i, available_i, d_taken_i, trR_i,
                     min_i, min_at_i, min_j_i) = state
                    valid = has_candidate & (column < block_size)
                    take = valid & (panel_taken[column] > 0.0)
                    slot = committed + column
                    new_column = jnp.where(take, W[:, column], 0.0)
                    L_i = L_i.at[:, slot].set(jnp.where(
                        valid, new_column, L_i[:, slot]))
                    piv_i = piv_i.at[slot].set(jnp.where(
                        valid, pivot_canonical[column], piv_i[slot]))
                    d_taken_i = d_taken_i.at[slot].set(jnp.where(
                        valid, panel_taken[column], d_taken_i[slot]))
                    raw = d_i - jnp.abs(new_column) ** 2
                    masked_raw = jnp.where(available_i, raw, plus_inf)
                    step_min = jnp.min(masked_raw)
                    step_at = jnp.min(jnp.where(
                        available_i & (masked_raw == step_min),
                        canonical_slab, far))
                    beats = take & (step_min < min_i)
                    min_i = jnp.where(beats, step_min, min_i)
                    min_at_i = jnp.where(beats, step_at, min_at_i)
                    min_j_i = jnp.where(
                        beats, slot.astype(jnp.int32), min_j_i)
                    d_next = jnp.maximum(jnp.where(take, raw, d_i), 0.0)
                    trR_i = trR_i.at[slot + 1].set(jnp.where(
                        valid, jnp.sum(d_next), trR_i[slot + 1]))
                    kill = valid & (global_rows == pivot_members[column])
                    available_i = available_i & ~kill
                    d_i = jnp.where(kill, 0.0, jnp.where(take, d_next, d_i))
                    return (d_i, L_i, piv_i, available_i, d_taken_i,
                            trR_i, min_i, min_at_i, min_j_i)

                (d, L, piv, available, d_taken, trR, d_min_raw,
                 d_min_at, d_min_j) = lax.fori_loop(
                    0, max_group, install,
                    (d, L, piv, available, d_taken, trR,
                     d_min_raw, d_min_at, d_min_j))
                chosen = chosen.at[candidate].set(
                    chosen[candidate] | has_candidate)
                committed = committed + block_size
                return (
                    iteration + 1, has_candidate, committed, d, L, piv,
                    available, chosen, d_taken, trR, d_min_raw,
                    d_min_at, d_min_j)

            (iteration, running, committed, d, L, piv, available, chosen,
             d_taken, trR, d_min_raw, d_min_at, d_min_j) = lax.while_loop(
                keep_going, open_group, init)
            del iteration, running, chosen
            # Match the rank-1 contract after structural exhaustion: trailing
            # trace slots hold the final residual rather than an artificial 0.
            trR = jnp.where(
                jnp.arange(storage + 1) > committed,
                jnp.sum(d), trR)
            trR = lax.psum(trR[:point_budget + 1],
                           axis_name=mesh_axis) / trG
            d = jnp.where(available, d, 0.0)
            rank = jnp.sum(d_taken[:point_budget] > floor).astype(jnp.int32)
            g_min = -lax.pmax(-d_min_raw, axis_name=mesh_axis)
            mine = d_min_raw == g_min
            g_at = -lax.pmax(
                -jnp.where(mine, d_min_at, far), axis_name=mesh_axis)
            g_j = -lax.pmax(
                -jnp.where(mine, d_min_j, far), axis_name=mesh_axis)
            return (
                piv[:point_budget], L[:, :point_budget], rank, d,
                d_taken[:point_budget], trR, (g_min, g_at, g_j))

        return shard_map(
            body_local, mesh=mesh,
            in_specs=(row_shard, row_shard_1d, row_shard_1d, row_shard_1d),
            out_specs=out_specs, check_vma=False,
        )(G, group_id, canonical_id, active_init)

    return step
