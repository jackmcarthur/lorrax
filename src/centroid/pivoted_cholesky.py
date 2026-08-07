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

from file_io import WfnLoader as WFNReader
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


@partial(jax.jit, static_argnames=('k_keep',))
def pivoted_cholesky_select(
    G: jnp.ndarray,
    k_keep: int,
    orbit_id: jnp.ndarray | None = None,
):
    """Greedy pivoted Cholesky on an Hermitian PSD ``G``. Always runs k_keep
    iterations. Returns ``(piv, L, rank, d_final, d_taken, trR_over_trG)``.

    When ``orbit_id`` is given (shape ``(M,)`` int), each pivot iteration
    marks the **whole orbit** of the picked point as inactive — i.e. one
    pivot per orbit. With a sym-invariant Gram (e.g. ρ-symmetric ISDF
    candidate Gram), all orbit members of the picked pivot have the same
    residual diagonal and the column update on any one of them is, by
    symmetry, the optimal full-orbit removal. The caller unfolds picked
    pivots through their orbits at output time to recover the full
    centroid set.
    """
    M = G.shape[0]
    real_dtype = G.real.dtype
    eps = jnp.finfo(real_dtype).eps
    minus_inf = jnp.array(-jnp.inf, dtype=real_dtype)
    if orbit_id is None:
        orbit_id = jnp.arange(M, dtype=jnp.int32)         # each point its own orbit

    diag0 = jnp.maximum(jnp.real(jnp.diag(G)), 0.0)
    trG = jnp.sum(diag0)

    init = (
        diag0,                                                       # d
        jnp.zeros((M, k_keep), dtype=G.dtype),                       # L
        -jnp.ones((k_keep,), dtype=jnp.int32),                       # piv
        jnp.ones((M,), dtype=bool),                                  # active
        jnp.zeros((k_keep,), dtype=real_dtype),                      # d_taken
        jnp.zeros((k_keep + 1,), dtype=real_dtype).at[0].set(1.0),   # trR/trG
    )
    col_ids = jnp.arange(k_keep)

    def body(j, carry):
        d, L, piv, active, d_taken, trR_over_trG = carry

        masked_d = jnp.where(active, d, minus_inf)
        p = jnp.argmax(masked_d)
        pivot_val = jnp.maximum(masked_d[p], eps)         # eps for div-safety

        # L[:, j] = (G[:, p] - Σ_{i<j} L[:, i] · conj(L[p, i])) / sqrt(d[p])
        prev_mask = (col_ids < j).astype(G.dtype)
        corr = L @ (jnp.conj(L[p, :]) * prev_mask)
        denom = jnp.sqrt(pivot_val)
        newcol = (G[:, p] - corr) / denom
        # Pivot entry exactly sqrt(d[p]) — kills rounding drift.
        newcol = newcol.at[p].set(denom.astype(G.dtype))

        L = L.at[:, j].set(newcol)
        piv = piv.at[j].set(p.astype(jnp.int32))
        d_taken = d_taken.at[j].set(pivot_val.astype(real_dtype))

        # Schur-complement update; d_new[p] ≈ 0 by the cleanup above.
        d_new = jnp.maximum(d - jnp.abs(newcol) ** 2, 0.0)
        trR_over_trG = trR_over_trG.at[j + 1].set(jnp.sum(d_new) / trG)
        # Mark p (or its whole orbit, if orbit_id was provided) inactive.
        kill_mask = orbit_id == orbit_id[p]
        d = jnp.where(kill_mask, minus_inf, d_new)
        active = active & ~kill_mask

        return d, L, piv, active, d_taken, trR_over_trG

    d, L, piv, _, d_taken, trR_over_trG = lax.fori_loop(0, k_keep, body, init)
    d_final = jnp.where(jnp.isfinite(d), d, 0.0)
    # Effective rank = #pivots above a fp-noise floor relative to ‖G‖.
    # The algorithm always runs k_keep iterations (residual decay is smooth
    # in the production use case); rank is only reported so callers can
    # detect rank-deficient synthetic inputs.
    floor = jnp.sqrt(eps) * jnp.max(jnp.real(jnp.diag(G)))
    rank = jnp.sum(d_taken > floor).astype(jnp.int32)
    # Zero out post-rank entries so callers can rely on d_taken[rank:] == 0.
    d_taken = jnp.where(jnp.arange(k_keep) < rank, d_taken, 0.0)
    return piv, L, rank, d_final, d_taken, trR_over_trG




# ═══════════════════════════════════════════════════════════════════════
# Step 4 — end-to-end wrapper
# ═══════════════════════════════════════════════════════════════════════


def prune_candidates_by_pivoted_cholesky(
    wfn: WFNReader,
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

    Returns ``(keep_idx, rank, G, d_final, d_taken, trR_over_trG)``.
    """
    M = int(cand_idx.shape[0])
    n_tot = int(wfn.nbands)
    asymmetric = (band_range_left is not None and band_range_right is not None)

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

    # Run select on the row-sharded Gram. Orbit-aware mode passes orbit_id
    # row-sharded the same way as G; the body marks the whole orbit
    # inactive after each pivot pick (orbit_id of the pivot is broadcast
    # via psum-with-mask, same idiom as the L[p, :] broadcast).
    with timing.section("prune.select"):
        select_step = make_sharded_pivoted_cholesky_select(
            mesh, M_pad, n_keep, mesh_axis=select_axis,
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
            piv, L, rank, d_final, d_taken, trR_over_trG = select_step(
                G, None, active_init)
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
            piv, L, rank, d_final, d_taken, trR_over_trG = select_step(
                G, orbit_id_jax, active_init)
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
    if n_pad:
        # HARD GUARD, not a comment: if the active mask ever failed, a pad
        # index would silently index past the candidate list (or wrap) and the
        # centroid set would be quietly wrong — the rc=0-with-garbage class.
        bad = piv_np[(piv_np >= M) | (piv_np < 0)]
        if bad.size:
            raise RuntimeError(
                f"pivoted-Cholesky picked {bad.size} PAD row(s) {bad[:8]} "
                f"(valid candidate range is [0,{M}); M_pad={M_pad}). The "
                f"active-mask that makes pad rows unpickable did not hold — "
                f"refusing to emit a centroid set built from padding.")
    if orbit_id is None:
        keep_idx = np.asarray(cand_idx)[piv_np]
    else:
        # Unfold: kept = union of orbits of picked pivots.
        orbit_id_np = np.asarray(orbit_id)
        picked_orbits = orbit_id_np[piv_np]
        in_kept = np.isin(orbit_id_np, picked_orbits)
        keep_idx = np.asarray(cand_idx)[in_kept]
        if verbose:
            print(f"[pivoted_cholesky] orbit-aware: {len(piv_np)} orbits picked "
                  f"→ {len(keep_idx)} unfolded centroids (orbit-closed)")
    d_final_np = np.asarray(_mh.process_allgather(d_final, tiled=True))[:M]
    if n_pad:
        G = G[:M, :M]        # hand back the LOGICAL Gram, not the padded one
    return keep_idx, int(rank), G, d_final_np, np.asarray(d_taken), np.asarray(trR_over_trG)


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
):
    """Sharded pivoted-Cholesky select on a row-sharded Gram. Always runs
    k_keep iterations (no tolerance early-stop). Returns the same tuple as
    ``pivoted_cholesky_select``: ``(piv, L, rank, d_final, d_taken,
    trR_over_trG)`` with shardings (replicated, row-sharded, replicated,
    row-sharded-1d, replicated, replicated)."""
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
    out_specs = (rep, row_shard, rep, row_shard_1d, rep, rep)

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
            local_diag = jnp.maximum(
                jnp.real(G_slab[jnp.arange(M_slab), col_ids_local]), 0.0,
            )
            trG = lax.psum(jnp.sum(local_diag), axis_name=mesh_axis)
            col_ids_k = jnp.arange(k_keep)

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
                jnp.zeros((k_keep + 1,), dtype=real_dtype).at[0].set(1.0),
            )

            def body(j, carry):
                d, L, piv, active, d_taken, trR_over_trG = carry

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
                pivot_val = jnp.maximum(global_pv, eps)

                # Column p of G (no collective: G is row-sharded).
                gcol_slab = G_slab[:, global_p]

                # Row p of L: broadcast from owning shard via masked psum.
                my_has_p = (global_p // M_slab == my_idx)
                local_p_rel = global_p - my_idx * M_slab
                safe_idx = jnp.clip(local_p_rel, 0, M_slab - 1)
                local_Lp = jnp.where(
                    my_has_p, L[safe_idx, :], jnp.zeros_like(L[safe_idx, :]),
                )
                L_p = lax.psum(local_Lp, mesh_axis)

                # New column.
                prev_mask = (col_ids_k < j).astype(G_slab.dtype)
                corr = L @ (jnp.conj(L_p) * prev_mask)
                denom = jnp.sqrt(pivot_val)
                newcol = (gcol_slab - corr) / denom
                # Pivot-row entry exactly sqrt(d[p]), only on the owner.
                fix_row_mask = my_has_p & (jnp.arange(M_slab) == local_p_rel)
                newcol = jnp.where(fix_row_mask, denom.astype(G_slab.dtype), newcol)

                L = L.at[:, j].set(newcol)
                piv = piv.at[j].set(global_p)
                d_taken = d_taken.at[j].set(pivot_val)

                # Schur update; mark p (or its whole orbit) inactive.
                d_new = jnp.maximum(d - jnp.abs(newcol) ** 2, 0.0)
                trR_over_trG = trR_over_trG.at[j + 1].set(
                    lax.psum(jnp.sum(d_new), axis_name=mesh_axis) / trG,
                )
                if orbit_id_slab is None:
                    kill_mask = my_has_p & (jnp.arange(M_slab) == local_p_rel)
                else:
                    # Broadcast orbit_id of the picked pivot via psum-with-mask
                    # (same idiom as the L[p, :] broadcast above), then mark
                    # all local orbit-mates inactive.
                    local_op_val = jnp.where(
                        my_has_p, orbit_id_slab[safe_idx], jnp.int32(0),
                    )
                    orbit_id_p = lax.psum(local_op_val, mesh_axis)
                    kill_mask = orbit_id_slab == orbit_id_p
                active = active & ~kill_mask
                d = jnp.where(kill_mask, minus_inf, d_new)

                return d, L, piv, active, d_taken, trR_over_trG

            d_final, L_out, piv_out, _, d_taken, trR_over_trG = lax.fori_loop(
                0, k_keep, body, init,
            )
            d_final = jnp.where(jnp.isfinite(d_final), d_final, 0.0)
            d0max_global = lax.pmax(jnp.max(local_diag), axis_name=mesh_axis)
            floor = jnp.sqrt(eps) * d0max_global
            rank = jnp.sum(d_taken > floor).astype(jnp.int32)
            d_taken = jnp.where(jnp.arange(k_keep) < rank, d_taken, 0.0)
            return piv_out, L_out, rank, d_final, d_taken, trR_over_trG

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
    wfn: WFNReader,
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
    if n_dev_total == 1:
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
        if col_block >= psi_l_rmu_Y.shape[3]:
            col_block = 0  # one full block == the original computation

    if col_block:
        M_cols = psi_l_rmu_Y.shape[3]
        if verbose:
            print(f"[pivoted_cholesky] column-blocked Gram: M={M_cols}, "
                  f"col_block={col_block} "
                  f"({-(-M_cols // col_block)} blocks; single-device path)")
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
                P_l_b = pair_density(
                    psi_l_rmuT_X, psi_l_rmu_Y[..., c0:c1], mesh_xy)
                P_r_b = pair_density(
                    psi_r_rmuT_X, psi_r_rmu_Y[..., c0:c1], mesh_xy)
                G_b = gram_q0_from_pair(P_l_b, P_r_b, kw, mesh_xy=mesh_xy,
                                        symmetrize=False)
                G_b.block_until_ready()
                g_blocks.append(G_b)
                del P_l_b, P_r_b
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
