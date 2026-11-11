"""Candidate-pool plumbing of the centroid prune: WFN faces, k batches, emission, refusals.

The prune itself is :mod:`centroid.sketch_select` (orbit-blocked QRCP on a
Khatri-Rao sketch of the q=0 pair Gram; ``docs/theory/centroid-selection.md``).
This module owns what it needs around that selection:

- :func:`load_candidate_faces`: one k batch's WFN faces at the candidate
  points through the shared sampler, in the sketch's flattened-mesh layout
  (``common.wfn_transforms.load_centroids_band_chunked(faces='cand')``);
- :func:`candidate_k_schedule`: the fixed-tile full-BZ k batches and weights;
- :func:`emit_complete_groups`: point pivots back to whole orbits;
- :func:`refuse_unless_select_certified`: the host-side refusal of an
  uncertifiable selection, shared with ``gw.downfold``'s pivoted-Cholesky
  basis selection;
- :func:`check_pair_space_window`: the band-window guard.

Shapes:

    psi_l, psi_r   (nk, nb, ns, M_pad) complex  P(None, None, None, ('y','x'))
    piv            (n,)                int32    point pivots, whole orbits
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh
from typing import TYPE_CHECKING

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

# ``WfnLoader`` and ``SymMaps`` are annotations only (SMALL_ISSUES row 38):
# importing them at module scope would make every consumer of the refusal
# depend on h5py.
if TYPE_CHECKING:                                                   # pragma: no cover
    from wfn_loader import WfnLoader
    from symmetry_maps import SymMaps
from common import timing
from common.gpu_utils import minimum_process_budget_gb
from common.log import get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# The select's INFO, read on the host
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
    """Raise unless the select delivered a set it is safe to write.

    A jitted kernel cannot raise, so it REPORTS and this REFUSES — the same
    division of labour LAPACK's ``pstrf`` makes with its ``INFO`` code, and
    the reason assessment R7 says to borrow ``distrib_la``'s refusal
    discipline rather than its dispatch.  Each condition below used to be a
    silent wrong answer that passed every downstream shape check; the
    measured before-behaviour is in the block comment above
    :func:`common.pivoted_cholesky.pivoted_cholesky_select`.  It serves both
    selectors that pivot on a residual diagonal: the centroid sketch
    (:mod:`centroid.sketch_select`, PSD by construction) and ``gw.downfold``'s
    pivoted Cholesky.

    **WHAT REFUSES AND WHAT REPORTS (2026-08-22).**  Three things are
    structurally unsafe and refuse: a non-PSD Gram (1), a pool that ran out
    of candidates (2), and a pivot outside the candidate range (3).  A pool
    that is merely numerically FLAT (2b) does not refuse: the delivered set
    is well defined, the arithmetic is safe (the ``pivot_val`` clamp removed
    the Inf/NaN blow-up that made 2026-08-07's refusal necessary), and rank
    is measured ANTI-correlated with BerkeleyGW agreement on the deck where
    it matters — the refusal blocked the most accurate configuration on
    record while passing one 20-56x worse.  It reports, loudly, with the
    downstream cost named.  ``docs/architecture/rank_truncation_policy.md`` §7
    owns this.

    It is a free function, and public, for one reason: every one of these
    refusals needs a constructible-FALSE twin, and building one through
    :func:`centroid.sketch_select.prune_candidates_by_sketch` would mean
    standing up a WFN and a symmetry table to test an arithmetic contract.  The tests
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

    # (2) THE POOL RAN OUT.  ``piv == -1`` now means exactly one thing — no
    # active candidate remained — and that is structural: there is no set of
    # the requested size in existence, so nothing can be delivered.
    unit = "orbits" if orbit_id is not None else "points"
    n_avail = (int(np.unique(np.asarray(orbit_id)).size)
               if orbit_id is not None else int(M))
    n_sel = int((piv_np >= 0).sum())
    if n_sel < int(n_keep):
        raise RuntimeError(
            f"pivoted-Cholesky REFUSES: asked for {int(n_keep)} {unit}, "
            f"delivered {n_sel}.\n"
            f"  cause : the select ran out of ACTIVE candidates — there was "
            f"nothing left to pick.  The pool holds {n_avail} {unit}; raise "
            f"--oversample for a richer one, or lower N.  (If that count "
            f"looks sufficient, a -1 sentinel survived the kernel's own "
            f"contract, which is a defect in the kernel, not in the deck.)\n"
            f"  effect: piv[{n_sel}:] is the -1 sentinel, so there is no set "
            f"of {int(n_keep)} {unit} to return.\n"
            f"Before 2026-08-07 this ran to k_keep regardless.  Past the "
            f"numerical rank the divisor clamped to sqrt(eps) and the "
            f"factor blew up geometrically to Inf and then NaN (MEASURED: "
            f"first non-finite column twelve iterations past a true rank of "
            f"10), after which argmax over NaN handed back the first "
            f"unpicked indices IN ARRAY ORDER — real candidate indices, so "
            f"the pad guard did not fire and nothing downstream noticed.  "
            f"In orbit mode the same exhaustion stayed finite and returned "
            f"index 0 repeated once per missing orbit.")

    # (2b) THE POOL IS NUMERICALLY FLAT — a different fact, and NOT an error.
    #
    # WHY THIS STOPPED BEING A REFUSAL (2026-08-22).  Rebuilding the shipped
    # Si 960-point anchor set's own documented recipe died here with "asked
    # for 960 points, certified 799", and that set scores sigTOT MAE
    # 0.644 meV — the best BerkeleyGW agreement on record for the deck —
    # while the orbit-mode arm the SAME gate passes at 960 is 20-56x worse.
    # So the refusal blocked the most accurate configuration measured and
    # waved through a much worse one, purely because orbit mode counts the
    # rank in ORBITS.  Retained rank is not basis quality, in either
    # direction (TASTE rule 12; ladder_rung1_notes R19.1).
    #
    # AND IT IS NOW SAFE TO DELIVER, which it was not on 2026-08-07.  The
    # refusal landed together with the ``pivot_val`` clamp, and it is the
    # CLAMP that removed the Inf/NaN blow-up quoted above: past the rank
    # ``take`` is false, ``newcol`` is exactly zero and ``d`` is unchanged,
    # so no divisor can run away.  The selection continues by the same
    # deterministic rule (largest frozen residual, ties to the lowest index)
    # and ``rank`` still counts only certified directions.
    #
    # docs/architecture/rank_truncation_policy.md §7 owns this ruling.
    if rank_i < int(n_keep):
        note = (
            f"pivoted-Cholesky: asked for {int(n_keep)} {unit}, DELIVERED "
            f"{n_sel}, but only {rank_i} of them add an independent "
            f"direction.\n"
            f"  what   : {n_avail} {unit} were available and the residual "
            f"Schur diagonal fell to the noise floor {floor:.3e} after "
            f"{rank_i} of them — the pool is numerically RANK-DEFICIENT, "
            f"not short.  The remaining {int(n_keep) - rank_i} were picked "
            f"by the same deterministic rule (largest frozen residual, ties "
            f"to the lowest index) and are QUADRATURE points, not certified "
            f"directions.\n"
            f"  effect : the zeta back-solve will truncate about "
            f"{int(n_keep) - rank_i} modes per q.  That is expected of an "
            f"over-complete interpolation set and is NOT by itself a defect "
            f"— on the Si anchor deck the 960-point set with ~160 dependent "
            f"points scores sigTOT MAE 0.644 meV, the best on record, while "
            f"the rank-clean orbit-mode arm at the same N is 20-56x worse.\n"
            f"  NOT the fix on this deck: widening the prune window "
            f"(a wider left leg in --fit-window) changes sigTOT by "
            f"<2x here and never recovers the orbit-mode loss.  If you want "
            f"a rank-clean set, LOWER N to {rank_i}.")
        for line in note.splitlines():
            log.info("  [select certificate] " + line)

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
# Emission and guards
# ═══════════════════════════════════════════════════════════════════════

def emit_complete_groups(cand_idx, dense_group_id, piv):
    """Map point pivots back to an exactly complete-group coordinate set.

    Returns ``(keep_idx, in_kept, order, piv_used)``: the kept candidate
    rows, their mask, the admitted groups in admission order and the
    non-sentinel pivots.  Refuses a partial group.
    """
    cand_idx = np.asarray(cand_idx)
    dense_group_id = np.asarray(dense_group_id, dtype=np.int32)
    piv_used = np.asarray(piv, dtype=np.int32)
    piv_used = piv_used[piv_used >= 0]
    picked_groups = dense_group_id[piv_used]
    order = picked_groups[np.sort(np.unique(
        picked_groups, return_index=True)[1])]
    in_kept = np.isin(dense_group_id, order)
    sizes = np.bincount(dense_group_id)
    pivot_counts = np.bincount(picked_groups, minlength=sizes.size)
    if not np.array_equal(pivot_counts[order], sizes[order]):
        raise RuntimeError(
            "the orbit-blocked select returned a partial group; "
            "the point-budget admission contract is broken")
    keep_idx = cand_idx[in_kept]
    if int(keep_idx.shape[0]) != int(piv_used.size):
        raise RuntimeError(
            "group-block pivot/emission count mismatch: "
            f"pivoted {piv_used.size} point rows but emitted "
            f"{keep_idx.shape[0]}")
    return keep_idx, in_kept, order, piv_used


def unpicked_over_floor(d_final, d0max, tol_rel) -> float:
    """Largest never-picked residual over the select's floor tol·max d0."""
    d_final = np.asarray(d_final)
    tol = (float(np.sqrt(np.finfo(d_final.dtype).eps)) if tol_rel is None
           else float(tol_rel))
    return float(np.max(d_final, initial=0.0)) / (tol * float(d0max))


def check_pair_space_window(wfn: "WfnLoader", max_band: int) -> None:
    """Refuse a band window past half the plane-wave basis.

    For centroid pruning to be meaningful, the pair-product space must be
    significantly smaller than the full plane-wave basis; once we include
    > 50 % of the available PW degrees of freedom, the candidate-vs-grid
    distinction blurs and the user should be pruning the real-space grid
    directly.
    """
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


# ═══════════════════════════════════════════════════════════════════════
# Candidate faces and k batches
# ═══════════════════════════════════════════════════════════════════════

def left_band_slice(left, right):
    """``(lo, hi)`` of a left window inside the right one, else None.

    The default feature pair (occupied + Σ conduction on the left, every
    band on the right) is nested.  Its left face is then the rows
    ``[lo, hi)`` of the right face: one load serves both windows.
    """
    l0, l1 = int(left[0]), int(left[1])
    r0, r1 = int(right[0]), int(right[1])
    if l0 < r0 or l1 > r1:
        return None
    return l0 - r0, l1 - r0


def candidate_band_carrier(width: int, mesh_xy: Mesh, band_chunk_size: int) -> int:
    """Band rows the shared sampler's accumulator holds for one window.

    The accumulator rounds a window of ``width`` bands up to whole transfer
    tiles of ``min(width, band_chunk_size)`` bands padded to the mesh size.
    """
    from runtime.padding import padded_axis
    tile = padded_axis(min(width, max(1, int(band_chunk_size))), int(mesh_xy.size),
                       name="candidate WFN band tile").carrier
    return padded_axis(width, tile, name="candidate WFN band accumulator").carrier


def candidate_k_schedule(wfn, sym, k_weights, per_k: float):
    """The fixed-tile full-BZ k batches of the candidate sketch.

    Parameters
    ----------
    wfn, sym
        The open loader and its ``SymMaps``.
    k_weights : (nk_tot,) float or None
        Normalized nonnegative full-BZ weights; ``None`` is uniform.
    per_k : float
        Per-rank bytes of one k row of the batch's WFN faces.

    Returns
    -------
    ``(rows, weights, nk, nk_full, k_batch, n_chunks)``: the positive-weight
    full-BZ rows in batch order and their weights, their count, the full-BZ
    count, the rows per batch (the most whose faces fit
    :data:`runtime.tiles.TILE_BYTES`) and the batch count.  Several batches
    hold whole symmetry stars, so the loader reads each raw parent once per
    band tile; one batch keeps the canonical full-BZ order.
    """
    from runtime.tiles import tile_units
    nk_full = int(sym.nk_tot)
    weights = (np.full(nk_full, 1.0/nk_full) if k_weights is None
               else np.asarray(k_weights, dtype=np.float64))
    if (weights.shape != (nk_full,) or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-14)):
        raise ValueError("the candidate sketch needs normalized nonnegative full-BZ k weights")
    selected_rows = np.flatnonzero(weights > 0)
    weights = weights[selected_rows]
    nk = int(selected_rows.size)
    k_limit = tile_units(per_k, nk)
    n_chunks = -(-nk // k_limit)
    k_batch = -(-nk // n_chunks)
    if n_chunks > 1:
        # Whole stars per batch: the loader reads a raw parent once per band
        # tile of the batch that holds its children.
        w_full = np.zeros(nk_full, dtype=np.float64)
        w_full[selected_rows] = weights
        selected_rows = np.concatenate([
            children for _, children in wfn.full_k_parent_groups(selected_rows)
        ]).astype(np.int64)
        weights = w_full[selected_rows]
    return selected_rows, weights, nk, nk_full, k_batch, n_chunks


def load_candidate_faces(
    wfn: "WfnLoader",
    sym: "SymMaps",
    cand_idx,
    mesh_xy: Mesh,
    *,
    left_range: tuple[int, int],
    right_range: tuple[int, int],
    bispinor: bool,
    band_norms,
    band_chunk_size: int,
    memory_per_device_gb: float | None,
    full_k_rows,
    verbose: bool,
):
    """One k batch's candidate WFN faces through the shared sampler.

    ``common.wfn_transforms.load_centroids_band_chunked(faces='cand')`` on a
    ``Meta`` for the windows: every rank holds its own candidate columns
    with every band.  The pseudoband clamp ``ψ /= max(norm, 1)`` is applied
    when ``band_norms`` is given.  A left window inside the right one is not
    loaded again: it is the right face's band rows ``left_slice``.

    Returns
    -------
    ``(meta, left_slice, psi_l, psi_r)``: ``psi_* (nk, nb, ns, M_pad)`` at
    ``P(None, None, None, ('y','x'))``; ``psi_l`` is ``None`` when
    ``left_slice`` is not.
    """
    from common.meta import Meta
    from common.wfn_transforms import load_centroids_band_chunked
    max_band = max(left_range[1], right_range[1])
    # Meta.nval is physical occupancy, not the width of the left feature
    # window.  Explicit windows may cross the occupied boundary.
    occupied = min(int(wfn.nelec), int(max_band))
    M = int(cand_idx.shape[0])
    cand_idx = jnp.asarray(cand_idx, dtype=jnp.int64)
    meta = Meta.from_system(
        wfn, sym,
        nval=occupied, ncond=int(max_band) - occupied,
        nband=max_band,
        n_rmu=M,
        bispinor=bispinor,
    )
    # The loader sizes its FFT scan from ``meta.memory_per_device_gb``; with
    # no pinned budget it takes the run's device budget.
    if memory_per_device_gb is None or memory_per_device_gb <= 0:
        from common.gpu_utils import device_budget_bytes
        memory_per_device_gb = device_budget_bytes() / 1e9
    meta.memory_per_device_gb = minimum_process_budget_gb(memory_per_device_gb)
    # One batch streams one raw parent at a time (k tile 1).  A k batch of
    # whole stars is one k tile: the loader's union read then takes each raw
    # parent once per band tile.
    k_tile = 1 if full_k_rows is None else len(full_k_rows)

    def norms(window):
        if band_norms is None:
            return None
        values = np.asarray(band_norms, dtype=np.float64)
        if values.shape[0] < max_band:
            raise ValueError(f"band_norms has {values.shape[0]} entries but "
                             f"the left/right windows touch band {max_band}")
        return jnp.asarray(np.maximum(values[window[0]:window[1]], 1.0))

    if verbose:
        log.debug(f"[candidates] faces via load_wfns: nk_tot={sym.nk_tot}, "
                  f"left={left_range}, right={right_range}, M={M}, "
                  f"norms={'on' if band_norms is not None else 'off'}, "
                  f"budget={meta.memory_per_device_gb:g} GB/device, "
                  f"band chunk={band_chunk_size}, transfer_k_tile={k_tile}")

    # A bispinor candidate set samples the current channel: its carrier is
    # the current lift (common.four_current_model).
    from common.four_current_model import resolve_four_current_representation
    lift = resolve_four_current_representation(bispinor, None).current_lift

    def load(window):
        face = load_centroids_band_chunked(
            wfn, sym, meta, cand_idx, bispinor, mesh_xy, window,
            band_chunk_size=band_chunk_size, k_chunk_size=k_tile,
            full_k_rows=full_k_rows, faces="cand", bispinor_lift=lift)
        scale = norms(window)
        if scale is not None:
            face = face / scale[None, :, None, None]
        return face.block_until_ready()

    left_slice = left_band_slice(left_range, right_range)
    with timing.section("left.load"):
        first = load(right_range if left_slice is not None else left_range)
    if left_slice is not None:
        return meta, left_slice, None, first
    with timing.section("right.load"):
        psi_r = load(right_range)
    return meta, None, first, psi_r


__all__ = [
    "candidate_band_carrier",
    "candidate_k_schedule",
    "check_pair_space_window",
    "emit_complete_groups",
    "left_band_slice",
    "load_candidate_faces",
    "refuse_unless_select_certified",
    "unpicked_over_floor",
]
