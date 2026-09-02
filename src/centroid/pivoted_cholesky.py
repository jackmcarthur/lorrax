"""Pivoted-Cholesky pruning of over-sampled ISDF candidate points.

Implements the q=0 candidate-pruning stage described in
``pivoted_cholesky.md`` (sandbox root). The idea: k-means gives a set of M
candidate points ``{r̃_a}`` (M > N_μ); the pair-product rows
``z_{a,(vck)} = φ*_{v,k}(r̃_a) ψ_{c,k}(r̃_a)`` define a Hermitian PSD Gram
matrix ``G^{(0)} ∈ ℂ^{M×M}``.  For the transverse bispinor channel the
features are stacked over all three current components and the Gram is
``G_perp = Σ_i Z_i Z_i†`` with equal component weights. Greedy pivoted
Cholesky picks the N_μ pivots
with the largest residual Schur-complement diagonal, and the corresponding
``r̃_a`` become the final ISDF points. This is strictly better than picking
on amplitude alone because it targets the coherence structure of the
valence-conduction pair-product space the ISDF fit will actually use.

Architectural map to ``gw/isdf_fitting.py``:

    pair_density                      ←→  per-k open-spin P^{(v/c)}_{αβ}(a,b)
                                          (rank-5; same einsum at candidates
                                          r̃_a not chosen r_μ)
    gram_q0_from_pair                 ←→  q=0 cross-product (no k→q FFT)
    common.pivoted_cholesky           ←→  numerical row selector + certificates

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
    piv            (k_keep,)         int32    pivots (−1 only on pool exhaustion)
    d_final        (M,)              real     Schur-complement residuals

``nv_eff`` / ``nc_eff`` fold the spinor axis into the band axis
(nv_eff = nv_bands · nspinor), matching the md's "assume spin has already
been folded" convention.
"""

from __future__ import annotations

import gc
import math
import os
from functools import lru_cache, partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import multihost_utils as _mh
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
from common.gpu_utils import worst_process_resident_bytes
from common.pivoted_cholesky import (
    make_sharded_group_block_pivoted_cholesky_select as _make_sharded_block_select,
    make_sharded_pivoted_cholesky_select as _make_sharded_select,
)
from runtime.padding import round_up

from . import distribution as dist

import symmetry_maps                                            # noqa: E402


_GRAM_MIN_COL_BLOCK = 256
_GRAM_COMPLEX_BYTES = 16
_GRAM_SEED_BUDGET_FRACTION = 0.25
_GRAM_FINAL_FOLD_SLOTS = 3
_CANDIDATE_GAMMA_MODES = ("charge", "transverse")


@lru_cache(maxsize=None)
def _candidate_gram_hermitian_fold_kernel(mesh_xy: Mesh):
    """Return the donated P(x,y)->P(x,y) terminal Gram fold.

    The transpose exchanges square-mesh owners, but no process may receive a
    complete Gram.  Keeping both input and output shardings explicit prevents
    eager ``G + G.T.conj()`` from resolving the public result to ``P()``.
    """
    xy = NamedSharding(mesh_xy, PartitionSpec("x", "y"))

    @partial(
        jax.jit,
        in_shardings=xy,
        out_shardings=xy,
        donate_argnums=(0,),
    )
    def _fold(G):
        return 0.5 * (G + jnp.conj(G.T))

    return _fold


@lru_cache(maxsize=None)
def _candidate_gram_zero_kernel(mesh_xy: Mesh, n_points: int):
    """Return the stable donated-Gram destination initializer."""
    xy = NamedSharding(mesh_xy, PartitionSpec("x", "y"))
    n_points = int(n_points)

    @jax.jit(out_shardings=xy)
    def _zero():
        return jnp.zeros((n_points, n_points), dtype=jnp.complex128)

    return _zero


def _resolve_candidate_gamma_mode(gamma_mode: str, *, bispinor: bool) -> str:
    """Validate the candidate feature family before loading wavefunctions."""
    mode = str(gamma_mode).strip().lower()
    if mode not in _CANDIDATE_GAMMA_MODES:
        raise ValueError(
            "gamma_mode must be 'charge' or 'transverse'; got "
            f"{gamma_mode!r}")
    if mode == "transverse" and not bispinor:
        raise ValueError(
            "gamma_mode='transverse' requires bispinor=True so the canonical "
            "four-component gamma algebra can be applied")
    return mode


def candidate_gram_q0_from_pair(
    P_l_k: jax.Array,
    P_r_k: jax.Array,
    k_weights: jax.Array,
    *,
    mesh_xy: Mesh,
    gamma_mode: str = "charge",
    symmetrize: bool = True,
) -> jax.Array:
    """Fold canonical pair densities into the selected candidate metric.

    ``charge`` delegates byte-for-byte to the historical scalar q=0 fold.
    ``transverse`` delegates to :mod:`isdf.core`'s fused three-component
    transition-feature fold, which computes the PSD sum
    ``G_perp = sum_i Z_i Z_i†`` without materialising ``Z_i`` or summing raw
    indefinite transverse CCTs.
    """
    mode = str(gamma_mode).strip().lower()
    if mode == "charge":
        from isdf import gram_q0_from_pair
        return gram_q0_from_pair(
            P_l_k, P_r_k, k_weights,
            mesh_xy=mesh_xy, symmetrize=symmetrize)
    if mode == "transverse":
        from isdf import transverse_gram_q0_from_pair
        return transverse_gram_q0_from_pair(
            P_l_k, P_r_k, k_weights,
            mesh_xy=mesh_xy, symmetrize=symmetrize)
    raise ValueError(
        "gamma_mode must be 'charge' or 'transverse'; got "
        f"{gamma_mode!r}")


def candidate_gram_q0_from_psi(
    psi_l_X: jax.Array,
    psi_l_Y: jax.Array,
    psi_r_X: jax.Array,
    psi_r_Y: jax.Array,
    k_weights: jax.Array,
    *,
    mesh_xy: Mesh,
    gamma_mode: str = "charge",
    symmetrize: bool = True,
) -> jax.Array:
    """Build one candidate Gram tile through the fused ISDF owner."""
    from isdf import gram_q0_from_psi_sm
    return gram_q0_from_psi_sm(
        psi_l_X, psi_l_Y, psi_r_X, psi_r_Y, k_weights,
        mesh_xy=mesh_xy, gamma_mode=gamma_mode, symmetrize=symmetrize,
    )


def gram_col_block_bytes(nk: int, nspinor: int, block_width: int) -> int:
    """Transient bytes priced for one open-spin Gram column block.

    Both left and right pair-density intermediates are complex128 and have
    ``nk * nspinor**2 * block_width**2`` elements.  Keep this formula here as
    the single source used by the auto planner, its refusal, and unit gates.
    """
    nk_i = int(nk)
    ns_i = int(nspinor)
    block_i = int(block_width)
    if nk_i < 1 or ns_i < 1 or block_i < 1:
        raise ValueError(
            "Gram block dimensions must be positive: "
            f"nk={nk_i}, nspinor={ns_i}, block_width={block_i}"
        )
    return (2 * nk_i * ns_i * ns_i * block_i * block_i
            * _GRAM_COMPLEX_BYTES)


def auto_gram_col_block_width(
    nk: int,
    nspinor: int,
    budget_bytes: int,
    *,
    divisor: int = 1,
    min_width: int = _GRAM_MIN_COL_BLOCK,
) -> int:
    """Largest mesh-aligned Gram seed tile whose square-law price fits.

    Auto widths align *down* so rounding for a mesh can never invalidate the
    memory bound.  Refuse before the pair-density allocation when even the
    supported minimum block cannot fit.
    """
    budget_i = int(budget_bytes)
    divisor_i = max(1, int(divisor))
    min_aligned = ((max(1, int(min_width)) + divisor_i - 1)
                   // divisor_i) * divisor_i
    if budget_i < 1:
        raise MemoryError(
            f"Gram tile planner has no positive seed budget: {budget_i} B"
        )
    coefficient = gram_col_block_bytes(nk, nspinor, 1)
    max_unaligned = math.isqrt(budget_i // coefficient)
    width = (max_unaligned // divisor_i) * divisor_i
    if width < min_aligned:
        required = gram_col_block_bytes(nk, nspinor, min_aligned)
        raise MemoryError(
            "Gram tile seed planner refuses before pair-density "
            f"allocation: nk={int(nk)}, nspinor={int(nspinor)}, minimum "
            f"mesh-aligned block={min_aligned} prices {required / 2**30:.2f} "
            f"GiB but the Gram transient budget is {budget_i / 2**30:.2f} "
            "GiB. Lower the candidate count/band window or raise the "
            "device-memory budget."
        )
    return width


def gram_col_block_device_bytes(
    nk: int,
    nspinor: int,
    n_rows: int,
    block_width: int,
    *,
    x_shards: int = 1,
    y_shards: int = 1,
) -> int:
    """Exact local bytes of the two sharded square pair-density tiles.

    ``n_rows`` remains in the signature for callers of the b6 pricing API;
    the live row extent is now bounded by the tile width rather than silently
    remaining the full candidate extent.  That makes the physical allocation
    agree with :func:`gram_col_block_bytes`' square law.
    """
    x_i = max(1, int(x_shards))
    y_i = max(1, int(y_shards))
    rows_local = (min(int(n_rows), int(block_width)) + x_i - 1) // x_i
    cols_local = (int(block_width) + y_i - 1) // y_i
    return gram_col_block_bytes(nk, nspinor, 1) * rows_local * cols_local


def gram_scan_live_set_bytes(
    *,
    resident_bytes: int,
    scan_resident_increment_bytes: int,
    gram_matrix_local_bytes: int,
) -> dict[str, int]:
    """Complete per-device live set for the one-dispatch tiled Gram scan.

    ``resident_bytes`` is sampled after both complete candidate-WFN windows
    exist.  The exact production executable reports only its bytes above
    those four inputs and the donated local ``P('x','y')`` Gram.  The final
    Hermitian fold can hold input, transpose/conjugate and output: three
    local-Gram slots, with no global concatenate.
    """
    resident = int(resident_bytes)
    gram_local = int(gram_matrix_local_bytes)
    stages = {
        "scan": (resident + gram_local
                 + int(scan_resident_increment_bytes)),
        "final_fold": resident + _GRAM_FINAL_FOLD_SLOTS * gram_local,
    }
    stages["peak"] = max(stages.values())
    return stages


def gram_tile_schedule(extent: int, width: int) -> tuple[int, int, float]:
    """Return tile count, padded extent and square-work inflation."""
    extent_i = int(extent)
    width_i = int(width)
    if extent_i < 1 or width_i < 1:
        raise ValueError(
            f"Gram extent and tile width must be positive; got "
            f"extent={extent_i}, width={width_i}")
    ntiles = -(-extent_i // width_i)
    executed = ntiles * width_i
    inflation = float(executed * executed) / float(extent_i * extent_i)
    return ntiles, executed, inflation


def _auto_gram_width_from_compiled_peaks(
    seed_width: int,
    *,
    max_width: int,
    divisor: int,
    budget_bytes: int,
    peak_for_width,
) -> tuple[int, dict[str, int]]:
    """Find a certified rung, then remove padding at the same tile count.

    Each rung compiles the SAME canonical tile executables production will
    run.  A geometric ladder avoids a dozen throw-away production-shape
    compilations. Once the largest feasible rung is known, its tile count is
    fixed and the width is reduced to the smallest mesh-aligned value that
    covers the logical extent in that many tiles. This preserves scan-iteration
    count while minimizing zero-padded pair-density work. The returned width
    itself is always queried and certified.
    """
    d = max(1, int(divisor))
    floor = ((max(1, _GRAM_MIN_COL_BLOCK) + d - 1) // d) * d
    ceiling = (int(max_width) // d) * d
    width = min(max(floor, (int(seed_width) // d) * d), ceiling)
    checked: dict[int, dict[str, int]] = {}

    def check(w):
        if w not in checked:
            checked[w] = peak_for_width(w)
        return checked[w]

    facts = check(width)
    while facts["peak"] > int(budget_bytes) and width > floor:
        width = max(floor, ((width // 2) // d) * d)
        facts = check(width)
    if facts["peak"] > int(budget_bytes):
        raise MemoryError(
            "Gram tile planner refuses before pair-density allocation: "
            f"the minimum mesh-aligned tile={floor} has a compiled full "
            f"live set of {facts['peak'] / 2**30:.2f} GiB/device, above "
            f"the {int(budget_bytes) / 2**30:.2f}-GiB/device target."
        )

    while width < ceiling:
        wider = min(ceiling, ((2 * width) // d) * d)
        if wider <= width:
            break
        wider_facts = check(wider)
        if wider_facts["peak"] > int(budget_bytes):
            break
        width, facts = wider, wider_facts

    # Largest-width is not a runtime optimum with fixed-shape tail padding.
    # Example: extent 3008, width 3004 executes two 3004-wide tiles per axis,
    # nearly 4x the useful square work.  Keep the SAME number of scan iterations
    # and shrink to the minimum aligned width that still covers the extent.
    ntiles, _, _ = gram_tile_schedule(ceiling, width)
    compact = round_up(-(-ceiling // ntiles), d)
    compact = min(width, max(floor, compact))
    if compact != width:
        compact_facts = check(compact)
        if compact_facts["peak"] <= int(budget_bytes):
            width, facts = compact, compact_facts
    return width, facts


# ═══════════════════════════════════════════════════════════════════════
# The pure reference and row-sharded selection recurrences are owned by
# ``common.pivoted_cholesky``.  This L1 module retains Gram construction,
# centroid policy/certification, and point-set reporting.

# ═══════════════════════════════════════════════════════════════════════
# The kernel's INFO, read on the host
# ═══════════════════════════════════════════════════════════════════════

#: What the select does when the pool is numerically flat BUT still has
#: candidates to hand back.  ``deliver`` (default) hands back the requested
#: set with a loud note naming the certified rank and its downstream cost;
#: ``strict`` restores the 2026-08-07 refusal verbatim.  See
#: :func:`refuse_unless_select_certified` guard (2b) and
#: ``docs/dev/rank_truncation_policy.md`` §7 for why the default moved.
#:
#: Deliberately NOT ``LORRAX_RANK_POLICY``: that dial governs a truncation's
#: CONDITIONING against a certified kappa ceiling, and this one governs
#: whether a rank-deficient candidate POOL is an error.  Measured, they point
#: opposite ways on the same deck, and one name for both would make either
#: setting wrong for the other.
SELECT_MODES = ("deliver", "strict")
SELECT_MODE_DEFAULT: str = "deliver"
SELECT_MODE_ENV = "LORRAX_CENTROID_SELECT"


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
    :func:`pivoted_cholesky_select`.

    **WHAT REFUSES AND WHAT REPORTS (2026-08-22).**  Three things are
    structurally unsafe and refuse: a non-PSD Gram (1), a pool that ran out
    of candidates (2), and a pivot outside the candidate range (3).  A pool
    that is merely numerically FLAT (2b) does not refuse: the delivered set
    is well defined, the arithmetic is safe (the ``pivot_val`` clamp removed
    the Inf/NaN blow-up that made 2026-08-07's refusal necessary), and rank
    is measured ANTI-correlated with BerkeleyGW agreement on the deck where
    it matters — the refusal blocked the most accurate configuration on
    record while passing one 20-56x worse.  It reports, loudly, with the
    downstream cost named, and ``LORRAX_CENTROID_SELECT=strict`` restores the
    old refusal.  ``docs/dev/rank_truncation_policy.md`` §7 owns this.

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
    # and ``rank`` still counts only certified directions.  The refusal is
    # kept verbatim behind LORRAX_CENTROID_SELECT=strict.
    #
    # docs/dev/rank_truncation_policy.md §7 owns this ruling.
    if rank_i < int(n_keep):
        mode = (os.environ.get(SELECT_MODE_ENV) or SELECT_MODE_DEFAULT
                ).strip().lower()
        if mode not in SELECT_MODES:
            raise ValueError(
                f"{SELECT_MODE_ENV}={mode!r} is not one of {SELECT_MODES}.  "
                f"A mis-spelled mode is not silently 'deliver'.")
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
            f"(--prune-n-cond / --prune-window vc_x_vc) changes sigTOT by "
            f"<2x here and never recovers the orbit-mode loss.  If you want "
            f"a rank-clean set, LOWER N to {rank_i}.\n"
            f"  strict : {SELECT_MODE_ENV}=strict restores the 2026-08-07 "
            f"refusal.")
        if mode == "strict":
            raise RuntimeError("pivoted-Cholesky REFUSES ("
                               + SELECT_MODE_ENV + "=strict):\n" + note)
        for line in note.splitlines():
            print("  [pivoted_cholesky] " + line)

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
    """Independent directions in a delivered point set.

    The production whole-orbit block selector computes this rank directly as
    it pivots every delivered point. This O(n³) host diagnostic remains for
    the explicit representative-group compatibility path, whose select rank
    counts representatives rather than emitted points.

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


def _report_point_granularity(G, keep_mask, rank_selected, *, unit,
                              tol_rel=None, verbose=True):
    """Say the certification at the granularity of the FILE being written.

    ``rank_selected`` is what the compatibility selector certified. ``unit``
    makes its granularity explicit; the production block path does not need
    this second eigensolve because it already pivots every emitted point.

    This reports; it never refuses.  A rank-deficient delivered set is a
    fact about an over-complete interpolation basis and is measured
    ANTI-correlated with BerkeleyGW agreement on the Si anchor deck — see
    ``refuse_unless_select_certified`` guard (2b).
    """
    pt_rank, n_pts, why = point_granularity_rank(G, keep_mask, tol_rel=tol_rel)
    if not verbose:
        return
    if pt_rank is None:
        print(f"  [point rank] {rank_selected} {unit} certified, {n_pts} "
              f"points delivered, independent directions NOT MEASURED — "
              f"{why}.  That is an absence, not a pass.")
        return
    print(f"  [point rank] {rank_selected} {unit} certified, {n_pts} points "
          f"delivered, {pt_rank} independent directions "
          f"({100.0 * pt_rank / max(1, n_pts):.1f}% of the points)")
    if pt_rank < n_pts:
        print(f"  [point rank] NOTE: {n_pts - pt_rank} of the "
              f"{n_pts} delivered points add no independent "
              f"direction at tol*max(diag G).  The zeta "
              f"back-solve will truncate about that many modes "
              f"per q; D3 shipped a 7 GiB restart file to learn "
              f"the same thing downstream.  This is NOT by itself a defect: "
              f"on the Si anchor deck the 960-point set with ~160 dependent "
              f"points is the most accurate one measured.")
    _note = point_rank_closure_note(G, keep_mask, pt_rank, tol_rel=tol_rel)
    print("  [point rank] closure: " + (
        _note if _note else
        "the rank cut falls in a gap — no degenerate block is "
        "sliced at this tolerance."))


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


def _emit_complete_groups(cand_idx, dense_group_id, piv):
    """Map point pivots back to an exactly complete-group coordinate set."""
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
            "group-block pivoted Cholesky returned a partial group; "
            "the point-budget admission contract is broken")
    keep_idx = cand_idx[in_kept]
    if int(keep_idx.shape[0]) != int(piv_used.size):
        raise RuntimeError(
            "group-block pivot/emission count mismatch: "
            f"pivoted {piv_used.size} point rows but emitted "
            f"{keep_idx.shape[0]}")
    return keep_idx, in_kept, order, piv_used


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
    gamma_mode: str = "charge",
    orbit_id: np.ndarray | None = None,
    tol_rel: float | None = None,
    n_point_budget: int | None = None,
    group_block: bool = False,
):
    """End-to-end pruning: gather wfns → Gram → pivoted Cholesky → keep.

    Requires a 2-D mesh ``('x', 'y')`` (single-device callers pass a 1×1
    mesh — same shape gw_jax uses). Wavefunction loading goes through
    ``load_centroids_band_chunked`` through ``WfnLoader``'s single automatic
    G-space path; the prune driver does not select an HDF5 implementation.

    When ``orbit_id`` is provided (one int per candidate, equal for sym-
    equivalent candidates), the returned set is a union of complete orbits.
    With ``group_block=False`` the compatibility path picks one representative
    per orbit and ``n_keep`` counts orbits.  With ``group_block=True``, every
    emitted orbit member participates in the Schur recurrence and
    ``n_point_budget`` is the selection budget; this is the physically correct
    centroid-pruning path.

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

    ``group_block=True`` requires both ``orbit_id`` and ``n_point_budget``.
    It admits an orbit only if the complete orbit fits, so the kernel itself
    returns the largest greedy whole-orbit set no larger than the point budget;
    there is no partial last block for the host to repair.

    ``tol_rel`` overrides the select's stopping tolerance (relative to the
    largest initial Gram diagonal).  ``None`` reads
    ``LORRAX_CENTROID_PC_TOL`` from the environment and falls back to
    ``sqrt(eps)`` — the number this kernel has always computed for its rank
    report, kept as the default so no existing deck's reported rank moves.
    LAPACK ``?pstrf``'s own policy is ``n·eps``; pass it explicitly to get
    it.

    REFUSES on a non-PSD Gram, structural pool exhaustion, or a pivot outside
    the candidate range.  Numerical rank deficiency is reported by default;
    ``LORRAX_CENTROID_SELECT=strict`` promotes it to a refusal.  The policy is
    owned once by :func:`refuse_unless_select_certified`.
    """
    if group_block and orbit_id is None:
        raise ValueError("group_block=True requires orbit_id")
    if group_block and n_point_budget is None:
        raise ValueError("group_block=True requires n_point_budget")
    gamma_mode = _resolve_candidate_gamma_mode(
        gamma_mode, bispinor=bispinor)
    if gamma_mode == "transverse" and band_norms is not None:
        raise ValueError(
            "transverse candidate pruning uses unit weight for every band "
            "in the requested left/right fit windows; band_norms must be None")

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
        print(f"[pivoted_cholesky] M={M}, n_keep={n_keep}, {window_tag}, "
              f"gamma_mode={gamma_mode} "
              f"(load_wfns 2-D, mesh axes {mesh.axis_names})")

    with timing.section("prune.gram"):
        G = build_gram_q0_via_loadwfns(
            wfn, sym, jnp.asarray(cand_idx),
            n_val=n_val, n_cond=n_cond,
            mesh_xy=mesh, bispinor=bispinor, verbose=verbose,
            band_range_left=band_range_left,
            band_range_right=band_range_right,
            band_norms=band_norms,
            k_weights=k_weights,
            gamma_mode=gamma_mode,
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

    if group_block:
        orbit_id_np = np.asarray(orbit_id, dtype=np.int32)
        labels, dense_id = np.unique(orbit_id_np, return_inverse=True)
        n_groups = int(labels.size)
        budget = min(int(n_point_budget), M)
        dense_pad = dense_id.astype(np.int32)
        active_init = None
        if n_pad:
            dense_pad = np.concatenate([
                dense_pad, np.full((n_pad,), n_groups, dtype=np.int32)])
            active_host = np.ones((M_pad,), dtype=bool)
            active_host[M:] = False
            active_init = device_put_process_local(
                active_host,
                NamedSharding(mesh, PartitionSpec(select_axis)),
            )
        group_id_jax = device_put_process_local(
            dense_pad, NamedSharding(mesh, PartitionSpec(select_axis)))
        with timing.section("prune.select"):
            select_step = _make_sharded_block_select(
                mesh, M_pad, budget, n_groups,
                mesh_axis=select_axis, tol_rel=tol_rel)
            (piv, L, rank, d_final, d_taken, trR_over_trG,
             psd_info) = select_step(G, group_id_jax, active_init)
            piv.block_until_ready()
        del L

        piv_np = np.asarray(piv)
        piv_used = piv_np[piv_np >= 0]
        if piv_used.size == 0:
            sizes = np.bincount(dense_id, minlength=n_groups)
            raise RuntimeError(
                f"pivoted-Cholesky REFUSES: point budget {budget} is smaller "
                f"than every complete group (minimum size {int(sizes.min())})")
        keep_idx, in_kept, order, piv_used = _emit_complete_groups(
            cand_idx, dense_id, piv_used)
        n_delivered = int(keep_idx.shape[0])
        rank_i = int(rank)
        diag_host = np.real(np.asarray(jnp.diag(G)))[:M]
        psd_host = (
            float(np.asarray(psd_info[0])), int(np.asarray(psd_info[1])),
            int(np.asarray(psd_info[2])))
        refuse_unless_select_certified(
            piv_used, rank_i, psd_host, n_keep=n_delivered, M=M,
            M_pad=M_pad, orbit_id=None, d0max=float(diag_host.max()),
            tol_rel=tol_rel)
        d_final_np = np.asarray(
            _mh.process_allgather(d_final, tiled=True))[:M]
        if n_pad:
            G = G[:M, :M]
        if verbose:
            last = max(n_delivered - 1, 0)
            print(
                f"[pivoted_cholesky] GROUP-BLOCK: {len(order)} groups -> "
                f"{n_delivered}/{budget} points, rank={rank_i}; every "
                f"delivered point updated the residual")
            print(f"[pivoted_cholesky] picked-pivot residuals: "
                  f"first={float(d_taken[0]):.3e}, "
                  f"last-delivered={float(d_taken[last]):.3e}")
            print(f"[pivoted_cholesky] tr(R_k)/tr(G): "
                  f"first={float(trR_over_trG[1]):.3e}, "
                  f"last-delivered={float(trR_over_trG[n_delivered]):.3e}")
        return (keep_idx, rank_i, G, d_final_np, np.asarray(d_taken),
                np.asarray(trR_over_trG), psd_host)

    # Compatibility selector: orbit-aware mode chooses one representative and
    # then emits its whole orbit. Production centroid pruning returns above
    # through the group-block selector, which pivots every emitted point.
    with timing.section("prune.select"):
        select_step = _make_sharded_select(
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
        # R2 — certify at POINT granularity in POINT MODE TOO.
        #
        # It used to run only in orbit mode, on the reasoning that "in point
        # mode ``rank`` is already the point count and there is nothing to
        # reconcile".  That reasoning held while a rank-deficient point-mode
        # select REFUSED — it could not deliver a set whose rank differed
        # from its size.  Since guard (2b) delivers, the two numbers can now
        # differ here as well, and the certification statement has to be made
        # at the granularity of the FILE being written in BOTH modes.  Making
        # it in only one is what let "18/18 directions certified — PASS" be
        # said over a delivered set of 768 points.
        _in_kept = np.zeros(int(M), dtype=bool)
        _in_kept[piv_np[piv_np >= 0]] = True
        _report_point_granularity(G, _in_kept, rank_i, unit="points",
                                  tol_rel=tol_rel, verbose=verbose)
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
        # FILE being written.  In ORBIT mode the select's ``rank`` counts
        # orbits, so it is not comparable to the point count at all.
        _report_point_granularity(G, in_kept, rank_i, unit="orbits",
                                  tol_rel=tol_rel, verbose=verbose)
    d_final_np = np.asarray(_mh.process_allgather(d_final, tiled=True))[:M]
    if n_pad:
        G = G[:M, :M]        # hand back the LOGICAL Gram, not the padded one
    return (keep_idx, rank_i, G, d_final_np, np.asarray(d_taken),
            np.asarray(trR_over_trG), psd_host)


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
# Uses full-BZ unfold.  The centroid driver expands each normalized IBZ
# parent weight uniformly over its star and gives both its cheap candidate
# metric and this exact Gram the same full-k quadrature table.


def _gram_meta_band_counts(wfn_nelec: int, max_band: int,
                           n_val: int | None, n_cond: int | None):
    """Keep physical occupancy separate from an explicit feature window."""
    if n_val is not None:
        return int(n_val), int(n_cond)
    occupied = min(int(wfn_nelec), int(max_band))
    return occupied, int(max_band) - occupied


def _band_window_slice(outer: tuple[int, int],
                       inner: tuple[int, int]) -> slice | None:
    """Return the local band slice when ``inner`` is contained in ``outer``."""
    if outer[0] <= inner[0] and inner[1] <= outer[1]:
        return slice(inner[0] - outer[0], inner[1] - outer[0])
    return None


def _slice_centroid_wfn_faces(psi_rmu_Y, psi_rmuT_X, band_slice: slice):
    """Slice the replicated band axis in the canonical Y/X WFN faces."""
    return (
        psi_rmu_Y[:, band_slice, :, :],
        psi_rmuT_X[:, :, band_slice, :],
    )


def build_gram_q0_via_loadwfns(
    wfn: "WfnLoader",
    sym: symmetry_maps.SymMaps,
    cand_idx: jnp.ndarray,
    n_val: int | None = None,
    n_cond: int | None = None,
    mesh_xy: Mesh | None = None,
    *,
    bispinor: bool = False,
    gamma_mode: str = "charge",
    verbose: bool = True,
    band_range_left: tuple[int, int] | None = None,
    band_range_right: tuple[int, int] | None = None,
    band_norms: np.ndarray | None = None,
    k_weights: np.ndarray | None = None,
    band_chunk_size: int = 64,
    memory_per_device_gb: float | None = None,
) -> jnp.ndarray:
    """Build the q=0 candidate Gram on a 2-D mesh using gw_jax's data path.

    Two band-window call modes:

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

    ``gamma_mode='charge'`` preserves the historical scalar pair-product
    Gram exactly. ``gamma_mode='transverse'`` requires ``bispinor=True`` and
    uses the equal-weight PSD sum of the three current transition-feature
    Grams. It never consumes occupations or ``band_norms``.

    The shared WFN transform service unfolds each band window onto candidate
    points. Small Grams retain the sequential low-residency pair route; large
    Grams keep both final WFN faces and fuse each bounded pair-density/q=0
    tile through :mod:`isdf`. k-weights are supplied in full-BZ order.
    ``None`` retains uniform ``1 / nk_tot`` for standalone callers.

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
        gamma_mode: ``'charge'`` or ``'transverse'``. The latter requires
            the four-component bispinor carrier and gives all three current
            components equal weight without per-component normalisation.
        verbose: print progress lines.
        band_range_left: optional explicit left window (start, end).
            When given, takes precedence over (n_val, n_cond).
        band_range_right: optional explicit right window.
        band_norms: optional (nbands,) array of band norms
            (``wfn.band_norms``) for pseudoband reweighting. When given,
            applied to both left and right ψ via
            ``ψ /= max(norm_slice, 1.0)`` before the pair-density
            einsum.
        k_weights: optional normalized full-BZ quadrature weights.  The
            centroid driver expands each stored IBZ parent weight uniformly
            over its star and passes the same table to candidate generation
            and pruning.

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
        gram_q0_tiled_from_psi_sm,
        gram_q0_tiled_from_psi_aot_resident_increment_bytes,
    )

    gamma_mode = _resolve_candidate_gamma_mode(
        gamma_mode, bispinor=bispinor)
    if gamma_mode == "transverse" and band_norms is not None:
        raise ValueError(
            "transverse candidate pruning uses unit weight for every band "
            "in the requested left/right fit windows; band_norms must be None")

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
    # Meta.nval is physical occupancy, not the width of the left feature
    # window.  Explicit windows may cross the occupied boundary.
    meta_nval, meta_ncond = _gram_meta_band_counts(
        wfn.nelec, max_band, n_val, n_cond)

    M = int(cand_idx.shape[0])
    cand_idx = jnp.asarray(cand_idx, dtype=jnp.int64)

    meta = Meta.from_system(
        wfn, sym,
        nval=meta_nval, ncond=meta_ncond,
        nband=max_band,
        n_rmu=M,
        bispinor=bispinor,
    )

    if k_weights is None:
        kw_np = np.full(int(sym.nk_tot), 1.0 / float(sym.nk_tot),
                        dtype=np.float64)
    else:
        kw_np = np.asarray(k_weights, dtype=np.float64)
        if kw_np.shape != (int(sym.nk_tot),):
            raise ValueError(
                "k_weights must have one entry per unfolded full-BZ k point; "
                f"got {kw_np.shape}, expected {(int(sym.nk_tot),)}")
        if (not np.all(np.isfinite(kw_np)) or np.any(kw_np < 0.0)
                or not np.isclose(kw_np.sum(), 1.0, rtol=1e-12, atol=1e-14)):
            raise ValueError(
                "k_weights must be finite, nonnegative, and sum to one; "
                f"got sum={kw_np.sum():.17g}")
    kw = jnp.asarray(kw_np, dtype=jnp.float64)

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

    # Prune must not retain the full-k G-flat WFN beside both final centroid
    # faces.  A one-k fixed tile is the hard memory bound; the shared
    # transform owner pads only the final tile and reuses one executable, so
    # this changes transfer scheduling, not the Gram or selection semantics.
    prune_k_tile = 1

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
              f"gamma_mode={gamma_mode}, "
              f"norms={'on' if band_norms is not None else 'off'}, "
              f"backend=WfnLoader(auto), "
              f"budget={meta.memory_per_device_gb:g} GB/device, "
              f"band_chunk_size={band_chunk_size}, "
              f"transfer_k_tile={prune_k_tile}")

    def _load_face(face_range, norms, timing_name):
        with timing.section(timing_name):
            psi_y, psi_x = load_centroids_band_chunked(
                wfn, sym, meta, cand_idx, bispinor, mesh_xy, face_range,
                band_chunk_size=band_chunk_size,
                k_chunk_size=prune_k_tile,
            )
            if norms is not None:
                # Y shape (nk, nb, ns, n_rmu); X shape (nk, n_rmu, nb, ns)
                psi_y = psi_y / norms[None, :, None, None]
                psi_x = psi_x / norms[None, None, :, None]
            psi_y.block_until_ready()
        return psi_y, psi_x

    # A nested pair of windows needs one WFN construction. Bands are
    # replicated in both final face layouts, so the smaller face is an exact
    # local slice of the larger one. This removes repeated parent I/O and
    # child FFTs without changing the one-k/band-tile memory policy.
    left_from_right = _band_window_slice(right_range, left_range)
    right_from_left = _band_window_slice(left_range, right_range)
    psi_r_rmu_Y = psi_r_rmuT_X = None
    if left_from_right is not None and left_range != right_range:
        psi_r_rmu_Y, psi_r_rmuT_X = _load_face(
            right_range, norms_r_j, "right.load")
        psi_l_rmu_Y, psi_l_rmuT_X = _slice_centroid_wfn_faces(
            psi_r_rmu_Y, psi_r_rmuT_X, left_from_right)
        if verbose:
            print(
                "[pivoted_cholesky] reused right WFN face for nested left "
                f"window {left_range} within {right_range}")
    else:
        psi_l_rmu_Y, psi_l_rmuT_X = _load_face(
            left_range, norms_l_j, "left.load")
        if right_from_left is not None:
            psi_r_rmu_Y, psi_r_rmuT_X = _slice_centroid_wfn_faces(
                psi_l_rmu_Y, psi_l_rmuT_X, right_from_left)
            if verbose:
                print(
                    "[pivoted_cholesky] reused left WFN face for nested "
                    f"right window {right_range} within {left_range}")

    # ---- 2-D tiled path (size-ladder wall fix) ----
    # The full open-spin pair tensors are (nk, ns, ns, M, M): 98 GB EACH at
    # M~9.8k (c7000 kmeans killed a 192 GB node — 2026-07-28 job 7878309).
    # Per-element contraction order is unchanged by tiling BOTH candidate
    # axes, so G is numerically the same map; only materialization moves.  The
    # two-axis tile is important: the nk-aware square law must price the same
    # object the pair-density compiler sees, never an unpriced M x block.
    n_dev_total = mesh_xy.devices.size
    n_x = int(mesh_xy.shape['x']) if 'x' in mesh_xy.axis_names else 1
    n_y = int(mesh_xy.shape['y']) if 'y' in mesh_xy.axis_names else 1
    col_block = 0
    # LORRAX_GRAM_COL_BLOCK: historical name, now an explicit square-tile
    # width; a falsy token
    # means "no override", i.e. the auto budget below.  This USED to be a
    # bare presence test — ``=0`` and ``=off`` are the two spellings a user
    # reaches for to DISABLE a knob, and they did the opposite or crashed.
    env_cb = os.environ.get("LORRAX_GRAM_COL_BLOCK", "").strip()
    if env_cb.lower() in ("", "0", "false", "no", "off"):
        env_cb = ""
    nk_, _, ns_, M_cols = (int(x) for x in psi_l_rmu_Y.shape)
    seed_budget_bytes = int(
        float(meta.memory_per_device_gb) * 1e9
        * _GRAM_SEED_BUDGET_FRACTION
    )
    tile_divisor = math.lcm(n_x, n_y)
    block_source = "auto"
    if env_cb:
        block_source = "LORRAX_GRAM_COL_BLOCK override"
        try:
            requested_block = int(env_cb)
            if requested_block <= 0:
                raise ValueError
            col_block = max(_GRAM_MIN_COL_BLOCK, requested_block)
        except ValueError:
            raise ValueError(
                f"LORRAX_GRAM_COL_BLOCK={env_cb!r} is neither a positive "
                f"integer tile width nor a falsy token "
                f"('', 0, false, no, off)."
            ) from None
        # A manual width keeps its historical floor and is rounded UP so the
        # now-square tile divides BOTH mesh axes.  It is an explicit override,
        # so it may exceed auto's target and the diagnostic below says so.
        col_block = round_up(col_block, tile_divisor)
    else:
        if gram_col_block_bytes(nk_, ns_, M_cols) <= seed_budget_bytes:
            col_block = M_cols
        else:
            col_block = auto_gram_col_block_width(
                nk_, ns_, seed_budget_bytes, divisor=tile_divisor,
            )
    if col_block >= M_cols:
        col_block = 0  # one full block == the original computation

    if col_block:
        if psi_r_rmu_Y is None:
            psi_r_rmu_Y, psi_r_rmuT_X = _load_face(
                right_range, norms_r_j, "right.load")

        # Compiler-aware width selection happens only after BOTH canonical
        # WFN windows exist.  The allocator reading is therefore the actual
        # resident floor (including loader tables and any unrelated live
        # arrays), not a second shape formula for the WFN service.
        gc.collect()
        from common.gpu_utils import _get_jax_gpu_memory_bytes
        _, live_now, _ = _get_jax_gpu_memory_bytes()
        if live_now is None and jax.default_backend() in ("gpu", "cuda"):
            from common.gpu_utils import (
                get_gpu_used_memory_bytes_nvidia_smi)
            live_now = get_gpu_used_memory_bytes_nvidia_smi()
            if live_now is not None:
                from runtime.aot_memory import announce_once
                announce_once(
                    "gram-live-nvidia-smi",
                    "allocator bytes_in_use unavailable for the Gram planner; "
                    "using this rank's conservative nvidia-smi whole-device "
                    "memory.used sample",
                )
        if live_now is None:
            # CPU/fallback accounting: sum the returned WFN shards.  Announce
            # that this is weaker because it cannot see service tables.
            resident_local_bytes = 0
            for arr in (psi_l_rmu_Y, psi_l_rmuT_X,
                        psi_r_rmu_Y, psi_r_rmuT_X):
                resident_local_bytes += sum(
                    int(np.asarray(sh.data).nbytes)
                    for sh in arr.addressable_shards
                )
            from runtime.aot_memory import announce_once
            announce_once(
                "gram-live-allocator-unavailable",
                "allocator bytes_in_use unavailable for the Gram planner; "
                "using the four canonical WFN output shards as a KNOWN-LOW "
                "resident floor",
            )
        else:
            resident_local_bytes = int(live_now)

        # The selected width controls static executable shapes and loop counts
        # on every process.  Allocator residency itself is rank-local, so price
        # from one shared worst-rank value before entering that host branch.
        resident_bytes = worst_process_resident_bytes(resident_local_bytes)

        target_bytes = int(float(meta.memory_per_device_gb) * 1e9)
        gram_local_bytes = (
            ((M_cols + n_x - 1) // n_x)
            * ((M_cols + n_y - 1) // n_y)
            * _GRAM_COMPLEX_BYTES
        )

        def _compiled_live_set(tile_width):
            tile_width = int(tile_width)
            scan_increment = (
                gram_q0_tiled_from_psi_aot_resident_increment_bytes(
                    mesh_xy=mesh_xy, nk=nk_, n_points=M_cols,
                    nb_l=nb_left, nb_r=nb_right, nspinor=ns_,
                    tile_width=tile_width, gamma_mode=gamma_mode,
                )
            )
            facts = gram_scan_live_set_bytes(
                resident_bytes=resident_bytes,
                scan_resident_increment_bytes=scan_increment,
                gram_matrix_local_bytes=gram_local_bytes,
            )
            facts["scan_increment"] = int(scan_increment)
            return facts

        # The sequential full-M path below still has the smaller WFN live set
        # and wins whenever the cheap square-law screen selected it. Once the
        # blocked route is entered, however, a full-width fused tile is valid
        # if its exact compiled live set fits; stopping at M-1 forced two
        # almost-full padded tiles per axis.
        max_tile_width = M_cols
        if not env_cb:
            col_block, live_facts = _auto_gram_width_from_compiled_peaks(
                col_block,
                max_width=max_tile_width,
                divisor=tile_divisor,
                budget_bytes=target_bytes,
                peak_for_width=_compiled_live_set,
            )
        else:
            live_facts = _compiled_live_set(col_block)

        if verbose:
            square_gib = (
                gram_col_block_bytes(nk_, ns_, col_block) / 2**30
            )
            local_gib = gram_col_block_device_bytes(
                nk_, ns_, M_cols, col_block,
                x_shards=n_x, y_shards=n_y,
            ) / 2**30
            ntiles, executed_extent, work_inflation = gram_tile_schedule(
                M_cols, col_block)
            print(
                f"[pivoted_cholesky] 2-D blocked Gram: M={M_cols}, "
                f"tile={col_block} ({ntiles}x{ntiles} tiles; "
                f"executed_extent={executed_extent}, "
                f"padded_work={work_inflation:.3f}x; "
                f"{n_dev_total}-device path; {block_source}; "
                f"square-law={square_gib:.2f} GiB global, "
                f"pair-workspace model={local_gib:.2f} GiB/device; "
                f"resident(two WFN windows; worst rank)="
                f"{resident_bytes / 2**30:.2f}, "
                f"compiled scan increment="
                f"{live_facts['scan_increment'] / 2**30:.2f}, "
                f"full-live peak={live_facts['peak'] / 2**30:.2f} "
                f"of target={target_bytes / 2**30:.2f} GiB/device)"
            )
        G = _candidate_gram_zero_kernel(mesh_xy, M_cols)()

        with timing.section("q0_sum.fused"):
            G = gram_q0_tiled_from_psi_sm(
                G, psi_l_rmuT_X, psi_l_rmu_Y,
                psi_r_rmuT_X, psi_r_rmu_Y, kw,
                mesh_xy=mesh_xy, tile_width=col_block,
                gamma_mode=gamma_mode,
            )
            # Same Hermitian symmetrization the unblocked kernel applies,
            # once, on the assembled square matrix.
            G = _candidate_gram_hermitian_fold_kernel(mesh_xy)(G)
            G.block_until_ready()
        del psi_l_rmu_Y, psi_l_rmuT_X, psi_r_rmu_Y, psi_r_rmuT_X
        return G

    with timing.section("left.pair"):
        P_l_k = pair_density(psi_l_rmuT_X, psi_l_rmu_Y, mesh_xy)
        P_l_k.block_until_ready()
    del psi_l_rmu_Y, psi_l_rmuT_X

    # ---- Right window ----
    if psi_r_rmu_Y is None:
        psi_r_rmu_Y, psi_r_rmuT_X = _load_face(
            right_range, norms_r_j, "right.load")
    with timing.section("right.pair"):
        P_r_k = pair_density(psi_r_rmuT_X, psi_r_rmu_Y, mesh_xy)
        P_r_k.block_until_ready()
    del psi_r_rmu_Y, psi_r_rmuT_X

    # ---- q=0 Gram: sum_k w_k · Σ_{αβ} conj(P_l_k,αβ) · P_r_k,αβ ----
    # γ̃ identity (charge channel) — open-spin Frobenius reduction.
    with timing.section("q0_sum.sequential"):
        G = candidate_gram_q0_from_pair(
            P_l_k, P_r_k, kw, mesh_xy=mesh_xy,
            gamma_mode=gamma_mode)
        G.block_until_ready()
    return G
