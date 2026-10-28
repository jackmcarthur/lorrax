"""CLI driver for the weighted k-means ISDF point selector.

Run as ``python3 -m centroid.kmeans_cli n_centroids [opts]``.

The physics, in order: sum unit-weight importance over the requested fit
bands; run weighted k-means over the real-space FFT grid in decorated-atom
space-group orbits (the crystal's symmetries; ``--no-symmetries`` selects
literal points); snap and unfold; then prune the oversampled pool with the
charge Gram or the three-channel transverse Gram for the same band windows.

Device meshes, sharding and placement live in :mod:`centroid.distribution`.
"""
from __future__ import annotations

import argparse


# ─────────────────────────────────────────────────────────────────────────
# Arg parser
# ─────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(allow_abbrev=False,
        description="Weighted k-means for ISDF sampling points.",
    )
    p.add_argument("n_centroids", type=int,
                   help="Number of centroids N_c (required).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true",
                   help="Emit a 3D matplotlib plot of centroids over ρ "
                        "(default off — most prod runs don't want a popup).")
    p.add_argument("--oversample", type=float, default=1.5,
                   help="k-means runs for ⌈n_centroids·oversample⌉ then "
                        "prunes via pivoted Cholesky (default 1.5). Set to "
                        "1.0 to disable pivoted-Cholesky pruning.")
    p.add_argument("-i", "--input", default=None,
                   help="GW deck. Its wfn_file is the WFN read (default "
                        "./WFN.h5) and its ncond sets the left window, every "
                        "occupied band plus the Sigma conduction bands; the "
                        "right window is every band in the WFN. Without a "
                        "deck the left window widens to every band.")
    p.add_argument("--fit-window", default=None, metavar="L0:L1,R0:R1",
                   help="Explicit 0-based half-open left/right band windows "
                        "shared by candidate construction and Gram pruning "
                        "(for example 0:16,0:28), in place of the deck's. "
                        "Physical occupancy still comes from WFN.nelec.")
    p.add_argument("--no-symmetries", action="store_true",
                   help="Select literal points without imposing the crystal's "
                        "symmetries. By default the points are closed under "
                        "every spatial Seitz operation of the decorated atoms "
                        "whenever there is more than one (electronic and TR "
                        "symmetry are not consulted).")
    p.add_argument("--density-mode",
                   choices=("scalar", "current"),
                   default="scalar",
                   help="Feature family used by both k-means and pruning. "
                        "'scalar' (default) uses the norm of charge "
                        "pair-density rows; 'current' uses the norm of the "
                        "three stacked Dirac-current rows. Both use the "
                        "resolved band windows with unit band weights. "
                        "Output files are "
                        "written with distinguishing suffixes ('' / "
                        "'_current') and a header comment naming the "
                        "density, so a downstream gw_jax run can read both "
                        "files unambiguously.")
    p.add_argument("--out-suffix", type=str, default=None,
                   help="Suffix appended to the output filename.  Default "
                        "is '' for --density-mode scalar and '_current' "
                        "for --density-mode current; pass explicitly to "
                        "override (e.g. multiple current-mode runs at "
                        "different n_centroids).")
    # Selector-study switches (ISDF_RANK_SCALING §10).  Each default is the
    # production selector, bit for bit.
    p.add_argument("--prune-q-shells", type=int, default=0, metavar="K",
                   help="Prune on the pair Gram summed over q = 0 and K "
                        "nonzero q of the k grid (one per |q| shell, spread "
                        "from the shortest shell to the longest; each q "
                        "weighted 1/(K+1)) instead of the q = 0 Gram alone. "
                        "Charge mode only. Default 0: q = 0.")
    p.add_argument("--weight-power", type=float, default=0.5, metavar="P",
                   help="k-means mass = (diag q=0 Gram)^P. 0.5 (default) is "
                        "the feature norm; a smaller P flattens the mass "
                        "toward low-density regions such as a slab's vacuum.")
    p.add_argument("--prune-normalize", type=float, default=0.0, metavar="G",
                   help="Prune on D^-G Gram D^-G, D = diag Gram (floored at "
                        "1e-12 max). 0 (default) is the plain Gram; 0.5 is "
                        "the amplitude-blind correlation matrix.")
    return p


if __name__ == "__main__":
    # Argv is answered before any runtime exists — runtime/cli_seam.py.
    from runtime.cli_seam import refuse_bad_argv
    refuse_bad_argv(build_parser())


# THE startup call (runtime module docstring): env defaults, fail-fast
# hook, jax.distributed, GPU-or-CPU resolution, the run's clique-warmed
# ('x','y') mesh, compile cache, rank-0 report.  MUST precede any import
# that pulls in jax, so JAX_ENABLE_X64 etc. take effect.  This driver's
# mesh POLICY stays in ``centroid.distribution.build_mesh`` (latency
# floor); when that policy shards, ``resolve_mesh`` hands it
# back this same startup mesh, not a second one.
from runtime import (
    debug_print,
    debug_print_enabled,
    initialize_communicator_stack,
)
RUNTIME = initialize_communicator_stack(print_fn=debug_print)

import gc
import os
import time
import warnings

import jax
import numpy as np

from wfn_loader import WfnLoader                                    # noqa: E402
from common import timing
from common.collectives import process_rank
from common.log import get_logger, message, record
from runtime.run_session import RunSession

from . import distribution as dist
from .kmeans_isdf import (
    BOHR_TO_ANG,
    _decide_init_method,
    _warn_dense_grid_regime,
    weighted_kmeans_jax,
    snap_centroids_to_grid,
    ensure_unique_centroids,
    _restore_orbit_pool,
)
from .production_output import (
    KMEANS_STAGES,
    format_centroid_header,
    kmeans_results_lines,
    kmeans_settings_rows,
    pool_rank_warning,
    prune_band_ranges,
    rank_law_estimate,
    resolve_sigma_ncond,
    validate_mode_policy,
)


def _release_lloyd_before_prune(*arrays) -> None:
    """Synchronize and release the completed Lloyd device state.

    Pivoted-Cholesky immediately loads a new WFN working set, and the prune
    consumes none of the returned labels or preparation grids: snapping has
    already copied the centroid coordinates and candidate indices to the host.
    Make that lifecycle boundary explicit and report BFC's measured effect.
    """
    def _bytes_in_use():
        used_values = []
        for device in jax.local_devices():
            stats = device.memory_stats() or {}
            used = stats.get("bytes_in_use")
            if used is not None:
                used_values.append(int(used))
        return max(used_values) if used_values else None

    before = _bytes_in_use()
    unique_arrays = []
    seen = set()
    for array in arrays:
        if isinstance(array, jax.Array) and id(array) not in seen:
            jax.block_until_ready(array)
            unique_arrays.append(array)
            seen.add(id(array))
    for array in unique_arrays:
        if not array.is_deleted():
            array.delete()
    del unique_arrays
    # Release Python cycles after deleting the finished array buffers. Keep
    # compiled programs available to the shared WFN and pruning routines.
    gc.collect()
    after = _bytes_in_use()
    memory_note = ""
    if before is not None and after is not None:
        memory_note = (
            f"; device bytes-in-use {before / 2**30:.2f} -> "
            f"{after / 2**30:.2f} GiB"
        )
    log.debug("Released completed Lloyd/preparation device state before prune "
              f"WFN transfer{memory_note}")


# Reachable through the one service-path bootstrap above; gated by
# tests/test_service_path_bootstrap.py.
import symmetry_maps                                            # noqa: E402

# Historical progress messages are forensic detail: DEBUG, in the rank log
# under LORRAX_DEBUG_PRINT.  The few facts a run's reader wants are INFO.
log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# The σ window — one resolver, two consumers
# ─────────────────────────────────────────────────────────────────────────

def _resolve_sigma_window(args, wfn) -> tuple[int, int]:
    """``(n_val, n_cond)`` of the σ window — the bands the ISDF must span.

    Single source of truth for both consumers: the pivoted-Cholesky prune
    band ranges and the ``band_range`` k-means weight.

    KEEP THESE TWO EXPLICIT.  ``n_cond`` defaulting to ``n_val`` gave every
    centroid set built before 2026-07-29 a prune window of (0, n_val) ×
    (0, 2·n_val) while Σ_c consumed the full band square: at nb=1024 the
    selection came back rank-deficient by 30 % (897 orbits requested, 630
    achieved, Gram diagonals at 7.6e-17) and the QP gap went NEGATIVE.
    Dose-response is monotone in the window: (0,52) → eqp0 0.3645,
    (0,256) → 3.1350, (0,1024) → 3.7227.  ``wk_REL/centroid_rank_gate.sh``
    reads this run's log and FAILS on the shortfall — do not reword the
    "target N orbits", "prune window:" or "After pruning ... rank=N" lines
    below without updating it.

    BUG FIX 2026-07-29 (size campaign, ladder notes R12).  The default was

        n_cond = min(n_val, nb_total - n_val)

    which CLAMPS the conduction extent to ``n_val`` and therefore has nothing
    to do with the window Σ_c actually consumes.  On the MoS2 4×4 deck
    (``nelec = 26``) it produced a 26×52 prune window for EVERY centroid set
    the size campaign built — b256 c2500, b512 c7000, b1024 c10000, b1024
    c15000 — while the b1024 deck's σ window is ``nval 26 + ncond 998 = 1024``.
    The ISDF basis was thus selected to resolve a 26×52 pair-density block and
    then used for a 1024×1024 one.  Measured cost of the clamp at
    (nb=1024, N=10000, M=13872), same WFN and same candidate pool:

        prune window (0,52)   -> Gram diag min 7.632e-17, rank 630 / 897
        prune window (0,256)  -> Gram diag min 7.189e-13, rank 897 / 897
        prune window (0,1024) -> Gram diag min 4.996e-12, rank 897 / 897

    i.e. the old default was **rank-deficient by 30%**: pivoted Cholesky was
    asked for 897 independent directions and could only certify 630, because
    the narrow window leaves most candidate centroids at numerically-zero Gram
    diagonal.  This is treated as a BUG, not a default change — every
    pre-fix centroid set is affected and should be regarded as selected for a
    different, much smaller problem than the one it was used on.

    The new default is the FULL conduction window present in the WFN, which is
    a superset of any deck's ncond and therefore always safe.  Narrowing is
    available only through ``--fit-window`` (the ``--prune-n-val`` /
    ``--prune-n-cond`` overrides were removed 2026-10-21).  Cost of the fix is
    small and was measured, not assumed: the (0,1024) build took 349 s against
    308 s for (0,52) at nb=1024 — +13% wall, +15 GB peak (81 GB of 186 GB).
    """
    n_val = int(wfn.nelec)
    return n_val, max(0, int(wfn.nbands) - n_val)


def _resolve_deck(args) -> tuple[dict | None, str]:
    """The input file's parameters and its ``wfn_file``.

    Without a deck: ``(None, "WFN.h5")``.  A relative ``wfn_file`` joins the
    deck's directory, as in ``gw.kin_ion_io``.  Resolved once in
    :func:`main`; :func:`_resolve_weight` resolves its Σ conduction count
    (``ncond``, or ``number_bands_protected`` as gw resolves it) against the
    WFN into ``args.sigma_ncond``, which ``prune_band_ranges`` reads for the
    left window.
    """
    if args.input is None:
        if args.fit_window is None:
            warnings.warn(
                "kmeans: no deck (-i), so the Sigma window is unknown; the "
                "left window widens to every band, a safe superset of "
                "occupied + Sigma conduction.", RuntimeWarning,
                stacklevel=2)
        return None, "WFN.h5"
    from file_io.paths import resolve_input_path
    from gw.gw_config import _DECK_NAMED_KEYS, read_lorrax_input, require_deck_keys
    deck = read_lorrax_input(args.input)
    require_deck_keys(deck[_DECK_NAMED_KEYS], "ncond_request")
    # Every planner of this run prices against the deck's device budget, as
    # in gwjax; 0 (the default) keeps the card's collective auto-detection.
    deck_gb = float(deck.get("memory_per_device_gb", 0.0) or 0.0)
    if deck_gb > 0:
        from common.gpu_utils import resolve_device_budget_gb
        resolve_device_budget_gb(deck_gb)
    return deck, resolve_input_path(
        os.path.dirname(os.path.abspath(args.input)), str(deck["wfn_file"]))


# ─────────────────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────────────────

def _resolve_symmetry(args, wfn):
    """The decorated atomic space group used only for centroid closure.

    Electronic density, magnetism, time reversal, occupations, and the WFN's
    symmetry authorization are intentionally absent.  Centroids are grid
    points, so their sampling set closes under every spatial Seitz operation
    preserving the lattice and mapping each atom to the same species.

    Returns ``(R, Rinv, tau, n_sym, orbit_aware)``.
    """
    if args.no_symmetries:
        return None, None, None, 1, False
    from symmetry_maps import recover_atomic_space_group
    R, Rinv, tau = recover_atomic_space_group(
        np.asarray(wfn.avec), np.asarray(wfn.atom_crys),
        np.asarray(wfn.atom_types))
    n_sym = int(R.shape[0])
    determinants = np.rint(np.linalg.det(Rinv)).astype(np.int32)
    n_improper = int(np.count_nonzero(determinants < 0))
    n_nonsymmorphic = int(np.count_nonzero(
        np.max(np.abs(tau - np.rint(tau)), axis=1) > 1.0e-8))
    log.debug(
        f"Centroid symmetry: decorated atoms give {n_sym} spatial Seitz "
        f"operations ({n_improper} improper, {n_nonsymmorphic} with "
        "fractional translation); electronic/TR data not consulted")
    if n_sym > 1:
        return R, Rinv, tau, n_sym, True
    return None, None, None, 1, False


def _resolve_weight(args, wfn, sym, R, tau, dist_mesh=None):
    """Build the feature-row norm used by both k-means and pruning.

    The cheap stage uses ``sqrt(diag(G))`` for the exact left/right windows
    of the final q=0 Gram. Occupations never enter either channel, and the
    band contraction is performed through local subspace projectors rather
    than explicit transition densities.

    Returns ``(weight, label, (left_range, right_range))``; the windows are
    carried unchanged into the centroid file's provenance header.

    WHY THE WINDOWS INCLUDE CONDUCTION BANDS ON BOTH LEGS: the weight of a
    point is ``Σ_{m∈L,n∈R} |ψ_m†ψ_n|²``, so a pair adds weight only where
    both of its states have density.  An occupied-only left leg gives
    ~zero weight wherever occupied states vanish, e.g. a slab's vacuum,
    however many conduction bands sit on the right.  The left leg therefore
    carries the Σ conduction window (deck ``ncond``) as well as every
    occupied band, so the c×c' pairs Σ reads are weighted too.
    """
    n_val, n_cond = _resolve_sigma_window(args, wfn)
    resolve_sigma_ncond(args, wfn, log.info)
    left_range, right_range, range_label = prune_band_ranges(
        args, n_val, n_cond)
    mode = "transverse" if args.density_mode == "current" else "charge"
    channel = ("Σ_i |Ψ_m†α_iΨ_n/α_fs|²" if mode == "transverse" else
               "|ψ_m†ψ_n|²")
    log.debug(
        f"k-means weight: sqrt(Σ_k w_k Σ_{{m∈{left_range},n∈{right_range}}} "
        f"{channel}); [{range_label}], unit band weights")
    from .sampling_metric import build_feature_metric_diagonal
    metric_diagonal = build_feature_metric_diagonal(
        wfn, sym, left_range, right_range, gamma_mode=mode,
        dist_mesh=dist_mesh,
        verbose=(debug_print_enabled() and process_rank() == 0),
    )
    if R is not None:
        from .charge_density import symmetrize_on_grid
        metric_diagonal = symmetrize_on_grid(metric_diagonal, R, tau)
    power = float(args.weight_power)
    if power == 0.5:
        weight = np.sqrt(metric_diagonal)
        mass = "sqrt"
    else:
        weight = np.maximum(metric_diagonal, 0.0) ** power
        mass = f"power {power:g} of"
    label = (
        f"{mass}(diag q=0 {mode} Gram), left={left_range}, "
        f"right={right_range}, unit band weights")
    return weight, label, (left_range, right_range)


def _snap_and_unfold(centroids_frac, fft_grid, weight, orbit_aware,
                     Rinv, tau, n_sym, M_cand):
    """Fractional centroids → unique FFT-grid points.

    Returns ``(indices, fractional, n_unique, orbit_id)``.
    """
    if orbit_aware:
        # Snap reps to the FFT grid FIRST. Lloyd produces off-grid fp64 reps,
        # and their fp64 sym images would round inconsistently — two
        # mathematically sym-related reps could land on different grid
        # cells. Snap-then-unfold guarantees on-grid orbit closure (because
        # R is integer and τ × fft_grid is integer for grid-commensurate τ).
        _, reps_snapped, _ = snap_centroids_to_grid(
            centroids_frac, fft_grid, deduplicate=False)
        # Unfold with Rinv = inv(mtrx): the BGW r-action is r' = Rinv·r + τ,
        # matching centroid_source_map_and_wrap and validate_atomic_symmetries.
        # A no-op vs forward S on symmorphic systems (CrI3, MoS2); critical
        # for Si Fd-3m.
        from symmetry_maps import unfold_orbit_unique_with_id
        unfolded, orbit_id = unfold_orbit_unique_with_id(
            reps_snapped, np.asarray(Rinv), np.asarray(tau))
        log.debug(f"\nUnfolded {centroids_frac.shape[0]} reps → "
                  f"{unfolded.shape[0]} distinct centroids (n_sym={n_sym})")
        indices, snapped, _ = snap_centroids_to_grid(
            unfolded, fft_grid, deduplicate=False)
        before = len(indices)
        indices = _restore_orbit_pool(
            indices, fft_grid, weight, Rinv, tau, M_cand)
        if len(indices) > before:
            # Reuse the symmetry service's orbit IDs for every point,
            # including the redistributed ones, before grouped pruning.
            snapped = indices.astype(float) / np.asarray(fft_grid)
            unfolded, orbit_id = unfold_orbit_unique_with_id(
                snapped, np.asarray(Rinv), np.asarray(tau))
            indices, snapped, _ = snap_centroids_to_grid(
                unfolded, fft_grid, deduplicate=False)
            log.info(f"Restored snapped candidate pool: {before} → "
                     f"{len(indices)} unique points in complete spatial orbits")
        return indices, snapped, indices.shape[0], orbit_id

    log.debug(f"\nSnapping {M_cand} centroids to FFT grid {fft_grid}...")
    indices, snapped, n_dups = snap_centroids_to_grid(
        centroids_frac, fft_grid, deduplicate=True)
    if n_dups == 0:
        log.debug(f"✓ All {indices.shape[0]} centroids on unique grid points.")
        return indices, snapped, indices.shape[0], None

    log.debug(f"⚠ {n_dups} duplicates; redistributing to nearby grid points...")
    snapped = ensure_unique_centroids(
        centroids_frac, fft_grid, rho=weight)
    indices = (np.round(snapped * np.asarray(fft_grid))
               .astype(np.int64) % np.asarray(fft_grid))
    return indices, snapped, snapped.shape[0], None


def _prune(args, wfn, sym, mesh, cand_idx, orbit_id, n_unique, N_c):
    """Pivoted-Cholesky prune of the over-sampled candidate pool to
    ``n_centroids`` (``N_c``).

    Greedy pivoting on the pair-density Gram keeps the candidates that add
    the most independent interpolation directions over the σ band window;
    the rank is the number it actually certified. Returns
    ``(indices, rank, unpicked)``, the last the largest never-picked residual
    over the floor (at most 1: the pool is spent).
    """
    from .pivoted_cholesky import prune_candidates_by_pivoted_cholesky
    from .sampling_metric import full_k_quadrature_weights

    # ``N_c`` is a point budget.  The block selector admits only whole orbits
    # that fit, then deflates by every point in each admitted orbit.
    log.debug(f"\nPivoted-Cholesky prune: {n_unique} → {N_c}"
              f"{' (whole-orbit blocks)' if orbit_id is not None else ''}")

    n_val, n_cond = _resolve_sigma_window(args, wfn)     # one resolver
    left_range, right_range, range_label = prune_band_ranges(
        args, n_val, n_cond)
    kwargs: dict = dict(
        wfn=wfn, sym=sym, cand_idx=cand_idx, n_keep=N_c, mesh=mesh,
        orbit_id=orbit_id,
        n_point_budget=(int(N_c) if orbit_id is not None else None),
        group_block=(orbit_id is not None),
        bispinor=(args.density_mode == "current"),
        gamma_mode=("transverse" if args.density_mode == "current"
                    else "charge"),
        k_weights=full_k_quadrature_weights(wfn, sym),
        verbose=(debug_print_enabled() and process_rank() == 0),
        band_range_left=left_range,
        band_range_right=right_range,
        gram_normalize=float(args.prune_normalize),
    )
    q_note = ""
    if int(args.prune_q_shells) > 0:
        from .pivoted_cholesky import prune_q_union
        kwargs["q_union"] = prune_q_union(wfn, sym, int(args.prune_q_shells))
        q_note = (" + q " + ", ".join(
            "(" + ",".join(f"{v:.4g}" for v in q) + ")"
            for q in kwargs["q_union"][0]) + f", each weight 1/"
            f"{len(kwargs['q_union'][0]) + 1}")
    log.info(f"  prune window: left={left_range} right={right_range} "
             f"[{range_label}]; Gram="
             f"{'Σ_i Z_i Z_i† (i=1,2,3)' if args.density_mode == 'current' else 'charge'}"
             f" at q = 0{q_note}"
             + (f"; normalized D^-{float(args.prune_normalize):g} G "
                f"D^-{float(args.prune_normalize):g}"
                if float(args.prune_normalize) else ""))

    with timing.section("prune"):
        keep_idx, rank, *_, unpicked = prune_candidates_by_pivoted_cholesky(
            **kwargs)
    indices = np.asarray(keep_idx, dtype=np.int64)
    log.info(f"After pruning: {indices.shape[0]} centroids (rank={rank})")
    return indices, int(rank), float(unpicked)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

class _NoStages:
    """Library calls of :func:`select_points` print no ``[i/N]`` banners."""

    total = 0

    def begin(self, *args, **kwargs) -> None:
        pass

    def result(self, text: str) -> None:
        pass

    @staticmethod
    def row(label, value) -> None:
        pass


def select_points(args, wfn_path, *, keep_weight=None, stages=None):
    """THE selection, as :func:`main` runs it: weight, weighted k-means in
    orbits, snap and unfold, then the pivoted-Cholesky prune of the
    over-sampled pool.  ``args`` is this module's parsed namespace (with
    ``sigma_ncond``); every rank must enter.  Returns a namespace with the
    FFT-grid ``indices`` (``(n, 3)``), their fractional ``snapped`` positions
    and the selection's receipts.  ``keep_weight`` (default ``args.plot``)
    keeps the weight array for the caller.  ``stages`` (the driver's
    ``common.preprocessing_output.StageSequence``) opens the main log's
    ``[i/N]`` stages; the last stays open for the caller's file write."""
    from types import SimpleNamespace
    from common.preprocessing_output import stage_row
    keep_weight = bool(args.plot) if keep_weight is None else bool(keep_weight)
    row = stage_row if stages is not None else _NoStages.row
    stages = stages if stages is not None else _NoStages()

    N_c = int(args.n_centroids)
    oversample = float(args.oversample)
    M_cand = int(np.ceil(N_c * oversample)) if oversample > 1.0 else N_c
    if M_cand != N_c:
        log.debug(f"Over-sampling: k-means M = {M_cand}, prune to "
                  f"n_centroids = {N_c} (ratio {oversample:g})")
    else:
        log.debug(f"Using n_centroids = {N_c} clusters (no pivoted-Cholesky "
                  "pruning)")

    stages.begin(1, "Feature weight", name="centroid.feature_weight")
    with timing.section("setup.wfn_io"):
        wfn = WfnLoader(wfn_path)
        sym = wfn.symmetry()

        n_rtot = int(np.prod(wfn.fft_grid))
        if N_c > n_rtot:
            raise ValueError(
                f"requested {N_c} centroids but the FFT grid contains only "
                f"{n_rtot} distinct points; request fewer centroids or a finer FFT grid")
        init_method, init_msg = _decide_init_method(N_c, n_rtot)
        if init_msg is not None:
            log.debug(init_msg)
        dense_warn = _warn_dense_grid_regime(M_cand, N_c, n_rtot)
        if dense_warn is not None:
            log.warning(
                dense_warn.removeprefix("WARNING: "),
                extra=message(
                    id="centroid_dense_grid",
                    found=("requested centroids",
                           f"n_centroids = {N_c} of {n_rtot} FFT-grid points"),
                    needs="candidate coverage and Gram conditioning checked",
                    fix="read the pivoted-Cholesky rank in the centroid header"))

    fft_grid = tuple(int(x) for x in wfn.fft_grid)

    avec_ang = np.asarray(wfn.avec) * float(wfn.alat) * BOHR_TO_ANG
    log.debug(f"WFN FFT-grid shape: {fft_grid}")
    log.debug(f"Lattice lengths: {np.linalg.norm(avec_ang, axis=1)} Å")

    mesh = dist.build_mesh(int(np.prod(fft_grid)))
    mesh_axis = dist.MESH_AXES

    # The loader was necessarily built mesh-less (the mesh is sized from
    # the FFT grid the file declares), which at P>1 pinned it to the
    # per-rank eager h5py read (scorecard BD.2).  Late-bind the mesh so
    # backend=auto can pick the collective phdf5 route for the ψ loads
    # (prune / rank gate / weight), as htransform already does.
    wfn.adopt_mesh(mesh)

    R, Rinv, tau, n_sym, orbit_aware = _resolve_symmetry(args, wfn)

    with timing.section("setup.weight"):
        weight, weight_label, weight_band_ranges = _resolve_weight(
            args, wfn, sym, R, tau, dist_mesh=mesh)
    row("windows", f"left {weight_band_ranges[0][0] + 1}-{weight_band_ranges[0][1]} x "
                   f"right {weight_band_ranges[1][0] + 1}-{weight_band_ranges[1][1]} "
                   f"over {int(np.asarray(sym.nk_tot))} k, unit band weights")
    stages.result(f"weight on {n_rtot} grid points ({fft_grid[0]} x {fft_grid[1]} "
                  f"x {fft_grid[2]})")

    if orbit_aware:
        # In orbit mode the SAMPLED count is M_cand; the OUTPUT after unfold
        # may inflate by up to n_sym. Adjust kmeans target so the final
        # unfolded centroid count is roughly N_c.
        # (Generic-position rep unfolds to n_sym distinct centroids.)
        M_cand_orbit = int(np.ceil(M_cand / n_sym))
        if M_cand_orbit < 1:
            raise ValueError(
                f"Symmetric selection: requested n_centroids={N_c} "
                f"(×oversample={oversample}, M_cand={M_cand}) gives < 1 "
                f"representative orbit at n_sym={n_sym}. Need n_centroids × "
                f"oversample >= n_sym (= {n_sym}) so kmeans can sample at "
                f"least one orbit; pass --no-symmetries or raise n_centroids."
            )
        log.debug(f"Orbit-aware mode: n_sym = {n_sym}, "
                  f"running kmeans for M_rep = {M_cand_orbit} representatives "
                  f"(unfolded ≈ {M_cand_orbit * n_sym} centroids)")
        kmeans_target = M_cand_orbit
    else:
        kmeans_target = M_cand

    # Seeding samples logical FFT points without replacement. Apply the cap
    # after converting the candidate point budget to orbit representatives;
    # an already-valid representative count must keep its existing seeds.
    if kmeans_target > n_rtot:
        log.info(
            f"Capping k-means initialization from {kmeans_target} to "
            f"{n_rtot} logical FFT-grid points")
        kmeans_target = n_rtot

    stages.begin(2, "Weighted k-means", name="centroid.kmeans")
    row("target", f"{M_cand} candidates = {kmeans_target} orbit representatives "
                  f"x {n_sym} operations" if orbit_aware else
                  f"{kmeans_target} candidates (literal points)")
    with timing.section("kmeans") as kmeans_section:
        labels, centroids, _lloyd_steps, _lloyd_move_sq = weighted_kmeans_jax(
            avec_ang, weight, N_c=kmeans_target, seed=args.seed,
            mesh=mesh, mesh_axis=mesh_axis,
            init_method=init_method,
            R=R, Rinv=Rinv, tau=tau,
            return_labels=False,
        )
        # The host materialises ``centroids`` immediately below.  Watch every
        # returned device value here so this phase measures completion rather
        # than asynchronous dispatch; this is one semantic boundary, not an
        # inner-loop synchronization point.
        kmeans_section.watch(
            labels, centroids, _lloyd_steps, _lloyd_move_sq)
    centroids_frac = np.asarray(centroids)

    with timing.section("snap_unfold"):
        centroid_indices, centroids_snapped, n_unique, orbit_id_arr = \
            _snap_and_unfold(centroids_frac, fft_grid, weight, orbit_aware,
                             Rinv, tau, n_sym, M_cand)

    stages.result(f"pool of {int(n_unique)} grid points after "
                  + ("orbit unfold and " if orbit_aware else "") + "FFT snap")
    pruned = False
    prune_rank = unpicked = None
    n_pool = int(n_unique)
    if stages.total == 3:
        stages.begin(3, "Pivoted-Cholesky selection", name="centroid.select")
    if oversample > 1.0 and n_unique > N_c:
        release_arrays = [labels, centroids]
        # ``weight`` is also the plot payload.  Retain it when plotting, but
        # otherwise release it before the prune reloads the WFN windows.
        if not keep_weight:
            release_arrays.append(weight)
        with timing.section("release_before_prune"):
            _release_lloyd_before_prune(*release_arrays)
        del release_arrays, labels, centroids
        if not keep_weight:
            weight = None
        centroid_indices, rank, unpicked = _prune(
            args, wfn, sym, mesh, centroid_indices, orbit_id_arr,
            n_unique, N_c)
        n_unique = int(centroid_indices.shape[0])
        centroids_snapped = centroid_indices.astype(float) / np.asarray(fft_grid)
        pruned = True
        prune_rank = int(rank)

        log.info(f"  [point rank] {rank}/{n_unique} delivered centroid "
                 "directions certified; rank deficiency is reported")
        row("pivots", f"{int(rank)} independent of {n_unique} kept, from the "
                      f"{n_pool}-point pool")
    elif stages.total == 3:
        row("pruning", f"not needed: the pool holds {n_pool} <= {N_c} points")

    return SimpleNamespace(
        wfn=wfn, indices=np.asarray(centroid_indices, dtype=np.int64),
        snapped=centroids_snapped, n_unique=int(n_unique), pool=int(n_pool),
        pruned=pruned, prune_rank=prune_rank, unpicked=unpicked,
        weight=weight, weight_label=weight_label,
        weight_band_ranges=weight_band_ranges, M_cand=M_cand,
        orbit_aware=orbit_aware, n_sym=n_sym, fft_grid=fft_grid, N_c=N_c)


def orbit_kmeans_points(wfn_path, n_points: int, *, fit_window, seed: int = 0):
    """FFT-grid indices ``(n, 3)`` of an orbit-closed, charge-density-weighted
    k-means selection of about ``n_points`` points over ``fit_window``
    (``((l0, l1), (r0, r1))`` bands), with the ISDF bundle's settings
    (the crystal's symmetries, the default over-sampling and prune).  Every
    rank must enter; the result is the same on every rank."""
    (l0, l1), (r0, r1) = fit_window
    args = build_parser().parse_args([
        str(int(n_points)), "--seed",
        str(int(seed)), "--fit-window", f"{l0}:{l1},{r0}:{r1}"])
    validate_mode_policy(args)
    args.sigma_ncond = None
    return select_points(args, str(wfn_path), keep_weight=False).indices


def _out_suffix(args) -> str:
    """The suffix of the centroid table and of its report (``''`` / ``_current``)."""
    return (args.out_suffix if args.out_suffix is not None
            else ("" if args.density_mode == "scalar" else "_current"))


def main():
    args = build_parser().parse_args()
    validate_mode_policy(args)
    # The report carries the SAME suffix as the table it describes.  A charge
    # and a current selection are routinely run in one directory (the deck
    # names both files), and a bare "kmeans.out" meant the second run silently
    # destroyed the first one's record -- its band window, seed, candidate
    # pool and achieved rank.
    from common.preprocessing_output import ScientificProductionReport, StageSequence
    report_path = os.path.abspath(f"kmeans{_out_suffix(args)}.out")
    with RunSession(RUNTIME, "kmeans", ScientificProductionReport, report_path,
                    stages=KMEANS_STAGES, driver="centroids",
                    purpose="ISDF centroid selection") as run, \
            StageSequence(3 if float(args.oversample) > 1.0 else 2) as stages:
        return _select_and_report(args, run, stages)


def _select_and_report(args, run, stages):
    """Select, write the centroid table, and close ``kmeans<suffix>.out``."""
    # (the compile cache was armed at import, by step 7 of
    # initialize_communicator_stack)
    report = run.report
    report.begin(input_file=args.input)
    report.devices(mesh_role="centroid selection")
    log.info(f"✓ JAX initialized: {dist.device_summary()}")

    selection_start = time.perf_counter()
    args.sigma_ncond = None
    args.sigma_deck, wfn_path = _resolve_deck(args)
    report.settings(kmeans_settings_rows(args, source=report.flag_source))
    sel = select_points(args, wfn_path, stages=stages)
    (wfn, _indices, centroids_snapped, n_unique, n_pool, pruned,
     prune_rank, unpicked, weight, weight_label, weight_band_ranges, M_cand,
     orbit_aware, n_sym, fft_grid, N_c) = (
        sel.wfn, sel.indices, sel.snapped, sel.n_unique, sel.pool, sel.pruned,
        sel.prune_rank, sel.unpicked, sel.weight, sel.weight_label,
        sel.weight_band_ranges, sel.M_cand, sel.orbit_aware, sel.n_sym,
        sel.fft_grid, sel.N_c)

    out_file = f"centroids_frac_{n_unique}{_out_suffix(args)}.txt"
    n_val_header, n_cond_header = _resolve_sigma_window(args, wfn)
    prune_left, prune_right, prune_label = prune_band_ranges(
        args, n_val_header, n_cond_header)
    if args.density_mode == "current":
        feature_fit = (
            f"left={weight_band_ranges[0]}, right={weight_band_ranges[1]}: "
            "sqrt(sum_k w_k sum_i,m,n "
            "|Psi_mk^dag alpha_i Psi_nk/alpha_fs|^2); unit band weights")
    else:
        feature_fit = (
            f"left={weight_band_ranges[0]}, right={weight_band_ranges[1]}: "
            "sqrt(sum_k w_k sum_m,n |psi_mk^dag psi_nk|^2); "
            "unit band weights")
    kgrid = tuple(int(v) for v in np.asarray(wfn.kgrid).reshape(-1)[:3])
    shift = tuple(float(v) for v in np.asarray(wfn.shift).reshape(-1)[:3])
    prune_state = "pivoted Cholesky" if pruned else "not applied"
    if pruned and int(args.prune_q_shells) > 0:
        prune_state += (f" on the q-union Gram (q = 0 + {int(args.prune_q_shells)}"
                        " |q| shells)")
    if pruned and float(args.prune_normalize):
        prune_state += f"; Gram normalized D^-{float(args.prune_normalize):g}"
    rank_law = (None if args.density_mode == "current" else
                rank_law_estimate(
                    left=prune_left, right=prune_right, nspinor=wfn.nspinor,
                    kgrid=kgrid, ng=wfn.ng, ecutwfc=wfn.ecutwfc,
                    ecutrho=wfn.ecutrho))
    rank_warning = pool_rank_warning(
        pool=n_pool, written=n_unique, unpicked=unpicked, rank_law=rank_law)
    if rank_warning is not None:
        # Graded: a WARNING only when the budget stop leaves the set below the
        # rank-law estimate (written/estimate < 1); at or above it, or with no
        # estimate (current mode), a NOTE.
        log.warning(
            f"pair-space rank not established: {n_unique} centroids are not a "
            "complete ISDF basis",
            extra=message(
                id="centroid_rank_not_established",
                found=("pivoted Cholesky", f"stopped by the point budget at "
                       f"{n_unique} points of a {n_pool}-point pool"),
                lines=(rank_warning,),
                fix="raise n_centroids or --oversample until the rank reads "
                    "'pool spent'",
                kind=(None if rank_law is not None
                      and n_unique / float(rank_law[0]) < 1.0 else "NOTE")))
    header = format_centroid_header(
        feature_fit=feature_fit, source_wfn=wfn_path,
        weight_label=weight_label,
        num_electrons=float(getattr(wfn, "num_electrons", np.nan)),
        occupied_boundary=int(wfn.nelec), fft_grid=fft_grid,
        kgrid=kgrid, shift=shift, seed=args.seed, requested=N_c, candidates=M_cand, written=n_unique,
        pruning=prune_state, prune_rank=prune_rank,
        prune_left=prune_left, prune_right=prune_right,
        prune_label=prune_label, orbit_aware=orbit_aware, n_sym=n_sym,
        density_mode=args.density_mode, pool=n_pool, unpicked=unpicked,
        rank_law=rank_law)
    # ONE writer.  Every rank used to reach this savetxt on the same shared
    # path.  It survived P=16 only because all ranks write identical bytes —
    # which is precisely the latent form of the bug that DID bite at P=64 in
    # ``bse_io.write_eigenvectors_stream`` (64 concurrent h5py creators,
    # rc=1 plus a structurally valid file; wk_REL S4.8).  ``centroids_snapped``
    # is a pure function of the WFN + seed + candidate list and is identical on
    # every rank, so rank 0's file is the file any rank would have written.
    # No collective below this point, so gating cannot deadlock.
    if process_rank() == 0:
        np.savetxt(
            out_file, centroids_snapped,
            header=header,
            fmt="%.6f", delimiter=" ", comments="# ",
        )
        log.info(f"Saved centroids to {out_file}")
        # The table's own header is the selection's full record.
        log.debug("centroid table header", extra=record(
            "centroids.header", file=out_file, lines=header.splitlines()))
    stages.result(f"{n_unique} centroids written to {out_file}"
                  + (" closed under the crystal's symmetries" if orbit_aware
                     else ""))
    stages.end()

    selection_wall = time.perf_counter() - selection_start
    if process_rank() == 0 and debug_print_enabled():
        timing.report(title="--- kmeans_cli timing (s) ---",
                      wall=selection_wall)

    report.emit_lines(kmeans_results_lines(
        requested=N_c, candidates=M_cand, pool=n_pool, written=n_unique,
        out_file=out_file, orbit_aware=orbit_aware, n_sym=n_sym,
        windows=(prune_left, prune_right), window_label=prune_label,
        density_mode=args.density_mode, prune_rank=prune_rank,
        unpicked=unpicked, rank_law=rank_law))

    if args.plot:
        from .kmeans_plot import plot_density_and_centroids, interpolate_density
        rho_plot = interpolate_density(weight, (1.0,) * 3)
        plot_density_and_centroids(wfn, rho_plot, centroids_snapped)
    # The shared footer without ``RunSession.complete``: that closes with a
    # barrier this driver never ran (print-only rule).
    report.timings(run.stage_rows(), wall=run.wall)
    report.warnings()
    report.files(run.file_rows([
        ("source wavefunctions", "read", wfn_path),
        *((("input deck", "read", args.input),) if args.input else ()),
        ("ISDF centroids", "written", os.path.abspath(out_file)),
    ]))
    report.finish(wall=run.wall)
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
