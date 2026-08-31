"""CLI driver for the weighted k-means ISDF point selector.

Run as ``python3 -m centroid.kmeans_cli N_C [opts]``.

The physics, in order: sum unit-weight importance over the requested fit
bands; run weighted k-means over the real-space FFT grid in decorated-atom
space-group orbits; snap and unfold; then prune the oversampled pool with the
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
    p.add_argument("N_c", type=int, nargs="?", default=400,
                   help="Number of centroids (default 400).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true",
                   help="Emit a 3D matplotlib plot of centroids over ρ "
                        "(default off — most prod runs don't want a popup).")
    p.add_argument("--plot-zoom", type=float, default=1.0,
                   help="Bicubic upsampling factor on the plot only.")
    p.add_argument("--no-shard", action="store_true",
                   help="Force single-device Lloyd even when multiple GPUs "
                        "are visible (default: shard across jax.devices()).")
    p.add_argument("--force-shard", action="store_true",
                   help="Override the per-shard-P auto-gate (useful for "
                        "benchmarking).")
    p.add_argument("--rho-power", type=float, default=1.0,
                   help="Raise the sampling weight to α (default α=1.0). "
                        "Per Gersho (3D), centroid number-density "
                        "asymptotically scales as w^(3α/5), where w is the "
                        "resolved feature-row norm. Thus α=1 gives w^0.6 "
                        "and α=5/3 gives w^1.")
    p.add_argument("--oversample", type=float, default=1.5,
                   help="k-means runs for ⌈N_c·oversample⌉ then prunes via "
                        "pivoted Cholesky (default 1.5). Set to 1.0 to "
                        "disable pivoted-Cholesky pruning.")
    p.add_argument("--prune-n-val", type=int, default=None,
                   help="Override pivoted-Cholesky n_val (default = wfn.nelec).")
    p.add_argument("--prune-n-cond", type=int, default=None,
                   help="Override pivoted-Cholesky n_cond (default = the FULL "
                        "conduction window in the WFN, nbands - n_val, which "
                        "is a superset of any deck's ncond and therefore "
                        "always safe). Narrowing this selects the ISDF basis "
                        "on fewer pair densities than Sigma_c will consume; "
                        "the rank gate will refuse if that costs independence. "
                        "Before 2026-07-29 the default was min(n_val, nbands - "
                        "n_val), which silently clamped the window to n_val.")
    p.add_argument("--prune-window", choices=("v_x_c", "v_x_vc", "vc_x_vc"),
                   default="v_x_vc",
                   help="Pivoted-Cholesky Gram band-window pair. "
                        "'v_x_c' (legacy) = left (0, n_val), right (n_val, "
                        "n_val+n_cond) — only val×cond pair densities. "
                        "'v_x_vc' (default) = left (0, n_val), right (0, "
                        "n_val+n_cond) — adds val×val (and val×cond), so "
                        "the centroids also span the |ψ_v|² and ψ_v*ψ_v' "
                        "diagonals that V_H / G_RI projections need. "
                        "'vc_x_vc' = left = right = (0, n_val+n_cond) — "
                        "full σ-window square Gram, also includes c×c "
                        "pair densities (|ψ_c|² diagonals). n_cond should "
                        "be the input file's ``ncond`` (σ-protected band "
                        "count), not the full ``nband`` summed over.")
    p.add_argument("--orbit", action="store_true",
                   help="Symmetry-adapted k-means: store orbit representatives,"
                        " unfold with the atom-derived spatial Seitz group. "
                        "Final centroids are closed under the decorated "
                        "crystal structure, independent of electronic/TR "
                        "symmetry.")
    p.add_argument("--no-orbit", action="store_true",
                   help="Force the literal-point (non-orbit) path even if "
                        "the decorated atoms have multiple spatial Seitz "
                        "operations. Overrides --orbit.")
    p.add_argument("--density-mode",
                   choices=("scalar", "current"),
                   default="scalar",
                   help="Feature family used by both k-means and pruning. "
                        "'scalar' (default) uses the norm of charge "
                        "pair-density rows; 'current' uses the norm of the "
                        "three stacked Dirac-current rows. Both use the "
                        "resolved --prune-window with unit band weights. "
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
                        "different N_c).")
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
# floor, --no-shard); when that policy shards, ``resolve_mesh`` hands it
# back this same startup mesh, not a second one.
from runtime import (
    debug_print,
    debug_print_enabled,
    initialize_communicator_stack,
    rank0_print,
)
RUNTIME = initialize_communicator_stack(print_fn=debug_print)

# Historical progress messages are forensic detail.  The one driver-wide
# switch owns all of them; the concise production report uses rank0_print.
print0 = debug_print

import gc
import os
import time

import jax
import numpy as np

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from wfn_loader import WfnLoader                                    # noqa: E402
from common import timing
from common.collectives import process_rank
from runtime.production_stream import ProductionStdout

from . import distribution as dist
from .kmeans_isdf import (
    BOHR_TO_ANG,
    _decide_init_method,
    _warn_dense_grid_regime,
    weighted_kmeans_jax,
    snap_centroids_to_grid,
    ensure_unique_centroids,
)
from .production_output import (
    format_centroid_header,
    format_kmeans_report,
    prune_band_ranges,
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
    # Lloyd's executable cache can retain device constants and the allocator
    # cannot reclaim deleted buffers while Python cycles still own them.
    jax.clear_caches()
    gc.collect()
    after = _bytes_in_use()
    memory_note = ""
    if before is not None and after is not None:
        memory_note = (
            f"; device bytes-in-use {before / 2**30:.2f} -> "
            f"{after / 2**30:.2f} GiB"
        )
    print("Released completed Lloyd/preparation device state before prune "
          f"WFN transfer{memory_note}")


# Reachable through the one service-path bootstrap above; gated by
# tests/test_service_path_bootstrap.py.
import symmetry_maps                                            # noqa: E402


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
    still available explicitly via ``--prune-n-cond``.  Cost of the fix is
    small and was measured, not assumed: the (0,1024) build took 349 s against
    308 s for (0,52) at nb=1024 — +13% wall, +15 GB peak (81 GB of 186 GB).
    """
    n_val = (int(args.prune_n_val) if args.prune_n_val is not None
             else int(wfn.nelec))
    nb_total = int(wfn.nbands)
    n_cond = (int(args.prune_n_cond) if args.prune_n_cond is not None
              else max(0, nb_total - n_val))
    return n_val, n_cond


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
    if args.no_orbit:
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
    print0(
        f"Centroid symmetry: decorated atoms give {n_sym} spatial Seitz "
        f"operations ({n_improper} improper, {n_nonsymmorphic} with "
        "fractional translation); electronic/TR data not consulted")
    if args.orbit or n_sym > 1:
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

    WHY THE WINDOW INCLUDES REQUESTED CONDUCTION BANDS: occupied-only ρ(r)
    is entirely inside the slab, so a valence-only k-means places ZERO
    centroids in the vacuum
    and the vacuum-localized far-conduction states have no quadrature
    support — ⟨nk|V_H|nk⟩ (a pure centroid sum) then comes back sign-wrong
    (+140 eV vs −140 eV on MoS2) and the whole error lands on
    Vxc = E_dft − kin_ion − V_H.
    """
    n_val, n_cond = _resolve_sigma_window(args, wfn)
    left_range, right_range, range_label = prune_band_ranges(
        args, n_val, n_cond)
    mode = "transverse" if args.density_mode == "current" else "charge"
    channel = ("Σ_i |Ψ_m†α_iΨ_n/α_fs|²" if mode == "transverse" else
               "|ψ_m†ψ_n|²")
    print0(
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
    weight = np.sqrt(metric_diagonal)
    label = (
        f"sqrt(diag q=0 {mode} Gram), left={left_range}, "
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
            centroids_frac, fft_grid, deduplicate=False, print_fn=print0)
        # Unfold with Rinv = inv(mtrx): the BGW r-action is r' = Rinv·r + τ,
        # matching centroid_source_map_and_wrap and validate_atomic_symmetries.
        # A no-op vs forward S on symmorphic systems (CrI3, MoS2); critical
        # for Si Fd-3m.
        from symmetry_maps import unfold_orbit_unique_with_id
        unfolded, orbit_id = unfold_orbit_unique_with_id(
            reps_snapped, np.asarray(Rinv), np.asarray(tau))
        print0(f"\nUnfolded {centroids_frac.shape[0]} reps → "
               f"{unfolded.shape[0]} distinct centroids (n_sym={n_sym})")
        indices, snapped, _ = snap_centroids_to_grid(
            unfolded, fft_grid, deduplicate=False, print_fn=print0)
        return indices, snapped, indices.shape[0], orbit_id

    print0(f"\nSnapping {M_cand} centroids to FFT grid {fft_grid}...")
    indices, snapped, n_dups = snap_centroids_to_grid(
        centroids_frac, fft_grid, deduplicate=True, print_fn=print0)
    if n_dups == 0:
        print0(f"✓ All {indices.shape[0]} centroids on unique grid points.")
        return indices, snapped, indices.shape[0], None

    print0(f"⚠ {n_dups} duplicates; redistributing to nearby grid points...")
    snapped = ensure_unique_centroids(
        centroids_frac, fft_grid, rho=weight, print_fn=print0)
    indices = (np.round(snapped * np.asarray(fft_grid))
               .astype(np.int64) % np.asarray(fft_grid))
    return indices, snapped, snapped.shape[0], None


def _prune(args, wfn, sym, mesh, cand_idx, orbit_id, n_unique, N_c):
    """Pivoted-Cholesky prune of the over-sampled candidate pool to N_c.

    Greedy pivoting on the pair-density Gram keeps the candidates that add
    the most independent interpolation directions over the σ band window;
    the achieved rank is the number it actually certified.  Returns
    ``(indices, n_kept, rank, n_orbit_keep, n_val, max_band)`` — the last
    four feed the rank gate in ``main()``.
    """
    from .pivoted_cholesky import prune_candidates_by_pivoted_cholesky
    from .sampling_metric import full_k_quadrature_weights

    # Orbit mode targets ORBITS, not points: the final centroid count is
    # Σ orbit_size over the picked orbits.  ``N_c`` is a POINT count the user
    # typed, so the orbit target is an ESTIMATE and the delivered point total
    # is FLOORED to N_c inside the kernel (``n_point_budget``) — owner ruling,
    # 2026-08-10.  The orbit TARGET below is unchanged, deliberately: it is
    # what ``refuse_unless_select_certified`` and the rank gate in ``main``
    # are both stated against, so moving it would move a refusal threshold as
    # a side effect of a budget change.  The floor can only ever TRUNCATE the
    # delivered set, so the generator now delivers at most N_c points and
    # never more, with every existing refusal reading exactly as before.
    n_orbits = len(np.unique(orbit_id)) if orbit_id is not None else n_unique
    n_orbit_keep = (max(1, int(np.ceil(N_c * n_orbits / n_unique)))
                    if orbit_id is not None else N_c)
    print0(f"\nPivoted-Cholesky prune: {n_unique} → {N_c}"
           f"{f' (target {n_orbit_keep} orbits)' if orbit_id is not None else ''}")

    n_val, n_cond = _resolve_sigma_window(args, wfn)     # one resolver
    max_band = n_val + n_cond
    left_range, right_range, range_label = prune_band_ranges(
        args, n_val, n_cond)
    kwargs: dict = dict(
        wfn=wfn, sym=sym, cand_idx=cand_idx, n_keep=n_orbit_keep, mesh=mesh,
        orbit_id=orbit_id,
        n_point_budget=(int(N_c) if orbit_id is not None else None),
        bispinor=(args.density_mode == "current"),
        gamma_mode=("transverse" if args.density_mode == "current"
                    else "charge"),
        k_weights=full_k_quadrature_weights(wfn, sym),
        verbose=(debug_print_enabled() and process_rank() == 0),
    )
    if args.prune_window == "v_x_vc":
        kwargs["band_range_left"] = left_range
        kwargs["band_range_right"] = right_range
    elif args.prune_window == "vc_x_vc":
        kwargs["band_range_left"] = left_range
        kwargs["band_range_right"] = right_range
    else:
        kwargs["n_val"] = n_val
        kwargs["n_cond"] = n_cond
    print0(f"  prune window: left={left_range} right={right_range} "
           f"[{range_label}]; Gram="
           f"{'Σ_i Z_i Z_i† (i=1,2,3)' if args.density_mode == 'current' else 'charge'}")

    with timing.section("prune"):
        keep_idx, rank, *_ = prune_candidates_by_pivoted_cholesky(**kwargs)
    indices = np.asarray(keep_idx, dtype=np.int64)
    print0(f"After pruning: {indices.shape[0]} centroids (rank={rank})")
    # The rank gate in main() compares rank against n_orbit_keep and names
    # the effective prune window in its refusal — return all four.
    return indices, indices.shape[0], int(rank), n_orbit_keep, n_val, max_band


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()
    validate_mode_policy(args)
    production_warnings = []
    production_stdout = ProductionStdout(
        debug=debug_print_enabled(), rank=RUNTIME.process_index,
        warning_fn=production_warnings.append)
    production_stdout.install()

    # (the compile cache was armed at import, by step 7 of
    # initialize_communicator_stack)
    print0(f"✓ JAX initialized: {dist.device_summary()}")

    timing.reset()
    selection_start = time.perf_counter()

    N_c = int(args.N_c)
    oversample = float(args.oversample)
    M_cand = int(np.ceil(N_c * oversample)) if oversample > 1.0 else N_c
    if M_cand != N_c:
        print0(f"Over-sampling: k-means M = {M_cand}, prune to N_c = {N_c} "
               f"(ratio {oversample:g})")
    else:
        print0(f"Using N_c = {N_c} clusters (no pivoted-Cholesky pruning)")

    with timing.section("setup.wfn_io"):
        wfn = WfnLoader("WFN.h5")
        sym = wfn.symmetry()

        n_rtot = int(np.prod(wfn.fft_grid))
        init_method, init_msg = _decide_init_method(N_c, n_rtot)
        if init_msg is not None:
            print0(init_msg)
        dense_warn = _warn_dense_grid_regime(M_cand, N_c, n_rtot)
        if dense_warn is not None:
            print0(dense_warn)

    fft_grid = tuple(int(x) for x in wfn.fft_grid)

    avec_ang = np.asarray(wfn.avec) * float(wfn.alat) * BOHR_TO_ANG
    print0(f"WFN FFT-grid shape: {fft_grid}")
    print0(f"Lattice lengths: {np.linalg.norm(avec_ang, axis=1)} Å")

    mesh = dist.build_mesh(int(np.prod(fft_grid)),
                           shard=not args.no_shard,
                           force_shard=args.force_shard,
                           print_fn=print0)
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

    # w^α re-weighting.  Per Gersho the asymptotic centroid number density
    # goes as w^(3α/5), so α > 1 pulls points into high-density regions.
    # Only the k-means sees the power; ``weight`` itself stays as measured,
    # because it is also what breaks ties when snapped centroids collide.
    kmeans_weight = weight
    if args.rho_power != 1.0:
        # Clip to non-negative first — QE's iFFT can leave tiny < 0 noise.
        kmeans_weight = np.maximum(
            np.asarray(weight, dtype=np.float64), 0.0) ** float(args.rho_power)
        print0(f"k-means weight: (weight)^{args.rho_power:g} "
               f"(asymptotic centroid density ∝ w^{0.6*args.rho_power:.3f})")

    if orbit_aware:
        # In orbit mode the SAMPLED count is M_cand; the OUTPUT after unfold
        # may inflate by up to n_sym. Adjust kmeans target so the final
        # unfolded centroid count is roughly N_c.
        # (Generic-position rep unfolds to n_sym distinct centroids.)
        M_cand_orbit = int(np.ceil(M_cand / n_sym))
        if M_cand_orbit < 1:
            raise ValueError(
                f"Orbit mode: requested N_c={N_c} (×oversample={oversample}, "
                f"M_cand={M_cand}) gives < 1 representative orbit at n_sym="
                f"{n_sym}. Need N_c × oversample >= n_sym (= {n_sym}) so "
                f"kmeans can sample at least one orbit; pass --no-orbit or "
                f"raise N_c."
            )
        print0(f"Orbit-aware mode: n_sym = {n_sym}, "
               f"running kmeans for M_rep = {M_cand_orbit} representatives "
               f"(unfolded ≈ {M_cand_orbit * n_sym} centroids)")
        kmeans_target = M_cand_orbit
    else:
        kmeans_target = M_cand

    with timing.section("kmeans"):
        labels, centroids, _lloyd_steps, _lloyd_move_sq = weighted_kmeans_jax(
            avec_ang, kmeans_weight, N_c=kmeans_target, seed=args.seed,
            mesh=mesh, mesh_axis=mesh_axis,
            init_method=init_method,
            R=R, Rinv=Rinv, tau=tau,
            print_fn=print0,
        )
    centroids_frac = np.asarray(centroids)

    with timing.section("snap_unfold"):
        centroid_indices, centroids_snapped, n_unique, orbit_id_arr = \
            _snap_and_unfold(centroids_frac, fft_grid, weight, orbit_aware,
                             Rinv, tau, n_sym, M_cand)

    pruned = False
    prune_rank = None
    if oversample > 1.0 and n_unique > N_c:
        release_arrays = [labels, centroids]
        for array in (weight, kmeans_weight):
            # ``weight`` is also the plot payload.  Retain that one reference
            # when plotting, but release every device alias before the prune
            # reloads the WFN windows.
            if not args.plot or array is not weight:
                release_arrays.append(array)
        _release_lloyd_before_prune(*release_arrays)
        del release_arrays, labels, centroids, kmeans_weight
        if not args.plot:
            del weight
        (centroid_indices, n_unique, rank, n_orbit_keep,
         _n_val_eff, _max_band) = _prune(
            args, wfn, sym, mesh, centroid_indices, orbit_id_arr,
            n_unique, N_c)
        centroids_snapped = centroid_indices.astype(float) / np.asarray(fft_grid)
        pruned = True
        prune_rank = int(rank)

        # --- HARD REFUSAL: the achieved rank must meet the request ----------
        # (size campaign 2026-07-29, ladder notes R12.4; owner-approved.)
        # ``rank`` is how many INDEPENDENT interpolation directions pivoted
        # Cholesky could actually certify.  If it falls short of the number of
        # orbits requested, the returned set is padded with directions the
        # Gram says are numerically null — the file still contains the
        # requested number of points, so nothing downstream notices, and the
        # ISDF silently under-resolves.  That is exactly what happened on the
        # b1024 rung: rank 630 against 897 requested, printed in every log for
        # the whole campaign and read by nobody, while Σ_c produced a QP gap
        # of 0.36 eV against a true ~3.2-3.6 eV.  Refuse instead.
        #
        # STATE THE SENSITIVITY OF THE PROXY (2026-08-22).  On the b1024 rung
        # the ROOT CAUSE was a band-window mismatch — the ISDF window was
        # clamped small and then used large — and rank deficiency was its
        # SYMPTOM (``docs/dev/isdf_basis_adequacy_at_large_nband.md``: the fix
        # was re-selecting the SAME NUMBER of centroids against a
        # representative window).  So this gate is a proxy for a window
        # mismatch, and it is not sensitive to accuracy in general: measured
        # on the Si anchor deck, the shipped 960-point set carries ~160
        # numerically dependent points and scores sigTOT MAE 0.644 meV, the
        # best BerkeleyGW agreement on record, while the rank-clean orbit-mode
        # arm at the same N is 20-56x worse.  Rank is not basis quality in
        # EITHER direction (TASTE rule 12; ladder_rung1_notes R19.1).  The
        # refusal is kept because the window-mismatch failure it catches costs
        # electron-volts and is invisible to every other gate — but its text
        # now says what it is a proxy FOR, and no longer offers advice that is
        # measured not to work on the deck that hits it.
        # docs/dev/rank_truncation_policy.md §7.
        _unit = 'orbits' if orbit_id_arr is not None else 'points'
        _rank_tol = float(os.environ.get("LORRAX_CENTROID_RANK_TOL", "0.01"))
        _rank_floor = int(np.ceil((1.0 - _rank_tol) * n_orbit_keep))
        if int(rank) < _rank_floor:
            raise SystemExit(
                f"\nFATAL: pivoted-Cholesky rank deficiency — the candidate "
                f"pool cannot supply the independence you asked for.\n"
                f"  requested : {n_orbit_keep} {_unit}\n"
                f"  achieved  : {rank} {_unit}   "
                f"({100.0 * rank / max(1, n_orbit_keep):.1f}%"
                f", floor {_rank_floor} at tol {_rank_tol:g})\n"
                f"  prune window: left=(0,{_n_val_eff}) right=(0,{_max_band}) "
                f"[{args.prune_window}]\n"
                f"WHAT THIS GATE IS A PROXY FOR: a prune window that does not "
                f"cover the pair densities Σ will consume.  That failure is\n"
                f"worth electron-volts and is invisible to every other gate "
                f"(b1024: rank 630/897, QP gap 0.36 eV against ~3.2-3.6 eV, "
                f"root-caused\nto a clamped ISDF band window).  It is NOT a "
                f"general accuracy statement: a rank-deficient set can be the "
                f"most accurate one\nmeasured (Si anchor 960 points, ~160 "
                f"dependent, sigTOT MAE 0.644 meV — the record best).\n"
                f"Fix, in order of likelihood on a real deck:\n"
                f"  * check the prune window matches the Σ window — "
                f"--prune-n-cond <ncond of your deck>.  If you NARROWED it\n"
                f"    deliberately, this is the gate telling you the cost.  "
                f"(On the Si anchor deck, widening at fixed orbit setting\n"
                f"    changes sigTOT by <2x and never recovers the orbit-mode "
                f"loss, so widening is not always the answer.)\n"
                f"  * --prune-window vc_x_vc to include c×c pair densities "
                f"(needed when ncond >> nval)\n"
                f"  * raise --oversample for a richer pool, or lower N to "
                f"{rank} {_unit} for a rank-clean set\n"
                f"  * LORRAX_CENTROID_RANK_TOL=<fraction> to accept a "
                f"deficient set deliberately — the named override, and the\n"
                f"    one to use when you have MEASURED that this set is the "
                f"accurate one.\n")
        print0(f"  [rank gate] {rank}/{n_orbit_keep} {_unit} certified "
               f"(floor {_rank_floor}, tol {_rank_tol:g}) — PASS")
        if orbit_id_arr is not None:
            # SAY WHAT THIS PASS DOES NOT CERTIFY.  In orbit mode the select
            # deflates by one direction per ORBIT while removing all n_sym
            # members from contention, so this number is not comparable to
            # the point count of the file about to be written — "42 of 42
            # directions certified — PASS" was once said over 1908 points
            # whose ζ back-solve then truncated ~24% of the modes per q.
            # The delivered-granularity number is the `[point rank]` line
            # printed by the select above.
            print0(f"  [rank gate] SCOPE: that PASS is stated in ORBITS.  It "
                   f"certifies nothing about the {n_unique} POINTS this file "
                   f"will contain — read the [point rank] line above for the "
                   f"delivered-granularity number.")

    # Default suffix follows --density-mode unless the user overrode it.
    out_suffix = (args.out_suffix
                  if args.out_suffix is not None
                  else ("" if args.density_mode == "scalar" else "_current"))
    out_file = f"centroids_frac_{n_unique}{out_suffix}.txt"
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
    header = format_centroid_header(
        feature_fit=feature_fit, source_wfn="WFN.h5",
        weight_label=weight_label,
        num_electrons=float(getattr(wfn, "num_electrons", np.nan)),
        occupied_boundary=int(wfn.nelec), fft_grid=fft_grid,
        kgrid=kgrid, shift=shift, seed=args.seed, rho_power=args.rho_power,
        requested=N_c, candidates=M_cand, written=n_unique,
        pruning=prune_state, prune_rank=prune_rank,
        prune_left=prune_left, prune_right=prune_right,
        prune_label=prune_label, orbit_aware=orbit_aware, n_sym=n_sym,
        density_mode=args.density_mode)
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
        print0(f"Saved centroids to {out_file}")

    if process_rank() == 0 and debug_print_enabled():
        timing.report(title="--- kmeans_cli timing (s) ---")

    if process_rank() == 0:
        report_file = "kmeans.out"
        report_text = format_kmeans_report(
            header=header, source_wfn="WFN.h5", centroid_file=out_file,
            report_file=report_file,
            wfn_backend=str(getattr(wfn, "backend", "unknown")),
            elapsed_s=time.perf_counter() - selection_start,
            runtime=RUNTIME, warnings=production_warnings)
        with open(report_file, "w", encoding="utf-8") as stream:
            stream.write(report_text)
        if debug_print_enabled():
            rank0_print(report_text, end="")
        else:
            production_stdout.emit(report_text, end="")

    if args.plot:
        from .kmeans_plot import plot_density_and_centroids, interpolate_density
        rho_plot = interpolate_density(weight, (args.plot_zoom,) * 3)
        plot_density_and_centroids(wfn, rho_plot, centroids_snapped)
    production_stdout.close()
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
