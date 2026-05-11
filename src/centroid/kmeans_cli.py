"""CLI driver for the weighted k-means ISDF point selector.

Run as ``python3 -m centroid.kmeans_cli N_C [opts]``. Builds the device
mesh, calls ``weighted_kmeans_jax``, snaps to the FFT grid, optionally
prunes via pivoted Cholesky, and optionally plots.
"""
from __future__ import annotations

# Must precede any jax import so JAX_ENABLE_X64 etc. take effect.
from runtime import set_default_env
set_default_env()

import argparse
import math

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from runtime import init_jax_distributed
init_jax_distributed()

from file_io import WfnLoader as WFNReader
from common import symmetry_maps, timing

from .charge_density import get_charge_density
from .kmeans_isdf import (
    BOHR_TO_ANG,
    _decide_init_method,
    _warn_dense_grid_regime,
    weighted_kmeans_jax,
    snap_centroids_to_grid,
    ensure_unique_centroids,
)


# ─────────────────────────────────────────────────────────────────────────
# Arg parser
# ─────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
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
    p.add_argument("--rho-source", choices=("auto", "qe_save", "wfn_ibz"),
                   default="auto",
                   help="Charge density source. 'auto' (default) prefers "
                        "QE's symmetrized rho when reachable, else falls "
                        "back to the WFN IBZ sum.")
    p.add_argument("--rho-power", type=float, default=1.0,
                   help="Use ρ(r)^α as the k-means weight (default α=1.0). "
                        "Per Gersho (3D), centroid number-density "
                        "asymptotically scales as ρ^(3α/5), so α=1 gives "
                        "ρ^0.6, α=5/3≈1.667 gives ρ^1, α=10/3≈3.333 gives "
                        "ρ^2 (closer to the |ψ_v|²|ψ_c|² distribution the "
                        "ISDF actually wants to fit). Bump above 1 if your "
                        "centroids are under-clustering high-density "
                        "regions.")
    p.add_argument("--qe-save", type=str, default=None,
                   help="Explicit QE <prefix>.save path. Default: "
                        "auto-detected from cwd via qe/scf/*.save or "
                        "qe/nscf/*.save (no need to pass when launching "
                        "from a normal sandbox run dir).")
    p.add_argument("--oversample", type=float, default=1.5,
                   help="k-means runs for ⌈N_c·oversample⌉ then prunes via "
                        "pivoted Cholesky (default 1.5). Set to 1.0 to "
                        "disable pivoted-Cholesky pruning.")
    p.add_argument("--prune-n-val", type=int, default=None,
                   help="Override pivoted-Cholesky n_val (default = wfn.nelec).")
    p.add_argument("--prune-n-cond", type=int, default=None,
                   help="Override pivoted-Cholesky n_cond (default = n_val).")
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
                        " unfold via the WFN's spatial sym ops at output. Final"
                        " centroid set is closed under the point group.")
    p.add_argument("--no-orbit", action="store_true",
                   help="Force the literal-point (non-orbit) path even if "
                        "WFN.h5 has multiple sym ops. Overrides --orbit.")
    p.add_argument("--use-phdf5", action="store_true",
                   help="Pull G-space wavefunctions through the parallel-HDF5 "
                        "FFI loader instead of the default WFNReader. "
                        "Necessary for WFN.h5 files that don't fit in host "
                        "RAM. Default off.")
    p.add_argument("--density-mode",
                   choices=("scalar", "current"),
                   default="scalar",
                   help="Which scalar field to use as the kmeans weight. "
                        "'scalar' (default) is the standard charge density "
                        "ρ(r), the right weight for charge-channel ISDF "
                        "(γ̃^0).  'current' is the squared Gordon-decomposed "
                        "Pauli current Σ_{n,k,i}(j^Gordon_{n,k,i})², the "
                        "right weight for the i-channel ISDF (γ̃^{1,2,3}) "
                        "in the bispinor pipeline.  Output files are "
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


# ─────────────────────────────────────────────────────────────────────────
# Mesh selection (factor n_dev as 2-D when possible).
# ─────────────────────────────────────────────────────────────────────────

_P_PER_SHARD_MIN = 100_000
"""NCCL-latency floor: shard only when each device sees ≥ this many points
(measured on Si 4×4×4: P=110k / 4 GPUs gave a slower run than single-device
because allreduce dominated the 1 ms local compute)."""


def _build_mesh(args, n_points: int) -> tuple[Mesh, tuple[str, ...]]:
    """Pick a 2-D device mesh ('x', 'y'), matching gw_jax's ISDF mesh.

    Single-device collapses to a 1×1 2-D mesh so the downstream pipeline
    (which uses ``load_centroids_band_chunked`` and friends) only has one
    codepath to worry about.
    """
    devices = jax.devices()
    n_dev = len(devices)
    multi_host = jax.process_count() > 1

    if args.no_shard or n_dev < 2:
        return Mesh(np.asarray(devices[:1]).reshape(1, 1), ("x", "y")), ("x", "y")

    # Most-square 2-D factorisation (same recipe as ``gw_jax._build_mesh``).
    nx = max(k for k in range(1, int(math.isqrt(n_dev)) + 1) if n_dev % k == 0)
    ny = n_dev // nx
    n_shards = nx * ny
    per_shard = n_points // n_shards

    # Never fall back to single-device when running multi-host: the other
    # ranks would sit on collectives against a mesh they aren't in and
    # the JAX distributed shutdown barrier would hang for minutes.
    if per_shard < _P_PER_SHARD_MIN and not args.force_shard and not multi_host:
        print(f"P/{n_shards} = {per_shard} < {_P_PER_SHARD_MIN} points per "
              "shard; falling back to single-device. Pass --force-shard to "
              "override.")
        return Mesh(np.asarray(devices[:1]).reshape(1, 1), ("x", "y")), ("x", "y")

    if multi_host and per_shard < _P_PER_SHARD_MIN:
        print(f"P/{n_shards} = {per_shard} < {_P_PER_SHARD_MIN}; sharding "
              "anyway (multi-host: single-device fallback would deadlock).")

    dev_grid = np.asarray(devices).reshape(nx, ny)
    print(f"Sharded mesh: ('x'={nx}, 'y'={ny}) over {n_dev} devices")
    return Mesh(dev_grid, ("x", "y")), ("x", "y")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    try:
        from common.jax_compile_cache import ensure_jax_compile_cache
        ensure_jax_compile_cache()
    except Exception as e:
        print(f"  [jax compile cache] skipped: {e}", flush=True)

    print(f"✓ JAX initialized: {len(jax.devices())} device(s) "
          f"(local: {len(jax.local_devices())}, "
          f"proc {jax.process_index()}/{jax.process_count()})")

    timing.reset()

    N_c = int(args.N_c)
    oversample = float(args.oversample)
    M_cand = int(np.ceil(N_c * oversample)) if oversample > 1.0 else N_c
    if M_cand != N_c:
        print(f"Over-sampling: k-means M = {M_cand}, prune to N_c = {N_c} "
              f"(ratio {oversample:g})")
    else:
        print(f"Using N_c = {N_c} clusters (no pivoted-Cholesky pruning)")

    with timing.section("setup.wfn_io"):
        wfn = WFNReader("WFN.h5")
        sym = symmetry_maps.SymMaps(wfn)

        n_rtot = int(np.prod(wfn.fft_grid))
        init_method, init_msg = _decide_init_method(N_c, n_rtot)
        if init_msg is not None:
            print(init_msg)
        dense_warn = _warn_dense_grid_regime(M_cand, N_c, n_rtot)
        if dense_warn is not None:
            print(dense_warn)

    with timing.section("setup.charge_density"):
        if args.density_mode == "current":
            from .current_density import build_current_density
            # n_occ convention: nelec for FR (nspinor=2), nelec/2 for scalar
            # (nspinor=1 — restricted Kohn-Sham, two electrons per band).
            n_occ = int(wfn.nelec) if int(wfn.nspinor) == 2 else int(wfn.nelec) // 2
            print(f"  density-mode=current: building bispinor j² weight "
                  f"with n_occ={n_occ} (nspinor_wfn={int(wfn.nspinor)})")
            charge_density = build_current_density(wfn, sym, n_occ)
        else:
            charge_density = get_charge_density(
                wfn=wfn, sym=sym,
                source=args.rho_source,
                save_dir=args.qe_save,
            )
    fft_grid = tuple(int(x) for x in charge_density.shape)
    if fft_grid != tuple(int(x) for x in wfn.fft_grid):
        raise ValueError(
            f"FFT-grid mismatch: ρ shape {fft_grid} vs WFN.h5 FFTgrid "
            f"{tuple(wfn.fft_grid)}. Requires ecutrho = 4·ecutwfc (norm-"
            f"conserving) or pw2bgw on the dense grid (USPP/PAW)."
        )

    avec_ang = np.asarray(wfn.avec) * float(wfn.alat) * BOHR_TO_ANG
    print(f"Charge density shape: {charge_density.shape}")
    print(f"Lattice lengths: {np.linalg.norm(avec_ang, axis=1)} Å")

    rho_jax = jnp.asarray(charge_density, dtype=jnp.float64)
    if args.rho_power != 1.0:
        # Clip to non-negative before power (QE iFFT can leave tiny < 0
        # noise) and tell the user we're using a non-default exponent.
        rho_jax = jnp.maximum(rho_jax, 0.0) ** float(args.rho_power)
        print(f"k-means weight: ρ(r)^{args.rho_power:g} "
              f"(asymptotic centroid density ∝ ρ^{0.6*args.rho_power:.3f})")
    avec_jax = jnp.asarray(avec_ang, dtype=jnp.float64)

    n_points = int(np.prod(fft_grid))
    mesh, mesh_axis = _build_mesh(args, n_points)

    # Decide orbit vs non-orbit. Default heuristic: orbit if --orbit explicit
    # OR WFN has > 1 spatial sym op AND --no-orbit not set.
    orbit_aware = args.orbit or (int(wfn.ntran) > 1 and not args.no_orbit)
    R = Rinv = tau = None
    n_sym = 1
    if orbit_aware:
        from .orbit_syms import build_real_space_syms
        R, Rinv, tau = build_real_space_syms(wfn, sym)
        n_sym = int(R.shape[0])
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
        print(f"Orbit-aware mode: n_sym = {n_sym}, "
              f"running kmeans for M_rep = {M_cand_orbit} representatives "
              f"(unfolded ≈ {M_cand_orbit * n_sym} centroids)")
        kmeans_target = M_cand_orbit
    else:
        kmeans_target = M_cand

    with timing.section("kmeans"):
        _, centroids_jax, _, _ = weighted_kmeans_jax(
            avec_jax, rho_jax, N_c=kmeans_target, seed=args.seed,
            mesh=mesh, mesh_axis=mesh_axis,
            init_method=init_method,
            R=R, Rinv=Rinv, tau=tau,
        )
        centroids_jax.block_until_ready()
    centroids_frac = np.asarray(centroids_jax)

    orbit_id_arr = None
    with timing.section("snap_unfold"):
        if orbit_aware:
            # Snap reps to the FFT grid FIRST. Lloyd produces off-grid fp64 reps,
            # and their fp64 sym images would round inconsistently — two
            # mathematically sym-related reps could land on different grid
            # cells. Snap-then-unfold guarantees on-grid orbit closure (because
            # R is integer and τ × fft_grid is integer for grid-commensurate τ).
            _, reps_snapped, _ = snap_centroids_to_grid(
                centroids_frac, fft_grid, deduplicate=False,
            )
            from .orbit_syms import unfold_orbit_unique_with_id
            unfolded, orbit_id_arr = unfold_orbit_unique_with_id(
                reps_snapped, np.asarray(Rinv), np.asarray(tau),
            )
            print(f"\nUnfolded {centroids_frac.shape[0]} reps → "
                  f"{unfolded.shape[0]} distinct centroids (n_sym={n_sym})")
            centroid_indices, centroids_snapped, _ = snap_centroids_to_grid(
                unfolded, fft_grid, deduplicate=False,
            )
            n_unique = centroid_indices.shape[0]
        else:
            print(f"\nSnapping {M_cand} centroids to FFT grid {fft_grid}...")
            centroid_indices, centroids_snapped, n_dups = snap_centroids_to_grid(
                centroids_frac, fft_grid, deduplicate=True
            )
            n_unique = centroid_indices.shape[0]
            if n_dups > 0:
                print(f"⚠ {n_dups} duplicates; redistributing to nearby grid points...")
                centroids_snapped = ensure_unique_centroids(
                    centroids_frac, fft_grid, rho=charge_density,
                )
                n_unique = centroids_snapped.shape[0]
                centroid_indices = (np.round(centroids_snapped * np.asarray(fft_grid))
                                    .astype(np.int64) % np.asarray(fft_grid))
            else:
                print(f"✓ All {n_unique} centroids on unique grid points.")

    if oversample > 1.0 and n_unique > N_c:
        from .pivoted_cholesky import prune_candidates_by_pivoted_cholesky
        # Orbit mode: target ORBITS not points; final centroid count is
        # Σ orbit_size of picked orbits (≈ N_c by construction).
        n_orbits = (len(np.unique(orbit_id_arr))
                    if orbit_id_arr is not None else n_unique)
        n_orbit_keep = (max(1, int(np.ceil(N_c * n_orbits / n_unique)))
                        if orbit_id_arr is not None else N_c)
        print(f"\nPivoted-Cholesky prune: {n_unique} → {N_c}"
              f"{f' (target {n_orbit_keep} orbits)' if orbit_id_arr is not None else ''}")
        # Resolve n_val/n_cond defaults the same way the legacy path did
        # so we can compute explicit band ranges for the v×(v+c) variant.
        _n_val_eff = (int(args.prune_n_val) if args.prune_n_val is not None
                      else int(wfn.nelec))
        _nb_total = int(wfn.nbands)
        _n_cond_eff = (int(args.prune_n_cond) if args.prune_n_cond is not None
                       else min(_n_val_eff, _nb_total - _n_val_eff))
        _max_band = _n_val_eff + _n_cond_eff
        _prune_kwargs: dict = dict(
            wfn=wfn, sym=sym, cand_idx=centroid_indices,
            n_keep=n_orbit_keep, mesh=mesh,
            orbit_id=orbit_id_arr,
            use_phdf5=args.use_phdf5,
        )
        if args.prune_window == "v_x_vc":
            _prune_kwargs["band_range_left"] = (0, _n_val_eff)
            _prune_kwargs["band_range_right"] = (0, _max_band)
            print(f"  prune window: v×(v+c)  left=(0,{_n_val_eff}) "
                  f"right=(0,{_max_band})  [covers |ψ_v|² + v×c]")
        elif args.prune_window == "vc_x_vc":
            _prune_kwargs["band_range_left"] = (0, _max_band)
            _prune_kwargs["band_range_right"] = (0, _max_band)
            print(f"  prune window: (v+c)×(v+c)  left=right=(0,{_max_band})"
                  f"  [full σ-window square Gram, covers |ψ_c|² too]")
        else:
            _prune_kwargs["n_val"] = _n_val_eff
            _prune_kwargs["n_cond"] = _n_cond_eff
            print(f"  prune window: v×c  left=(0,{_n_val_eff}) "
                  f"right=({_n_val_eff},{_max_band})  [legacy]")
        with timing.section("prune"):
            keep_idx, rank, *_ = prune_candidates_by_pivoted_cholesky(
                **_prune_kwargs,
            )
        centroid_indices = np.asarray(keep_idx, dtype=np.int64)
        centroids_snapped = centroid_indices.astype(float) / np.asarray(fft_grid)
        n_unique = centroid_indices.shape[0]
        print(f"After pruning: {n_unique} centroids (rank={rank})")

    # Default suffix follows --density-mode unless the user overrode it.
    out_suffix = (args.out_suffix
                  if args.out_suffix is not None
                  else ("" if args.density_mode == "scalar" else "_current"))
    out_file = f"centroids_frac_{n_unique}{out_suffix}.txt"
    if args.density_mode == "scalar":
        density_label = "scalar charge density ρ(r)"
    else:
        density_label = ("Gordon-decomposed Pauli current "
                         "Σ_{n,k,i}(j^Gordon_{n,k,i}(r))²")
    header = (
        f"x y z (snapped to FFT grid {fft_grid}, {n_unique} unique)\n"
        f"density: {args.density_mode}\n"
        f"weight: {density_label}\n"
        f"intended channels: "
        f"{'γ̃^0 (charge) ISDF' if args.density_mode == 'scalar' else 'γ̃^{1,2,3} (current) ISDF'}"
    )
    np.savetxt(
        out_file, centroids_snapped,
        header=header,
        fmt="%.6f", delimiter=" ", comments="# ",
    )
    print(f"Saved centroids to {out_file}")

    if jax.process_index() == 0:
        timing.report(title="--- kmeans_cli timing (s) ---")

    if args.plot:
        from .kmeans_plot import plot_density_and_centroids, interpolate_density
        rho_plot = interpolate_density(charge_density, (args.plot_zoom,) * 3)
        plot_density_and_centroids(wfn, rho_plot, centroids_snapped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
