"""CLI driver for the weighted k-means ISDF point selector.

Run as ``python3 -m centroid.kmeans_cli N_C [opts]``. Builds the device
mesh, calls ``weighted_kmeans_jax``, snaps to the FFT grid, optionally
prunes via pivoted Cholesky, and optionally plots.
"""
from __future__ import annotations

# Canonical JAX GPU/CPU bootstrap — single-sourced in runtime.bootstrap()
# (env defaults + jax.distributed init + CPU fallback; all idempotent).
# MUST precede this module's own `import jax` so JAX_ENABLE_X64 etc. take
# effect.  NOTE: this used to be set_default_env() + init_jax_distributed()
# only; bootstrap() adds fallback_to_cpu_if_no_gpu_backend(), so on a node
# with no usable GPU backend this CLI now falls back to CPU instead of
# dying at the first jax call.
from runtime import bootstrap
bootstrap()

import argparse
import math
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

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
    p.add_argument("--centroid-weight",
                   choices=("charge_density", "band_range"),
                   default=None,
                   help="WHICH density weights the k-means, i.e. where the "
                        "ISDF quadrature gets points. 'band_range' (DEFAULT "
                        "for --density-mode scalar) = Σ_{n∈range} Σ_k w_k "
                        "|ψ_nk(r)|² over the bands the calculation actually "
                        "uses. 'charge_density' = the ground-state OCCUPIED "
                        "ρ(r) (the historical weight, and the default for "
                        "--density-mode current). Occupied-"
                        "only weighting is entirely inside a slab, so a 2D "
                        "system gets ZERO centroids in the vacuum and its "
                        "vacuum-localized far-conduction states have no "
                        "quadrature support — ⟨nk|V_H|nk⟩ comes back "
                        "sign-wrong (+140 eV vs −140 eV on MoS2) and the "
                        "error lands on Vxc. Use 'band_range' when the σ "
                        "window reaches into vacuum-like states.")
    p.add_argument("--weight-bands", type=str, default=None,
                   metavar="LO:HI",
                   help="Explicit 0-based half-open band range for "
                        "--centroid-weight band_range. Default is the σ "
                        "window (0, n_val+n_cond) resolved exactly like the "
                        "pivoted-Cholesky prune window (see --prune-n-val / "
                        "--prune-n-cond). Sweep HI from n_val (occupied-"
                        "only) to n_val+n_cond to trade slab resolution "
                        "against vacuum support.")
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


def _resolve_sigma_window(args, wfn) -> tuple[int, int]:
    """``(n_val, n_cond)`` of the σ window — the bands the ISDF must span.

    Single source of truth for both consumers: the pivoted-Cholesky prune
    band ranges and the ``band_range`` k-means weight.

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


def _build_mesh(args, n_points: int) -> tuple[Mesh, tuple[str, ...]]:
    """Pick a 2-D device mesh ('x', 'y'), matching gw_jax's ISDF mesh.

    Single-device collapses to a 1×1 2-D mesh so the downstream pipeline
    (which uses ``load_centroids_band_chunked`` and friends) only has one
    codepath to worry about.
    """
    from common.wfn_transforms import process_local_mesh

    devices = jax.devices()
    n_dev = len(devices)
    multi_host = jax.process_count() > 1

    # Single-device fallbacks use THIS PROCESS's device, never
    # ``jax.devices()[0]`` (the global list's first entry = process 0's device
    # on every rank, a mesh no other rank can compute on).  At P=1 the two are
    # the same object, so single-process behaviour is byte-identical.
    if args.no_shard or n_dev < 2:
        return process_local_mesh(), ("x", "y")

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
        return process_local_mesh(), ("x", "y")

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
    avec_jax = jnp.asarray(avec_ang, dtype=jnp.float64)

    n_points = int(np.prod(fft_grid))
    mesh, mesh_axis = _build_mesh(args, n_points)

    # Decide orbit vs non-orbit.  Unless --no-orbit, build the closure sym
    # group with charge-density point-group recovery, then enable orbit mode
    # when it has more than the identity.  Gating on the *recovered* group
    # (not raw ``wfn.ntran``) makes orbit closure the default for any WFN
    # whose density carries point-group symmetry — including reduced WFNs
    # (non-collinear SOC stores only {E, σ_h}; a nosym run stores {E}) whose
    # stored ntran understates the crystal symmetry and would otherwise leave
    # ⟨nk|V_H|nk⟩ C3-broken across the k-star.
    R = Rinv = tau = None
    n_sym = 1
    orbit_aware = False
    if not args.no_orbit:
        from .orbit_syms import build_real_space_syms
        R, Rinv, tau = build_real_space_syms(
            wfn, sym, charge_density=charge_density)
        n_sym = int(R.shape[0])
        orbit_aware = args.orbit or n_sym > 1
    if not orbit_aware:
        R = Rinv = tau = None
        n_sym = 1

    # ── k-means weight ───────────────────────────────────────────────────
    # WHY band_range EXISTS: occupied-only ρ(r) is entirely inside the slab,
    # so the weighted k-means places ZERO centroids in the vacuum and the
    # vacuum-localized far-conduction states have no quadrature support —
    # ⟨nk|V_H|nk⟩ (a pure centroid sum) is then sign-wrong and the whole
    # error lands on Vxc = E_dft − kin_ion − V_H.
    with timing.section("setup.weight"):
        if args.centroid_weight is None:      # scalar defaults to band_range
            args.centroid_weight = ("band_range" if args.density_mode == "scalar"
                                    else "charge_density")
        if args.centroid_weight == "band_range":
            if args.density_mode != "scalar":
                raise ValueError(
                    "--centroid-weight band_range applies to the scalar "
                    "(charge) channel; --density-mode current already "
                    "weights by its own occupied-state current.")
            if args.weight_bands is not None:
                b_lo, b_hi = (int(v) for v in args.weight_bands.split(":"))
            else:
                _nv, _nc = _resolve_sigma_window(args, wfn)
                b_lo, b_hi = 0, _nv + _nc
            # τ=0 is required for the plain-index grid symmetrization; the
            # recovered density point group is symmorphic by construction,
            # a WFN group may not be.
            _ops = (np.asarray(Rinv) if Rinv is not None
                    and np.allclose(np.asarray(tau), 0.0, atol=1e-8) else None)
            print(f"k-means weight: band_range Σ_{{n∈[{b_lo},{b_hi})}} "
                  f"Σ_k w_k|ψ_nk|²"
                  f"{'' if _ops is None else f' (symmetrized, {len(_ops)} ops)'}")
            from .charge_density import rho_from_band_range
            weight = rho_from_band_range(wfn, (b_lo, b_hi), sym_ops=_ops)
            weight_label = (f"band-range density Σ_{{n∈[{b_lo},{b_hi})}} "
                            f"Σ_k w_k|ψ_nk(r)|²")
        else:
            weight = charge_density
            weight_label = ("scalar charge density ρ(r)"
                            if args.density_mode == "scalar" else
                            "Gordon-decomposed Pauli current "
                            "Σ_{n,k,i}(j^Gordon_{n,k,i}(r))²")

    rho_jax = jnp.asarray(weight, dtype=jnp.float64)
    if args.rho_power != 1.0:
        # Clip to non-negative before power (QE iFFT can leave tiny < 0
        # noise) and tell the user we're using a non-default exponent.
        rho_jax = jnp.maximum(rho_jax, 0.0) ** float(args.rho_power)
        print(f"k-means weight: (weight)^{args.rho_power:g} "
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
            # Pass Rinv = inv(mtrx).  BGW r-action is r' = Rinv·r + τ;
            # this matches the direction used by compute_centroid_sym_perm
            # and validate_atomic_symmetries.  No-op vs forward S on
            # symmorphic systems (CrI3, MoS2); critical for Si Fd-3m.
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
                    centroids_frac, fft_grid, rho=weight,
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
        # Same σ window the band_range weight uses (one resolver).
        _n_val_eff, _n_cond_eff = _resolve_sigma_window(args, wfn)
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
        _rank_tol = float(os.environ.get("LORRAX_CENTROID_RANK_TOL", "0.01"))
        _rank_floor = int(np.ceil((1.0 - _rank_tol) * n_orbit_keep))
        if int(rank) < _rank_floor:
            raise SystemExit(
                f"\nFATAL: pivoted-Cholesky rank deficiency — the candidate "
                f"pool cannot supply the independence you asked for.\n"
                f"  requested : {n_orbit_keep} "
                f"{'orbits' if orbit_id_arr is not None else 'points'}\n"
                f"  achieved  : {rank}   ({100.0 * rank / max(1, n_orbit_keep):.1f}%"
                f", floor {_rank_floor} at tol {_rank_tol:g})\n"
                f"  prune window: left=(0,{_n_val_eff}) right=(0,{_max_band}) "
                f"[{args.prune_window}]\n"
                f"The centroids that WOULD have been written are padded with "
                f"numerically-null directions; an ISDF built on them will\n"
                f"under-resolve Σ and can be wrong by electron-volts without "
                f"failing any other gate.  Fix one of:\n"
                f"  * widen the prune window  — --prune-n-cond <ncond of your "
                f"deck>  (this is the usual cause; the default is now the\n"
                f"    full WFN conduction window, so a shortfall here means "
                f"the window was narrowed explicitly)\n"
                f"  * use --prune-window vc_x_vc to include c×c pair "
                f"densities (needed when ncond >> nval)\n"
                f"  * raise --oversample so the candidate pool is richer, or "
                f"lower N so you ask for fewer directions\n"
                f"To override deliberately (NOT recommended for production): "
                f"LORRAX_CENTROID_RANK_TOL=<fraction>.\n")
        print(f"  [rank gate] {rank}/{n_orbit_keep} directions certified "
              f"(floor {_rank_floor}, tol {_rank_tol:g}) — PASS")

    # Default suffix follows --density-mode unless the user overrode it.
    out_suffix = (args.out_suffix
                  if args.out_suffix is not None
                  else ("" if args.density_mode == "scalar" else "_current"))
    out_file = f"centroids_frac_{n_unique}{out_suffix}.txt"
    header = (
        f"x y z (snapped to FFT grid {fft_grid}, {n_unique} unique)\n"
        f"density: {args.density_mode}  centroid-weight: {args.centroid_weight}\n"
        f"weight: {weight_label}\n"
        f"intended channels: "
        f"{'γ̃^0 (charge) ISDF' if args.density_mode == 'scalar' else 'γ̃^{1,2,3} (current) ISDF'}"
    )
    # ONE writer.  Every rank used to reach this savetxt on the same shared
    # path.  It survived P=16 only because all ranks write identical bytes —
    # which is precisely the latent form of the bug that DID bite at P=64 in
    # ``bse_io.write_eigenvectors_stream`` (64 concurrent h5py creators,
    # rc=1 plus a structurally valid file; wk_REL S4.8).  ``centroids_snapped``
    # is a pure function of the WFN + seed + candidate list and is identical on
    # every rank, so rank 0's file is the file any rank would have written.
    # No collective below this point, so gating cannot deadlock.
    if jax.process_index() == 0:
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
