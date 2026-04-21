"""Baseline DoE presets per kernel.

Each kernel exposes a ``points_<kernel_name>(preset) -> list`` factory that
returns the list of ``(sys, knobs, mesh)`` points to sweep.
"""
from __future__ import annotations

from dataclasses import replace

from .core import Knobs, MeshSpec, SysDims
from .doe import build_doe_axes


# ---------------------------------------------------------------------------
# chi0_tau_step
# ---------------------------------------------------------------------------

def points_chi0_tau_step(preset: str):
    """DoE for the chi0 _tau_step kernel.

    Factors: n_k, n_rmu, n_b_v, n_b_c, and mesh shape.  Knobs: none (no
    chunking in the current implementation — future work).  Product-only
    check: none at this stage (all the Si μ=2400 buffers scale as μ²,
    so we don't expect a ``μ · n_b`` primitive; band count enters build_G
    but gets contracted out before the chi accumulator).

    Note: at current mem16 scales, a full Si 4×4×4 μ=2400 lower on 4 GPUs
    would allocate ~22 GB temp — but the AOT pipeline never instantiates
    buffers, so the sweep will NOT crash.  We still pin μ to realistic
    values; ``memory_analysis`` scales linearly so extrapolation is fine.
    """
    if preset == "si444_60Ry":
        baseline = SysDims(
            kgrid=(4, 4, 4), n_rmu=2400, n_s=2,
            n_b_v=20, n_b_c=276, n_b=0,
        )
        kgrid_axes = [(2, 2, 2), (4, 4, 2)]  # nk ∈ {8, 32} vs baseline 64
        rmu_axes = [1200, 3600]
    elif preset == "si444_60Ry_lean":
        # Identifiability-focused DoE — drops high-leverage points that
        # trigger rematerialization (μ=3600 saturates >40 GiB on a 2×2 mesh
        # and XLA rewrites the allocation plan, violating linear scaling).
        # Two n_s=1 points are the key lever for breaking Gbuf/chi
        # collinearity: at n_s=1 Gbuf and chi have the same per-device size.
        baseline = SysDims(
            kgrid=(4, 4, 4), n_rmu=2400, n_s=2,
            n_b_v=20, n_b_c=276, n_b=0,
        )
        lean_points = [
            (baseline, Knobs.of(), MeshSpec(2, 2)),
            (replace(baseline, kgrid=(2, 2, 2)), Knobs.of(), MeshSpec(2, 2)),
            (replace(baseline, kgrid=(4, 4, 2)), Knobs.of(), MeshSpec(2, 2)),
            (replace(baseline, n_rmu=1200), Knobs.of(), MeshSpec(2, 2)),
            (replace(baseline, n_b_c=552), Knobs.of(), MeshSpec(2, 2)),
            (replace(baseline, n_s=1), Knobs.of(), MeshSpec(2, 2)),
            (replace(baseline, n_s=1, n_rmu=1200), Knobs.of(), MeshSpec(2, 2)),
            (baseline, Knobs.of(), MeshSpec(1, 4)),
        ]
        return lean_points
    elif preset == "mos2_3x3":
        baseline = SysDims(
            kgrid=(3, 3, 1), n_rmu=1600, n_s=1,
            n_b_v=30, n_b_c=50, n_b=0,
        )
        kgrid_axes = [(2, 2, 1), (4, 4, 1)]
        rmu_axes = [800, 2400]
    elif preset == "si222_tiny":
        # Cheap baseline for dev-loop iteration (~1 s per AOT compile).
        baseline = SysDims(
            kgrid=(2, 2, 2), n_rmu=400, n_s=2,
            n_b_v=8, n_b_c=40, n_b=0,
        )
        kgrid_axes = [(2, 2, 1), (3, 2, 2)]  # nk ∈ {4, 12} vs baseline 8
        rmu_axes = [200, 800]
    else:
        raise ValueError(f"Unknown preset {preset!r} for chi0_tau_step")

    knobs = Knobs.of()
    mesh = MeshSpec(p_x=2, p_y=2)

    # Axis sweep — each axis breaks one primitive's scaling:
    #   - kgrid varies n_k: scales every primitive linearly (product-only
    #     check against μ²)
    #   - n_rmu: μ²  vs μ (psi) — identifies Gbuf vs psi
    #   - n_b_c: μ·nb — identifies psi alone (doesn't touch Gbuf/chi)
    #   - n_s:   n_s² vs n_s (Gbuf vs psi) — BREAKS the chi/Gbuf collinearity
    #   - mesh (1,4) and (4,1): probes p_x/p_y asymmetry in the psi shards
    sys_axes = {
        "kgrid":  kgrid_axes,
        "n_rmu":  rmu_axes,
        "n_b_c":  [max(4, baseline.n_b_c // 2), baseline.n_b_c * 2],
        "n_s":    [1 if baseline.n_s == 2 else 2],  # flip n_s
    }
    mesh_axes = [MeshSpec(1, 4), MeshSpec(4, 1)]

    return build_doe_axes(baseline, knobs, mesh,
                          sys_axes=sys_axes, mesh_axes=mesh_axes)


# ---------------------------------------------------------------------------
# pair_density_traced
# ---------------------------------------------------------------------------

def points_pair_density_traced(preset: str):
    baseline = SysDims(kgrid=(4, 4, 4), n_rmu=2400, n_s=2, n_b=296)
    return [
        (baseline, Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, kgrid=(2, 2, 2)), Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, n_rmu=1200), Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, n_b=148), Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, n_b=592), Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, n_s=1), Knobs.of(), MeshSpec(2, 2)),
        (baseline, Knobs.of(), MeshSpec(1, 4)),
        (baseline, Knobs.of(), MeshSpec(4, 1)),
    ]


# ---------------------------------------------------------------------------
# cct_lr
# ---------------------------------------------------------------------------

def points_cct_lr(preset: str):
    baseline = SysDims(kgrid=(4, 4, 4), n_rmu=2400)
    return [
        (baseline, Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, kgrid=(2, 2, 2)), Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, kgrid=(4, 4, 2)), Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, n_rmu=1200), Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, n_rmu=3200), Knobs.of(), MeshSpec(2, 2)),
        (baseline, Knobs.of(), MeshSpec(1, 4)),
        (baseline, Knobs.of(), MeshSpec(4, 1)),
    ]


# ---------------------------------------------------------------------------
# zct_lr  (knob: chunk_r)
# ---------------------------------------------------------------------------

def points_zct_lr(preset: str):
    baseline = SysDims(kgrid=(4, 4, 4), n_rmu=2400, n_r=12672)
    return [
        (baseline, Knobs.of(chunk_r=12672), MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=6336),  MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=25344), MeshSpec(2, 2)),
        (replace(baseline, kgrid=(2, 2, 2)), Knobs.of(chunk_r=12672), MeshSpec(2, 2)),
        (replace(baseline, n_rmu=1200), Knobs.of(chunk_r=12672), MeshSpec(2, 2)),
        (replace(baseline, n_rmu=3200), Knobs.of(chunk_r=12672), MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=12672), MeshSpec(1, 4)),
    ]


# ---------------------------------------------------------------------------
# solve_q  (knobs: chunk_r, q_chunk)
# ---------------------------------------------------------------------------

def points_solve_q(preset: str):
    baseline = SysDims(kgrid=(4, 4, 4), n_rmu=2400, n_r=12672)
    return [
        (baseline, Knobs.of(chunk_r=12672, q_chunk=1),  MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=6336,  q_chunk=1),  MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=12672, q_chunk=4),  MeshSpec(2, 2)),
        (replace(baseline, kgrid=(2, 2, 2)), Knobs.of(chunk_r=12672, q_chunk=1), MeshSpec(2, 2)),
        (replace(baseline, n_rmu=1200), Knobs.of(chunk_r=12672, q_chunk=1), MeshSpec(2, 2)),
        (replace(baseline, n_rmu=3200), Knobs.of(chunk_r=12672, q_chunk=1), MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=12672, q_chunk=1), MeshSpec(1, 4)),
    ]


# ---------------------------------------------------------------------------
# vq_mu_chunk  — single-GPU, kgrid = FFT box (knob: mu_chunk)
# ---------------------------------------------------------------------------

def points_vq_mu_chunk(preset: str):
    baseline = SysDims(kgrid=(48, 48, 48), n_rmu=2400, n_r=48 ** 3)
    mesh = MeshSpec(1, 1)
    return [
        (baseline, Knobs.of(mu_chunk=128),  mesh),
        (baseline, Knobs.of(mu_chunk=256),  mesh),
        (baseline, Knobs.of(mu_chunk=512),  mesh),
        (baseline, Knobs.of(mu_chunk=1024), mesh),
        (replace(baseline, kgrid=(36, 36, 36), n_r=36 ** 3),
         Knobs.of(mu_chunk=128), mesh),
        (replace(baseline, kgrid=(60, 60, 60), n_r=60 ** 3),
         Knobs.of(mu_chunk=128), mesh),
    ]


# ---------------------------------------------------------------------------
# sigma_kij
# ---------------------------------------------------------------------------

def points_sigma_kij(preset: str):
    baseline = SysDims(kgrid=(4, 4, 4), n_rmu=2400, n_s=2, n_b=40)
    # 4 n_s=1 points confirm that the 2.67/2.33 Gmid/Vmid split is not
    # a DoE collinearity artifact — it's XLA's actual buffer scheduling.
    # Adding more n_s=1 points leaves the fit unchanged.  The
    # constraint structure is 4α+β (n_s=2) and α+β (n_s=1), uniquely
    # determined with 2 data points; extra points add robustness but
    # don't shift the answer.
    return [
        (baseline, Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, kgrid=(2, 2, 2)), Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, n_rmu=1200), Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, n_b=20), Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, n_b=80), Knobs.of(), MeshSpec(2, 2)),
        # n_s=1 axis, multiple points so Gmid/Vmid could separate *in
        # principle*; they don't because the fit is already identified.
        (replace(baseline, n_s=1), Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, n_s=1, n_rmu=1200), Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, n_s=1, n_b=80), Knobs.of(), MeshSpec(2, 2)),
        (replace(baseline, n_s=1, kgrid=(2, 2, 2)), Knobs.of(), MeshSpec(2, 2)),
        (baseline, Knobs.of(), MeshSpec(1, 4)),
        (baseline, Knobs.of(), MeshSpec(4, 1)),
    ]


# ---------------------------------------------------------------------------
# load_psi_rchunk_fft  (knobs: chunk_r, nb_pad)
# ---------------------------------------------------------------------------

def points_load_psi_rchunk_fft(preset: str):
    # FFT grid is kgrid here (we repurpose SysDims.kgrid as the real-space
    # FFT grid; n_r = prod(kgrid)).  System has Si 4x4x4 64 k-points.
    # DoE sweeps k_chunk × band_chunk × chunk_r + system dims (n_s, n_r).
    baseline = SysDims(kgrid=(24, 24, 24), n_rmu=0, n_s=2,
                       n_b=16, n_r=24 ** 3)
    # sys.n_k is derived from kgrid=(24,24,24) → 13824, way bigger than we
    # want for a per-call jit.  We override via k_chunk knob instead.
    # So all DoE points specify (k_chunk, band_chunk).
    return [
        # --- baseline: k=64 (Si 4x4x4-like), b=16 ---
        (baseline, Knobs.of(k_chunk=64, band_chunk=16, chunk_r=3456), MeshSpec(2, 2)),
        # --- vary band_chunk at fixed k ---
        (baseline, Knobs.of(k_chunk=64, band_chunk=8,  chunk_r=3456), MeshSpec(2, 2)),
        (baseline, Knobs.of(k_chunk=64, band_chunk=32, chunk_r=3456), MeshSpec(2, 2)),
        # --- vary k_chunk at fixed band ---
        (baseline, Knobs.of(k_chunk=16, band_chunk=16, chunk_r=3456), MeshSpec(2, 2)),
        (baseline, Knobs.of(k_chunk=32, band_chunk=16, chunk_r=3456), MeshSpec(2, 2)),
        # --- vary chunk_r ---
        (baseline, Knobs.of(k_chunk=64, band_chunk=16, chunk_r=1728), MeshSpec(2, 2)),
        (baseline, Knobs.of(k_chunk=64, band_chunk=16, chunk_r=6912), MeshSpec(2, 2)),
        # --- vary n_s (for spin scaling) ---
        (replace(baseline, n_s=1),
            Knobs.of(k_chunk=64, band_chunk=16, chunk_r=3456), MeshSpec(2, 2)),
        # --- vary kgrid / n_r ---
        (replace(baseline, kgrid=(32, 32, 32), n_r=32**3),
            Knobs.of(k_chunk=64, band_chunk=16, chunk_r=3456), MeshSpec(2, 2)),
        # --- mesh asymmetry ---
        (baseline, Knobs.of(k_chunk=64, band_chunk=16, chunk_r=3456), MeshSpec(1, 4)),
        (baseline, Knobs.of(k_chunk=64, band_chunk=16, chunk_r=3456), MeshSpec(4, 1)),
    ]


# ---------------------------------------------------------------------------
# load_psi_rchunk_reshard  (knobs: chunk_r, nb_pad)
# ---------------------------------------------------------------------------

def points_load_psi_rchunk_reshard(preset: str):
    baseline = SysDims(kgrid=(4, 4, 4), n_rmu=0, n_s=2,
                       n_b=296, n_r=24 ** 3)
    # DoE: vary k_chunk × band_chunk × chunk_r × mesh.  Sys.n_k=64 at
    # baseline; k_chunk knob controls how many of those each jit call sees.
    return [
        (baseline, Knobs.of(k_chunk=64, band_chunk=32, chunk_r=3456), MeshSpec(2, 2)),
        (baseline, Knobs.of(k_chunk=64, band_chunk=16, chunk_r=3456), MeshSpec(2, 2)),
        (baseline, Knobs.of(k_chunk=32, band_chunk=32, chunk_r=3456), MeshSpec(2, 2)),
        (baseline, Knobs.of(k_chunk=16, band_chunk=32, chunk_r=3456), MeshSpec(2, 2)),
        (baseline, Knobs.of(k_chunk=64, band_chunk=32, chunk_r=1728), MeshSpec(2, 2)),
        (baseline, Knobs.of(k_chunk=64, band_chunk=32, chunk_r=6912), MeshSpec(2, 2)),
        (replace(baseline, n_s=1),
            Knobs.of(k_chunk=64, band_chunk=32, chunk_r=3456), MeshSpec(2, 2)),
        (baseline, Knobs.of(k_chunk=64, band_chunk=32, chunk_r=3456), MeshSpec(1, 4)),
        (baseline, Knobs.of(k_chunk=64, band_chunk=32, chunk_r=3456), MeshSpec(4, 1)),
    ]


# ---------------------------------------------------------------------------
# fit_one_rchunk  — composite driver jit
#
# Memory features that scale independently:
#   chunk_r     -> Pacc / PrBc / psiBcY / (all r-scale primitives)
#   band_chunk  -> psiBc / psiBcY (per-call temps; NOT psi_cent nor psiG_total)
#   n_b         -> psi_cent, psiG_total (centroid + cache volume)
#   n_rmu       -> Pacc, PrBc, psi_cent, L_q
#   kgrid       -> n_k, n_r
#   mesh shape  -> all primitives (p_x × p_y distribution)
# Goal: separate the above via one-at-a-time axis sweep around MoS2 3×3.
# ---------------------------------------------------------------------------

def points_fit_one_rchunk(preset: str):
    """Driver-level r-chunk-body AOT DoE."""
    if preset == "mos2_ortho":
        # Orthogonal DoE for the post-cleanup primitives — every primitive
        # varies independently on at least one axis so NNLS β's come out
        # as the small-integer counts seen in the zct_lr / cct_lr fits
        # rather than the 1.65 / 5.02 dumps my first DoE produced.
        #
        # Axes that each primitive is identifiable on:
        #   T_pair       ∝ n_k·n_rmu·cr/P       — cr axis, n_rmu axis, kgrid axis
        #   T_psiG_cache ∝ n_k·n_b·n_s·n_r/P    — n_b, n_r (via fft_grid), n_s
        #   T_psiG_bc    ∝ n_k·bc·n_s·n_r/P     — bc axis at fixed n_b
        #   T_psiY_bc    ∝ n_k·bc·n_s·cr/p_y    — bc·cr cross, mesh asymmetry
        #   T_centroid   ∝ n_k·n_rmu·n_b·n_s/p_x — n_rmu at fixed n_r, mesh
        #   T_Lq_*       ∝ n_q·n_rmu²/P or /1   — n_rmu quadratic (vs linear)
        base = SysDims(
            kgrid=(3, 3, 1), fft_grid=(80, 72, 8),
            n_rmu=320, n_s=1, n_b=40, n_r=80 * 72 * 8,
        )
        kn0 = Knobs.of(chunk_r=10000, band_chunk=20)
        mesh0 = MeshSpec(2, 2)

        pts = [(base, kn0, mesh0)]  # baseline

        def P(s=None, k=None, m=None):
            return (s if s is not None else base,
                    k if k is not None else kn0,
                    m if m is not None else mesh0)

        # One-at-a-time axis sweeps
        pts += [  # n_rmu
            P(replace(base, n_rmu=160)),
            P(replace(base, n_rmu=640)),
        ]
        pts += [  # n_b (40 baseline)
            P(replace(base, n_b=20)),
            P(replace(base, n_b=80)),
        ]
        pts += [  # n_s (1 baseline)
            P(replace(base, n_s=2)),
        ]
        pts += [  # chunk_r
            P(k=Knobs.of(chunk_r=5000, band_chunk=20)),
            P(k=Knobs.of(chunk_r=46080, band_chunk=20)),
        ]
        pts += [  # band_chunk
            P(k=Knobs.of(chunk_r=10000, band_chunk=8)),
            P(k=Knobs.of(chunk_r=10000, band_chunk=40)),
        ]
        pts += [  # kgrid → varies n_k (n_q collinear)
            P(replace(base, kgrid=(2, 2, 1))),
            P(replace(base, kgrid=(4, 4, 1))),
        ]
        pts += [  # mesh asymmetry at fixed total_devices=4
            P(m=MeshSpec(1, 4)),
            P(m=MeshSpec(4, 1)),
        ]
        pts += [  # fft_grid → varies n_r at fixed n_rmu; breaks
                 # psiG_cache vs centroid collinearity cleanly
            P(replace(base, fft_grid=(40, 72, 8), n_r=40 * 72 * 8)),
            P(replace(base, fft_grid=(80, 72, 16), n_r=80 * 72 * 16)),
        ]

        # Cross-terms — probe product-only interactions that might
        # escape the one-at-a-time sweep
        pts += [
            # big n_rmu × big cr: stresses T_pair identification
            P(replace(base, n_rmu=640),
              Knobs.of(chunk_r=46080, band_chunk=20)),
            # bc × cr: identifies T_psiY_bc
            P(k=Knobs.of(chunk_r=46080, band_chunk=40)),
            P(k=Knobs.of(chunk_r=5000, band_chunk=8)),
            # n_s × n_b: scales anything with n_s·n_b jointly
            P(replace(base, n_s=2, n_b=80)),
            # mesh × n_rmu: stresses centroid mesh dependence
            P(replace(base, n_rmu=640), m=MeshSpec(4, 1)),
        ]
        return pts

    if preset == "mos2_3x3":
        # Realistic per-r-chunk shape at MoS2 3×3.  fft_grid=(80,72,8)
        # → n_r=46080 matching the production smoke test.  Those axes are
        # all divisible by 2 so the (2×2) mesh partitions cleanly.
        baseline = SysDims(
            kgrid=(3, 3, 1), fft_grid=(80, 72, 8),
            n_rmu=640, n_s=2, n_b=80, n_r=80 * 72 * 8,
        )
        chunk_r_axes  = [5000, 20000, 46080]
        band_chunk_axes = [8, 32]
    elif preset == "si444_60Ry":
        # Si 4×4×4 60 Ry — fft_grid=24³ = 13824 per production runs.
        baseline = SysDims(
            kgrid=(4, 4, 4), fft_grid=(24, 24, 24),
            n_rmu=2400, n_s=2, n_b=296, n_r=24 ** 3,
        )
        chunk_r_axes  = [1728, 3456, 6912]
        band_chunk_axes = [16, 32]
    else:
        raise ValueError(f"Unknown preset {preset!r} for fit_one_rchunk")

    base_knobs = Knobs.of(chunk_r=chunk_r_axes[1], band_chunk=band_chunk_axes[0])
    mesh = MeshSpec(2, 2)

    sys_axes = {
        "n_rmu":  [baseline.n_rmu // 2, baseline.n_rmu * 2],
        "n_b":    [max(8, baseline.n_b // 2), baseline.n_b * 2],
        "n_s":    [1 if baseline.n_s == 2 else 2],
    }
    knob_axes = {
        "chunk_r":    chunk_r_axes,
        "band_chunk": band_chunk_axes,
    }
    mesh_axes = [MeshSpec(1, 4), MeshSpec(4, 1)]

    return build_doe_axes(baseline, base_knobs, mesh,
                          sys_axes=sys_axes, knob_axes=knob_axes,
                          mesh_axes=mesh_axes)


# ---------------------------------------------------------------------------
# slab_write  (knob: chunk_r)
# ---------------------------------------------------------------------------

def points_slab_write(preset: str):
    baseline = SysDims(kgrid=(4, 4, 4), n_rmu=2400, n_r=12672)
    return [
        (baseline, Knobs.of(chunk_r=12672), MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=6336),  MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=25344), MeshSpec(2, 2)),
        (replace(baseline, n_rmu=1200), Knobs.of(chunk_r=12672), MeshSpec(2, 2)),
        (replace(baseline, n_rmu=3200), Knobs.of(chunk_r=12672), MeshSpec(2, 2)),
        (replace(baseline, kgrid=(2, 2, 2)), Knobs.of(chunk_r=12672), MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=12672), MeshSpec(1, 4)),
    ]
