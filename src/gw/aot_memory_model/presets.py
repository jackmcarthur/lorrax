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
    # FFT grid here is kgrid (we repurpose SysDims.kgrid as the real-space
    # FFT grid; n_r := prod(kgrid)).  Typical Si 4x4x4 60 Ry fft_grid is
    # (24, 24, 24) but we go smaller for reasonable compile wall.
    baseline = SysDims(kgrid=(24, 24, 24), n_rmu=0, n_s=2,
                       n_b=16, n_r=24 ** 3)
    return [
        (baseline, Knobs.of(chunk_r=3456, nb_pad=16),  MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=1728, nb_pad=16),  MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=6912, nb_pad=16),  MeshSpec(2, 2)),
        (replace(baseline, n_b=32, n_r=24**3),
         Knobs.of(chunk_r=3456, nb_pad=32), MeshSpec(2, 2)),
        (replace(baseline, kgrid=(32, 32, 32), n_r=32**3),
         Knobs.of(chunk_r=3456, nb_pad=16), MeshSpec(2, 2)),
        (replace(baseline, n_s=1),
         Knobs.of(chunk_r=3456, nb_pad=16), MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=3456, nb_pad=16), MeshSpec(1, 4)),
        (baseline, Knobs.of(chunk_r=3456, nb_pad=16), MeshSpec(4, 1)),
    ]


# ---------------------------------------------------------------------------
# load_psi_rchunk_reshard  (knobs: chunk_r, nb_pad)
# ---------------------------------------------------------------------------

def points_load_psi_rchunk_reshard(preset: str):
    baseline = SysDims(kgrid=(4, 4, 4), n_rmu=0, n_s=2,
                       n_b=296, n_r=24 ** 3)
    return [
        (baseline, Knobs.of(chunk_r=3456, nb_pad=296),  MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=1728, nb_pad=296),  MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=6912, nb_pad=296),  MeshSpec(2, 2)),
        (replace(baseline, n_b=148),
         Knobs.of(chunk_r=3456, nb_pad=148), MeshSpec(2, 2)),
        (replace(baseline, kgrid=(2, 2, 2)),
         Knobs.of(chunk_r=3456, nb_pad=296), MeshSpec(2, 2)),
        (replace(baseline, n_s=1),
         Knobs.of(chunk_r=3456, nb_pad=296), MeshSpec(2, 2)),
        (baseline, Knobs.of(chunk_r=3456, nb_pad=296), MeshSpec(1, 4)),
        (baseline, Knobs.of(chunk_r=3456, nb_pad=296), MeshSpec(4, 1)),
    ]


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
