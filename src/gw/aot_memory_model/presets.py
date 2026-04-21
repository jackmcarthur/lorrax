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
