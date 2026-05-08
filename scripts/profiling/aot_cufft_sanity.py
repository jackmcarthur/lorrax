"""Sanity check: compare predicted cuFFT scratch + compiled_peak against
observed behavior on CrI3 and MoS2 production V_q FFT shapes.

Run on a GPU-equipped node (login node works on Perlmutter — JAX sees
the visible CUDA device).

Numbers come from existing run logs and the v_q_consolidation /
v_q_memory_model_open reports.

CrI3 6×6×1 80 Ry, 16-GPU 4×4 mesh:
    n_rmu = 1504,  μ_local per rank = 1504/16 = 94
    fft_grid = (75, 75, 200)
    n_G_sphere = 62649
    Per-rank FFT batch at q_chunk Q = Q × 94 (c128)
    Observed:
        Q=8  : compiled_peak 40.84 GB, ran fine, total predicted ≤ 80 GB ✓
        Q=12 : compiled_peak 61.22 GB, historically ran ≤ 80 GB ceiling ✓
        Q=13 : compiled_peak 66.32 GB, cuFFT plan creation FAILED → > 80 GB
        Q=18 : compiled_peak 91.0  GB, JAX OOM → > 80 GB

MoS2 3×3 80 Ry, 4-GPU 2×2 mesh:
    n_rmu = 640,  μ_local per rank = 640/4 = 160
    fft_grid = (60, 60, 200)
    Per-rank FFT batch at q_chunk Q = Q × 160 (c128)
    Observed at Q=9: compiled_peak 3.21 GB, actual XLA peak 2.99 GiB
    → cuFFT scratch is essentially zero on MoS2.

The script prints predicted scratch (from cufftMakePlanMany on jaxlib's
loaded libcufft.so) for each (system, Q) and verifies total = compiled
+ scratch matches the qualitative outcome.
"""

from __future__ import annotations

import sys

# Force x64 so c128 stays c128.
import os as _os
_os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp


import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _scratch_for(transform_shape, batch, dtype="c128"):
    from runtime.aot_memory import FftSpec, _query_one_plan_workspace_bytes
    spec = FftSpec(
        rank=len(transform_shape),
        transform_shape=tuple(transform_shape),
        batch=int(batch),
        dtype=dtype,
        fft_type="FFT",
    )
    return _query_one_plan_workspace_bytes(spec)


def main():
    print("=" * 72)
    print("CrI3 6×6×1 80 Ry, 4×4 mesh — fft (75,75,200) c128, μ_local=94/rank")
    print("=" * 72)

    cri3_compiled_peak_gb = {
        # q_chunk -> compiled_peak GB (from v_q_memory_model_open report
        #            and run logs).  These are *per rank*.
        8:  40.84,
        12: 61.22,
        13: 66.32,
        18: 91.0,    # full-kernel AOT predicted; slope+intercept saw 56.7
    }

    print(f"{'Q':>4} {'batch':>8} {'compiled':>10} {'scratch':>10} "
          f"{'total':>10} {'≤80GB?':>7} {'observed':>20}")
    for q in (8, 12, 13, 18):
        batch = q * 94
        scratch = _scratch_for((75, 75, 200), batch, "c128")
        compiled = cri3_compiled_peak_gb[q] * 1e9
        total = compiled + scratch
        fits = total <= 80e9
        obs = {8: "ran fine", 12: "ran fine",
               13: "cuFFT plan FAIL", 18: "OOM at runtime"}[q]
        print(f"{q:>4} {batch:>8} {compiled/1e9:>9.2f}G "
              f"{scratch/1e9:>9.2f}G {total/1e9:>9.2f}G "
              f"{'YES' if fits else 'NO':>7} {obs:>20}")

    print()
    print("=" * 72)
    print("MoS2 3×3 80 Ry, 2×2 mesh — fft (60,60,200) c128, μ_local=160/rank")
    print("=" * 72)
    print(f"{'Q':>4} {'batch':>8} {'compiled':>10} {'scratch':>10} "
          f"{'total':>10} {'observed XLA':>16}")
    # Q=9 covers all 9 q-points in one batch on MoS2
    for q in (9,):
        batch = q * 160
        scratch = _scratch_for((60, 60, 200), batch, "c128")
        compiled = 3.21 * 1e9   # from earlier MoS2 measurement
        total = compiled + scratch
        obs_gib = 2.99
        print(f"{q:>4} {batch:>8} {compiled/1e9:>9.2f}G "
              f"{scratch/1e9:>9.2f}G {total/1e9:>9.2f}G "
              f"{f'{obs_gib:.2f} GiB':>16}")

    print()
    print("=" * 72)
    print("Linearity check: scratch / batch at fixed transform shape")
    print("=" * 72)
    for shape, label in [((75, 75, 200), "CrI3 box"),
                         ((60, 60, 200), "MoS2 box"),
                         ((128, 128, 128), "power-of-2 cube")]:
        sizes = [(b, _scratch_for(shape, b, "c128")) for b in (50, 100, 200, 500)]
        print(f"  shape={shape} ({label}):")
        for b, s in sizes:
            per = s / b if b else 0
            print(f"    batch={b:>4}  scratch={s/1e6:>10.2f} MB  "
                  f"({per/1e6:>6.3f} MB/batch)")


if __name__ == "__main__":
    sys.exit(main())
