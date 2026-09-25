"""GEMM-vs-FFT crossover sweep for ``common.fourier_plan`` (one GPU, complex128).

Every arm is one jitted executable that applies the transform ``CHAIN`` times
in sequence (a round trip for the restricted arms), so kernel launches are
timed as they run inside a production jit, not as Python dispatches.  It is
timed with ``block_until_ready`` after warm-up, ``inner`` back-to-back calls per
repetition; the reported ``us`` is per transform, the median over ``REPS``
repetitions, with min/max.  ``pass`` is one HBM read+write of the input array
at the measured copy bandwidth: the floor a transform stage cannot beat.

Parts (``--part``):
  1d      full N→N on one axis, last axis (contiguous lines) and first axis
          (batch minor), library FFT vs stored-matrix GEMM, three batches.
  comp    full 2-D/3-D composites: one multidimensional jnp.fft vs all-GEMM.
  sparse  one restricted axis (K = N/2, N/4): round trips K→N→K, the plan's FFT
          arm (gather embed + FFT, FFT + take) vs its GEMM arm.
  sparse_comp  2-D/3-D sphere-box round trips: box FFT, staged FFTs, GEMM
          first stage, all-GEMM.
Output: one JSON object per arm on stdout (and ``--out`` file).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
# Production XLA flags (the runtime's GPU autotune level), set before jax loads.
from runtime import _gpu_is_present, set_default_xla_gpu_autotune  # noqa: E402
set_default_xla_gpu_autotune(platform="gpu" if _gpu_is_present() else "cpu")

import numpy as np
import jax
import jax.numpy as jnp

from common import fourier_plan
from common.fourier_plan import LocalFourierPlan

REPS = 7
CHAIN = 16
SIZES = (list(range(2, 33)) + [36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 90, 96, 100, 108,
                               120, 125, 128, 135, 144, 150, 160, 180, 192, 200, 216, 225,
                               240, 250, 256])
GEMM, FFT = "__gemm__", "__fft__"
fourier_plan.GEMM_CROSSOVER[GEMM] = (range(1 << 30),) * 2


def chain(f, reps=CHAIN):
    def run(x):
        # The barrier keeps XLA from cancelling layout transposes or fusing
        # stages across iterations: each transform is priced on its own.
        for _ in range(reps):
            x = jax.lax.optimization_barrier(f(x))
        return x
    return jax.jit(run)


def timeit(f, x, target_s=0.03, per=CHAIN):
    y = f(x)
    y.block_until_ready()
    y = f(x)
    y.block_until_ready()
    t0 = time.perf_counter()
    f(x).block_until_ready()
    one = max(time.perf_counter() - t0, 1e-6)
    inner = int(min(400, max(1, target_s / one)))
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for _ in range(inner):
            y = f(x)
        y.block_until_ready()
        ts.append((time.perf_counter() - t0) / inner / per * 1e6)
    return {"us": statistics.median(ts), "min": min(ts), "max": max(ts), "inner": inner}


def crandn(shape, seed=0):
    r = np.random.default_rng(seed)
    return jnp.asarray(r.standard_normal(shape) + 1j * r.standard_normal(shape))


_BW = {}


def pass_us(x):
    """One read+write of ``x`` at the copy bandwidth measured on 1 GiB."""
    if not _BW:
        big = crandn((1 << 26,))
        t = timeit(jax.jit(lambda v: v * (1.0 + 0.5j)), big, per=1)["us"]
        _BW["Bps"] = 2 * big.nbytes / (t * 1e-6)
        del big
    return 2 * x.nbytes / _BW["Bps"] * 1e6


def emit(rec, fh):
    line = json.dumps(rec)
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def plan(extents, axes, kind, **kw):
    return LocalFourierPlan(extents, axes, sign=-1, norm="backward", device_kind=kind, **kw)


def part_1d(fh):
    for n in SIZES:
        for lines in (1_000, 10_000, 100_000):
            for layout in ("last", "first"):
                shape = (lines, n) if layout == "last" else (n, lines)
                ax = -1 if layout == "last" else 0
                x = crandn(shape)
                rec = {"part": "1d", "n": n, "lines": lines, "layout": layout,
                       "pass": pass_us(x)}
                rec["fft"] = timeit(chain(plan((n,), (ax,), FFT)), x)
                rec["gemm"] = timeit(chain(plan((n,), (ax,), GEMM)), x)
                rec["ratio_gemm_over_fft"] = rec["gemm"]["us"] / rec["fft"]["us"]
                emit(rec, fh)
                del x


COMPOSITES = (
    # (extents, batch): small k-grids (batch = μ-like lines) and plane boxes
    [((n, n, n), b) for n, b in [(2, 100_000), (3, 30_000), (4, 20_000), (5, 8_000),
                                 (6, 5_000), (7, 3_000), (8, 2_000), (9, 1_400), (10, 1_000),
                                 (11, 800), (12, 600), (13, 450), (14, 360), (16, 250),
                                 (17, 200), (18, 170), (19, 150), (20, 125), (23, 80),
                                 (24, 70), (32, 30)]]
    + [((n, n), b) for n, b in [(4, 60_000), (6, 30_000), (8, 16_000), (12, 7_000),
                                (16, 4_000), (24, 2_000), (27, 1_400), (32, 1_000),
                                (40, 600), (45, 500), (54, 2_000), (54, 200), (60, 300),
                                (64, 250), (72, 1_000), (72, 200), (80, 1_000), (80, 150),
                                (96, 100), (128, 60)]]
)


def part_comp(fh):
    for extents, batch in COMPOSITES:
        d = len(extents)
        axes = tuple(range(1, d + 1))
        x = crandn((batch,) + extents)
        rec = {"part": "comp", "extents": extents, "batch": batch, "pass": pass_us(x)}
        rec["fft"] = timeit(chain(plan(extents, axes, FFT)), x)
        rec["gemm"] = timeit(chain(plan(extents, axes, GEMM)), x)
        rec["ratio_gemm_over_fft"] = rec["gemm"]["us"] / rec["fft"]["us"]
        emit(rec, fh)
        del x


def _centred(n, k):
    return np.arange(-(k // 2), k - k // 2) % n


def _seq(fs, x):
    for f in fs:
        x = f(x)
    return x


def _round_trip(f_in, f_out):
    return lambda xc: f_out(f_in(xc))


def _sparse_arms(fh):
    def arms(extents, axes, sup, xc, rec):
        for kind, key in ((FFT, "fft"), (GEMM, "gemm")):
            rec[key] = timeit(chain(_round_trip(plan(extents, axes, kind, in_support=sup),
                                                plan(extents, axes, kind, out_support=sup))), xc)
        rec["ratio_gemm_over_fft"] = rec["gemm"]["us"] / rec["fft"]["us"]
        emit(rec, fh)
    return arms


def part_sparse(fh):
    """Round trips compact → full → compact through the plan (in_support, then
    out_support), FFT arm (gather-embed + FFT, FFT + take) against GEMM arm.
    ``us`` is per round trip."""
    arms = _sparse_arms(fh)

    for n in SIZES:
        if n < 4:
            continue
        for frac in (2, 4):
            k = max(1, n // frac)
            for lines in (10_000, 100_000):
                xc = crandn((lines, k))
                arms((n,), (1,), {1: _centred(n, k)}, xc,
                     {"part": "sparse1d", "n": n, "k": k, "lines": lines,
                      "pass_full": pass_us(xc) * n / k})


def part_sparse_comp(fh):
    """Composite sphere-box geometry, K = N/2 on every axis: box FFT (gather
    embed + one multidimensional FFT + take), tight-box staged FFTs, a GEMM
    first stage on the minor axis, and all-GEMM.  ``us`` is per round trip."""
    arms = _sparse_arms(fh)
    for extents, batch in [((12, 12, 12), 500), ((16, 16, 16), 200), ((24, 24), 5_000),
                           ((32, 32), 3_000), ((24, 24, 24), 64), ((32, 32, 32), 32),
                           ((48, 48, 48), 8),
                           ((54, 54), 27_648), ((54, 54), 2_000), ((72, 72), 1_000),
                           ((80, 80), 12_000), ((80, 80), 1_000), ((96, 96, 96), 4),
                           ((128, 128), 500), ((64, 64, 64), 8)]:
        d = len(extents)
        axes = tuple(range(1, d + 1))
        sup = {a: _centred(n, n // 2) for n, a in zip(extents, axes)}
        xc = crandn((batch,) + tuple(n // 2 for n in extents))
        rec = {"part": "sparse_comp", "extents": extents, "batch": batch,
               "pass_full": pass_us(xc) * 2 ** d}
        # Tight-box staging with library FFTs: one axis at a time on the
        # growing box (last axis first), the reverse order on the way back.
        ins = [plan((n,), (a,), FFT, in_support={a: sup[a]}) for n, a in zip(extents, axes)]
        outs = [plan((n,), (a,), FFT, out_support={a: sup[a]}) for n, a in zip(extents, axes)]
        rec["staged_fft"] = timeit(chain(lambda v: _seq(outs, _seq(ins[::-1], v))), xc)
        # A restricted GEMM first stage (last axis), library FFT for the rest.
        g_in = plan(extents[-1:], axes[-1:], GEMM, in_support={axes[-1]: sup[axes[-1]]})
        g_out = plan(extents[-1:], axes[-1:], GEMM, out_support={axes[-1]: sup[axes[-1]]})
        r_sup = {a: sup[a] for a in axes[:-1]}
        f_in = plan(extents[:-1], axes[:-1], FFT, in_support=r_sup)
        f_out = plan(extents[:-1], axes[:-1], FFT, out_support=r_sup)
        rec["gemm_first"] = timeit(chain(lambda v: g_out(f_out(f_in(g_in(v))))), xc)
        arms(extents, axes, sup, xc, rec)
        del xc


def part_oneway(fh):
    """One-way sphere→box into row-major output (no round trip to fold the
    layout away): the plane sites and 3-D boxes, K = N/2; the FFT arm, the
    GEMM arm from the natural (B, K, …) layout, and cuFFT alone on a filled box."""
    for extents, batch in [((54, 54), 27_648), ((80, 80), 12_288), ((24, 24), 5_000),
                           ((24, 24, 24), 64), ((32, 32, 32), 32), ((64, 64, 64), 8)]:
        d = len(extents)
        axes = tuple(range(1, d + 1))
        sup = {a: _centred(n, n // 2) for n, a in zip(extents, axes)}
        xc = crandn((batch,) + tuple(n // 2 for n in extents))
        rec = {"part": "oneway", "extents": extents, "batch": batch}
        for kind, key in ((FFT, "fft"), (GEMM, "gemm")):
            rec[key] = timeit(jax.jit(plan(extents, axes, kind, in_support=sup)), xc, per=1)
        rec["ratio_gemm_over_fft"] = rec["gemm"]["us"] / rec["fft"]["us"]
        del xc
        xf = crandn((batch,) + extents)
        rec["fft_only_full_box"] = timeit(jax.jit(plan(extents, axes, FFT)), xf, per=1)
        emit(rec, fh)
        del xf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["1d", "comp", "sparse", "sparse_comp", "oneway"])
    ap.add_argument("--leg", choices=["xla", "ffi"], default=None,
                    help="force the plan's leg (default: the platform's)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.leg == "ffi":
        so = os.environ["LORRAX_FOURIER_PLAN_SO"]
        import ctypes
        lib = ctypes.CDLL(so)
        jax.ffi.register_ffi_target("lorrax_fourier_plan",
                                    jax.ffi.pycapsule(lib.LorraxFourierPlanCudaFfi),
                                    platform="CUDA")
        main.lib = lib
    if a.leg:
        fourier_plan._default_leg = lambda: a.leg
    dev = jax.devices()[0]
    print(json.dumps({"device_kind": dev.device_kind, "platform": dev.platform,
                      "jax": jax.__version__, "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
                      "leg": a.leg or fourier_plan._default_leg()}), flush=True)
    fh = open(a.out, "a") if a.out else None
    {"1d": part_1d, "comp": part_comp, "sparse": part_sparse,
     "sparse_comp": part_sparse_comp, "oneway": part_oneway}[a.part](fh)


if __name__ == "__main__":
    sys.exit(main())
