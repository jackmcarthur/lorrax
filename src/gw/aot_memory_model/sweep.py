"""Sweep driver: run AOT lowering over a DoE and record memory_analysis.

Standard invocation:

    lxrun python3 -m gw.aot_memory_model.sweep \\
        --kernel chi0_tau_step --preset si444_60Ry --tag default

See ``presets.py`` for the baseline configurations per kernel.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .core import (
    MeshSpec, SysDims, Knobs,
    aot_measure, fit_nnls, get_kernel, save_samples, save_fit, _sample_to_dict,
    load_samples, samples_to_fit_input,
)
from .doe import build_doe_axes
from . import presets


def _make_mesh(mesh_spec: MeshSpec):
    import jax
    devs = jax.devices()
    total = mesh_spec.p_x * mesh_spec.p_y
    if len(devs) < total:
        raise RuntimeError(
            f"Need {total} devices for mesh {mesh_spec}, have {len(devs)}")
    from jax.sharding import Mesh
    arr = np.array(devs[:total]).reshape(mesh_spec.p_x, mesh_spec.p_y)
    return Mesh(arr, ("x", "y"))


def _is_rank0() -> bool:
    import jax
    try:
        return jax.process_index() == 0
    except Exception:
        return True


def run_sweep(kernel_name: str, points, *, tag: str = "default", verbose: bool = True):
    kernel = get_kernel(kernel_name)
    rank0 = _is_rank0()
    samples = []
    for i, (sys, knobs, mesh_spec) in enumerate(points):
        t0 = time.perf_counter()
        mesh = _make_mesh(mesh_spec)
        try:
            # All ranks call aot_measure: the result is a deterministic
            # compiler estimate.  Each rank computes it independently; only
            # rank 0 keeps/writes the sample.
            meas = aot_measure(kernel, sys, knobs, mesh)
        except Exception as e:
            if verbose and rank0:
                print(f"[{i+1}/{len(points)}] FAILED {sys} {knobs} {mesh_spec}: {e}",
                      flush=True)
            continue
        elapsed = time.perf_counter() - t0
        if rank0:
            samples.append(_sample_to_dict(sys, knobs, mesh_spec, meas))
            if verbose:
                gb = meas["total"] / 1e9
                print(f"[{i+1}/{len(points)}] mesh={mesh_spec.p_x}x{mesh_spec.p_y} "
                      f"nk={sys.n_k} μ={sys.n_rmu} nbv={sys.n_b_v} nbc={sys.n_b_c} "
                      f"-> {gb:.3f} GB (lower {meas['t_lower_s']:.1f}s "
                      f"compile {meas['t_compile_s']:.1f}s total {elapsed:.1f}s)",
                      flush=True)
    if rank0:
        save_path = save_samples(kernel_name, samples, tag=tag)
        if verbose:
            print(f"\nWrote {len(samples)} samples -> {save_path}", flush=True)
    return samples


def fit_from_saved(kernel_name: str, tag: str = "default", *, save: bool = True):
    """Rank 0 only — loads JSON, NNLS-fits, writes fit artifact."""
    if not _is_rank0():
        return None
    kernel = get_kernel(kernel_name)
    samples = load_samples(kernel_name, tag=tag)
    tuples = samples_to_fit_input(kernel, samples)
    fit = fit_nnls(kernel, tuples)
    if save:
        save_fit(fit, tag=tag)
    print(f"\nFit for kernel={fit.kernel}: n_samples={fit.n_samples}")
    print(f"  intercept = {fit.intercept:.3e} bytes")
    for name, coef in zip(fit.feature_names, fit.coefs):
        print(f"  β[{name:12s}] = {coef:.3f}")
    print(f"  residual RMS = {fit.residual_rms:.3e} bytes "
          f"({fit.residual_rms / 1e9:.3f} GB)")
    return fit


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, help="e.g. chi0_tau_step")
    ap.add_argument("--preset", default="si444_60Ry",
                    help="baseline preset name (see presets.py)")
    ap.add_argument("--tag", default="default", help="artifacts file suffix")
    ap.add_argument("--mode", choices=("sweep", "fit", "both"), default="both")
    ap.add_argument("--dry-run", action="store_true",
                    help="print DoE points and exit")
    args = ap.parse_args(argv)

    from . import kernels  # noqa: F401  — ensures registry populated

    points_fn = getattr(presets, f"points_{args.kernel}", None)
    if points_fn is None:
        raise SystemExit(f"No preset point factory for kernel {args.kernel}. "
                         f"Add ``points_{args.kernel}`` to presets.py.")
    points = points_fn(args.preset)

    if args.dry_run:
        for i, (s, k, m) in enumerate(points):
            print(f"[{i}] sys={s} knobs={k} mesh={m}")
        return 0

    if args.mode in ("sweep", "both"):
        _init_jax_for_sweep()
        run_sweep(args.kernel, points, tag=args.tag)
    if args.mode in ("fit", "both"):
        fit_from_saved(args.kernel, tag=args.tag)
    return 0


def _init_jax_for_sweep():
    """Set up JAX just like gw.gw_jax for multi-process + X64."""
    import os
    import jax
    jax.config.update("jax_enable_x64", True)
    # Cray MPICH default: one GPU per rank.  Even when world=1, pinning
    # local_device_ids=[0] matches production's layout.
    try:
        cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        n_local = len(cv.split(",")) if cv else 1
        if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
            jax.distributed.initialize(local_device_ids=list(range(n_local)))
    except Exception:
        pass


if __name__ == "__main__":
    sys_exit = main()
    if sys_exit:
        raise SystemExit(sys_exit)
