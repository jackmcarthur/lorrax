"""Profile cuSOLVERMp batched potrf/potrs with GWJAX-like call shapes.

This is a benchmark/profiling driver, not a correctness unit test.  It lives
next to the cuSOLVERMp FFI because the goal is to produce Nsight traces that
transfer directly to the in-situ GWJAX calls:

* one process per GPU under ``lxrun``;
* a 2x2 JAX mesh over 4 GPUs;
* batched matrices sharded ``P(None, 'x', 'y')``;
* default MoS2 3x3-ish shape: ``nq=9, n=640, dtype=complex128``;
* the same Python wrappers used by GWJAX.

Typical Perlmutter usage from this repository:

```
export SLURM_JOBID=<one-node allocation>
cd /global/homes/j/jackm/scratchperl/lorrax_sandbox/sources/lorrax_D
LORRAX_NGPU=4 lxrun python3 -u -m ffi.cusolvermp.profile_batched \\
    --mode potrf --nq 9 -n 640 --iters 8 --warmup 2
```

For Nsight Systems, use ``--cuda-profiler-api`` and wrap the command:

```
export OUT=/path/to/nsys
LORRAX_NGPU=4 lxrun bash -lc 'nsys profile --trace=cuda,nvtx,osrt \\
    --capture-range=cudaProfilerApi --capture-range-end=stop \\
    --force-overwrite=true -o "$OUT/potrf_rank${SLURM_PROCID}" \\
    python3 -u -m ffi.cusolvermp.profile_batched \\
    --mode potrf --nq 9 -n 640 --iters 8 --warmup 2 --cuda-profiler-api'
```
"""
from __future__ import annotations

from runtime import set_default_env

set_default_env()

import argparse
import ctypes
import ctypes.util
import os
import statistics
import sys
import time
from contextlib import contextmanager

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from runtime import fallback_to_cpu_if_no_gpu_backend, init_jax_distributed
from ffi.cusolvermp import (
    batched_distributed_cholesky,
    batched_distributed_potrs,
)


def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def _parse_mesh(spec: str) -> tuple[int, int]:
    try:
        px_s, py_s = spec.lower().split("x")
        return int(px_s), int(py_s)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"mesh must look like 2x2, got {spec!r}") from exc


def _dtype(name: str):
    return jnp.complex128 if name == "c128" else jnp.float64


def _load_cudart():
    libname = ctypes.util.find_library("cudart") or "libcudart.so"
    return ctypes.CDLL(libname)


@contextmanager
def cuda_profiler_api(enabled: bool):
    """Bracket only the measured loop for nsys --capture-range=cudaProfilerApi."""
    if not enabled:
        yield
        return
    cudart = _load_cudart()
    start = cudart.cudaProfilerStart
    stop = cudart.cudaProfilerStop
    start.restype = ctypes.c_int
    stop.restype = ctypes.c_int
    jax.block_until_ready(jnp.asarray(0.0))
    multihost_utils.sync_global_devices("before_cuda_profiler_start")
    if start() != 0:
        raise RuntimeError("cudaProfilerStart failed")
    try:
        yield
    finally:
        jax.block_until_ready(jnp.asarray(0.0))
        multihost_utils.sync_global_devices("before_cuda_profiler_stop")
        if stop() != 0:
            raise RuntimeError("cudaProfilerStop failed")


@contextmanager
def trace_annotation(name: str):
    profiler = getattr(jax, "profiler", None)
    trace_cls = getattr(profiler, "TraceAnnotation", None) if profiler else None
    if trace_cls is None:
        yield
    else:
        with trace_cls(name):
            yield


def make_mesh(mesh_spec: str) -> Mesh:
    px, py = _parse_mesh(mesh_spec)
    if px * py != jax.process_count():
        raise SystemExit(
            f"mesh {mesh_spec} has {px * py} cells, but "
            f"jax.process_count()={jax.process_count()}")
    devices = np.asarray(jax.devices()).reshape(px, py)
    return Mesh(devices, axis_names=("x", "y"))


def build_inputs(
    *,
    nq: int,
    n: int,
    mrhs: int,
    dtype,
    seed: int,
    mesh: Mesh,
) -> tuple[jax.Array, jax.Array]:
    """Build reusable sharded base inputs.

    Timed iterations add a tiny diagonal shift before calling the FFI.  That
    creates a fresh buffer for donation while preserving the same shape and
    dataflow as the in-situ call.
    """
    sh = NamedSharding(mesh, P(None, "x", "y"))

    @jax.jit
    def _build():
        k0, k1, k2, k3 = jax.random.split(jax.random.key(seed), 4)
        ar = jax.random.normal(k0, (nq, n, n), dtype=jnp.float64)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            ai = jax.random.normal(k1, (nq, n, n), dtype=jnp.float64)
            z = (ar + 1j * ai).astype(dtype)
            a = 0.5 * (z + jnp.conj(jnp.swapaxes(z, -1, -2)))
        else:
            a = 0.5 * (ar + jnp.swapaxes(ar, -1, -2)).astype(dtype)
        # Strongly positive-definite, like the CCT charge-channel solve.
        a = a + (2.0 * n) * jnp.eye(n, dtype=dtype)[None, :, :]

        br = jax.random.normal(k2, (nq, n, mrhs), dtype=jnp.float64)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            bi = jax.random.normal(k3, (nq, n, mrhs), dtype=jnp.float64)
            b = (br + 1j * bi).astype(dtype)
        else:
            b = br.astype(dtype)

        return (
            jax.lax.with_sharding_constraint(a, sh),
            jax.lax.with_sharding_constraint(b, sh),
        )

    a0, b0 = _build()
    jax.block_until_ready(a0)
    jax.block_until_ready(b0)
    return a0, b0


def fresh_for_donation(a0: jax.Array, b0: jax.Array, step: int):
    """Return fresh donated buffers without changing the benchmark shape.

    The FFI wrappers donate their inputs.  Reusing the exact same ``a0``/``b0``
    buffers across iterations would either fail or silently measure donation
    side effects.  Adding a scalar zero-ish diagonal shift forces XLA to make
    a fresh A buffer; adding ``0 * step`` to B does the same for B when potrs
    is timed.
    """
    n = int(a0.shape[-1])
    eps = jnp.asarray(step, dtype=a0.real.dtype) * jnp.asarray(1e-12, a0.real.dtype)
    eye = jnp.eye(n, dtype=a0.dtype)[None, :, :]
    a = a0 + eps.astype(a0.dtype) * eye
    b = b0 + jnp.asarray(0, dtype=b0.dtype)
    return a, b


def prebuild_donated_inputs(
    a0: jax.Array,
    b0: jax.Array,
    count: int,
    *,
    step0: int,
    sync_label: str,
) -> list[tuple[jax.Array, jax.Array]]:
    """Materialize one donated input pair per measured call before profiling.

    The wrappers donate A (and B for potrs).  Real GWJAX receives fresh arrays
    from upstream kernels; this microbenchmark has to manufacture those fresh
    buffers.  Doing it inside the profiler range pollutes the trace with JAX
    add/copy allocation noise, so the default is to build the per-iteration
    buffers before timing.
    """
    pairs: list[tuple[jax.Array, jax.Array]] = []
    for i in range(count):
        a, b = fresh_for_donation(a0, b0, step0 + i)
        jax.block_until_ready(a)
        jax.block_until_ready(b)
        pairs.append((a, b))
    multihost_utils.sync_global_devices(sync_label)
    return pairs


def timed_call(mode: str, a: jax.Array, b: jax.Array, mesh: Mesh) -> None:
    if mode == "potrf":
        l = batched_distributed_cholesky(a, mesh=mesh)
        jax.block_until_ready(l.raw)
    elif mode == "potrf_potrs":
        l = batched_distributed_cholesky(a, mesh=mesh)
        x = batched_distributed_potrs(l, b, mesh=mesh)
        jax.block_until_ready(x)
    else:
        raise ValueError(f"unknown mode {mode!r}")


def summarize_ms(samples: list[float]) -> str:
    samples_sorted = sorted(samples)
    n = len(samples_sorted)
    p50 = samples_sorted[n // 2]
    p90 = samples_sorted[min(n - 1, int(0.9 * (n - 1)))]
    return (
        f"mean={statistics.mean(samples):.3f} ms, "
        f"median={p50:.3f} ms, p90={p90:.3f} ms, "
        f"min={min(samples):.3f} ms, max={max(samples):.3f} ms"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--mode", choices=["potrf", "potrf_potrs"], default="potrf")
    parser.add_argument("--mesh", default="2x2")
    parser.add_argument("--nq", type=int, default=9)
    parser.add_argument("-n", type=int, default=640)
    parser.add_argument("--mrhs", type=int, default=640)
    parser.add_argument("--dtype", choices=["f64", "c128"], default="c128")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-prebuild-inputs", action="store_true",
                        help="build fresh donated A/B inside timed loop "
                             "(mostly useful to estimate caller-side overhead)")
    parser.add_argument("--sync-each-iter", action="store_true",
                        help="insert a multihost barrier before/after each "
                             "timed iteration; useful for wall timing, but "
                             "pollutes Nsight traces with psum collectives")
    parser.add_argument("--cuda-profiler-api", action="store_true",
                        help="call cudaProfilerStart/Stop around timed iters")
    parser.add_argument("--sync-label", default="cusolvermp_profile_batched",
                        help="prefix for multihost sync labels")
    args = parser.parse_args(argv)

    init_jax_distributed()
    fallback_to_cpu_if_no_gpu_backend()
    jax.config.update("jax_enable_x64", True)

    mesh = make_mesh(args.mesh)
    px, py = _parse_mesh(args.mesh)
    if args.n % px or args.n % py or args.mrhs % py:
        raise SystemExit(
            f"shape must divide mesh: n={args.n}, mrhs={args.mrhs}, "
            f"mesh={args.mesh}")
    dtype = _dtype(args.dtype)

    _log(
        "=== cuSOLVERMp FFI batched profile ===\n"
        f"mode={args.mode} nq={args.nq} n={args.n} mrhs={args.mrhs} "
        f"dtype={args.dtype} mesh={args.mesh} "
        f"processes={jax.process_count()} local_devices={jax.local_device_count()}"
    )

    with trace_annotation("build_inputs"):
        a0, b0 = build_inputs(
            nq=args.nq, n=args.n, mrhs=args.mrhs,
            dtype=dtype, seed=args.seed, mesh=mesh)
    multihost_utils.sync_global_devices(f"{args.sync_label}_inputs_ready")

    _log(f"warmup iters: {args.warmup}")
    warmup_pairs = prebuild_donated_inputs(
        a0, b0, args.warmup, step0=0,
        sync_label=f"{args.sync_label}_warmup_inputs_ready")
    for i, (a, b) in enumerate(warmup_pairs):
        with trace_annotation(f"warmup_{i}"):
            timed_call(args.mode, a, b, mesh)
        multihost_utils.sync_global_devices(f"{args.sync_label}_warmup_{i}")

    timed_pairs = None
    if not args.no_prebuild_inputs:
        _log("prebuilding donated inputs outside timed/profiler range")
        timed_pairs = prebuild_donated_inputs(
            a0, b0, args.iters, step0=1000,
            sync_label=f"{args.sync_label}_timed_inputs_ready")

    samples_ms: list[float] = []
    _log(f"timed iters: {args.iters}")
    profile_cm = cuda_profiler_api(args.cuda_profiler_api)
    with profile_cm:
        for i in range(args.iters):
            if args.sync_each_iter:
                multihost_utils.sync_global_devices(
                    f"{args.sync_label}_iter_{i}_start")
            with trace_annotation(f"timed_{args.mode}_{i}"):
                if timed_pairs is None:
                    a, b = fresh_for_donation(a0, b0, i + 1000)
                else:
                    a, b = timed_pairs[i]
                t0 = time.perf_counter()
                timed_call(args.mode, a, b, mesh)
                dt_ms = (time.perf_counter() - t0) * 1000.0
            if args.sync_each_iter:
                multihost_utils.sync_global_devices(
                    f"{args.sync_label}_iter_{i}_done")
            samples_ms.append(dt_ms)
            _log(f"iter {i:03d}: {dt_ms:.3f} ms")

    _log(f"summary: {summarize_ms(samples_ms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
