"""Measure the exact complex128 vertex GEMMs at fixture/production shapes.

This is benchmark scaffolding, not a production alternative.  It compares
the current einsums, an explicit k-batched ``dot_general`` spelling, and a
standalone ``cublasZgemmStridedBatched`` FFI probe.  The reported time is the
whole jitted contraction, including layout packing required by each spelling.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

TARGET = "lorrax_probe_cublas_zgemm_strided_batched"
SIZES = {
    "fixture": {"mu": 399, "nk": 9, "nc": 20, "nv": 26, "spin": 2},
    "production": {"mu": 648, "nk": 64, "nc": 56, "nv": 4, "spin": 1},
}


def _register_probe(path: str) -> None:
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    fn = getattr(lib, "LorraxVertexZgemmStridedBatchedProbeFfi")
    try:
        jax.ffi.register_ffi_target(
            TARGET, jax.ffi.pycapsule(fn), platform="CUDA")
    except Exception as exc:
        if "already registered" not in str(exc).lower():
            raise
    # Keep the shared object alive for the process lifetime.
    _register_probe.lib = lib


def _ffi_gemm(a, b, math_mode: int = 1):
    out = jax.ShapeDtypeStruct((a.shape[0], a.shape[1], b.shape[2]), a.dtype)
    return jax.ffi.ffi_call(TARGET, out)(a, b, math_mode=math_mode)


def _dot_batch(a, b):
    return jax.lax.dot_general(
        a,
        b,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        precision=jax.lax.Precision.DEFAULT,
    )


def _encode_baseline(psi, r):
    return jnp.einsum("kctM,cksN->MNtsk", psi, r)


def _encode_operands(psi, r):
    nk, nc, nt, mu = psi.shape
    ns, nu = r.shape[2], r.shape[3]
    a = jnp.broadcast_to(
        psi.transpose(0, 2, 3, 1)[:, :, None, :, :],
        (nk, nt, ns, mu, nc),
    ).reshape(nk * nt * ns, mu, nc)
    b = jnp.broadcast_to(
        r.transpose(1, 2, 0, 3)[:, None, :, :, :],
        (nk, nt, ns, nc, nu),
    ).reshape(nk * nt * ns, nc, nu)
    return a, b, (nk, nt, ns, mu, nu)


def _encode_batched(psi, r):
    a, b, (nk, nt, ns, mu, nu) = _encode_operands(psi, r)
    out = _dot_batch(a, b).reshape(nk, nt, ns, mu, nu)
    return out.transpose(3, 4, 1, 2, 0)


def _encode_ffi(psi, r):
    a, b, (nk, nt, ns, mu, nu) = _encode_operands(psi, r)
    out = _ffi_gemm(a, b).reshape(nk, nt, ns, mu, nu)
    return out.transpose(3, 4, 1, 2, 0)


def _sigma_right_baseline(o, psi):
    return jnp.einsum("ksxty,ktyn->ksxn", o, psi)


def _sigma_right_operands(o, psi):
    nk, ns, mu, nt, nu = o.shape
    n = psi.shape[-1]
    return o.reshape(nk, ns * mu, nt * nu), psi.reshape(nk, nt * nu, n)


def _sigma_right_batched(o, psi):
    return _dot_batch(*_sigma_right_operands(o, psi)).reshape(
        o.shape[0], o.shape[1], o.shape[2], psi.shape[-1])


def _sigma_right_ffi(o, psi):
    return _ffi_gemm(*_sigma_right_operands(o, psi)).reshape(
        o.shape[0], o.shape[1], o.shape[2], psi.shape[-1])


def _sigma_left_baseline(psi, right):
    return jnp.einsum("kmsx,ksxn->kmn", jnp.conj(psi), right)


def _sigma_left_operands(psi, right):
    nk, m, ns, mu = psi.shape
    n = right.shape[-1]
    return jnp.conj(psi).reshape(nk, m, ns * mu), right.reshape(
        nk, ns * mu, n)


def _sigma_left_batched(psi, right):
    return _dot_batch(*_sigma_left_operands(psi, right))


def _sigma_left_ffi(psi, right):
    return _ffi_gemm(*_sigma_left_operands(psi, right))


def _decode_mu_baseline(psi_c, u):
    return jnp.einsum("kctM,MNtsk->cNsk", jnp.conj(psi_c), u)


def _decode_mu_operands(psi_c, u):
    nk, nc, nt, mu = psi_c.shape
    nu, ns = u.shape[1], u.shape[3]
    a = jnp.broadcast_to(
        jnp.conj(psi_c)[:, None, :, :, :],
        (nk, ns, nc, nt, mu),
    ).reshape(nk * ns, nc, nt * mu)
    b = u.transpose(4, 3, 2, 0, 1).reshape(nk * ns, nt * mu, nu)
    return a, b, (nk, ns, nc, nu)


def _decode_mu_batched(psi_c, u):
    a, b, (nk, ns, nc, nu) = _decode_mu_operands(psi_c, u)
    return _dot_batch(a, b).reshape(nk, ns, nc, nu).transpose(2, 3, 1, 0)


def _decode_mu_ffi(psi_c, u):
    a, b, (nk, ns, nc, nu) = _decode_mu_operands(psi_c, u)
    return _ffi_gemm(a, b).reshape(nk, ns, nc, nu).transpose(2, 3, 1, 0)


def _decode_nu_baseline(psi_v, a):
    return jnp.einsum("kvsN,cNsk->cvk", psi_v, a)


def _decode_nu_operands(psi_v, a):
    nk, nv, ns, nu = psi_v.shape
    nc = a.shape[0]
    lhs = a.transpose(3, 0, 2, 1).reshape(nk, nc, ns * nu)
    rhs = psi_v.transpose(0, 2, 3, 1).reshape(nk, ns * nu, nv)
    return lhs, rhs


def _decode_nu_batched(psi_v, a):
    return _dot_batch(*_decode_nu_operands(psi_v, a)).transpose(1, 2, 0)


def _decode_nu_ffi(psi_v, a):
    return _ffi_gemm(*_decode_nu_operands(psi_v, a)).transpose(1, 2, 0)


def _data(shape, phase):
    n = 1
    for d in shape:
        n *= d
    x = jnp.arange(n, dtype=jnp.float64).reshape(shape)
    return jnp.sin(x * 0.013 + phase) + 1j * jnp.cos(x * 0.017 - phase)


def _case(op: str, cfg: dict):
    nk, mu, nc, nv, spin = (
        cfg["nk"], cfg["mu"], cfg["nc"], cfg["nv"], cfg["spin"])
    if op == "bse_encode":
        args = (_data((nk, nc, spin, mu), 0.1),
                _data((nc, nk, spin, mu), 0.4))
        gemm = (nk * spin * spin, mu, nc, mu)
        funcs = (_encode_baseline, _encode_batched, _encode_ffi)
    elif op == "bse_decode_mu":
        args = (_data((nk, nc, spin, mu), 0.3),
                _data((mu, mu, spin, spin, nk), 0.6))
        gemm = (nk * spin, nc, spin * mu, mu)
        funcs = (_decode_mu_baseline, _decode_mu_batched, _decode_mu_ffi)
    elif op == "bse_decode_nu":
        args = (_data((nk, nv, spin, mu), 0.3),
                _data((nc, mu, spin, nk), 0.6))
        gemm = (nk, nc, spin * mu, nv)
        funcs = (_decode_nu_baseline, _decode_nu_batched, _decode_nu_ffi)
    elif op == "sigma_right":
        n = nc
        args = (_data((nk, spin, mu, spin, mu), 0.2),
                _data((nk, spin, mu, n), 0.5))
        gemm = (nk, spin * mu, spin * mu, n)
        funcs = (_sigma_right_baseline, _sigma_right_batched, _sigma_right_ffi)
    elif op == "sigma_left":
        n = nc
        args = (_data((nk, nc, spin, mu), 0.3),
                _data((nk, spin, mu, n), 0.6))
        gemm = (nk, nc, spin * mu, n)
        funcs = (_sigma_left_baseline, _sigma_left_batched, _sigma_left_ffi)
    else:
        raise ValueError(op)
    return args, gemm, dict(zip(("baseline", "batched", "ffi"), funcs))


def _block(tree):
    jax.tree.map(lambda x: x.block_until_ready(), tree)
    return tree


def _timings(fn: Callable, args, warmup: int, samples: int):
    compiled = jax.jit(fn).lower(*args).compile()
    for _ in range(warmup):
        _block(compiled(*args))
    values = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        _block(compiled(*args))
        values.append((time.perf_counter_ns() - start) * 1e-6)
    return compiled, values


def _numerics(ref, got):
    abs_err = jnp.max(jnp.abs(got - ref))
    scale = jnp.max(jnp.abs(ref))
    l2_rel = jnp.linalg.norm((got - ref).ravel()) / jnp.linalg.norm(ref.ravel())
    abs_err, scale, l2_rel = jax.device_get((abs_err, scale, l2_rel))
    return {
        "max_abs": float(abs_err),
        "max_rel": float(abs_err / max(float(scale), 1e-300)),
        "l2_rel": float(l2_rel),
    }


def _hlo_summary(compiled, path: Path | None):
    text = compiled.as_text()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    calls = re.findall(r"custom[_-]call[_-]target=\"([^\"]+)\"", text)
    return {"custom_call_targets": sorted(set(calls)),
            "custom_call_count": len(calls),
            "cutlass_mentions": text.lower().count("cutlass")}


def _profile(fn, args, warmup: int, reps: int):
    compiled = jax.jit(fn).lower(*args).compile()
    for _ in range(warmup):
        _block(compiled(*args))
    cudart = ctypes.CDLL("libcudart.so")
    if cudart.cudaProfilerStart() != 0:
        raise RuntimeError("cudaProfilerStart failed")
    for _ in range(reps):
        _block(compiled(*args))
    if cudart.cudaProfilerStop() != 0:
        raise RuntimeError("cudaProfilerStop failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=(*SIZES, "both"), default="both")
    parser.add_argument(
        "--ops", default="bse_encode,bse_decode_mu,bse_decode_nu,"
                         "sigma_right,sigma_left")
    parser.add_argument("--arms", default="baseline,batched,ffi")
    parser.add_argument("--ffi-so")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--hlo-dir")
    parser.add_argument("--json")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-reps", type=int, default=20)
    args = parser.parse_args()

    arms = args.arms.split(",")
    if "ffi" in arms:
        if not args.ffi_so:
            parser.error("--ffi-so is required when arm 'ffi' is selected")
        _register_probe(args.ffi_so)
    sizes = SIZES if args.size == "both" else {args.size: SIZES[args.size]}
    ops = args.ops.split(",")
    results = {
        "environment": {
            "jax": jax.__version__,
            "devices": [str(x) for x in jax.devices()],
            "BENCH_ALLOC": os.environ.get("BENCH_ALLOC"),
            "XLA_PYTHON_CLIENT_ALLOCATOR": os.environ.get(
                "XLA_PYTHON_CLIENT_ALLOCATOR"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        },
        "sizes": sizes,
        "results": [],
    }
    print("ENV", json.dumps(results["environment"], sort_keys=True), flush=True)

    for size_name, cfg in sizes.items():
        for op in ops:
            op_args, (batch, m, k, n), funcs = _case(op, cfg)
            flops = 8.0 * batch * m * k * n
            print(f"CASE size={size_name} op={op} gemm=batch{batch} "
                  f"M{m} K{k} N{n} flops={flops:.0f}", flush=True)
            ref = _block(jax.jit(funcs["baseline"])(*op_args))
            if args.profile:
                for arm in arms:
                    _profile(funcs[arm], op_args, args.warmup,
                             args.profile_reps)
                continue

            # Compile first, then warm and measure in round-robin order.  A100
            # clocks and a shared node power envelope otherwise bias an arm
            # simply because it ran first or last.
            compiled_arms = {
                arm: jax.jit(funcs[arm]).lower(*op_args).compile()
                for arm in arms
            }
            for _ in range(args.warmup):
                for arm in arms:
                    _block(compiled_arms[arm](*op_args))
            arm_times = {arm: [] for arm in arms}
            for sample in range(args.samples):
                order = arms[sample % len(arms):] + arms[:sample % len(arms)]
                for arm in order:
                    start = time.perf_counter_ns()
                    _block(compiled_arms[arm](*op_args))
                    arm_times[arm].append(
                        (time.perf_counter_ns() - start) * 1e-6)

            for arm in arms:
                compiled = compiled_arms[arm]
                times = arm_times[arm]
                got = _block(compiled(*op_args))
                numerics = _numerics(ref, got)
                median_ms = statistics.median(times)
                row = {
                    "size": size_name,
                    "op": op,
                    "arm": arm,
                    "batch": batch,
                    "m": m,
                    "k": k,
                    "n": n,
                    "flops": flops,
                    "median_ms": median_ms,
                    "min_ms": min(times),
                    "tflops_median": flops / (median_ms * 1e9),
                    "samples_ms": times,
                    "numerics": numerics,
                }
                hlo_path = None
                if args.hlo_dir:
                    hlo_path = Path(args.hlo_dir) / (
                        f"{size_name}_{op}_{arm}.hlo.txt")
                row["hlo"] = _hlo_summary(compiled, hlo_path)
                results["results"].append(row)
                print("RESULT", json.dumps(row, sort_keys=True), flush=True)
                if numerics["max_rel"] > 1e-14:
                    raise AssertionError(
                        f"1e-15-class gate failed: {size_name}/{op}/{arm}: "
                        f"max_rel={numerics['max_rel']:.3e}")
            del ref, op_args

    if args.json and not args.profile:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
