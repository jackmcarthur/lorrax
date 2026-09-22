"""P4 proof for the one-time Sigma band-interval census and prepared GEMM.

Run this file directly under four runtime-initialized processes.  It checks
the production replicated parent-energy layout, records the census HLO and
memory analysis, and compares the dynamic and prepared Green contractions
with non-unit signed weights.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    _TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _REPO = os.path.dirname(_TESTS)
    for _svc in ("lxkit", "distrib_la"):
        _src = os.path.join(_REPO, "services", _svc, "src")
        if os.path.isdir(_src) and _src not in sys.path:
            sys.path.insert(0, _src)
    from lxkit.gate import platform_from_env
    from runtime import initialize_communicator_stack
    _plat = platform_from_env()
    _RUNTIME = initialize_communicator_stack(
        platform="gpu" if _plat == "CUDA" else "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from distrib_la import gemm_plan
from gw.greens_function_kernel import build_G_tau, prepare_tau_band_range


def _put(value, mesh, spec):
    return device_put_process_local(
        np.asarray(value), NamedSharding(mesh, spec))


def _local_max_difference(left, right):
    return max(
        float(np.max(np.abs(np.asarray(a.data) - np.asarray(b.data)), initial=0.0))
        for a, b in zip(left.addressable_shards, right.addressable_shards))


def run_gate(mesh):
    rng = np.random.default_rng(396)
    nq, ns, mu, nb = 4, 1, 4, 8
    energy = np.asarray([
        [3.0, 0.0, 900.0, -2.0, 4.0, 0.0, 7.0, -1.0],
        [8.0, 3.0, 0.0, -1.0, 900.0, 2.0, 0.0, 6.0],
        [5.0, 0.0, -3.0, 900.0, 1.0, 4.0, 0.0, 2.0],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ])
    weight = np.asarray([
        [0.0, -0.25, 0.5, 0.0, -0.75, 0.125, 0.0, 0.0],
        [0.0, 0.0, 0.2, -0.4, 0.0, 0.8, -0.1, 0.0],
        [0.0, 0.3, 0.0, -0.6, 0.9, 0.0, 0.0, -0.2],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    times = np.asarray([0.0, 1.0, np.nan], dtype=np.complex128)
    rep = P(None, None)
    energy_d = _put(energy, mesh, rep)
    weight_d = _put(weight, mesh, rep)
    times_d = _put(times, mesh, P())
    ref_d = _put(np.asarray(0.0), mesh, P())
    count_d = _put(np.asarray(2, np.int32), mesh, P())

    lowered = prepare_tau_band_range.lower(
        energy_d, weight_d, ref_d, times_d, count_d)
    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    forbidden = (
        "all-gather", "all-reduce", "all-to-all", "collective-permute",
        "reduce-scatter",
    )
    assert "while" in hlo.lower()
    assert not any(token in hlo.lower() for token in forbidden)
    assert "pred[3,4,8]" not in hlo.lower()
    assert "tensor<3x4x8xi1>" not in hlo.lower()
    compiled = lowered.compile()
    memory = compiled.memory_analysis()
    lo, hi, invariant = compiled(
        energy_d, weight_d, ref_d, times_d, count_d)
    np.testing.assert_array_equal(np.asarray(lo), [1, 2, 1, 0])
    np.testing.assert_array_equal(np.asarray(hi), [6, 7, 8, 0])
    assert bool(np.asarray(invariant))

    left = (rng.standard_normal((nq, ns, mu, nb))
            + 1j * rng.standard_normal((nq, ns, mu, nb)))
    right = (rng.standard_normal((nq, nb, ns, mu))
             + 1j * rng.standard_normal((nq, nb, ns, mu)))
    left_d = _put(left, mesh, P(None, None, "x", "y"))
    right_d = _put(right, mesh, P(None, "x", None, "y"))
    plan = gemm_plan(
        mesh, m=mu, k=nb, n=mu, nq=nq, dtype=jnp.complex128,
        layout="face", enable_active_range=True)
    prepared_gemm = plan.prepare_active_range(
        np.asarray(lo), np.asarray(hi))
    evolution_time = _put(np.asarray(0.35), mesh, P())
    dynamic = build_G_tau(
        left_d, right_d, energy_d, evolution_time, gemm=plan,
        band_weight=weight_d, trim_zero_bands=True)
    prepared = build_G_tau(
        left_d, right_d, energy_d, evolution_time, gemm=plan,
        band_weight=weight_d, trim_zero_bands=True,
        prepared_active_gemm=prepared_gemm)
    jax.block_until_ready((dynamic, prepared))
    error = _local_max_difference(prepared, dynamic)
    scale = max(
        float(np.max(np.abs(np.asarray(shard.data)), initial=0.0))
        for shard in dynamic.addressable_shards)
    assert error <= 2.0e-13 * max(scale, 1.0), (error, scale)
    assert prepared.sharding.spec == P(None, "x", None, "y", None)

    return {
        "schema": "lorrax.prepared_tau_band_range.p4.v1",
        "rank": jax.process_index(),
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "backend": jax.default_backend(),
        # Parent energies and occupations are intentionally replicated in the
        # production Green carrier.  The census therefore needs no band-axis
        # collective; this differs from kernels whose full band axis is split.
        "energy_selector_sharding": "P(None,None), replicated on the 2x2 mesh",
        "interval_lo": np.asarray(lo).tolist(),
        "interval_hi": np.asarray(hi).tolist(),
        "invariant": bool(np.asarray(invariant)),
        "green_max_abs_error": error,
        "green_scale": scale,
        "census_memory": {
            name: int(getattr(memory, name))
            for name in (
                "argument_size_in_bytes", "output_size_in_bytes",
                "temp_size_in_bytes", "alias_size_in_bytes")
        },
        "forbidden_collectives": list(forbidden),
        "forbidden_collectives_found": [],
        "tau_history_buffer_found": False,
    }, hlo


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if jax.process_count() != 4 or jax.device_count() != 4:
        raise RuntimeError(
            "prepared_tau_band_range_p4 requires four processes and four devices")
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    record, hlo = run_gate(mesh)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rank = jax.process_index()
    (output / f"receipt_rank{rank}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")
    if rank == 0:
        (output / "census.hlo.txt").write_text(hlo)
        print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_main)
