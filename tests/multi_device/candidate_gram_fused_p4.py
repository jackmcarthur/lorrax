"""P=4 gate for the fused q=0 candidate-Gram tile.

Run as four processes with one GPU each on a 2x2 mesh. Both charge and
transverse modes must match the canonical staged pair-density route on every
addressable output tile, and the output must remain distributed over X,Y.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
))

from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402


def _fail(message: str) -> None:
    print(f"[candidate-gram-fused-p4] FAIL: {message}", flush=True)
    raise SystemExit(1)


def main() -> None:
    from centroid.pivoted_cholesky import (
        candidate_gram_q0_from_pair,
        candidate_gram_q0_from_psi,
    )
    from common.collectives import device_put_process_local
    from isdf import pair_density
    from isdf.core import (
        _gammas_perm,
        _gammas_phase,
        _gram_q0_from_psi_kernel,
    )
    from runtime.aot_memory import aot_kernel_peak_bytes

    if jax.process_count() != 4 or jax.device_count() != 4:
        _fail(
            "needs exactly four processes and four global devices; got "
            f"{jax.process_count()} process(es), {jax.device_count()} device(s)"
        )
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    source_path = os.path.realpath(sys.modules["isdf.core"].__file__)
    expected_root = os.path.realpath(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))) + os.sep
    if not source_path.startswith(expected_root):
        _fail(f"wrong source checkout: {source_path}")
    if jax.process_index() == 0:
        print(
            "[candidate-gram-fused-p4] source="
            f"{source_path}",
            flush=True,
        )
    x_sh = NamedSharding(mesh, P(None, "x", None, None))
    y_sh = NamedSharding(mesh, P(None, None, None, "y"))
    rep = NamedSharding(mesh, P())

    rng = np.random.default_rng(20260901)
    nk, nl, nr, ns, npoint = 3, 5, 7, 4, 12
    left = (rng.standard_normal((nk, nl, ns, npoint))
            + 1j * rng.standard_normal((nk, nl, ns, npoint)))
    right = (rng.standard_normal((nk, nr, ns, npoint))
             + 1j * rng.standard_normal((nk, nr, ns, npoint)))
    weights = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)

    left_x = device_put_process_local(
        np.conj(left).transpose(0, 3, 1, 2), x_sh)
    left_y = device_put_process_local(left, y_sh)
    right_x = device_put_process_local(
        np.conj(right).transpose(0, 3, 1, 2), x_sh)
    right_y = device_put_process_local(right, y_sh)
    kw = device_put_process_local(weights, rep)

    P_l = pair_density(left_x, left_y, mesh)
    P_r = pair_density(right_x, right_y, mesh)
    memory_rows = []
    for mode in ("charge", "transverse"):
        staged = candidate_gram_q0_from_pair(
            P_l, P_r, kw, mesh_xy=mesh, gamma_mode=mode)
        fused = candidate_gram_q0_from_psi(
            left_x, left_y, right_x, right_y, kw,
            mesh_xy=mesh, gamma_mode=mode)
        fused.block_until_ready()
        if tuple(fused.sharding.spec) != ("x", "y"):
            _fail(f"{mode}: output layout is {fused.sharding.spec}")
        for shard, staged_shard in zip(
                fused.addressable_shards, staged.addressable_shards):
            if shard.index != staged_shard.index:
                _fail(
                    f"{mode}: mismatched addressable indices "
                    f"{shard.index} != {staged_shard.index}")
            got = np.asarray(shard.data)
            want = np.asarray(staged_shard.data)
            scale = max(float(np.max(np.abs(want))), 1.0)
            rel = float(np.max(np.abs(got - want))) / scale
            if rel > 3.0e-13:
                _fail(f"{mode}: rank={jax.process_index()} rel={rel:.3e}")

        if mode == "transverse":
            perm, phase = _gammas_perm, _gammas_phase
        else:
            perm = jnp.arange(ns, dtype=jnp.int32)
            phase = jnp.ones(ns, dtype=left_x.dtype)
        compiled = _gram_q0_from_psi_kernel(
            mesh, nk, npoint, npoint, nl, nr, ns,
            gamma_mode=mode, symmetrize=False,
        ).lower(
            left_x, left_y, right_x, right_y, kw,
            perm, phase, perm, phase,
        ).compile()
        hlo = compiled.as_text().lower()
        forbidden = [op for op in (
            "all-gather", "all-to-all", "all-reduce", "reduce-scatter",
            "collective-permute",
        ) if re.search(rf"\b{re.escape(op)}(?:-start|-done)?\(", hlo)]
        if forbidden:
            _fail(f"{mode}: unexpected collective(s) {forbidden}")
        mem = aot_kernel_peak_bytes(compiled)
        memory_rows.append(
            (mode, int(mem.total), int(mem.resident_increment)))

    if jax.process_index() == 0:
        print(
            "[candidate-gram-fused-p4] PASS: charge/transverse parity; "
            f"P('x','y') output; zero collectives; memory={memory_rows}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        finalize_process()
