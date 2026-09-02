"""P=4 parity/HLO gate for the one-dispatch tiled candidate Gram.

The production-like fixture has 600 candidates and nine square tiles, with a
real tail on both local axes.  Both feature families must be bit-identical to
the incumbent extract/Gram/update dispatch sequence.  Optimized HLO must
contain two pair-density contraction definitions, no gather-class collective,
and an alias for the donated local Gram shard.  The matched Nsight gate owns
their dynamic execution count because definition count alone cannot expose a
contraction sunk into the transverse component scan.
"""
from __future__ import annotations

import os
import re
import sys
from functools import partial

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
    print(f"[candidate-gram-tiled-scan-p4] FAIL: {message}", flush=True)
    raise SystemExit(1)


def _rep_i32(value, rep_sh):
    from common.collectives import device_put_process_local
    return device_put_process_local(np.asarray(value, dtype=np.int32), rep_sh)


def main() -> None:
    from common.collectives import device_put_process_local
    from common.staged_reshard import shard_local_slice_pad, shard_local_update
    from isdf import (
        gram_q0_from_psi_sm,
        gram_q0_from_psi_aot_peak_bytes,
        gram_q0_tiled_from_psi_sm,
        gram_q0_tiled_from_psi_aot_resident_increment_bytes,
    )
    from isdf.core import (
        _gammas_perm,
        _gammas_phase,
        _gram_q0_tiled_from_psi_kernel,
    )
    from runtime.aot_memory import aot_kernel_peak_bytes
    from centroid.pivoted_cholesky import (
        _candidate_gram_hermitian_fold_kernel,
    )

    if jax.process_count() != 4 or jax.device_count() != 4:
        _fail(
            "needs exactly four processes and four global devices; got "
            f"{jax.process_count()} process(es), {jax.device_count()} device(s)")
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    source_path = os.path.realpath(sys.modules["isdf.core"].__file__)
    expected_root = os.path.realpath(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))) + os.sep
    if not source_path.startswith(expected_root):
        _fail(f"wrong source checkout: {source_path}")
    if jax.process_index() == 0:
        print(
            f"[candidate-gram-tiled-scan-p4] source={source_path}",
            flush=True)

    x_spec = P(None, "x", None, None)
    y_spec = P(None, None, None, "y")
    xy_spec = P("x", "y")
    x_sh = NamedSharding(mesh, x_spec)
    y_sh = NamedSharding(mesh, y_spec)
    xy_sh = NamedSharding(mesh, xy_spec)
    rep_sh = NamedSharding(mesh, P())

    rng = np.random.default_rng(20260901)
    nk, nl, nr, ns, npoint, width = 3, 16, 20, 4, 600, 256
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
    kw = device_put_process_local(weights, rep_sh)

    @partial(jax.jit, out_shardings=xy_sh)
    def _zero_gram():
        return jnp.zeros((npoint, npoint), dtype=jnp.complex128)

    extract_x = shard_local_slice_pad(
        mesh, spec=x_spec, axis=1, mesh_axis="x", local_size=width // 2)
    extract_y = shard_local_slice_pad(
        mesh, spec=y_spec, axis=3, mesh_axis="y", local_size=width // 2)
    update_g = shard_local_update(mesh, spec=xy_spec)
    local_extent = npoint // 2
    local_tile = width // 2

    def _incumbent(mode):
        G = _zero_gram()
        for c0 in range(0, local_extent, local_tile):
            c_start = _rep_i32(c0, rep_sh)
            for r0 in range(0, local_extent, local_tile):
                r_start = _rep_i32(r0, rep_sh)
                tile = gram_q0_from_psi_sm(
                    extract_x(left_x, r_start),
                    extract_y(left_y, c_start),
                    extract_x(right_x, r_start),
                    extract_y(right_y, c_start),
                    kw, mesh_xy=mesh, gamma_mode=mode, symmetrize=False,
                )
                tile.block_until_ready()
                G = update_g(G, tile, _rep_i32((r0, c0), rep_sh))
                G.block_until_ready()
        return G

    memory_rows = []
    for mode in ("charge", "transverse"):
        expected = _incumbent(mode)
        got = gram_q0_tiled_from_psi_sm(
            _zero_gram(), left_x, left_y, right_x, right_y, kw,
            mesh_xy=mesh, tile_width=width, gamma_mode=mode)
        got.block_until_ready()
        if tuple(got.sharding.spec) != ("x", "y"):
            _fail(f"{mode}: output layout is {got.sharding.spec}")
        for shard, expected_shard in zip(
                got.addressable_shards, expected.addressable_shards):
            if shard.index != expected_shard.index:
                _fail(
                    f"{mode}: mismatched addressable indices "
                    f"{shard.index} != {expected_shard.index}")
            if not np.array_equal(
                    np.asarray(shard.data), np.asarray(expected_shard.data)):
                delta = np.asarray(shard.data) - np.asarray(expected_shard.data)
                _fail(
                    f"{mode}: not bit-identical on rank={jax.process_index()}; "
                    f"max_abs={np.max(np.abs(delta)):.3e}")

        fold_kernel = _candidate_gram_hermitian_fold_kernel(mesh)
        folded = fold_kernel(got)
        folded.block_until_ready()
        if tuple(folded.sharding.spec) != ("x", "y"):
            _fail(f"{mode}: terminal fold layout is {folded.sharding.spec}")
        fold_compiled = fold_kernel.lower(expected).compile()
        fold_hlo = fold_compiled.as_text().lower()
        fold_forbidden = [op for op in ("all-gather", "all-to-all")
                          if re.search(rf"\b{op}(?:-start|-done)?\(", fold_hlo)]
        if fold_forbidden:
            _fail(
                f"{mode}: terminal fold replicated a Gram via "
                f"{fold_forbidden}")
        local_g_bytes = (npoint // 2) ** 2 * np.dtype(np.complex128).itemsize
        fold_alias_bytes = int(
            fold_compiled.memory_analysis().alias_size_in_bytes)
        if fold_alias_bytes < local_g_bytes:
            _fail(
                f"{mode}: donated terminal-fold alias={fold_alias_bytes} B "
                f"is smaller than one local Gram shard={local_g_bytes} B")

        if mode == "transverse":
            perm, phase = _gammas_perm, _gammas_phase
        else:
            perm = jnp.arange(ns, dtype=jnp.int32)
            phase = jnp.ones(ns, dtype=left_x.dtype)
        compiled = _gram_q0_tiled_from_psi_kernel(
            mesh, nk, npoint, nl, nr, ns, width, gamma_mode=mode,
        ).lower(
            _zero_gram(), left_x, left_y, right_x, right_y, kw,
            perm, phase, perm, phase,
        ).compile()
        hlo = compiled.as_text().lower()
        forbidden = [op for op in (
            "all-gather", "all-to-all", "all-reduce", "reduce-scatter",
            "collective-permute",
        ) if re.search(rf"\b{re.escape(op)}(?:-start|-done)?\(", hlo)]
        if forbidden:
            _fail(f"{mode}: unexpected collective(s) {forbidden}")
        if not re.search(r"\bwhile(?:-start|-done)?\(", hlo):
            _fail(f"{mode}: optimized HLO has no scan/while loop")
        if re.search(r"\bscatter\(", hlo):
            _fail(
                f"{mode}: contiguous tile store lowered back to scatter")

        dot_ops = len(re.findall(r"\bdot(?:-general)?\(", hlo))
        gemm_ops = len(re.findall(
            r"custom_call_target=\"[^\"]*(?:gemm|blas)[^\"]*\"", hlo))
        pair_ops = gemm_ops if gemm_ops else dot_ops
        if pair_ops != 2:
            _fail(
                f"{mode}: expected exactly two pair-density dot/GEMM ops "
                f"inside the loop, got dot={dot_ops}, gemm={gemm_ops}; "
                "a per-tile repeated fusion path may have reappeared")

        analysis = compiled.memory_analysis()
        alias_bytes = int(analysis.alias_size_in_bytes)
        if alias_bytes < local_g_bytes:
            _fail(
                f"{mode}: donated G alias={alias_bytes} B is smaller than "
                f"one local Gram shard={local_g_bytes} B")
        mem = aot_kernel_peak_bytes(compiled)
        helper_increment = (
            gram_q0_tiled_from_psi_aot_resident_increment_bytes(
                mesh_xy=mesh, nk=nk, n_points=npoint, nb_l=nl, nb_r=nr,
                nspinor=ns, tile_width=width, gamma_mode=mode))
        if helper_increment != int(mem.resident_increment):
            _fail(
                f"{mode}: AOT helper={helper_increment} B but compiled "
                f"resident_increment={int(mem.resident_increment)} B")
        incumbent_tile_peak = gram_q0_from_psi_aot_peak_bytes(
            mesh_xy=mesh, nk=nk, n_rows=width, n_cols=width,
            nb_l=nl, nb_r=nr, nspinor=ns, gamma_mode=mode,
            symmetrize=False)
        if helper_increment > incumbent_tile_peak:
            _fail(
                f"{mode}: scan resident_increment={helper_increment} B "
                "exceeds the incumbent complete tile-kernel peak="
                f"{incumbent_tile_peak} B")

        # Same tile/body at a much larger logical M must keep the same
        # temporary class.  A second full local Gram would add 2.75 MiB here;
        # allow only 256 KiB for loop-schedule constants/compiler bookkeeping.
        large_increment = (
            gram_q0_tiled_from_psi_aot_resident_increment_bytes(
                mesh_xy=mesh, nk=nk, n_points=1024, nb_l=nl, nb_r=nr,
                nspinor=ns, tile_width=width, gamma_mode=mode))
        if abs(large_increment - helper_increment) > 256 * 1024:
            _fail(
                f"{mode}: fixed-width resident_increment grew with M: "
                f"M={npoint} -> {helper_increment} B, M=1024 -> "
                f"{large_increment} B; expected tile-class temporaries")
        memory_rows.append((
            mode, int(mem.total), int(mem.resident_increment), alias_bytes,
            incumbent_tile_peak, large_increment, dot_ops, gemm_ops,
        ))

    if jax.process_index() == 0:
        print(
            "[candidate-gram-tiled-scan-p4] PASS: charge/transverse "
            "bit parity; P('x','y'); one loop-body pair path; zero "
            f"collectives; donated G; memory={memory_rows}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        finalize_process()
