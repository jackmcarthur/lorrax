"""P4 parity gate for whole-group pivoted-Cholesky selection."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
))

from runtime import initialize_communicator_stack, run_main_and_finalize  # noqa: E402

RUNTIME = initialize_communicator_stack()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402


def _fail(message: str) -> None:
    print(f"[centroid-group-block-p4] FAIL: {message}", flush=True)
    raise SystemExit(1)


def main() -> None:
    from common.collectives import device_put_process_local
    from common.pivoted_cholesky import (
        group_block_pivoted_cholesky_select,
        make_sharded_group_block_pivoted_cholesky_select,
    )

    if jax.process_count() != 4 or jax.device_count() != 4:
        _fail(
            "needs exactly four processes and four global devices; got "
            f"{jax.process_count()} process(es), {jax.device_count()} device(s)")
    mesh = RUNTIME.mesh
    if tuple(mesh.axis_names) != ("x", "y") or mesh.devices.shape != (2, 2):
        _fail(f"expected a 2x2 ('x','y') mesh, got {mesh}")

    logical_sizes = np.asarray(
        [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 11],
        dtype=np.int32)
    logical_M = int(logical_sizes.sum())
    M = 64
    if logical_M != 62:
        _fail(f"fixture arithmetic drifted: logical_M={logical_M}")
    n_groups = int(logical_sizes.size)
    group_id = np.repeat(
        np.arange(n_groups, dtype=np.int32), logical_sizes)
    group_id_pad = np.concatenate([
        group_id, np.full((M - logical_M,), n_groups, dtype=np.int32)])
    active = np.arange(M) < logical_M

    rng = np.random.default_rng(260831)
    A = (rng.standard_normal((logical_M, 2 * logical_M))
         + 1j * rng.standard_normal((logical_M, 2 * logical_M)))
    G_logical = A @ A.conj().T
    G_host = np.zeros((M, M), dtype=np.complex128)
    G_host[:logical_M, :logical_M] = 0.5 * (
        G_logical + G_logical.conj().T)
    budget = 31

    ref = group_block_pivoted_cholesky_select(
        jnp.asarray(G_host), budget, jnp.asarray(group_id_pad),
        n_groups=n_groups, active_init=jnp.asarray(active))
    row = NamedSharding(mesh, P(("x", "y"), None))
    row1 = NamedSharding(mesh, P(("x", "y")))
    G = device_put_process_local(G_host, row)
    gid = device_put_process_local(group_id_pad, row1)
    active_dev = device_put_process_local(active, row1)
    step = make_sharded_group_block_pivoted_cholesky_select(
        mesh, M, budget, n_groups, mesh_axis=("x", "y"))
    got = step(G, gid, active_dev)
    jax.block_until_ready(got)

    piv_ref = np.asarray(ref[0])
    piv_got = np.asarray(got[0])
    if not np.array_equal(piv_ref, piv_got):
        _fail(f"pivot order differs: ref={piv_ref}, got={piv_got}")
    if int(ref[2]) != int(got[2]):
        _fail(f"rank differs: ref={int(ref[2])}, got={int(got[2])}")
    np.testing.assert_allclose(
        np.asarray(ref[4]), np.asarray(got[4]), rtol=2e-12, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(ref[5]), np.asarray(got[5]), rtol=2e-12, atol=2e-15)

    used = piv_got[piv_got >= 0]
    if np.any(used >= logical_M):
        _fail(f"inactive padding row selected: {used[used >= logical_M]}")
    picked = np.unique(group_id[used])
    counts = np.bincount(group_id[used], minlength=n_groups)
    if not np.array_equal(counts[picked], logical_sizes[picked]):
        _fail(
            "partial group selected: counts="
            f"{counts[picked].tolist()} sizes={logical_sizes[picked].tolist()}")
    if used.size > budget:
        _fail(f"delivered {used.size} points over budget {budget}")
    remaining = budget - used.size
    unpicked = np.setdiff1d(np.arange(n_groups), picked)
    if np.any(logical_sizes[unpicked] <= remaining):
        _fail(
            f"stopped with {remaining} points while a complete group fit")
    if int(got[2]) != used.size:
        _fail(f"full-rank fixture delivered {used.size}, rank={int(got[2])}")

    if jax.process_index() == 0:
        print(
            f"[centroid-group-block-p4] delivered={used.size}/{budget} "
            f"groups={picked.size}/{n_groups} rank={int(got[2])} "
            f"logical/padded={logical_M}/{M}",
            flush=True)
        print("[centroid-group-block-p4] PASS", flush=True)


if __name__ == "__main__":
    run_main_and_finalize(main)
